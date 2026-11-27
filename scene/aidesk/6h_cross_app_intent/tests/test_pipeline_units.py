import unittest

import tests  # noqa: F401
from src.orchestrator import build_model_input, resolve_window, make_sample_id
from src.validators import validate_generated, validate_final
from src.taxonomy import Taxonomy
from src.utils import load_yaml
from datetime import datetime
from zoneinfo import ZoneInfo
import random


def _related(n=15):
    return [{"event_id": f"e{i:03d}", "timestamp": f"2026-12-01T{9 + (i * 20) // 60:02d}:{(i * 20) % 60:02d}:00+08:00",
             "app": "hotel_app", "action": "search", "content": {"city": "成都"}, "relation": "supporting",
             "intent_ids": ["i001"]} for i in range(1, n + 1)]


def _noise(n=15):
    return [{"pool_id": f"p{i}", "timestamp": f"2026-12-01T{9 + (i * 25 + 7) // 60:02d}:{(i * 25 + 7) % 60:02d}:00+08:00",
             "app": "news_app", "action": "read_article", "content": {"title": "新闻"}, "hard_negative": False}
            for i in range(n)]


class RenumberTest(unittest.TestCase):
    def test_ids_follow_time_and_do_not_leak_origin(self):
        model_input, labeled, mapping = build_model_input("s", "2026-12-01T09:00:00+08:00", "2026-12-01T15:00:00+08:00",
                                                          _related(), _noise())
        ids = [e["event_id"] for e in model_input["events"]]
        self.assertEqual(ids, [f"e{i:03d}" for i in range(1, 31)])
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), 30)
        # the first 15 ids are no longer all "related"
        origins = [l["origin"] for l in labeled[:15]]
        self.assertIn("noise", origins)
        self.assertIn("related", origins)
        # mapping rewires generator ids to final ids
        first_final = mapping["e001"]
        self.assertIn(first_final, ids)
        self.assertEqual(next(l for l in labeled if l["event_id"] == first_final)["generator_event_id"], "e001")
        # model input carries no label fields
        for e in model_input["events"]:
            self.assertEqual(set(e), {"event_id", "timestamp", "app", "action", "content"})
        # internal labels keep generator relation for related events and unrelated for noise
        self.assertTrue(all(l["relation"] == "unrelated" for l in labeled if l["origin"] == "noise"))
        self.assertTrue(all(l["relation"] == "supporting" for l in labeled if l["origin"] == "related"))

    def test_sample_id_uses_template(self):
        a = make_sample_id({"template_id": "C01-S04"})
        b = make_sample_id({"template_id": "C01-S04"})
        self.assertTrue(a.startswith("c01-s04-"))
        self.assertNotEqual(a, b)


class WindowTest(unittest.TestCase):
    def test_fixed_window_honours_start(self):
        cfg = load_yaml("config/dataset_config.yaml")
        s, e = resolve_window({"window_start": "2026-12-01T09:00:00+08:00"}, cfg, random.Random(1))
        self.assertEqual(s, "2026-12-01T09:00:00+08:00")
        self.assertEqual(e, "2026-12-01T15:00:00+08:00")

    def test_jitter_keeps_six_hours_and_is_seeded(self):
        cfg = load_yaml("config/dataset_config.yaml")
        s1, e1 = resolve_window({"window_start": "2026-12-01T09:00:00+08:00", "jitter_window": True}, cfg, random.Random(3))
        s2, e2 = resolve_window({"window_start": "2026-12-01T09:00:00+08:00", "jitter_window": True}, cfg, random.Random(3))
        s3, _ = resolve_window({"window_start": "2026-12-01T09:00:00+08:00", "jitter_window": True}, cfg, random.Random(4))
        self.assertEqual((s1, e1), (s2, e2))
        self.assertNotEqual(s1, s3)
        self.assertEqual((datetime.fromisoformat(e1) - datetime.fromisoformat(s1)).total_seconds(), 6 * 3600)


class ValidateGeneratedTest(unittest.TestCase):
    def setUp(self):
        self.start, self.end = "2026-12-01T09:00:00+08:00", "2026-12-01T15:00:00+08:00"
        self.spec = {"intents": [{"intent_id": "i001", "intent_name": "成都出行"}]}

    def test_valid_batch_passes(self):
        self.assertEqual(validate_generated({"events": _related()}, self.start, self.end, spec=self.spec), [])

    def test_catches_count_style_vocab_and_leak(self):
        events = _related()
        events[0]["action"] = "搜索航班"
        events[1]["app"] = "brand_app"
        events[2]["content"] = {"城市": "成都"}
        events[3]["content"] = {"city": "intent 成都出行"}
        events[4]["timestamp"] = events[5]["timestamp"]
        events = events[:-1]  # 14 events
        errors = validate_generated({"events": events}, self.start, self.end, spec=self.spec,
                                    app_vocabulary={"hotel_app"})
        joined = "\n".join(errors)
        self.assertIn("related event count must be 15", joined)
        self.assertIn("must be snake_case English", joined)
        self.assertIn("not in app_vocabulary", joined)
        self.assertIn("content keys must be snake_case", joined)
        self.assertIn("leaked verbatim", joined)
        self.assertIn("less than 60s apart", joined)

    def test_unknown_intent_id(self):
        events = _related()
        events[0]["intent_ids"] = ["i999"]
        errors = validate_generated({"events": events}, self.start, self.end, spec=self.spec)
        self.assertTrue(any("not defined in scenario spec" in e for e in errors))


class ValidateFinalTest(unittest.TestCase):
    def setUp(self):
        self.taxonomy = Taxonomy.load()
        self.events = [{"event_id": f"e{i:03d}", "timestamp": f"2026-12-01T10:00:00+08:00", "app": "a", "action": "b", "content": {}}
                       for i in range(1, 31)]
        self.model_input = {"sample_id": "s", "events": self.events}
        sup = [f"e{i:03d}" for i in range(1, 9)]
        ri = [f"e{i:03d}" for i in range(9, 16)]
        unrelated = [f"e{i:03d}" for i in range(16, 31)]
        self.gold = {
            "sample_id": "s",
            "intents": [{"intent_id": "i001", "intent_name": "成都出行", "category": "travel", "purpose": "business",
                         "status": "planned", "confidence_level": "high", "supporting_event_ids": sup,
                         "related_interest_event_ids": ri, "contradicting_event_ids": [],
                         "confirmed_facts": [{"fact": "已支付机票", "source_event_ids": ["e001"]}],
                         "inferred_facts": [{"fact": "可能出差", "source_event_ids": ["e009"]}],
                         "unknown_information": ["同行人"], "conflicts": []}],
            "event_labels": [{"event_id": e, "relation": "supporting", "intent_ids": ["i001"]} for e in sup] +
                            [{"event_id": e, "relation": "related_interest", "intent_ids": ["i001"]} for e in ri] +
                            [{"event_id": e, "relation": "unrelated", "intent_ids": []} for e in unrelated],
            "unrelated_event_ids": unrelated,
            "global_observations": {"intent_count": 1, "unrelated_event_count": 15},
        }

    def _check(self, gold=None):
        return validate_final(self.model_input, gold or self.gold, self.taxonomy)

    def test_valid_gold_passes(self):
        res = self._check()
        self.assertEqual(res["errors"], [])
        self.assertEqual(res["warnings"], [])
        self.assertEqual(res["stats"]["unrelated"], 15)

    def test_mirror_inconsistency_detected(self):
        gold = __import__("copy").deepcopy(self.gold)
        gold["intents"][0]["supporting_event_ids"] = gold["intents"][0]["supporting_event_ids"][:7]
        errors = self._check(gold)["errors"]
        self.assertTrue(any("inconsistent with event_labels" in e for e in errors))

    def test_taxonomy_violations(self):
        gold = __import__("copy").deepcopy(self.gold)
        gold["intents"][0]["purpose"] = "holiday"
        gold["intents"][0]["status"] = "maybe"
        errors = self._check(gold)["errors"]
        self.assertTrue(any("not allowed for 'travel'" in e for e in errors))
        self.assertTrue(any("status 'maybe'" in e for e in errors))

    def test_fact_reference_rules(self):
        gold = __import__("copy").deepcopy(self.gold)
        gold["intents"][0]["confirmed_facts"][0]["source_event_ids"] = ["e009"]  # related_interest only
        gold["intents"][0]["inferred_facts"].append({"fact": "x", "source_event_ids": ["e020"]})  # unrelated
        errors = self._check(gold)["errors"]
        self.assertTrue(any("cites events not related" in e for e in errors))
        self.assertEqual(sum("cites events not related" in e for e in errors), 2)

    def test_unrelated_mismatch_and_bounds(self):
        gold = __import__("copy").deepcopy(self.gold)
        gold["unrelated_event_ids"] = gold["unrelated_event_ids"][:-1]
        errors = self._check(gold)["errors"]
        self.assertTrue(any("unrelated_event_ids inconsistent" in e for e in errors))

    def test_composition_deviation_is_warning_not_error(self):
        # adjudicator relabels 4 noise events as related_interest → 11 unrelated
        gold = __import__("copy").deepcopy(self.gold)
        moved = ["e016", "e017", "e018", "e019"]
        for lab in gold["event_labels"]:
            if lab["event_id"] in moved:
                lab.update(relation="related_interest", intent_ids=["i001"])
        gold["intents"][0]["related_interest_event_ids"] = sorted(gold["intents"][0]["related_interest_event_ids"] + moved)
        gold["unrelated_event_ids"] = [e for e in gold["unrelated_event_ids"] if e not in moved]
        gold["global_observations"]["unrelated_event_count"] = 11
        res = self._check(gold)
        self.assertEqual(res["errors"], [])
        self.assertTrue(any("differs from target" in w for w in res["warnings"]))


if __name__ == "__main__":
    unittest.main()

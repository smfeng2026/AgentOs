import copy
import json
import unittest

import tests  # noqa: F401
from src import evaluator
from src.evaluator import evaluate, evaluate_sample, match_intents, Aggregator


def _gold(n_intents=1, purpose="business"):
    events = [{"event_id": f"e{i:03d}", "timestamp": f"2026-12-01T10:00:00+08:00", "app": "a", "action": "b", "content": {"k": i}}
              for i in range(1, 31)]
    intents, labels = [], []
    for k in range(n_intents):
        sup = [f"e{3 * k + 1:03d}", f"e{3 * k + 2:03d}"]
        ri = [f"e{3 * k + 3:03d}"]
        intents.append({"intent_id": f"i00{k + 1}", "intent_name": f"意图{k}", "category": "travel", "purpose": purpose,
                        "status": "planned", "confidence_level": "high", "supporting_event_ids": sup,
                        "related_interest_event_ids": ri, "contradicting_event_ids": [],
                        "confirmed_facts": [{"fact": "f", "source_event_ids": sup[:1]}], "inferred_facts": [],
                        "unknown_information": [], "conflicts": []})
        labels += [{"event_id": e, "relation": "supporting", "intent_ids": [f"i00{k + 1}"]} for e in sup]
        labels += [{"event_id": e, "relation": "related_interest", "intent_ids": [f"i00{k + 1}"]} for e in ri]
    rest = [e["event_id"] for e in events if e["event_id"] not in {l["event_id"] for l in labels}]
    labels += [{"event_id": e, "relation": "unrelated", "intent_ids": []} for e in rest]
    gold = {"intents": intents, "event_labels": labels, "unrelated_event_ids": rest,
            "global_observations": {"intent_count": n_intents, "unrelated_event_count": len(rest)}}
    return {"sample_id": "s", "model_input": {"sample_id": "s", "events": events}, "gold_label": gold,
            "meta": {"category": "travel", "difficulty": "D2_multi_app_fusion", "template_id": "C01-S04"}}


def _pred(gold, **changes):
    p = copy.deepcopy(gold["gold_label"])
    p["intents"][0].update(changes)
    return p


class MatchingTest(unittest.TestCase):
    def test_match_is_by_evidence_overlap_not_name(self):
        g = _gold()["gold_label"]["intents"]
        p = copy.deepcopy(g)
        p[0]["intent_id"] = "zzz"
        p[0]["intent_name"] = "完全不同的名字"
        self.assertEqual(match_intents(g, p), [(0, 0, 1.0)])


class EvaluateSampleTest(unittest.TestCase):
    def test_oracle_scores_perfect(self):
        row = _gold()
        s = evaluate_sample(row["gold_label"], copy.deepcopy(row["gold_label"]), [e["event_id"] for e in row["model_input"]["events"]])
        agg = Aggregator(); agg.add(s)
        m = agg.metrics()
        self.assertEqual(m["intent"]["f1"], 1.0)
        self.assertEqual(m["relation_macro_f1"], 1.0)
        self.assertEqual(m["purpose_accuracy"], 1.0)
        self.assertEqual(m["intent_count_accuracy"], 1.0)
        self.assertEqual(m["fact_grounding_precision"], 1.0)

    def test_all_unrelated_baseline_fails_on_intents(self):
        row = _gold()
        pred = {"intents": [], "event_labels": [{"event_id": e["event_id"], "relation": "unrelated", "intent_ids": []}
                                                 for e in row["model_input"]["events"]],
                "unrelated_event_ids": [e["event_id"] for e in row["model_input"]["events"]],
                "global_observations": {"intent_count": 0, "unrelated_event_count": 30}}
        m = evaluate([row], [{"sample_id": "s", "prediction": pred}])["overall"]
        self.assertEqual(m["intent"]["recall"], 0.0)
        self.assertEqual(m["intent"]["f1"], 0.0)
        self.assertLess(m["relation_macro_f1"], 0.5)

    def test_purpose_over_inference_counted(self):
        row = _gold(purpose="unknown")
        pred = _pred(row, purpose="business")
        m = evaluate([row], [{"sample_id": "s", "prediction": pred}])["overall"]
        self.assertEqual(m["purpose_over_inference_rate"], 1.0)
        self.assertEqual(m["purpose_under_inference_rate"], None)      # no known-purpose gold pair
        self.assertEqual(m["purpose_accuracy"], 0.0)

    def test_purpose_under_inference_counted(self):
        row = _gold(purpose="business")
        pred = _pred(row, purpose="unknown")
        m = evaluate([row], [{"sample_id": "s", "prediction": pred}])["overall"]
        self.assertEqual(m["purpose_under_inference_rate"], 1.0)
        self.assertEqual(m["purpose_over_inference_rate"], None)       # no unknown-purpose gold pair

    def test_merge_and_split(self):
        g2 = _gold(n_intents=2)
        merged = copy.deepcopy(g2["gold_label"])
        merged["intents"][0]["supporting_event_ids"] += merged["intents"][1]["supporting_event_ids"]
        merged["intents"][0]["related_interest_event_ids"] += merged["intents"][1]["related_interest_event_ids"]
        merged["intents"] = [merged["intents"][0]]
        for lab in merged["event_labels"]:
            lab["intent_ids"] = [x if x != "i002" else "i001" for x in lab["intent_ids"]]
        merged["global_observations"]["intent_count"] = 1
        m = evaluate([g2], [{"sample_id": "s", "prediction": merged}])["overall"]
        self.assertEqual(m["multi_intent"]["merge_error_rate"], 0.5)
        self.assertLess(m["intent"]["recall"], 1.0)
        self.assertEqual(m["intent_count_accuracy"], 0.0)

        g1 = _gold(n_intents=1)
        split = copy.deepcopy(g1["gold_label"])
        second = copy.deepcopy(split["intents"][0]); second["intent_id"] = "i002"
        second["supporting_event_ids"], second["related_interest_event_ids"] = ["e002"], []
        split["intents"][0]["supporting_event_ids"] = ["e001"]
        split["intents"].append(second)
        for lab in split["event_labels"]:
            if lab["event_id"] == "e002":
                lab["intent_ids"] = ["i002"]
        split["global_observations"]["intent_count"] = 2
        m = evaluate([g1], [{"sample_id": "s", "prediction": split}])["overall"]
        self.assertEqual(m["multi_intent"]["split_error_rate"], 0.5)

    def test_missing_intent_marks_format_error(self):
        row = _gold()
        res = evaluate([row], [{"sample_id": "s", "prediction": {"event_labels": []}}])
        self.assertEqual(len(res["format_errors"]), 1)
        self.assertEqual(res["meta"]["n_missing_predictions"], 0)
        self.assertEqual(res["meta"]["format_valid_rate"], 0.0)
        res = evaluate([row], [])
        self.assertEqual(res["meta"]["n_missing_predictions"], 1)


if __name__ == "__main__":
    unittest.main()

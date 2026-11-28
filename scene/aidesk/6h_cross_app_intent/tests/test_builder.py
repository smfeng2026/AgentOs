import json
import unittest

import tests  # noqa: F401
from tests.helpers import DataTestCase, make_sample, persist, sid
from src import dataset_builder, manifest as manifest_store
from src.utils import load_jsonl, load_json, load_yaml
from src.taxonomy import Taxonomy

TEMPLATES = {"C01-S04": "travel", "C01-S05": "travel", "C02-S01": "work", "C02-S02": "work", "C03-S01": "shopping",
             "C03-S02": "shopping", "C04-S01": "social_dining", "C05-S01": "health_service", "C06-S01": "learning",
             "C07-S01": "entertainment", "C08-S01": "family", "C09-S01": "finance_admin", "C10-S01": "fitness",
             "C11-S01": "local_transport"}

class ApprovalGateTest(DataTestCase):
    def test_only_approved_by_default(self):
        a = persist(make_sample(sid("gate-a"), approval="approved", tag="a"))
        persist(make_sample(sid("gate-b"), approval="pending", tag="b"))
        persist(make_sample(sid("gate-c"), approval="rejected", tag="c"))
        summary = dataset_builder.build(out_dir="data/splits_gate", force=True)
        self.assertEqual(summary["n_records"], 1)
        self.assertEqual(a["sample_id"], next(iter(summary["assignment"])))
        self.assertEqual(sorted(x["reason"] for x in summary["skipped"]), ["not_approved"] * 2)
        summary2 = dataset_builder.build(out_dir="data/splits_gate2", allow_unapproved=True, force=True)
        self.assertEqual(summary2["n_records"], 3)
        self.assertEqual(summary2["unapproved_included"], 2)
        self.assertIn("未经人工批准", dataset_builder.render_markdown(summary2))

    def test_invalid_gold_is_skipped_even_if_approved(self):
        s = make_sample(sid("gate-bad"), approval="approved", tag="bad")
        s["gold"]["intents"][0]["purpose"] = "not-a-purpose"
        persist(s)
        summary = dataset_builder.build(out_dir="data/splits_gate3", force=True)
        self.assertEqual(summary["n_records"], 0)
        self.assertEqual(summary["skipped"][0]["reason"], "invalid_gold")


class SplitIsolationTest(DataTestCase):
    def test_whole_template_group_stays_together_and_entities_connect_groups(self):
        names = [(tpl, cat) for tpl, cat in TEMPLATES.items()]
        names += [("C01-S04", "travel")] * 3                      # 4 samples of one template
        names += [("C02-S01", "work")]                             # shares entity with nothing → own component
        for i, (tpl, cat) in enumerate(names):
            entities = ("成都",) if tpl in ("C01-S04", "C02-S01") else (f"实体{tpl}",)
            persist(make_sample(sid(f"iso-{i}-{tpl}-{entities[0]}"), category=cat, template_id=tpl,
                                entities=entities, tag=f"{tpl}{i}"))
        summary = dataset_builder.build(out_dir="data/splits_iso", force=True)
        assignment = summary["assignment"]
        splits_of_c01 = {v["split"] for v in assignment.values() if v["group"] == "C01-S04"}
        self.assertEqual(len(splits_of_c01), 1, "samples of one template must share a split")
        # C01-S04 and C02-S01 both contain destination 成都 → same component → same base split
        base = lambda g: {v["split"] for v in assignment.values() if v["group"] == g}  # noqa: E731
        self.assertEqual(base("C01-S04"), base("C02-S01"))
        self.assertEqual(summary["split_info"]["errors"], [])
        # every split that exists has samples from more than one category
        self.assertGreater(len(summary["stats"]["by_split_category"]["train"]), 1)
        for split in ("validation", "test"):
            self.assertTrue(summary["stats"]["by_split"].get(split), f"{split} is empty")
        total = sum(summary["stats"]["by_split"].values())
        self.assertEqual(total, len(names))

    def test_explicit_conflicting_split_is_an_error(self):
        persist(make_sample(sid("conf-1"), template_id="C06-S01", category="learning", split="train", entities=("甲",)))
        persist(make_sample(sid("conf-2"), template_id="C06-S01", category="learning", split="test", entities=("乙",)))
        with self.assertRaises(SystemExit):
            dataset_builder.build(out_dir="data/splits_conf")
        summary = dataset_builder.build(out_dir="data/splits_conf", force=True)
        self.assertTrue(any("train/validation and test" in e or "conflicting explicit" in e for e in summary["split_info"]["errors"]))

    def test_challenge_difficulty_moves_test_to_challenge(self):
        persist(make_sample(sid("chal-1"), template_id="C12-M01", category="travel", split="test",
                            difficulty="D5_multi_intent", entities=("丙",)))
        summary = dataset_builder.build(out_dir="data/splits_chal", force=True)
        self.assertEqual(summary["assignment"][sid("chal-1")]["split"], "challenge_test")
        self.assertTrue((tests.TEST_ROOT / "data/splits_chal/challenge_test.jsonl").exists())

    def test_leakage_report_flags_duplicates_and_shared_entities(self):
        s1 = make_sample(sid("leak-1"), template_id="C07-S01", category="entertainment", entities=("三亚",), tag="x")
        s2 = make_sample(sid("leak-2"), template_id="C08-S01", category="family", entities=("三亚",), tag="x")
        for s in (s1, s2):
            s["model_input"]["events"] = [dict(e) for e in s1["model_input"]["events"]]
        s1["split"], s2["split"] = "train", "test"
        s1["group_key"], s2["group_key"] = "g1", "g2"
        s1["manifest"] = {"task": {}, "spec_summary": {}}
        s2["manifest"] = s1["manifest"]
        report = dataset_builder.leakage_report([s1, s2])
        self.assertTrue(any("exact duplicate input across train/test" in e for e in report["errors"]))
        self.assertTrue(any("strong entities appear in both train and test" in w for w in report["warnings"]))


class ExportFormatTest(DataTestCase):
    def test_sft_and_test_rows(self):
        persist(make_sample(sid("fmt-1"), template_id="C12-M01", category="travel", split="train", entities=("丁",)))
        persist(make_sample(sid("fmt-2"), template_id="C12-M02", category="travel", split="test", entities=("戊",)))
        summary = dataset_builder.build(out_dir="data/splits_fmt", force=True, include_meta=True)
        self.assertEqual(summary["split_info"]["errors"], [])
        train = load_jsonl("data/splits_fmt/train.jsonl")
        test = load_jsonl("data/splits_fmt/test.jsonl")
        self.assertEqual([r["sample_id"] for r in test], [sid("fmt-2")])
        row = train[0]
        self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])
        user = json.loads(row["messages"][1]["content"])
        target = json.loads(row["messages"][2]["content"])
        self.assertEqual(user["sample_id"], sid("fmt-1"))
        self.assertEqual(set(user["events"][0]), {"event_id", "timestamp", "app", "action", "content"})
        self.assertNotIn("sample_id", target)
        self.assertEqual(list(target)[:4], ["intents", "event_labels", "unrelated_event_ids", "global_observations"])
        self.assertEqual(row["meta"]["split"], "train")
        self.assertEqual(test[0]["gold_label"]["global_observations"]["unrelated_event_count"], 15)
        self.assertEqual(test[0]["meta"]["template_id"], "C12-M02")
        self.assertTrue((tests.TEST_ROOT / "data/splits_fmt/DATASET_REPORT.md").exists())
        self.assertTrue((tests.TEST_ROOT / "data/splits_fmt/split_manifest.json").exists())
    def test_fixed_plan_does_not_reallocate_existing_groups(self):
        persist(make_sample(sid("fixed-a"), template_id="C01-S04", category="travel", entities=("甲",)))
        cfg = load_yaml("config/dataset_config.yaml")
        taxonomy = Taxonomy.load()
        templates = {t["template_id"]: t for t in load_json("config/templates.json")["templates"]}
        records, _ = dataset_builder.load_records(cfg, taxonomy)
        first = dataset_builder.assign_splits(records, templates, cfg)["group_split"]
        persist(make_sample(sid("fixed-b"), template_id="C02-S01", category="work", entities=("乙",)))
        persist(make_sample(sid("fixed-c"), template_id="C03-S01", category="shopping", entities=("丙",)))
        records, _ = dataset_builder.load_records(cfg, taxonomy)
        second = dataset_builder.assign_splits(records, templates, cfg)["group_split"]
        self.assertEqual(first["C01-S04"], second["C01-S04"])
        self.assertEqual(second["C01-S04"], load_json("config/split_plan.json")["assignments"]["C01-S04"])

    def test_entity_link_across_fixed_splits_is_an_error(self):
        plan = load_json("config/split_plan.json")["assignments"]
        self.assertNotEqual(plan["C12-M01"], plan["C12-M02"])
        persist(make_sample(sid("fixed-x"), template_id="C12-M01", category="travel", entities=("相同目的地",)))
        persist(make_sample(sid("fixed-y"), template_id="C12-M02", category="travel", entities=("相同目的地",)))
        cfg = load_yaml("config/dataset_config.yaml")
        records, _ = dataset_builder.load_records(cfg, Taxonomy.load())
        templates = {t["template_id"]: t for t in load_json("config/templates.json")["templates"]}
        result = dataset_builder.assign_splits(records, templates, cfg)
        self.assertTrue(any("crosses planned splits" in e for e in result["errors"]))

    def test_release_enforces_coverage_and_sft_uses_shared_policy(self):
        persist(make_sample(sid("release-small"), template_id="C01-S04", category="travel", entities=("丁",)))
        with self.assertRaises(SystemExit):
            dataset_builder.build(out_dir="data/splits_release", release=True)
        summary = dataset_builder.build(out_dir="data/splits_release", release=True, force=True, include_meta=True)
        self.assertTrue(summary["coverage"]["errors"])
        train = load_jsonl("data/splits_release/train.jsonl")
        self.assertIn("D6 状态变化的唯一口径", train[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()

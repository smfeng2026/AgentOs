import copy
import unittest

import tests  # noqa: F401
from tests.helpers import DataTestCase, _purpose, make_sample, persist, sid
from src import dataset_report


class ReportTest(DataTestCase):
    def test_report_sections_and_agreement(self):
        good = make_sample(sid("rep-good"), template_id="C01-S04", category="travel", approval="approved")
        good["manifest_flags"] = []
        persist(good)
        noisy = make_sample(sid("rep-noisy"), template_id="C03-S01", category="shopping", approval="pending", tag="n")
        # the adjudicator relabels 2 injected noise events as related_interest (still consistent → valid)
        moved = noisy["gold"]["unrelated_event_ids"][:2]
        for lab in noisy["gold"]["event_labels"]:
            if lab["event_id"] in moved:
                lab.update(relation="related_interest", intent_ids=["i001"])
        noisy["gold"]["intents"][0]["related_interest_event_ids"] += moved
        noisy["gold"]["unrelated_event_ids"] = [e for e in noisy["gold"]["unrelated_event_ids"] if e not in moved]
        noisy["gold"]["global_observations"]["unrelated_event_count"] -= 2
        noisy["gold"]["intents"][0]["purpose"] = "unknown"
        # blind annotation keeps the evidence but infers a purpose → purpose over-inference
        blind = copy.deepcopy(noisy["gold"])
        blind["intents"][0]["purpose"] = _purpose("shopping")
        noisy["audit"]["blind_annotation"] = blind
        noisy["audit"]["review"] = {"passed": False, "issues": [{"severity": "medium", "issue_type": "over_split_intent",
                                                                 "description": "x"}]}
        persist(noisy)
        # mark manifest flags/usage the report aggregates
        m = dataset_report.manifest_store.load_manifest(sid("rep-noisy"))
        m["flags"] = ["composition_deviation", "blind_low_agreement"]
        m["attempts"] = 2
        m["usage"] = {"generator": {"calls": 2, "total_tokens": 1000}, "adjudicator": {"calls": 1, "total_tokens": 500}}
        m["elapsed_s"] = 12.5
        dataset_report.manifest_store.write_manifest(m)
        m2 = dataset_report.manifest_store.load_manifest(sid("rep-good"))
        m2["usage"] = {"generator": {"calls": 1, "total_tokens": 500}}
        m2["elapsed_s"] = 7.5
        dataset_report.manifest_store.write_manifest(m2)

        r = dataset_report.build_report()
        self.assertEqual(r["n_loaded"], 2)
        self.assertEqual(r["overview"]["approval"], {"approved": 1, "pending": 1})
        self.assertEqual(r["overview"]["flags"], {"composition_deviation": 1, "blind_low_agreement": 1})
        self.assertEqual(r["overview"]["tokens_per_sample"]["mean"], 1000)
        self.assertEqual(r["agreement"]["blind_vs_gold"]["intent"]["f1"], 1.0)        # same intent, same evidence
        self.assertEqual(r["agreement"]["blind_vs_gold"]["purpose_over_inference_rate"], 1.0)
        self.assertEqual(r["agreement"]["intended_vs_gold"]["intent"]["f1"], 1.0)
        # the 2 relabelled noise events are counted against the pool
        self.assertEqual(r["noise"]["noise_events_total"], 30)
        self.assertAlmostEqual(r["noise"]["noise_relabel_rate"], 2 / 30, places=3)
        self.assertEqual(r["composition"]["samples_with_composition_deviation"], 1)
        self.assertEqual(r["reviewer"]["passed"], {"True": 1, "False": 1})
        md = dataset_report.render_markdown(r)
        for needle in ("与 Gold 的一致性", "噪声池", "捷径风险", "审查员"):
            self.assertIn(needle, md)
        self.assertIn("没有可加载的样本", dataset_report.render_markdown({"n_loaded": 0, "n_manifests": 0}))


if __name__ == "__main__":
    unittest.main()

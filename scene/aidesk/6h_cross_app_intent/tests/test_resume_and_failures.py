import argparse
import copy
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import tests  # noqa: F401
from tests.helpers import DataTestCase
from src import cli, manifest as manifest_store
from src.noise import load_pool
from src.orchestrator import run_one
from src.taxonomy import Taxonomy
from src.utils import load_yaml, save_json, load_json


class _LLM:
    model = "test-model"
    host = "test.invalid"


class _Planner:
    def __init__(self, *args):
        pass

    def run(self, payload):
        return {
            "title": "失败路径测试", "scenario_category": "travel", "difficulty": payload["difficulty"],
            "key_entities": {"home_city": "北京", "destinations": ["成都"], "products": []},
            "intents": [{"intent_id": "i001", "intent_name": "测试出行", "category": "travel",
                         "purpose": "unknown", "status": "emerging", "confidence_level": "low"}],
        }


class _InvalidGenerator:
    def __init__(self, *args):
        pass

    def run(self, spec, task):
        return {"events": [], "noise_events": []}


class GenerationFailureTest(DataTestCase):
    def test_exhausted_generation_is_persisted_and_cannot_be_approved(self):
        cfg = load_yaml("config/dataset_config.yaml")
        cfg["max_regeneration_attempts"] = 1
        task = {"scenario_category": "travel", "template_id": "C01-S01", "difficulty": "D3_weak_evidence",
                "window_start": "2026-12-01T09:00:00+08:00", "fixed_window": True,
                "task_instance_id": "failure-instance", "run_id": "test"}
        with patch("src.orchestrator.PlannerAgent", _Planner), \
                patch("src.orchestrator.GeneratorAgent", _InvalidGenerator):
            result = run_one(_LLM(), _LLM(), task, cfg, Taxonomy.load(), load_pool())
        self.assertEqual(result["status"], "NEEDS_HUMAN_REVIEW")
        self.assertEqual(result["run_state"], "failed")
        self.assertEqual(result["attempts"], 2)
        manifest = manifest_store.load_manifest(result["sample_id"])
        self.assertEqual(manifest["run_state"], "failed")
        audit = load_json(manifest["artefacts"]["audit_report"]["path"])
        self.assertIsNone(audit["preset_relation_agreement"])
        self.assertTrue(all(a["stage"] == "generation" for a in audit["attempts"]))
        self.assertIn("generation failed after retries", manifest["validation"]["errors"][0])
        with self.assertRaises(ValueError):
            manifest_store.set_approval(result["sample_id"], "approved", "tester")


class _AlignedGenerator:
    def __init__(self, *args):
        pass

    def run(self, spec, task):
        related, noise = [], []
        for i in range(1, 16):
            related.append({
                "event_id": f"e{i:03d}", "timestamp": f"2026-12-01T{9 + (i * 15) // 60:02d}:{(i * 15) % 60:02d}:00+08:00",
                "app": "hotel_app", "action": "search",
                "content": {"city": "成都", "detail": f"目标场景的详细搜索条件{i}", "kind": "target"},
                "intent_ids": ["i001"], "relation": "related_interest",
            })
            noise.append({
                "event_id": f"n{i:03d}", "app": "hotel_app", "action": "search",
                "content": {"city": "本地", "detail": f"日常住宿之外的无关搜索记录{i}", "kind": "background"},
                "hard_negative": True, "surface_match_event_id": f"e{i:03d}",
            })
        return {"events": related, "noise_events": noise}


def _blind_from_input(model_input):
    related = [e["event_id"] for e in model_input["events"] if e["content"].get("kind") == "target"]
    unrelated = [e["event_id"] for e in model_input["events"] if e["event_id"] not in related]
    return {
        "intents": [{"intent_id": "i001", "intent_name": "测试出行", "category": "travel", "purpose": "unknown",
                     "status": "emerging", "confidence_level": "low", "supporting_event_ids": [],
                     "related_interest_event_ids": related, "contradicting_event_ids": [],
                     "confirmed_facts": [], "inferred_facts": [], "unknown_information": ["同行人"], "conflicts": []}],
        "event_labels": [{"event_id": e["event_id"],
                          "relation": "related_interest" if e["event_id"] in related else "unrelated",
                          "intent_ids": ["i001"] if e["event_id"] in related else []}
                         for e in model_input["events"]],
        "unrelated_event_ids": unrelated,
        "global_observations": {"intent_count": 1, "unrelated_event_count": len(unrelated)},
    }


class _BlindAnnotator:
    def __init__(self, *args):
        pass

    def run(self, model_input, taxonomy=None):
        return _blind_from_input(model_input)


class _Adjudicator:
    def __init__(self, *args):
        pass

    def run(self, spec, labeled, model_input, blind, review, taxonomy=None):
        return {"decision": "accept", "gold_label": copy.deepcopy(blind), "rationale": "ok",
                "requires_regeneration": False}


class _NoReviewer:
    def __init__(self, *args):
        pass

    def run(self, *args):
        raise AssertionError("reviewer should be skipped for identical preset/blind labels")


class AlignedPipelineTest(DataTestCase):
    def test_aligned_noise_runs_through_complete_pipeline(self):
        cfg = load_yaml("config/dataset_config.yaml")
        cfg["max_regeneration_attempts"] = 0
        task = {"scenario_category": "travel", "template_id": "C01-S01", "difficulty": "D3_weak_evidence",
                "window_start": "2026-12-01T09:00:00+08:00", "fixed_window": True,
                "task_instance_id": "aligned-instance", "run_id": "test"}
        with patch("src.orchestrator.PlannerAgent", _Planner), \
                patch("src.orchestrator.GeneratorAgent", _AlignedGenerator), \
                patch("src.orchestrator.AnnotatorAgent", _BlindAnnotator), \
                patch("src.orchestrator.ReviewerAgent", _NoReviewer), \
                patch("src.orchestrator.AdjudicatorAgent", _Adjudicator):
            result = run_one(_LLM(), _LLM(), task, cfg, Taxonomy.load(), load_pool())
        self.assertEqual(result["status"], "HUMAN_APPROVAL_PENDING")
        self.assertEqual(result["run_state"], "completed")
        self.assertIn("reviewer_skipped", result["flags"])
        manifest = manifest_store.load_manifest(result["sample_id"])
        self.assertEqual(manifest["noise"]["mode"], "generated_aligned")
        self.assertEqual(manifest["noise"]["surface_alignment"]["paired_fraction"], 1.0)
        model_input = load_json(manifest["artefacts"]["model_input"]["path"])
        internal = load_json(manifest["artefacts"]["internal_labels"]["path"])
        self.assertEqual(len(model_input["events"]), 30)
        noise = next(e for e in internal["events"] if e["origin"] == "noise")
        self.assertIsNotNone(noise["surface_match_event_id"])


class ResumeIdentityTest(DataTestCase):
    def _args(self, resume=False):
        return argparse.Namespace(
            tasks_file="config/resume_tasks.json", template_id=None, category="travel", count=1,
            difficulty=None, intent_count=None, split=None, timezone="Asia/Shanghai",
            start="2026-12-01T09:00:00+08:00", fixed_window=True, jitter_window=False,
            run_id="resume-test", resume=resume, workers=1,
        )

    def setUp(self):
        super().setUp()
        save_json("config/resume_tasks.json", [{"template_id": "C01-S01", "count": 2},
                                                {"template_id": "C02-S01", "count": 1}])

    def test_ids_are_stable_and_order_independent(self):
        args = self._args()
        templates = cli.load_templates()
        first = cli.expand_tasks(args, templates)
        second = cli.expand_tasks(args, templates)
        self.assertEqual([t["task_instance_id"] for t in first], [t["task_instance_id"] for t in second])
        self.assertEqual(len({t["task_instance_id"] for t in first}), 3)
        reordered = copy.deepcopy(args)
        save_json("config/resume_tasks.json", [{"template_id": "C02-S01", "count": 1},
                                                {"template_id": "C01-S01", "count": 2}])
        third = cli.expand_tasks(reordered, templates)
        self.assertEqual({t["task_instance_id"] for t in first}, {t["task_instance_id"] for t in third})

    def test_resume_skips_completed_without_constructing_llm(self):
        args = self._args(resume=True)
        task = cli.expand_tasks(args, cli.load_templates())[0]
        manifest_store.write_manifest({
            "sample_id": "c01-s01-complete", "task": task, "run_state": "completed",
            "status": "HUMAN_APPROVAL_PENDING", "approval": {"state": "pending"},
        })
        with patch("src.cli.OpenAICompatibleClient.from_env", side_effect=AssertionError("must not construct LLM")):
            # Two instances remain, so this resume invocation does construct clients; use an all-complete file.
            save_json("config/resume_tasks.json", [{"template_id": "C01-S01", "count": 1}])
            args = self._args(resume=True)
            with redirect_stdout(io.StringIO()):
                result = cli.cmd_generate(args)
        self.assertEqual(result, 0)

    def test_rerun_without_resume_rejects_existing_instances(self):
        args = self._args()
        task = cli.expand_tasks(args, cli.load_templates())[0]
        manifest_store.write_manifest({"sample_id": "existing", "task": task, "run_state": "completed",
                                       "status": "HUMAN_APPROVAL_PENDING"})
        with self.assertRaises(SystemExit):
            cli.cmd_generate(args)


if __name__ == "__main__":
    unittest.main()

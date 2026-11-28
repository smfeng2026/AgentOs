import json
import tempfile
import unittest
from pathlib import Path

from src import runlog


class TestNaming(unittest.TestCase):
    def test_sanitize(self):
        self.assertEqual(runlog.sanitize("qwen3-0.6b__sft/pilot v1"), "qwen3-0.6b__sft-pilot-v1")

    def test_run_id(self):
        rid = runlog.make_run_id("qwen3-0.6b__official", "bfloat16", "pilot_v1", day="2026-11-28")
        self.assertEqual(rid, "2026-11-28__qwen3-0.6b__official__bfloat16__pilot_v1")


class TestRunLifecycle(unittest.TestCase):
    def setUp(self):
        self._orig = runlog.MODEL_EVAL_ROOT
        self.tmp = Path(tempfile.mkdtemp())
        runlog.MODEL_EVAL_ROOT = self.tmp
        (self.tmp / "runs").mkdir()

    def tearDown(self):
        runlog.MODEL_EVAL_ROOT = self._orig

    def test_create_finalize(self):
        d = runlog.create_run("r1", {"artifact": {"artifact_id": "a__official"}})
        rec = json.loads((d / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["status"], "running")
        self.assertIn("hardware", rec)
        runlog.finalize_run(d, "completed", {"n_samples": 12})
        rec = json.loads((d / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["status"], "completed")
        self.assertEqual(rec["n_samples"], 12)
        self.assertIn("finished_at", rec)

    def test_no_overwrite(self):
        runlog.create_run("r1", {})
        with self.assertRaises(FileExistsError):
            runlog.create_run("r1", {})


if __name__ == "__main__":
    unittest.main()
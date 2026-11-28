import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src import cli


class TestSnapshot(unittest.TestCase):
    def setUp(self):
        self._orig = cli.MODEL_EVAL_ROOT
        self.tmp = Path(tempfile.mkdtemp())
        cli.MODEL_EVAL_ROOT = self.tmp
        (self.tmp / "datasets").mkdir()
        self.src = self.tmp / "splits"
        self.src.mkdir()
        (self.src / "test.jsonl").write_text('{"sample_id": "a", "model_input": {}, "gold_label": {}}\n', encoding="utf-8")
        (self.src / "challenge_test.jsonl").write_text('{"sample_id": "b"}\n', encoding="utf-8")
        (self.src / "split_manifest.json").write_text(
            json.dumps({"builder_version": "1.1.0", "generated_at": "x", "allow_unapproved": True, "n_records": 2}),
            encoding="utf-8")
        self.args = argparse.Namespace(name="t1", source=str(self.src),
                                       files=["test.jsonl", "challenge_test.jsonl"], force=False)

    def tearDown(self):
        cli.MODEL_EVAL_ROOT = self._orig

    def test_snapshot_freeze(self):
        self.assertEqual(cli.cmd_snapshot(self.args), 0)
        d = self.tmp / "datasets" / "t1"
        meta = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
        expect = hashlib.sha256((self.src / "test.jsonl").read_bytes()).hexdigest()
        self.assertEqual(meta["files"]["test.jsonl"]["sha256"], expect)
        self.assertTrue(meta["provenance"]["allow_unapproved"])
        source_md = (d / "SOURCE.md").read_text(encoding="utf-8")
        self.assertIn("尚未人工批准", source_md)
        self.assertIn(expect, source_md)
        # 源文件后续变化不影响快照
        (self.src / "test.jsonl").write_text("changed\n", encoding="utf-8")
        self.assertEqual(hashlib.sha256((d / "test.jsonl").read_bytes()).hexdigest(), expect)

    def test_no_overwrite_without_force(self):
        self.assertEqual(cli.cmd_snapshot(self.args), 0)
        self.assertEqual(cli.cmd_snapshot(self.args), 1)
        self.args.force = True
        self.assertEqual(cli.cmd_snapshot(self.args), 0)


if __name__ == "__main__":
    unittest.main()
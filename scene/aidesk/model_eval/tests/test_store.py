import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src import store


class StoreTestBase(unittest.TestCase):
    def setUp(self):
        self._orig = store.STORE_ROOT
        self.tmp = Path(tempfile.mkdtemp())
        store.STORE_ROOT = self.tmp / "model_store"
        (store.STORE_ROOT / "base" / "demo-0.5b").mkdir(parents=True)
        (store.STORE_ROOT / "derived").mkdir(parents=True)
        cfg_dir = self.tmp / "model_eval" / "configs" / "bases"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "demo-0.5b.yaml").write_text(
            "base_id: demo-0.5b\nrepo_id: org/demo-0.5b\ntrust_remote_code: false\ntorch_dtype: bfloat16\n",
            encoding="utf-8")
        self.up = {"base_id": "demo-0.5b", "repo": "org/demo-0.5b", "revision": None, "status": "not_downloaded"}
        (store.STORE_ROOT / "base" / "demo-0.5b" / "upstream.json").write_text(json.dumps(self.up), encoding="utf-8")

    def tearDown(self):
        store.STORE_ROOT = self._orig

    def make_derived(self, artifact_id="demo-0.5b__sft-t1__lora-r16", with_file=True):
        d = store.STORE_ROOT / "derived" / artifact_id
        d.mkdir(parents=True)
        files = []
        if with_file:
            p = d / "adapter_model.safetensors"
            p.write_bytes(b"fake-weights")
            files.append({"path": "adapter_model.safetensors",
                          "sha256": hashlib.sha256(b"fake-weights").hexdigest()})
        m = {"artifact_id": artifact_id, "kind": "lora_adapter",
             "base": {"repo": "org/demo-0.5b", "revision": None, "store_path": "base/demo-0.5b"},
             "parent_artifact": None, "training_run": "model_finetune/runs/x",
             "train_data": {"snapshot": "t1", "sha256": "ab" * 32},
             "hyperparams": {"lora_r": 16}, "files": files, "created_at": "2026-11-27T00:00:00+08:00"}
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        return d, m


class TestResolveOfficial(StoreTestBase):
    def test_not_downloaded(self):
        r = store.resolve_artifact("demo-0.5b__official")
        self.assertEqual(r["kind"], "official")
        self.assertIsNone(r["weights_dir"])
        self.assertEqual(r["repo_id"], "org/demo-0.5b")
        self.assertEqual(r["base_config"]["torch_dtype"], "bfloat16")

    def test_with_weights(self):
        (store.STORE_ROOT / "base" / "demo-0.5b" / "model.safetensors").write_bytes(b"w")
        r = store.resolve_artifact("demo-0.5b__official")
        self.assertIsNotNone(r["weights_dir"])
        self.assertEqual(len(r["weight_files"]), 1)
        self.assertEqual(r["weight_files"][0]["sha256"], hashlib.sha256(b"w").hexdigest())

    def test_unknown_base(self):
        with self.assertRaises(FileNotFoundError):
            store.resolve_artifact("nope__official")


class TestResolveDerived(StoreTestBase):
    def test_ok(self):
        self.make_derived()
        r = store.resolve_artifact("demo-0.5b__sft-t1__lora-r16")
        self.assertEqual(r["kind"], "lora_adapter")
        self.assertEqual(r["base_id"], "demo-0.5b")
        self.assertIsNotNone(r["manifest_sha256"])
        self.assertEqual(r["manifest"]["train_data"]["snapshot"], "t1")

    def test_unregistered(self):
        with self.assertRaises(FileNotFoundError):
            store.resolve_artifact("demo-0.5b__sft-t9__lora-r8")

    def test_missing_field(self):
        d, m = self.make_derived()
        del m["files"]
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaises(ValueError):
            store.resolve_artifact("demo-0.5b__sft-t1__lora-r16")

    def test_id_mismatch(self):
        d, m = self.make_derived()
        m["artifact_id"] = "other__id"
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaises(ValueError):
            store.resolve_artifact("demo-0.5b__sft-t1__lora-r16")

    def test_unsupported_format(self):
        d, m = self.make_derived(with_file=False)
        (d / "model.gguf").write_bytes(b"g")
        m["files"] = [{"path": "model.gguf", "sha256": hashlib.sha256(b"g").hexdigest()}]
        m["kind"] = "quantized"
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        r = store.resolve_artifact("demo-0.5b__sft-t1__lora-r16")
        self.assertEqual(store.unsupported_formats(r), ["model.gguf"])


class TestList(StoreTestBase):
    def test_list(self):
        self.make_derived()
        rows = store.list_artifacts()
        ids = {r["artifact_id"] for r in rows}
        self.assertIn("demo-0.5b__official", ids)
        self.assertIn("demo-0.5b__sft-t1__lora-r16", ids)


if __name__ == "__main__":
    unittest.main()
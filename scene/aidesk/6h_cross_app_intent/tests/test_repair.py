import unittest

import tests  # noqa: F401
from tests.helpers import DataTestCase, make_sample, persist, sid
from src import manifest as manifest_store, repair
from src.utils import load_json, sha256_file


class RepairPlaceholderTest(DataTestCase):
    def _persist_with_placeholder(self, sample_id, approval="pending"):
        s = make_sample(sample_id, approval=approval)
        s["model_input"]["events"][-1]["content"] = {"city": "当前所在城市", "date": "今天"}
        persist(s)
        return sample_id

    def test_dry_run_changes_nothing(self):
        sample_id = self._persist_with_placeholder(sid("rep-dry"))
        res = repair.repair_noise_placeholder(dry_run=True)
        self.assertEqual(len(res["repaired"]), 1)
        self.assertIn("当前所在城市", str(load_json(f"data/model_inputs/{sample_id}.json")))

    def test_repair_updates_hash_and_logs_provenance(self):
        sample_id = self._persist_with_placeholder(sid("rep-fix"))
        before = manifest_store.load_manifest(sample_id)
        old_hash = before["artefacts"]["model_input"]["sha256"]
        res = repair.repair_noise_placeholder()
        m = manifest_store.load_manifest(sample_id)
        self.assertEqual(res["repaired"][0]["home_city"], "北京")
        self.assertNotEqual(m["artefacts"]["model_input"]["sha256"], old_hash)
        self.assertEqual(m["artefacts"]["model_input"]["sha256"], sha256_file(f"data/model_inputs/{sample_id}.json"))
        self.assertNotIn("当前所在城市", str(load_json(f"data/model_inputs/{sample_id}.json")))
        self.assertEqual(m["repairs"][0]["tool"], "repair_noise_placeholder")
        self.assertEqual(m["approval"]["state"], "pending")
        self.assertEqual(repair.repair_noise_placeholder()["repaired"], [])   # idempotent

    def test_approved_samples_are_untouched_by_default(self):
        sample_id = self._persist_with_placeholder(sid("rep-appr"), approval="approved")
        res = repair.repair_noise_placeholder()
        self.assertEqual(res["repaired"], [])
        self.assertEqual(res["skipped"][0]["reason"], "already approved")
        self.assertIn("当前所在城市", str(load_json(f"data/model_inputs/{sample_id}.json")))
        res = repair.repair_noise_placeholder(include_approved=True)
        self.assertEqual(len(res["repaired"]), 1)


if __name__ == "__main__":
    unittest.main()

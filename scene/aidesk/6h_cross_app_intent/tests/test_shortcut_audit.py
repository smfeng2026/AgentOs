import unittest

from src.shortcut_audit import PROBES, audit_records


def record(split, event_id, relation, content, action="open", app="mail", timestamp="2026-11-28T09:00:00+08:00"):
    return {
        "split": split,
        "model_input": {"events": [{
            "event_id": event_id, "timestamp": timestamp, "app": app, "action": action, "content": content,
        }]},
        "gold": {"event_labels": [{"event_id": event_id, "relation": relation}]},
    }


class ShortcutAuditTest(unittest.TestCase):
    def test_obvious_length_shortcut_exceeds_threshold(self):
        records = []
        for split in ("train", "test"):
            for i in range(2):
                records.append(record(split, f"{split}-r{i}", "supporting", {"text": "很长的相关内容" * 8}))
                records.append(record(split, f"{split}-u{i}", "unrelated", {"text": "短"}))
        result = audit_records(records, {"shortcut_audit": {"thresholds": {"content_length": 0.75}}})
        probe = result["probes"]["content_length"]
        self.assertEqual(probe["accuracy"], 1.0)
        self.assertEqual(probe["balanced_accuracy"], 1.0)
        self.assertEqual(probe["n_train"], 4)
        self.assertEqual(probe["n_eval"], 4)
        self.assertIn(probe["direction"], ("gte", "lte"))
        self.assertTrue(any("content_length" in error and "exceeds" in error for error in result["errors"]))

    def test_action_majority_vote_and_unseen_fallback(self):
        records = [
            record("train", "r1", "supporting", {}, action="read"),
            record("train", "r2", "related_interest", {}, action="read"),
            record("train", "u1", "unrelated", {}, action="delete"),
            record("train", "u2", "unrelated", {}, action="delete"),
            record("test", "r3", "supporting", {}, action="read"),
            record("challenge_test", "u3", "unrelated", {}, action="delete"),
            record("test", "u4", "unrelated", {}, action="never-seen"),
        ]
        result = audit_records(records, {})
        probe = result["probes"]["action"]
        self.assertEqual(probe["n_train"], 4)
        self.assertEqual(probe["n_eval"], 3)
        self.assertEqual(probe["accuracy"], 1.0)
        self.assertEqual(probe["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(probe["majority_accuracy"], 2 / 3, places=4)
        self.assertEqual(probe["n_unseen_features"], 1)
        self.assertAlmostEqual(result["info"]["train_holdout_exact_duplicate_single_events"]["holdout_ratio"], 2 / 3, places=4)

    def test_no_data_and_malformed_fields_do_not_crash(self):
        result = audit_records([], {"shortcut_audit": {"thresholds": {"action": 0.5}}})
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["warnings"]), 2)
        self.assertEqual(set(result["probes"]), set(PROBES))
        for probe in result["probes"].values():
            self.assertEqual(probe["n_train"], 0)
            self.assertEqual(probe["n_eval"], 0)
            self.assertIsNone(probe["accuracy"])

        malformed = [None, {"split": "train"}, {"split": "test", "model_input": {"events": [None]},
                                                 "gold": {"event_labels": [None]}}]
        result = audit_records(malformed, None)
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["warnings"]), 2)


if __name__ == "__main__":
    unittest.main()

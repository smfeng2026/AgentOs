import unittest

from src.postprocess import extract_json, parse_prediction

GOOD = '{"intents": [], "event_labels": [], "unrelated_event_ids": [], "global_observations": {"intent_count": 0, "unrelated_event_count": 30}}'


class TestExtract(unittest.TestCase):
    def test_direct(self):
        self.assertEqual(extract_json(GOOD)["intents"], [])

    def test_fenced(self):
        self.assertIn("intents", extract_json("```json\n" + GOOD + "\n```"))

    def test_embedded_in_prose(self):
        self.assertIn("intents", extract_json("好的，结果如下：\n" + GOOD + "\n以上。"))

    def test_cjk_values(self):
        obj = extract_json('{"intents": [{"intent_name": "成都出行"}], "event_labels": []}')
        self.assertEqual(obj["intents"][0]["intent_name"], "成都出行")

    def test_braces_in_prose_after(self):
        self.assertIn("intents", extract_json(GOOD + " 解释：{无关}"))

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json("抱歉，我无法回答。")


class TestParsePrediction(unittest.TestCase):
    def test_direct_ok(self):
        r = parse_prediction(GOOD)
        self.assertTrue(r["format_ok"])
        self.assertFalse(r["repair_used"])
        self.assertEqual(r["method"], "direct")

    def test_fenced_repair_flag(self):
        r = parse_prediction("```json\n" + GOOD + "\n```")
        self.assertTrue(r["format_ok"])
        self.assertTrue(r["repair_used"])

    def test_trailing_comma_repair(self):
        bad = '{"intents": [], "event_labels": [],}'
        r = parse_prediction(bad)
        self.assertTrue(r["format_ok"])
        self.assertEqual(r["method"], "comma_repair")

    def test_structure_missing_not_ok(self):
        r = parse_prediction('{"foo": 1}')
        self.assertFalse(r["format_ok"])
        self.assertIsNotNone(r["prediction"])

    def test_garbage(self):
        r = parse_prediction("完全不是 JSON")
        self.assertFalse(r["format_ok"])
        self.assertIsNone(r["prediction"])
        self.assertEqual(r["method"], "failed")

    def test_empty(self):
        self.assertFalse(parse_prediction("")["format_ok"])
        self.assertFalse(parse_prediction(None)["format_ok"])


if __name__ == "__main__":
    unittest.main()
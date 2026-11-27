import random
import unittest

import tests  # noqa: F401
from src.noise import load_pool, sample_noise, assign_noise_timestamps, hard_negative_fraction_for
from src.utils import parse_time


def _contents(events):
    return sorted((__import__("json").dumps(e["content"], ensure_ascii=False, sort_keys=True), e["pool_id"]) for e in events)


class NoiseSamplerTest(unittest.TestCase):
    def setUp(self):
        self.pool = load_pool()

    def test_deterministic_for_same_seed(self):
        a = sample_noise(self.pool, 15, random.Random(7), ["travel"], {"destinations": ["成都"]})
        b = sample_noise(self.pool, 15, random.Random(7), ["travel"], {"destinations": ["成都"]})
        self.assertEqual(_contents(a), _contents(b))
        c = sample_noise(self.pool, 15, random.Random(8), ["travel"], {"destinations": ["成都"]})
        self.assertNotEqual(_contents(a), _contents(c))

    def test_semantic_exclusion_by_category(self):
        avoid = {t["id"] for t in self.pool["events"] if "travel" in t.get("avoid_categories", [])}
        for seed in range(30):
            events = sample_noise(self.pool, 15, random.Random(seed), ["travel", "shopping"], {})
            self.assertFalse(avoid & {e["pool_id"] for e in events}, f"seed {seed} used an avoided template")
            self.assertEqual(len(events), 15)

    def test_entity_collision_guard(self):
        for seed in range(50):
            events = sample_noise(self.pool, 15, random.Random(seed), ["travel"],
                                  {"destinations": ["杭州", "西安"], "products": ["咖啡机"]},
                                  difficulty="D4_noise_interference", context={"home_city": "北京", "other_date": "12月21日"})
            text = " ".join(str(v) for e in events for v in e["content"].values())
            self.assertNotIn("杭州", text)
            self.assertNotIn("西安", text)
            self.assertNotIn("咖啡机", text)

    def test_hard_negatives_use_context_and_distractors(self):
        events = sample_noise(self.pool, 15, random.Random(3), ["travel"], {"destinations": ["成都"]},
                              hard_negative_fraction=0.4, context={"home_city": "北京", "other_date": "12月21日"})
        hard = [e for e in events if e["hard_negative"]]
        self.assertEqual(len(hard), 6)
        text = " ".join(str(v) for e in hard for v in e["content"].values())
        self.assertIn("北京", text)
        self.assertIn("12月21日", text)
        self.assertNotIn("当前所在城市", text)
        self.assertNotIn("成都", text)
        for vague in ("下个月中旬", "下周末", "国庆假期", "年底假期"):
            self.assertNotIn(vague, text)

    def test_no_hard_negatives_for_untagged_category(self):
        events = sample_noise(self.pool, 15, random.Random(3), ["health_service"], {}, hard_negative_fraction=0.4)
        self.assertFalse(any(e["hard_negative"] for e in events))
        events = sample_noise(self.pool, 15, random.Random(3), ["finance_admin"], {}, hard_negative_fraction=0.4)
        self.assertEqual(sum(e["hard_negative"] for e in events), 1)   # exactly one finance hard negative exists

    def test_timestamps_inside_window_with_min_gap(self):
        rng = random.Random(5)
        related = [f"2026-12-01T{10 + i // 60:02d}:{i % 60:02d}:00+08:00" for i in range(0, 15)]
        noise = sample_noise(self.pool, 15, rng, [], {})
        noise = assign_noise_timestamps(noise, rng, "2026-12-01T09:00:00+08:00", "2026-12-01T15:00:00+08:00", related)
        stamps = sorted(parse_time(x) for x in related + [e["timestamp"] for e in noise])
        self.assertTrue(all(parse_time("2026-12-01T09:00:00+08:00") <= t <= parse_time("2026-12-01T15:00:00+08:00") for t in stamps))
        self.assertTrue(all((b - a).total_seconds() >= 60 for a, b in zip(stamps, stamps[1:])))
        self.assertEqual(len(stamps), 30)

    def test_difficulty_map(self):
        self.assertGreater(hard_negative_fraction_for("D4_noise_interference"), hard_negative_fraction_for("D1_single_strong_evidence"))


if __name__ == "__main__":
    unittest.main()

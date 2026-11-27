import unittest

import tests  # noqa: F401
from tests.helpers import DataTestCase, make_sample, persist, sid
from src import review_view


class ReviewViewTest(DataTestCase):
    def test_render_and_pending_list(self):
        persist(make_sample(sid("rev-1"), approval="pending", tag="a"))
        persist(make_sample(sid("rev-2"), approval="approved", tag="b"))
        self.assertEqual(review_view.pending_sample_ids(), [sid("rev-1")])
        text = review_view.render_review(review_view.load_sample(sid("rev-1")))
        self.assertIn(sid("rev-1"), text)
        self.assertIn("| e001 |", text)
        self.assertIn("审核清单", text)
        self.assertIn("### Gold 意图", text)
        out, ids = review_view.export("data/review/review_sheet.md")
        self.assertEqual(ids, [sid("rev-1")])
        self.assertTrue(out.exists())
        self.assertIn("共 1 条待审", out.read_text(encoding="utf-8"))
        with self.assertRaises(Exception):
            review_view.load_sample(sid("missing"))


if __name__ == "__main__":
    unittest.main()

"""Shared fixtures: synthetic valid samples written through the real writer + manifest store."""
import shutil
import unittest

import tests  # noqa: F401  (must run first: sets DATASET_ROOT)

from src import manifest as manifest_store
from src.dataset_writer import write_sample
from src.taxonomy import Taxonomy
from src.utils import stable_hash

TAXONOMY = Taxonomy.load()
PIPELINE_VERSION = "test"


class DataTestCase(unittest.TestCase):
    """Every test starts from an empty data/ directory inside the throwaway root."""

    def setUp(self):
        shutil.rmtree(tests.TEST_ROOT / "data", ignore_errors=True)
        (tests.TEST_ROOT / "data").mkdir()


def _purpose(category):
    options = [p for p in TAXONOMY.purpose.get(category, ["unknown"]) if p != "unknown"]
    return options[0] if options else "unknown"


def make_sample(sample_id, category="travel", template_id="C01-S04", difficulty="D2_multi_app_fusion",
                n_unrelated=15, entities=("成都",), approval="approved", split="unassigned", tag=""):
    """Return the dict passed to persist(); override fields before persisting."""
    events = [{"event_id": f"e{i:03d}", "timestamp": f"2026-12-01T{9 + i // 60:02d}:{i % 60:02d}:00+08:00",
               "app": "flight_booking_app" if i <= 30 - n_unrelated else "news_app",
               "action": "search" if i <= 30 - n_unrelated else "read_article",
               "content": {"k": f"{tag}{i}"}} for i in range(1, 31)]
    n_rel = 30 - n_unrelated
    sup = [e["event_id"] for e in events[:n_rel // 2]]
    ri = [e["event_id"] for e in events[n_rel // 2:n_rel]]
    unrelated = [e["event_id"] for e in events[n_rel:]]
    intent = {
        "intent_id": "i001", "intent_name": f"测试意图{tag}", "category": category, "purpose": _purpose(category),
        "status": "planned", "confidence_level": "high",
        "supporting_event_ids": sup, "related_interest_event_ids": ri, "contradicting_event_ids": [],
        "confirmed_facts": [{"fact": "已确认", "source_event_ids": sup[:1]}],
        "inferred_facts": [{"fact": "可推断", "source_event_ids": ri[:1]}],
        "unknown_information": ["是否有同行人"], "conflicts": [],
    }
    labels = [{"event_id": e, "relation": "supporting", "intent_ids": ["i001"]} for e in sup] + \
             [{"event_id": e, "relation": "related_interest", "intent_ids": ["i001"]} for e in ri] + \
             [{"event_id": e, "relation": "unrelated", "intent_ids": []} for e in unrelated]
    gold = {"sample_id": sample_id, "intents": [intent], "event_labels": labels, "unrelated_event_ids": unrelated,
            "global_observations": {"intent_count": 1, "unrelated_event_count": n_unrelated}}
    model_input = {"sample_id": sample_id, "task": "cross_app_intent_reasoning",
                   "window": {"start": "2026-12-01T09:00:00+08:00", "end": "2026-12-01T15:00:00+08:00", "duration_hours": 6},
                   "history_intents": [], "events": events}
    spec = {"scenario_id": sample_id, "title": f"测试场景{tag}", "template_id": template_id, "scenario_category": category,
            "key_entities": {"home_city": "北京", "destinations": list(entities), "products": [], "people": [], "dates": []},
            "intents": [intent]}
    internal = {"sample_id": sample_id, "spec_intents": [intent],
                "events": [{"event_id": e["event_id"], "origin": "related" if i < n_rel else "noise",
                            "pool_id": None if i < n_rel else "news_read",
                            "hard_negative": False, "relation": "unrelated" if i >= n_rel else "supporting",
                            "intent_ids": [] if i >= n_rel else ["i001"]} for i, e in enumerate(events)]}
    audit = {"sample_id": sample_id, "status": "HUMAN_APPROVAL_PENDING", "decision": "accept", "flags": [],
             "attempts": [{"attempt": 1, "stage": "adjudication", "decision": "accept"}],
             "blind_annotation": gold, "review": {"passed": True, "issues": []},
             "adjudication": {"decision": "accept", "rationale": "ok"},
             "validation": {"errors": [], "warnings": [], "stats": {"intents": 1, "unrelated": n_unrelated}}}
    return {"sample_id": sample_id, "spec": spec, "internal": internal, "model_input": model_input, "gold": gold,
            "audit": audit, "category": category, "template_id": template_id, "difficulty": difficulty,
            "approval": approval, "split": split, "entities": entities, "tag": tag}


def persist(sample):
    artefacts = write_sample(sample["sample_id"], sample["spec"], sample["internal"], sample["model_input"],
                             sample["gold"], sample["audit"])
    manifest = {
        "sample_id": sample["sample_id"], "pipeline_version": PIPELINE_VERSION, "created_at": "2026-11-27T10:00:00+08:00",
        "task": {"scenario_category": sample["category"], "template_id": sample["template_id"], "subscenario": None,
                 "difficulty": sample["difficulty"], "intent_count": 1, "split": sample["split"],
                 "window_start": "2026-12-01T09:00:00+08:00", "window_end": "2026-12-01T15:00:00+08:00", "seed": 1},
        "spec_summary": {"title": sample["spec"]["title"], "intent_names": ["测试意图"], "intent_categories": [sample["category"]],
                         "key_entities": sample["spec"]["key_entities"]},
        "composition": {"total_events": 30, "gold_intents": 1, "gold_unrelated": len(sample["gold"]["unrelated_event_ids"])},
        "noise": {"pool_version": "test", "pool_ids": ["news_read"]},
        "provenance": {"generator_model": "g", "judge_model": "j"},
        "decision": "accept", "status": "HUMAN_APPROVAL_PENDING", "flags": [],
        "validation": {"errors": [], "warnings": [], "stats": {"intents": 1, "unrelated": len(sample["gold"]["unrelated_event_ids"])}},
        "attempts": 1, "usage": {}, "elapsed_s": 1.0,
        "approval": {"state": sample["approval"], "by": None, "at": None, "note": None},
        "artefacts": artefacts,
    }
    manifest_store.write_manifest(manifest)
    return manifest


def sid(name):
    return f"{name}-{stable_hash(name, 8)}"

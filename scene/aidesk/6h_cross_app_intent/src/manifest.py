"""Per-sample manifest: provenance, status, split assignment and the human approval record.

The dataset builder only ingests samples whose manifest says approval.state == "approved".
"""
import json
from pathlib import Path

from src.utils import ROOT, save_json, load_json, append_jsonl, now_iso

MANIFEST_DIR = "data/manifests"
APPROVAL_STATES = ("pending", "approved", "rejected")


def manifest_path(sample_id):
    return ROOT / MANIFEST_DIR / f"{sample_id}.json"


def write_manifest(data):
    data["updated_at"] = now_iso()
    return save_json(f"{MANIFEST_DIR}/{data['sample_id']}.json", data)


def load_manifest(sample_id):
    return load_json(f"{MANIFEST_DIR}/{sample_id}.json")


def list_manifests():
    directory = ROOT / MANIFEST_DIR
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def set_approval(sample_id, state, by, note=None):
    if state not in APPROVAL_STATES:
        raise ValueError(f"approval state must be one of {APPROVAL_STATES}")
    data = load_manifest(sample_id)
    data["approval"] = {"state": state, "by": by, "at": now_iso(), "note": note}
    write_manifest(data)
    append_jsonl("logs/approval_logs.jsonl", {
        "sample_id": sample_id, "state": state, "by": by, "note": note, "at": data["approval"]["at"]
    })
    return data["approval"]

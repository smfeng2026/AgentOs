"""Persist one sample as six artefacts + return their paths/hashes for the manifest.

data/scenario_specs/   internal scenario plan (never a model input)
data/internal_labels/  generator's per-event relation/intent_ids + noise origin (never a model input)
data/model_inputs/     what the on-device model sees (no labels inside)
data/gold_labels/      adjudicated Gold Label (may be invalid unless status says otherwise)
data/audit_reports/    blind annotation, review, adjudication, attempts, validation, usage
"""
from src.utils import ROOT, save_json, sha256_file

ARTEFACT_DIRS = {
    "scenario_spec": "data/scenario_specs",
    "internal_labels": "data/internal_labels",
    "model_input": "data/model_inputs",
    "gold_label": "data/gold_labels",
    "audit_report": "data/audit_reports",
}


def artefact_path(kind, sample_id):
    return f"{ARTEFACT_DIRS[kind]}/{sample_id}.json"


def write_sample(sample_id, spec, internal_labels, model_input, gold, audit):
    payloads = {
        "scenario_spec": spec,
        "internal_labels": internal_labels,
        "model_input": model_input,
        "gold_label": gold,
        "audit_report": audit,
    }
    artefacts = {}
    for kind, payload in payloads.items():
        path = save_json(artefact_path(kind, sample_id), payload)
        artefacts[kind] = {
            "path": str(path.relative_to(ROOT)) if hasattr(path, "relative_to") else str(path),
            "sha256": sha256_file(path),
        }
    return artefacts

"""Provenance-preserving repairs on already-generated samples.

`repair noise-placeholder` — the seed noise pool (version 2026-11-27.1 before the weather fix) carried the
literal placeholder "当前所在城市" in two weather templates. The label (unrelated) is unaffected, but the text is
unnatural. This rewrites the literal to the sample's `key_entities.home_city`, updates the artefact hash and
records the edit in `manifest["repairs"]` so nothing is changed silently. Samples already approved by a human
are left alone unless --include-approved is given.
"""
import argparse

from src import manifest as manifest_store
from src.utils import load_json, save_json, sha256_file, now_iso

PLACEHOLDER = "当前所在城市"


def _rewrite(value, home_city):
    """Return (new_value, n_replacements) for nested dict / list / str content."""
    if isinstance(value, dict):
        out, n = {}, 0
        for k, v in value.items():
            out[k], m = _rewrite(v, home_city)
            n += m
        return out, n
    if isinstance(value, list):
        out, n = [], 0
        for v in value:
            x, m = _rewrite(v, home_city)
            out.append(x)
            n += m
        return out, n
    if isinstance(value, str) and PLACEHOLDER in value:
        return value.replace(PLACEHOLDER, home_city), 1
    return value, 0


def repair_noise_placeholder(dry_run=False, include_approved=False):
    changed, skipped = [], []
    for m in manifest_store.list_manifests():
        sid = m.get("sample_id")
        artefacts = m.get("artefacts") or {}
        if "model_input" not in artefacts:
            continue
        state = (m.get("approval") or {}).get("state", "pending")
        model_input = load_json(artefacts["model_input"]["path"])
        text_hits = sum(1 for e in model_input.get("events", []) if PLACEHOLDER in str(e.get("content")))
        if not text_hits:
            continue
        if state == "approved" and not include_approved:
            skipped.append({"sample_id": sid, "reason": "already approved"})
            continue
        home_city = ((m.get("spec_summary") or {}).get("key_entities") or {}).get("home_city")
        if not home_city:
            spec_path = (artefacts.get("scenario_spec") or {}).get("path")
            home_city = ((load_json(spec_path).get("key_entities") or {}).get("home_city")) if spec_path else None
        if not home_city:
            skipped.append({"sample_id": sid, "reason": "no home_city in spec"})
            continue
        if any(PLACEHOLDER in home_city for _ in [0]):
            skipped.append({"sample_id": sid, "reason": "home_city is itself a placeholder"})
            continue
        new_events, total = [], 0
        for e in model_input.get("events", []):
            content, n = _rewrite(e.get("content"), home_city)
            total += n
            new_events.append({**e, "content": content})
        model_input["events"] = new_events
        changed.append({"sample_id": sid, "events_with_placeholder": text_hits, "replaced": total, "home_city": home_city})
        if dry_run:
            continue
        path = artefacts["model_input"]["path"]
        save_json(path, model_input)
        artefacts["model_input"]["sha256"] = sha256_file(path)
        m.setdefault("repairs", []).append({
            "tool": "repair_noise_placeholder", "at": now_iso(), "from": PLACEHOLDER, "to": home_city,
            "replacements": total, "note": "label unaffected; pool fixed in 2026-11-27.1+ (weather_now / weather_forecast)",
        })
        manifest_store.write_manifest(m)
    return {"dry_run": dry_run, "repaired": changed, "skipped": skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Repair noise placeholder literals in generated samples.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-approved", action="store_true")
    args = parser.parse_args(argv)
    result = repair_noise_placeholder(args.dry_run, args.include_approved)
    print({"dry_run": result["dry_run"], "n_repaired": len(result["repaired"]), "n_skipped": len(result["skipped"])})
    for row in result["repaired"]:
        print(" ", row)
    for row in result["skipped"]:
        print("  skipped", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

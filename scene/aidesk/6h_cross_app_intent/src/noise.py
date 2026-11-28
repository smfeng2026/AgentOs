"""Unrelated-event (noise) sampler.

Replaces the fixed 15-event list of the MVP:
- per-sample random draw from a slot-based pool (content varies per sample),
- semantic exclusion by intent category (`avoid_categories`),
- entity-collision guard against the sample's own key entities,
- hard negatives (other city / other product / other date) drawn from distractor entities,
- non-uniform timestamps with a minimum gap from every other event.
All randomness comes from a seeded rng → reproducible per sample.
"""
import json
import random
import re
import statistics
from datetime import timedelta

from src.utils import load_json, parse_time, flatten_entities

ENTITY_PREFIX = "entities."
DEFAULT_POOL_PATH = "config/noise_pool.json"

# Share of noise drawn from hard negatives by difficulty (design doc §11.4 — confusable noise is more valuable).
HARD_NEGATIVE_FRACTION = {
    "D1_single_strong_evidence": 0.10,
    "D2_multi_app_fusion": 0.20,
    "D3_weak_evidence": 0.15,
    "D4_noise_interference": 0.40,
    "D5_multi_intent": 0.20,
    "D6_conflict_state_change": 0.20,
}
DEFAULT_HARD_NEGATIVE_FRACTION = 0.15

# Time-of-day mix over the 6h window: hours 0-1 / 1-2 / 2-4 / 4-6 (design doc §3.2 table, normalised).
_HOUR_BUCKET_EDGES = [0, 1, 2, 4, 6]
_HOUR_BUCKET_WEIGHTS = [0.17, 0.17, 0.33, 0.33]


def load_pool(path=DEFAULT_POOL_PATH):
    return load_json(path)


_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def _content_chars(event):
    return len(json.dumps(event.get("content"), ensure_ascii=False, separators=(",", ":")))


def prepare_generated_noise(generated, related_events, expected_count, generation_cfg=None, app_vocabulary=None):
    """Validate and normalize Generator-produced surface-aligned noise.

    A claimed surface pair must genuinely share app, action and content keys. The checks are deliberately
    syntactic; semantic unrelatedness remains the blind annotator/reviewer/adjudicator's responsibility.
    """
    cfg = generation_cfg or {}
    raw = generated.get("noise_events") if isinstance(generated, dict) else None
    errors = []
    if not isinstance(raw, list):
        return [], {"errors": ["generated.noise_events must be a list"], "warnings": [], "stats": {}}
    if len(raw) != expected_count:
        errors.append(f"noise event count must be {expected_count}, got {len(raw)}")

    related_by_id = {e.get("event_id"): e for e in related_events if isinstance(e, dict)}
    vocabulary = set(app_vocabulary) if app_vocabulary is not None else None
    noise, ids, pair_refs, valid_pair_refs, signatures = [], [], [], [], set()
    for idx, event in enumerate(raw, 1):
        if not isinstance(event, dict):
            errors.append(f"noise #{idx} must be an object")
            continue
        eid = event.get("event_id")
        if not isinstance(eid, str) or not eid:
            errors.append(f"noise #{idx}: event_id missing")
            eid = f"n{idx:03d}"
        ids.append(eid)
        if set(event) & {"relation", "intent_ids", "timestamp"}:
            errors.append(f"{eid}: noise must not contain relation/intent_ids/timestamp")
        for field in ("app", "action"):
            value = event.get(field)
            if not isinstance(value, str) or not _SNAKE.match(value):
                errors.append(f"{eid}: {field} must be snake_case English")
        if vocabulary is not None and event.get("app") not in vocabulary:
            errors.append(f"{eid}: app {event.get('app')!r} not in app_vocabulary")
        content = event.get("content")
        if not isinstance(content, dict) or not content:
            errors.append(f"{eid}: content must be a non-empty object")
            content = content if isinstance(content, dict) else {}
        bad_keys = [k for k in content if not isinstance(k, str) or not _SNAKE.match(k)]
        if bad_keys:
            errors.append(f"{eid}: content keys must be snake_case English: {bad_keys[:4]}")
        allowed_purpose_fields = {"purpose", "payment_purpose", "pay_purpose", "cancel_reason", "refund_reason", "reason"}
        leak_keys = [k for k in content if isinstance(k, str)
                     and any(token in k for token in ("purpose", "intent", "goal")) and k not in allowed_purpose_fields]
        if leak_keys:
            errors.append(f"{eid}: non-natural purpose/intent field(s) leak the label: {leak_keys[:4]}")
        if not isinstance(event.get("hard_negative"), bool):
            errors.append(f"{eid}: hard_negative must be boolean")
        ref = event.get("surface_match_event_id")
        if ref is not None:
            pair_refs.append(ref)
            counterpart = related_by_id.get(ref)
            if counterpart is None:
                errors.append(f"{eid}: unknown surface_match_event_id {ref!r}")
            else:
                same = (event.get("app") == counterpart.get("app")
                        and event.get("action") == counterpart.get("action")
                        and set(content) == set(counterpart.get("content") or {}))
                if not same:
                    errors.append(f"{eid}: surface pair {ref} must share app/action/content keys")
                else:
                    valid_pair_refs.append(ref)
        signature = json.dumps({"app": event.get("app"), "action": event.get("action"), "content": content},
                               ensure_ascii=False, sort_keys=True)
        if signature in signatures:
            errors.append(f"{eid}: duplicate noise event within sample")
        signatures.add(signature)
        noise.append({
            "pool_id": f"generated:{eid}", "hard_negative": bool(event.get("hard_negative")),
            "app": event.get("app"), "action": event.get("action"), "content": content,
            "surface_match_event_id": ref,
        })
    if len(ids) != len(set(ids)):
        errors.append("duplicate noise event_id")
    if len(pair_refs) != len(set(pair_refs)):
        errors.append("surface-matched noise must use distinct related counterparts")

    paired = len(valid_pair_refs)
    paired_fraction = paired / expected_count if expected_count else 0.0
    min_paired = float(cfg.get("min_paired_fraction", 0.6))
    if paired_fraction < min_paired:
        errors.append(f"surface-paired fraction {paired_fraction:.3f} below {min_paired:.3f}")
    related_lengths = [_content_chars(e) for e in related_events if isinstance(e, dict)]
    noise_lengths = [_content_chars(e) for e in noise]
    ratio = None
    if related_lengths and noise_lengths:
        related_median = statistics.median(related_lengths)
        noise_median = statistics.median(noise_lengths)
        ratio = noise_median / related_median if related_median else None
        if ratio is not None:
            low = float(cfg.get("min_median_length_ratio", 0.65))
            high = float(cfg.get("max_median_length_ratio", 1.6))
            if not low <= ratio <= high:
                errors.append(f"noise/related median content length ratio {ratio:.3f} outside [{low}, {high}]")
    stats = {"paired": paired, "paired_fraction": round(paired_fraction, 3),
             "median_length_ratio": round(ratio, 3) if ratio is not None else None}
    return noise, {"errors": errors, "warnings": [], "stats": stats}


def hard_negative_fraction_for(difficulty):
    return HARD_NEGATIVE_FRACTION.get(difficulty, DEFAULT_HARD_NEGATIVE_FRACTION)


def _text_of(content):
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _collides(text, exclude_entities):
    t = str(text).casefold()
    return any(e.casefold() in t for e in exclude_entities)


def _fill(template, rng, pool, exclude_entities, context=None):
    context = context or {}
    slots = template.get("slots", {}) or {}
    values = {}
    for name, spec in slots.items():
        if isinstance(spec, str) and spec.startswith(ENTITY_PREFIX):
            candidates = pool["distractor_entities"][spec[len(ENTITY_PREFIX):]]
            candidates = [c for c in candidates if not _collides(c, exclude_entities)]
        elif spec == "home_city":
            candidates = [context.get("home_city")] if context.get("home_city") else ["当前所在城市"]
        elif spec == "window.other_date":
            candidates = [context.get("other_date") or "下个月"]
        else:
            candidates = list(spec)
        values[name] = rng.choice(candidates) if candidates else None
    content = {}
    for key, val in template["content"].items():
        if isinstance(val, str):
            name = val[1:-1] if val.startswith("{") and val.endswith("}") else None
            if name is not None and name in values:
                content[key] = values[name]
            elif "{" in val:
                content[key] = val.format(**values)
            else:
                content[key] = val
        else:
            content[key] = val
    return content


def _materialise(kind, template, rng, pool, exclude_entities, context=None):
    content = None
    for _ in range(4):
        candidate = _fill(template, rng, pool, exclude_entities, context)
        if not _collides(_text_of(candidate), exclude_entities):
            content = candidate
            break
    if content is None:
        content = _fill(template, rng, pool, exclude_entities, context)
    return {
        "pool_id": template["id"],
        "hard_negative": kind == "hard",
        "app": template["app"],
        "action": template["action"],
        "content": content,
    }


def _sample_base(templates, n, rng):
    """Round-robin over shuffled event types so one sample mixes many behaviour kinds."""
    by_type = {}
    for t in templates:
        by_type.setdefault(t.get("type", "misc"), []).append(t)
    for group in by_type.values():
        rng.shuffle(group)
    types = list(by_type)
    rng.shuffle(types)
    picked = []
    pass_idx = 0
    while len(picked) < n and by_type:
        progressed = False
        for ty in types:
            group = by_type.get(ty)
            if not group:
                continue
            picked.append(group.pop(0))
            progressed = True
            if len(picked) >= n:
                break
        if not progressed:
            break
        pass_idx += 1
    return picked


def sample_noise(pool, n, rng, categories, key_entities, hard_negative_fraction=None, difficulty=None,
                 context=None):
    """Return n noise events (no timestamps yet). Deterministic for a given rng state.

    context: {"home_city": "北京", "other_date": "12月21日"} — fills the hard-negative placeholders
    so distractors look like real events instead of template text.
    """
    categories = set(categories or [])
    exclude_entities = [e for e in flatten_entities(key_entities) if e and len(str(e)) >= 2]
    if hard_negative_fraction is None:
        hard_negative_fraction = hard_negative_fraction_for(difficulty)

    base = [t for t in pool["events"] if not (categories & set(t.get("avoid_categories", [])))]
    hard = [t for t in pool["hard_negatives"] if categories & set(t["applies_to"])]
    n_hard = min(len(hard), int(round(n * hard_negative_fraction))) if hard else 0
    n_base = n - n_hard

    chosen = [("hard", t) for t in rng.sample(hard, n_hard)]
    chosen.extend(("base", t) for t in _sample_base(base, n_base, rng))

    events = []
    for kind, template in chosen:
        events.append(_materialise(kind, template, rng, pool, exclude_entities, context))
    rng.shuffle(events)
    return events


def assign_noise_timestamps(noise_events, rng, window_start, window_end, related_timestamps, min_gap_s=60):
    """Give each noise event a timestamp inside the window, ≥ min_gap_s away from every other event."""
    start = parse_time(window_start)
    end = parse_time(window_end)
    duration = (end - start).total_seconds()
    taken = [(parse_time(t) - start).total_seconds() for t in related_timestamps]
    edges = _HOUR_BUCKET_EDGES
    for ev in noise_events:
        offset = None
        for _ in range(30):
            bucket = rng.choices(range(len(_HOUR_BUCKET_WEIGHTS)), weights=_HOUR_BUCKET_WEIGHTS)[0]
            lo = edges[bucket] * 3600 + 30
            hi = min(edges[bucket + 1] * 3600, duration) - 30
            candidate = rng.uniform(lo, max(lo, hi))
            if all(abs(candidate - t) >= min_gap_s for t in taken):
                offset = candidate
                break
        if offset is None:
            span = max(duration - 120, 1)
            offset = rng.uniform(30, span)
            for _ in range(400):
                if all(abs(offset - t) >= min_gap_s for t in taken):
                    break
                offset = 30 + ((offset + min_gap_s) % span)
        taken.append(offset)
        ev["timestamp"] = (start + timedelta(seconds=int(offset))).isoformat()
    noise_events.sort(key=lambda e: e["timestamp"])
    return noise_events

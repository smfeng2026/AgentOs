"""Structural + label-consistency validation.

validate_generated: the generator's 15 related events (before noise injection).
validate_final:     the adjudicated Gold Label against the final model input.

Both return plain lists / dicts — no exceptions — so the orchestrator can log every problem.
"""
from collections import Counter
import re

from src.utils import parse_time

RELATIONS = ("supporting", "related_interest", "contradicting", "unrelated")
RELATED_RELATIONS = RELATIONS[:3]
RELATION_FIELDS = {
    "supporting": "supporting_event_ids",
    "related_interest": "related_interest_event_ids",
    "contradicting": "contradicting_event_ids",
}
MIN_EVENT_GAP_S = 60
# Noise-pool events use snake_case app/action/keys; generated events must too, otherwise the *style*
# of a field would leak the label (Chinese action = related, English action = noise).
_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def _as_list(value):
    return value if isinstance(value, list) else []


def _event_text(event):
    content = event.get("content")
    if isinstance(content, str):
        return content
    import json
    return json.dumps(content, ensure_ascii=False)


def validate_generated(generated, window_start, window_end, expected_related=15, spec=None, app_vocabulary=None):
    errors = []
    events = generated.get("events") if isinstance(generated, dict) else None
    if not isinstance(events, list):
        return ["generated.events must be a list"]

    if len(events) != expected_related:
        errors.append(f"related event count must be {expected_related}, got {len(events)}")

    spec_intents = [i for i in _as_list((spec or {}).get("intents")) if isinstance(i, dict)]
    spec_intent_ids = {i.get("intent_id") for i in spec_intents}
    intent_names = [str(i.get("intent_name")) for i in spec_intents if i.get("intent_name")]

    ids, stamps = [], []
    for e in events:
        if not isinstance(e, dict):
            errors.append("event entry must be an object")
            continue
        eid = e.get("event_id")
        if not isinstance(eid, str) or not eid:
            errors.append("event_id missing")
            continue
        ids.append(eid)
        if e.get("relation") not in RELATED_RELATIONS:
            errors.append(f"{eid}: invalid relation {e.get('relation')!r}")
        intent_ids = e.get("intent_ids")
        if not isinstance(intent_ids, list) or not intent_ids:
            errors.append(f"{eid}: related event must have non-empty intent_ids")
        elif spec_intent_ids and not set(intent_ids) <= spec_intent_ids:
            errors.append(f"{eid}: intent_ids {intent_ids} not defined in scenario spec")
        for field in ("app", "action"):
            value = e.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{eid}: {field} missing")
            elif not _SNAKE.match(value):
                errors.append(f"{eid}: {field} {value!r} must be snake_case English")
        if app_vocabulary is not None and isinstance(e.get("app"), str) and e["app"] not in app_vocabulary:
            errors.append(f"{eid}: app {e['app']!r} not in app_vocabulary")
        content = e.get("content")
        if isinstance(content, dict) and not content or isinstance(content, str) and not content.strip() \
                or not isinstance(content, (dict, str)):
            errors.append(f"{eid}: content must be a non-empty object")
        if isinstance(content, dict):
            bad_keys = [k for k in content if not (isinstance(k, str) and _SNAKE.match(k))]
            if bad_keys:
                errors.append(f"{eid}: content keys must be snake_case English: {bad_keys[:4]}")
        text = " ".join([str(e.get("app", "")), str(e.get("action", "")), _event_text(e)])
        for name in intent_names:
            if len(name) >= 2 and name in text:
                errors.append(f"{eid}: intent name '{name}' leaked verbatim into the event")
        try:
            t = parse_time(e["timestamp"])
            if t < parse_time(window_start) or t > parse_time(window_end):
                errors.append(f"{eid}: timestamp outside window")
            stamps.append((t, eid))
        except Exception:  # noqa: BLE001
            errors.append(f"{eid}: invalid timestamp")

    if len(ids) != len(set(ids)):
        errors.append("duplicate event_id")
    stamps.sort()
    for (t1, id1), (t2, id2) in zip(stamps, stamps[1:]):
        if (t2 - t1).total_seconds() < MIN_EVENT_GAP_S:
            errors.append(f"{id1}/{id2}: events less than {MIN_EVENT_GAP_S}s apart")
    return errors


def validate_final(model_input, gold, taxonomy=None, expected_total=30, min_unrelated=10,
                   max_unrelated=20, target_unrelated=15):
    errors, warnings = [], []
    events = _as_list(model_input.get("events"))
    event_ids = [e.get("event_id") for e in events if isinstance(e, dict)]
    event_id_set = set(event_ids)
    if len(events) != expected_total:
        errors.append(f"model input event count must be {expected_total}, got {len(events)}")
    if not isinstance(gold, dict):
        return {"errors": ["gold_label must be an object"], "warnings": [], "stats": {}}
    if gold.get("sample_id") not in (None, model_input.get("sample_id")):
        errors.append("gold sample_id does not match model input")

    intents = [i for i in _as_list(gold.get("intents")) if isinstance(i, dict)]
    if not isinstance(gold.get("intents"), list):
        errors.append("intents must be a list")
    intent_ids = []
    rel_sets = {}  # intent_id -> {relation: set(event ids)} from the per-intent lists
    for it in intents:
        iid = it.get("intent_id")
        if not isinstance(iid, str) or not iid:
            errors.append("intent missing intent_id")
            continue
        intent_ids.append(iid)
        if not isinstance(it.get("intent_name"), str) or not it["intent_name"].strip():
            errors.append(f"{iid}: intent_name missing")
        if taxonomy is not None:
            if not taxonomy.valid_category(it.get("category")):
                errors.append(f"{iid}: category {it.get('category')!r} not in taxonomy")
            elif not taxonomy.valid_purpose(it.get("category"), it.get("purpose")):
                errors.append(f"{iid}: purpose {it.get('purpose')!r} not allowed for {it.get('category')!r}")
            if it.get("status") not in taxonomy.intent_status:
                errors.append(f"{iid}: status {it.get('status')!r} not in taxonomy")
            if it.get("confidence_level") not in taxonomy.confidence:
                errors.append(f"{iid}: confidence_level {it.get('confidence_level')!r} not in taxonomy")
        sets = {}
        for rel, field in RELATION_FIELDS.items():
            values = it.get(field)
            if not isinstance(values, list):
                errors.append(f"{iid}: {field} must be a list")
                values = []
            bad = [x for x in values if x not in event_id_set]
            if bad:
                errors.append(f"{iid}: {field} references unknown event ids {bad}")
            sets[rel] = set(values)
        rel_sets[iid] = sets
        if not (sets["supporting"] | sets["related_interest"] | sets["contradicting"]):
            errors.append(f"{iid}: intent has no related events")
        unknowns = it.get("unknown_information")
        if not isinstance(unknowns, list) or any(not str(u).strip() for u in unknowns):
            errors.append(f"{iid}: unknown_information must be a list of non-empty strings")
        if not isinstance(it.get("conflicts"), list):
            errors.append(f"{iid}: conflicts must be a list")
        for field in ("confirmed_facts", "inferred_facts"):
            facts = it.get(field)
            if not isinstance(facts, list):
                errors.append(f"{iid}: {field} must be a list")
                continue
            allowed = sets["supporting"] | sets["contradicting"] if field == "confirmed_facts" \
                else sets["supporting"] | sets["related_interest"] | sets["contradicting"]
            for fact in facts:
                text = fact.get("fact") if isinstance(fact, dict) else None
                if not isinstance(text, str) or not text.strip():
                    errors.append(f"{iid}: {field} entry needs a non-empty 'fact'")
                    continue
                refs = fact.get("source_event_ids")
                if not isinstance(refs, list) or not refs:
                    errors.append(f"{iid}: {field} '{text[:20]}' has no source_event_ids")
                elif not set(refs) <= event_id_set:
                    errors.append(f"{iid}: {field} '{text[:20]}' cites unknown event ids")
                elif not set(refs) <= allowed:
                    errors.append(f"{iid}: {field} '{text[:20]}' cites events not related to this intent")
    if len(intent_ids) != len(set(intent_ids)):
        errors.append("duplicate intent_id")
    intent_id_set = set(intent_ids)

    labels = gold.get("event_labels")
    if not isinstance(labels, list):
        errors.append("event_labels must be a list")
        labels = []
    label_ids = []
    by_intent = {iid: {rel: set() for rel in RELATED_RELATIONS} for iid in intent_id_set}
    actual_unrelated = set()
    for lab in labels:
        if not isinstance(lab, dict):
            errors.append("event label entry must be an object")
            continue
        eid = lab.get("event_id")
        rel = lab.get("relation")
        label_ids.append(eid)
        if rel not in RELATIONS:
            errors.append(f"{eid}: invalid relation {rel!r}")
            continue
        iids = lab.get("intent_ids")
        if not isinstance(iids, list):
            errors.append(f"{eid}: intent_ids must be a list")
            iids = []
        if not set(iids) <= intent_id_set:
            errors.append(f"{eid}: unknown intent_id in {iids}")
        if rel == "unrelated":
            if iids:
                errors.append(f"{eid}: unrelated must have empty intent_ids")
            actual_unrelated.add(eid)
        else:
            if not iids:
                errors.append(f"{eid}: {rel} event must reference at least one intent")
            for iid in iids:
                if iid in by_intent:
                    by_intent[iid][rel].add(eid)
    if set(label_ids) != event_id_set or len(label_ids) != len(set(label_ids)):
        errors.append("event_labels must cover every event exactly once")

    for iid, sets in rel_sets.items():
        for rel, field in RELATION_FIELDS.items():
            if sets[rel] != by_intent[iid][rel]:
                errors.append(f"{iid}: {field} inconsistent with event_labels")

    unrelated_ids = gold.get("unrelated_event_ids")
    if not isinstance(unrelated_ids, list):
        errors.append("unrelated_event_ids must be a list")
        unrelated_ids = []
    if set(unrelated_ids) != actual_unrelated:
        errors.append("unrelated_event_ids inconsistent with event_labels")
    n_unrelated = len(actual_unrelated)
    if n_unrelated < min_unrelated or n_unrelated > max_unrelated:
        errors.append(f"unrelated count {n_unrelated} outside [{min_unrelated}, {max_unrelated}]")
    elif n_unrelated != target_unrelated:
        warnings.append(f"unrelated count {n_unrelated} differs from target {target_unrelated}")

    observations = gold.get("global_observations")
    if not isinstance(observations, dict):
        errors.append("global_observations missing")
    else:
        if observations.get("intent_count") != len(intents):
            errors.append(f"global_observations.intent_count should be {len(intents)}")
        if observations.get("unrelated_event_count") != n_unrelated:
            errors.append(f"global_observations.unrelated_event_count should be {n_unrelated}")

    relation_counts = Counter(lab.get("relation") for lab in labels if isinstance(lab, dict))
    stats = {
        "intents": len(intents),
        "unrelated": n_unrelated,
        "supporting": relation_counts.get("supporting", 0),
        "related_interest": relation_counts.get("related_interest", 0),
        "contradicting": relation_counts.get("contradicting", 0),
        "purposes": dict(Counter(it.get("purpose") for it in intents)),
        "statuses": dict(Counter(it.get("status") for it in intents)),
        "categories": dict(Counter(it.get("category") for it in intents)),
    }
    return {"errors": errors, "warnings": warnings, "stats": stats}

"""Train-only probes for detecting superficial event-label shortcuts.

Each row is one labelled event.  The probes deliberately use no content semantics:
they expose how well a small surface feature alone predicts related/unrelated.
"""
import json
import math
from collections import Counter, defaultdict
from datetime import datetime

PROBES = ("content_length", "action", "app_action", "content_keys", "event_position", "time_bucket")
RELATIONS = {"supporting", "related_interest", "contradicting", "unrelated"}


def _json_text(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        try:
            return json.dumps(str(value), ensure_ascii=False)
        except Exception:  # noqa: BLE001 - malformed user objects must not abort an audit
            return '"<unserializable>"'


def _feature(value):
    return _json_text(value)


def _time_bucket(value):
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).hour
    except (ValueError, OverflowError):
        return "unknown"


def _events(records, splits):
    rows = []
    for record in records or []:
        if not isinstance(record, dict) or record.get("split") not in splits:
            continue
        model_input = record.get("model_input")
        gold = record.get("gold")
        if not isinstance(gold, dict):
            gold = record.get("gold_label")
        if not isinstance(model_input, dict) or not isinstance(gold, dict):
            continue
        events = model_input.get("events")
        labels = gold.get("event_labels")
        if not isinstance(events, list) or not isinstance(labels, list):
            continue
        relations = {}
        for label in labels:
            if not isinstance(label, dict) or label.get("event_id") is None:
                continue
            relation = label.get("relation")
            if relation not in RELATIONS:
                continue
            try:
                relations[label["event_id"]] = int(relation != "unrelated")
            except TypeError:
                continue
        for position, event in enumerate(events):
            if not isinstance(event, dict) or event.get("event_id") is None:
                continue
            try:
                label = relations.get(event["event_id"])
            except TypeError:
                continue
            if label is not None:
                rows.append({"event": event, "label": label, "position": position})
    return rows


def _value(row, probe):
    event = row["event"]
    if probe == "content_length":
        return len(_json_text(event.get("content")))
    if probe == "action":
        return _feature(event.get("action"))
    if probe == "app_action":
        return (_feature(event.get("app")), _feature(event.get("action")))
    if probe == "content_keys":
        content = event.get("content")
        return tuple(sorted(str(k) for k in content)) if isinstance(content, dict) else ()
    if probe == "event_position":
        return row["position"]
    if probe == "time_bucket":
        return _time_bucket(event.get("timestamp"))
    raise ValueError(f"unknown probe: {probe}")


def _majority(counts):
    if not counts:
        return None
    # A deterministic unrelated fallback when the training labels are tied.
    return max(counts, key=lambda label: (counts[label], -label))


def _score(train, evaluate, predict):
    out = {"accuracy": None, "balanced_accuracy": None, "majority_accuracy": None,
           "n_train": len(train), "n_eval": len(evaluate)}
    global_majority = _majority(Counter(row["label"] for row in train))
    if not train or not evaluate or global_majority is None:
        return out
    truth = [row["label"] for row in evaluate]
    guesses = [predict(row) for row in evaluate]
    out["accuracy"] = round(sum(int(g == y) for g, y in zip(guesses, truth)) / len(truth), 4)
    recalls = []
    for label in (0, 1):
        support = truth.count(label)
        if support:
            recalls.append(sum(int(g == y == label) for g, y in zip(guesses, truth)) / support)
    out["balanced_accuracy"] = round(sum(recalls) / len(recalls), 4)
    out["majority_accuracy"] = round(sum(int(y == global_majority) for y in truth) / len(truth), 4)
    return out


def _length_probe(train, evaluate):
    values = sorted({_value(row, "content_length") for row in train})
    best = None
    if values:
        candidates = [values[0] - 1] + values + [values[-1] + 1]
        for direction in ("gte", "lte"):
            for threshold in candidates:
                def predict(row, direction=direction, threshold=threshold):
                    length = _value(row, "content_length")
                    return int(length >= threshold if direction == "gte" else length <= threshold)
                correct = sum(int(predict(row) == row["label"]) for row in train)
                accuracy = correct / len(train)
                candidate = (accuracy, direction == "gte", -threshold, direction, threshold, predict)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
    if best is None:
        return _score(train, evaluate, lambda row: 0) | {"threshold": None, "direction": None,
                                                        "train_accuracy": None}
    accuracy, _, _, direction, threshold, predict = best
    return _score(train, evaluate, predict) | {"threshold": threshold, "direction": direction,
                                                "train_accuracy": round(accuracy, 4)}


def _categorical_probe(name, train, evaluate):
    tables = defaultdict(Counter)
    for row in train:
        tables[_value(row, name)][row["label"]] += 1
    global_majority = _majority(Counter(row["label"] for row in train))

    def predict(row):
        return _majority(tables.get(_value(row, name), Counter())) if _value(row, name) in tables \
            else (global_majority if global_majority is not None else 0)

    metrics = _score(train, evaluate, predict)
    metrics["n_unseen_features"] = sum(int(_value(row, name) not in tables) for row in evaluate)
    return metrics


def _duplicate_info(train, evaluate):
    # Event ids and timestamps necessarily differ across samples; surface duplication is about the
    # reusable app/action/content payload instead.
    def signature(row):
        event = row["event"]
        return _json_text({"app": event.get("app"), "action": event.get("action"),
                           "content": event.get("content")})
    train_signatures = {signature(row) for row in train}
    duplicates = sum(int(signature(row) in train_signatures) for row in evaluate)
    return {"train_events": len(train), "holdout_events": len(evaluate), "duplicates": duplicates,
            "holdout_ratio": round(duplicates / len(evaluate), 4) if evaluate else None}


def audit_records(records, cfg):
    """Fit simple probes on train events and evaluate on test + challenge_test."""
    train = _events(records, {"train"})
    evaluate = _events(records, {"test", "challenge_test"})
    result = {"probes": {}, "errors": [], "warnings": [], "info": {}}
    if not train:
        result["warnings"].append("shortcut audit: no usable train events; probes were not fitted")
    if not evaluate:
        result["warnings"].append("shortcut audit: no usable test/challenge_test events; holdout metrics unavailable")

    for name in PROBES:
        result["probes"][name] = _length_probe(train, evaluate) if name == "content_length" \
            else _categorical_probe(name, train, evaluate)

    audit_cfg = cfg.get("shortcut_audit", {}) if isinstance(cfg, dict) else {}
    if not isinstance(audit_cfg, dict):
        result["warnings"].append("shortcut audit: shortcut_audit config must be an object")
        audit_cfg = {}
    thresholds = audit_cfg.get("thresholds", {})
    if not isinstance(thresholds, dict):
        result["warnings"].append("shortcut audit: thresholds must be an object")
        thresholds = {}
    for name, limit in thresholds.items():
        if name not in result["probes"]:
            result["warnings"].append(f"shortcut audit: threshold configured for unknown probe '{name}'")
            continue
        try:
            limit = float(limit)
            if not math.isfinite(limit):
                raise ValueError
        except (TypeError, ValueError):
            result["warnings"].append(f"shortcut audit: invalid accuracy threshold for '{name}': {limit!r}")
            continue
        accuracy = result["probes"][name]["accuracy"]
        majority = result["probes"][name]["majority_accuracy"]
        min_lift = float(audit_cfg.get("min_majority_lift", 0.05))
        if accuracy is not None and accuracy > limit and (majority is None or accuracy > majority + min_lift):
            result["errors"].append(
                f"shortcut probe '{name}' accuracy {accuracy:.4f} exceeds maximum allowed {limit:.4f}")

    result["info"]["train_holdout_exact_duplicate_single_events"] = _duplicate_info(train, evaluate)
    return result

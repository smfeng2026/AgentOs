"""Configurable release coverage quotas for the training split."""
from collections import Counter


def coverage_report(records, cfg):
    train = [r for r in records if r.get("split") == "train"]
    relations, statuses, confidences = Counter(), Counter(), Counter()
    difficulties = Counter(str(r.get("difficulty")) for r in train)
    unknown_purpose = 0
    multi_intent_samples = 0
    for record in train:
        gold = record.get("gold") or {}
        intents = [i for i in gold.get("intents", []) if isinstance(i, dict)]
        multi_intent_samples += int(len(intents) >= 2)
        for label in gold.get("event_labels", []):
            if isinstance(label, dict):
                relations[label.get("relation")] += 1
        for intent in intents:
            statuses[intent.get("status")] += 1
            confidences[intent.get("confidence_level")] += 1
            unknown_purpose += int(intent.get("purpose") == "unknown")
    metrics = {
        "train_samples": len(train),
        "validation_samples": sum(r.get("split") == "validation" for r in records),
        "test_samples": sum(r.get("split") in ("test", "challenge_test") for r in records),
        "train_templates": len({r.get("group_key") for r in train if r.get("group_key")}),
        "train_difficulties": dict(difficulties),
        "train_relation_counts": dict(relations),
        "train_status_counts": dict(statuses),
        "train_confidence_counts": dict(confidences),
        "train_multi_intent_samples": multi_intent_samples,
        "train_unknown_purpose": unknown_purpose,
    }
    requirements = cfg.get("release") or {}
    errors = []
    simple_minimums = (
        ("min_train_samples", "train_samples"),
        ("min_validation_samples", "validation_samples"),
        ("min_test_samples", "test_samples"),
        ("min_train_templates", "train_templates"),
        ("min_train_multi_intent_samples", "train_multi_intent_samples"),
        ("min_train_unknown_purpose", "train_unknown_purpose"),
    )
    for requirement, metric in simple_minimums:
        if requirement in requirements and metrics[metric] < int(requirements[requirement]):
            errors.append(f"release coverage: {metric}={metrics[metric]} below {requirements[requirement]}")
    for difficulty in requirements.get("required_train_difficulties", []):
        if not difficulties.get(difficulty):
            errors.append(f"release coverage: missing train difficulty {difficulty}")
    for requirement, counts in (
        ("min_train_relation_counts", relations),
        ("min_train_status_counts", statuses),
        ("min_train_confidence_counts", confidences),
    ):
        for label, minimum in (requirements.get(requirement) or {}).items():
            if counts.get(label, 0) < int(minimum):
                errors.append(f"release coverage: {label}={counts.get(label, 0)} below {minimum} ({requirement})")
    return {"metrics": metrics, "requirements": requirements, "errors": errors, "warnings": []}

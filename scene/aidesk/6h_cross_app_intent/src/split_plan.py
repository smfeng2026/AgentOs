"""Deterministic, versioned template-group split plans.

The plan covers every registered template, including templates with no generated samples yet. Builds may
add samples without reallocating existing groups. Strong-entity links crossing planned splits are reported
by dataset_builder instead of silently moving a group.
"""
from collections import defaultdict

from src.utils import load_json, now_iso, save_json, stable_hash


def _largest_remainder(n, ratios):
    names = [k for k in ("train", "validation", "test") if k in ratios]
    total = sum(ratios[k] for k in names)
    raw = {k: n * ratios[k] / total for k in names}
    counts = {k: int(raw[k]) for k in names}
    order = sorted(names, key=lambda k: (-(raw[k] - counts[k]), -ratios[k], 0 if k != "train" else 1))
    for k in order[:n - sum(counts.values())]:
        counts[k] += 1
    return counts


def allocate_template_groups(templates, ratios):
    by_category = defaultdict(list)
    for template in templates:
        by_category[template["category"]].append(template["template_id"])
    queues = {cat: sorted(groups, key=lambda g: stable_hash([cat, g]))
              for cat, groups in sorted(by_category.items())}
    targets = _largest_remainder(sum(len(v) for v in queues.values()), ratios)
    filled = {k: 0 for k in targets}
    assignments = {}
    remaining = sum(len(v) for v in queues.values())
    while remaining:
        for cat, queue in queues.items():
            if not queue:
                continue
            group = queue.pop(0)
            best = max(targets, key=lambda k: ((targets[k] - filled[k]) / targets[k] if targets[k] else -1.0,
                                                k == "train", k == "test"))
            assignments[group] = best
            filled[best] += 1
            remaining -= 1
    return assignments


def create_plan(templates, cfg):
    ratios = cfg.get("split_ratios", {"train": 0.7, "validation": 0.1, "test": 0.2})
    return {
        "version": 1,
        "created_at": now_iso(),
        "split_ratios": ratios,
        "assignments": allocate_template_groups(templates, ratios),
    }


def load_plan(cfg):
    plan_cfg = cfg.get("split_plan") or {}
    path = plan_cfg.get("path")
    if not path:
        return None
    try:
        plan = load_json(path)
    except FileNotFoundError:
        if plan_cfg.get("required"):
            raise
        return None
    if not isinstance(plan.get("assignments"), dict):
        raise ValueError(f"invalid split plan {path}: assignments must be an object")
    return plan


def write_plan(path, templates, cfg):
    plan = create_plan(templates, cfg)
    save_json(path, plan)
    return plan

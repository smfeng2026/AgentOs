"""Dataset builder — turns approved (model_input, gold_label) pairs into train/validation/test files.

Steps:
1. scan manifests → keep approval.state == "approved" (unless --allow-unapproved, flagged loudly),
2. pair model_inputs / gold_labels by sample_id, re-run validate_final (invalid → skipped, reported),
3. split by *template group* using the versioned fixed split plan. Groups sharing a strong entity
   (destination / product) are connected; a connection across planned splits is an error rather than a
   silent reassignment. Test-split samples with a challenge difficulty go to challenge_test,
4. export SFT `messages` JSONL (train/validation) and test JSONL (model_input + gold + meta),
5. report leakage, release coverage quotas and surface-only related/unrelated shortcut probes;
   `build --release` turns quota/probe failures into hard errors.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from src import manifest as manifest_store
from src.utils import (ROOT, load_json, load_yaml, write_jsonl, save_json, compact_json,
                       canonical_json, stable_hash, now_iso, estimate_tokens)
from src.validators import validate_final
from src.taxonomy import Taxonomy
from src.split_plan import load_plan
from src.labeling_policy import sft_system_prompt
from src.release_checks import coverage_report
from src.shortcut_audit import audit_records

BUILDER_VERSION = "1.2.0"
SPLITS = ("train", "validation", "test", "challenge_test")
STRONG_ENTITY_KEYS = ("destinations", "products")   # entity pools used for isolation (people/dates are too generic)
TARGET_KEY_ORDER = ("intents", "event_labels", "unrelated_event_ids", "global_observations")
INTENT_KEY_ORDER = ("intent_id", "intent_name", "category", "purpose", "status", "confidence_level",
                    "supporting_event_ids", "related_interest_event_ids", "contradicting_event_ids",
                    "confirmed_facts", "inferred_facts", "unknown_information", "conflicts")


def _ordered(obj, order):
    if not isinstance(obj, dict):
        return obj
    out = {k: obj[k] for k in order if k in obj}
    for k, v in obj.items():
        if k not in out:
            out[k] = v
    return out


def canonical_target(gold):
    target = {k: gold[k] for k in TARGET_KEY_ORDER if k in gold}
    if isinstance(target.get("intents"), list):
        target["intents"] = [_ordered(i, INTENT_KEY_ORDER) for i in target["intents"]]
    if isinstance(target.get("event_labels"), list):
        target["event_labels"] = [_ordered(x, ("event_id", "relation", "intent_ids")) for x in target["event_labels"]]
        target["event_labels"].sort(key=lambda x: str(x.get("event_id")))
    if isinstance(target.get("unrelated_event_ids"), list):
        target["unrelated_event_ids"] = sorted(target["unrelated_event_ids"])
    return target


# ---- 1–2: load + re-validate -----------------------------------------------------------------------

def load_records(cfg, taxonomy, allow_unapproved=False, require_approval=True):
    records, skipped = [], []
    for m in manifest_store.list_manifests():
        sid = m.get("sample_id")
        approval = (m.get("approval") or {}).get("state", "pending")
        if require_approval and approval != "approved" and not allow_unapproved:
            skipped.append({"sample_id": sid, "reason": "not_approved", "detail": approval})
            continue
        if m.get("run_state") == "failed" or m.get("status") == "NEEDS_HUMAN_REVIEW":
            skipped.append({"sample_id": sid, "reason": "pipeline_not_ready", "detail": m.get("status")})
            continue
        paths = {k: v.get("path") for k, v in (m.get("artefacts") or {}).items()}
        try:
            model_input = load_json(paths["model_input"])
            gold = load_json(paths["gold_label"])
            spec = load_json(paths["scenario_spec"]) if paths.get("scenario_spec") else {}
        except Exception as exc:  # noqa: BLE001
            skipped.append({"sample_id": sid, "reason": "missing_artefact", "detail": str(exc)[:200]})
            continue
        check = validate_final(
            model_input, gold, taxonomy, expected_total=int(cfg.get("total_events", 30)),
            min_unrelated=int(cfg.get("min_unrelated_events", 10)),
            max_unrelated=int(cfg.get("max_unrelated_events", 20)),
            target_unrelated=int(cfg.get("unrelated_events", 15)),
        )
        if check["errors"]:
            skipped.append({"sample_id": sid, "reason": "invalid_gold", "detail": check["errors"][:5]})
            continue
        records.append({
            "sample_id": sid, "model_input": model_input, "gold": gold, "spec": spec, "manifest": m,
            "approval": approval, "validation": check,
        })
    return records, skipped


# ---- 3: split --------------------------------------------------------------------------------------

class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_key(record):
    task = record["manifest"].get("task", {})
    tid = task.get("template_id") or record["spec"].get("template_id")
    if tid:
        return str(tid)
    spec = record["spec"]
    label = spec.get("title") or spec.get("subscenario") or ",".join(
        str(i.get("intent_name")) for i in spec.get("intents", []) if isinstance(i, dict)) or record["sample_id"]
    return f"{task.get('scenario_category', 'unknown')}|auto:{stable_hash(label, 8)}"


def strong_entities(record):
    ke = record["spec"].get("key_entities") or record["manifest"].get("spec_summary", {}).get("key_entities")
    if not isinstance(ke, dict):
        return set()
    out = set()
    for key in STRONG_ENTITY_KEYS:
        for v in ke.get(key) or []:
            if isinstance(v, str) and len(v) >= 2:
                out.add(v.strip())
    return out


def _explicit_split(record, templates):
    task_split = (record["manifest"].get("task") or {}).get("split")
    if task_split in SPLITS:
        return task_split
    tpl_split = templates.get(group_key(record), {}).get("split")
    return tpl_split if tpl_split in SPLITS else None


def _largest_remainder(n, ratios):
    names = [k for k in ("train", "validation", "test") if k in ratios]
    total = sum(ratios[k] for k in names)
    raw = {k: n * ratios[k] / total for k in names}
    counts = {k: int(raw[k]) for k in names}
    for k in sorted(names, key=lambda k: (-(raw[k] - counts[k]), -ratios[k], 0 if k != "train" else 1))[: n - sum(counts.values())]:
        counts[k] += 1
    return counts


def allocate_components(comps_by_category, ratios):
    """Largest-remainder 70/10/20 over components, filled round-robin across categories.

    Splitting per category would starve validation/test of small categories (2 templates → 1 train / 1 test);
    a global quota with category round-robin keeps both the ratio and the category coverage.
    """
    n = sum(len(v) for v in comps_by_category.values())
    targets = _largest_remainder(n, ratios)
    filled = {k: 0 for k in targets}
    queues = {cat: sorted(roots, key=lambda r: stable_hash([cat, r])) for cat, roots in sorted(comps_by_category.items())}
    out, remaining = {}, n
    while remaining:
        for cat, queue in queues.items():
            if not queue:
                continue
            root = queue.pop(0)
            best = max(targets, key=lambda k: ((targets[k] - filled[k]) / targets[k] if targets[k] else -1.0,
                                                k == "train", k == "test"))
            out[root] = best
            filled[best] += 1
            remaining -= 1
    return out


def assign_splits(records, templates, cfg):
    ratios = cfg.get("split_ratios", {"train": 0.7, "validation": 0.1, "test": 0.2})
    challenge_difficulties = set(cfg.get("challenge_difficulties", []))
    errors, warnings, release_errors = [], [], []
    plan_cfg = cfg.get("split_plan") or {}
    try:
        plan = load_plan(cfg)
    except (FileNotFoundError, ValueError) as exc:
        plan = None
        errors.append(str(exc) if isinstance(exc, ValueError) else f"required split plan missing: {plan_cfg.get('path')}")
    planned_splits = (plan or {}).get("assignments", {})

    for r in records:
        r["group_key"] = group_key(r)
        r["category"] = (r["manifest"].get("task") or {}).get("scenario_category") or \
            r["spec"].get("scenario_category") or "unknown"

    # Components: groups connected through a shared strong entity stay together.
    uf = _UnionFind()
    entity_groups = defaultdict(set)
    for r in records:
        uf.find(r["group_key"])
        for e in strong_entities(r):
            entity_groups[e].add(r["group_key"])
    for e, groups in entity_groups.items():
        groups = sorted(groups)
        for g in groups[1:]:
            uf.union(groups[0], g)
    components = defaultdict(list)
    for g in sorted(uf.parent):
        components[uf.find(g)].append(g)
    comp_of_group = {g: uf.find(g) for g in uf.parent}

    # Explicit splits must agree inside a component.
    comp_explicit = {}
    for r in records:
        exp = _explicit_split(r, templates)
        if exp:
            root = comp_of_group[r["group_key"]]
            if root in comp_explicit and comp_explicit[root] != exp:
                errors.append(f"conflicting explicit splits in component {root}: {comp_explicit[root]} vs {exp}")
            comp_explicit[root] = exp

    # A fixed plan is authoritative for existing groups. Explicit task splits remain useful for tests and
    # overrides, but a disagreement is always reported and blocks --release.
    comp_split = {}
    auto_by_category = defaultdict(list)
    for root, groups in components.items():
        cat = next((r["category"] for r in records if comp_of_group[r["group_key"]] == root), "unknown")
        planned = {planned_splits[g] for g in groups if g in planned_splits}
        missing = [g for g in groups if g not in planned_splits]
        if planned_splits and missing:
            message = f"component {root} has groups missing from split plan: {sorted(missing)}"
            (errors if plan_cfg.get("required", True) else warnings).append(message)
        if len(planned) > 1:
            errors.append(f"strong-entity component {root} crosses planned splits: {sorted(planned)}")
        explicit = comp_explicit.get(root)
        planned_split = sorted(planned)[0] if planned else None
        if explicit and planned_split and explicit != planned_split:
            message = f"explicit split {explicit} conflicts with planned split {planned_split} for {root}"
            warnings.append(message)
            release_errors.append(message)
        chosen = explicit or planned_split
        if chosen:
            comp_split[root] = chosen
        else:
            auto_by_category[cat].append(root)
    comp_split.update(allocate_components(auto_by_category, ratios))
    group_split = {g: comp_split[root] for g, root in comp_of_group.items()}

    for r in records:
        base = group_split[r["group_key"]]
        difficulty = (r["manifest"].get("task") or {}).get("difficulty") or r["spec"].get("difficulty")
        r["difficulty"] = difficulty
        if base == "test" and (difficulty in challenge_difficulties or r["category"] == "multi_intent"):
            base = "challenge_test"
        r["split"] = base

    # Group overlap must be empty by construction; explicit conflicts are reported above.
    group_splits = defaultdict(set)
    for r in records:
        group_splits[r["group_key"]].add(r["split"])
    for g, s in group_splits.items():
        holdout = s & {"test", "challenge_test"}
        if "test" in s and "challenge_test" in s:
            warnings.append(f"group {g} is split between test and challenge_test (difficulty mix)")
        if len(s - {"test", "challenge_test"}) > 0 and holdout:
            errors.append(f"group {g} appears in train/validation and test: {sorted(s)}")
    return {"group_split": group_split, "components": {k: sorted(v) for k, v in components.items()},
            "split_plan": {"path": plan_cfg.get("path"), "version": (plan or {}).get("version"),
                           "fixed_groups": len(planned_splits)},
            "errors": errors, "warnings": warnings, "release_errors": release_errors}


# ---- 5: leakage checks -----------------------------------------------------------------------------

def _event_tokens(event):
    content = event.get("content")
    return canonical_json({"a": event.get("action"), "c": content})


def input_signature(model_input):
    return stable_hash([_event_tokens(e) for e in model_input.get("events", [])], 16)


def _token_set(model_input):
    import re
    text = canonical_json([e.get("content") for e in model_input.get("events", [])])
    return set(re.findall(r"[\u4e00-\u9fff]{2}|[A-Za-z_]{3,}|\d{2,}", text))


def leakage_report(records):
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    holdout = [r for r in records if r["split"] in ("test", "challenge_test")]
    report = {"errors": [], "warnings": [], "info": {}}

    def cross(a_list, b_list, label):
        sigs = defaultdict(list)
        for r in a_list:
            sigs[input_signature(r["model_input"])].append(r["sample_id"])
        for r in b_list:
            if input_signature(r["model_input"]) in sigs:
                report["errors"].append(f"exact duplicate input across {label}: {r['sample_id']} ~ {sigs[input_signature(r['model_input'])]}")

    cross(train, holdout, "train/test")
    cross(train, val, "train/validation")
    cross(val, holdout, "validation/test")

    # Near duplicates via inverted index on content tokens (candidates sharing ≥ 6 tokens).
    def near(a_list, b_list, label):
        index = defaultdict(set)
        for r in a_list:
            for tok in _token_set(r["model_input"]):
                index[tok].add(r["sample_id"])
        by_id = {r["sample_id"]: r for r in a_list}
        pairs = []
        for r in b_list:
            cand = Counter()
            for tok in _token_set(r["model_input"]):
                for sid in index.get(tok, ()):
                    cand[sid] += 1
            for sid, shared in cand.items():
                if shared < 6:
                    continue
                a, b = _token_set(by_id[sid]["model_input"]), _token_set(r["model_input"])
                j = len(a & b) / len(a | b)
                if j >= 0.5:
                    pairs.append((sid, r["sample_id"], round(j, 2)))
            if len(pairs) >= 50:
                break
        for a, b, j in pairs[:50]:
            report["warnings"].append(f"near-duplicate input across {label}: {a} ~ {b} (token Jaccard {j})")

    near(train, holdout, "train/test")
    near(train, val, "train/validation")

    ent_train, ent_holdout = Counter(), Counter()
    for r in train:
        for e in strong_entities(r):
            ent_train[e] += 1
    for r in holdout:
        for e in strong_entities(r):
            ent_holdout[e] += 1
    shared = sorted(((e, ent_train[e], ent_holdout[e]) for e in ent_train if e in ent_holdout),
                    key=lambda x: -(x[1] + x[2]))
    if shared:
        report["warnings"].append(f"{len(shared)} strong entities appear in both train and test (top: {shared[:10]})")
    report["info"]["entity_overlap_top"] = [{"entity": e, "train": a, "test": b} for e, a, b in shared[:30]]

    apps = {name: Counter() for name in SPLITS}
    for r in records:
        for e in r["model_input"].get("events", []):
            apps[r["split"]][e.get("app")] += 1
    train_apps = set(apps["train"])
    report["info"]["apps_only_in_holdout"] = sorted((set(apps["test"]) | set(apps["challenge_test"])) - train_apps)
    report["info"]["apps_only_in_train"] = sorted(train_apps - set(apps["test"]) - set(apps["challenge_test"]))
    report["info"]["app_frequency"] = {k: dict(v.most_common()) for k, v in apps.items()}
    return report


# ---- 4: export + stats -----------------------------------------------------------------------------

def sft_row(record, system_prompt, include_meta=False):
    row = {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": compact_json(record["model_input"])},
        {"role": "assistant", "content": compact_json(canonical_target(record["gold"]))},
    ]}
    if include_meta:
        row["meta"] = record_meta(record)
    return row


def record_meta(record):
    m = record["manifest"]
    stats = record.get("validation", {}).get("stats", {})
    return {
        "sample_id": record["sample_id"], "split": record["split"], "group_key": record["group_key"],
        "category": record["category"], "difficulty": record.get("difficulty"),
        "template_id": (m.get("task") or {}).get("template_id"),
        "intent_count": stats.get("intents"), "unrelated_count": stats.get("unrelated"),
        "approval": record.get("approval"),
    }


def test_row(record):
    return {
        "sample_id": record["sample_id"], "split": record["split"],
        "model_input": record["model_input"], "gold_label": canonical_target(record["gold"]),
        "meta": record_meta(record),
    }


def _pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def dataset_stats(records):
    by_split = Counter(r["split"] for r in records)
    by_split_cat = defaultdict(Counter)
    by_split_diff = defaultdict(Counter)
    relations, purposes, statuses, confidences, categories = Counter(), Counter(), Counter(), Counter(), Counter()
    intents_per_sample, unrelated_counts = Counter(), Counter()
    in_len, out_len, in_tok, out_tok = [], [], [], []
    system_prompt = sft_system_prompt()
    for r in records:
        by_split_cat[r["split"]][r["category"]] += 1
        by_split_diff[r["split"]][str(r.get("difficulty"))] += 1
        gold = r["gold"]
        intents_per_sample[len(gold.get("intents", []))] += 1
        unrelated_counts[len(gold.get("unrelated_event_ids", []))] += 1
        for lab in gold.get("event_labels", []):
            relations[lab.get("relation")] += 1
        for it in gold.get("intents", []):
            purposes[(it.get("category"), it.get("purpose"))] += 1
            statuses[it.get("status")] += 1
            confidences[it.get("confidence_level")] += 1
            categories[it.get("category")] += 1
        user, assistant = compact_json(r["model_input"]), compact_json(canonical_target(gold))
        in_len.append(len(user)); out_len.append(len(assistant))
        in_tok.append(estimate_tokens(system_prompt) + estimate_tokens(user)); out_tok.append(estimate_tokens(assistant))
    total_tok = [a + b for a, b in zip(in_tok, out_tok)]
    return {
        "by_split": dict(by_split),
        "by_split_category": {k: dict(v) for k, v in by_split_cat.items()},
        "by_split_difficulty": {k: dict(v) for k, v in by_split_diff.items()},
        "group_counts": {"groups": len({r["group_key"] for r in records}),
                         "groups_per_split": {s: len({r["group_key"] for r in records if r["split"] == s}) for s in SPLITS}},
        "intents_per_sample": dict(sorted(intents_per_sample.items())),
        "unrelated_per_sample": dict(sorted(unrelated_counts.items())),
        "relation_counts": dict(relations),
        "intent_categories": dict(categories),
        "purpose_by_category": {f"{c}->{p}": n for (c, p), n in sorted(purposes.items(), key=lambda kv: str(kv[0]))},
        "status": dict(statuses), "confidence": dict(confidences),
        "length": {
            "input_chars": {"p50": _pct(in_len, .5), "p90": _pct(in_len, .9), "max": max(in_len) if in_len else 0},
            "target_chars": {"p50": _pct(out_len, .5), "p90": _pct(out_len, .9), "max": max(out_len) if out_len else 0},
            "est_tokens_total": {"p50": _pct(total_tok, .5), "p90": _pct(total_tok, .9), "max": max(total_tok) if total_tok else 0},
            "est_tokens_input": {"p50": _pct(in_tok, .5), "p90": _pct(in_tok, .9), "max": max(in_tok) if in_tok else 0},
            "est_tokens_target": {"p50": _pct(out_tok, .5), "p90": _pct(out_tok, .9), "max": max(out_tok) if out_tok else 0},
        },
        "est_tokens_note": "heuristic estimate (src/utils.estimate_tokens), not a tokenizer count",
    }


def build(out_dir="data/splits", allow_unapproved=False, include_meta=False, force=False, release=False):
    cfg = load_yaml("config/dataset_config.yaml")
    taxonomy = Taxonomy.load()
    templates = {t["template_id"]: t for t in load_json("config/templates.json")["templates"]}
    if release and allow_unapproved:
        raise SystemExit("--release cannot be combined with --allow-unapproved")
    records, skipped = load_records(cfg, taxonomy, allow_unapproved=allow_unapproved)
    out_root = Path(out_dir) if Path(out_dir).is_absolute() else ROOT / out_dir
    summary = {"builder_version": BUILDER_VERSION, "generated_at": now_iso(), "allow_unapproved": allow_unapproved,
               "release": release, "n_records": len(records), "skipped": skipped}
    if not records:
        if release:
            summary["release_errors"] = ["release requires at least one approved, valid sample"]
        out_root.mkdir(parents=True, exist_ok=True)
        save_json(f"{out_root}/split_manifest.json", summary)
        if summary.get("release_errors") and not force:
            raise SystemExit("Release checks failed: no usable records")
        return summary
    split_info = assign_splits(records, templates, cfg)
    leak = leakage_report(records)
    stats = dataset_stats(records)
    coverage = coverage_report(records, cfg)
    shortcuts = audit_records(records, cfg)
    system_prompt = sft_system_prompt()

    out_root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        rows = [r for r in records if r["split"] == split]
        if split in ("train", "validation"):
            data = [sft_row(r, system_prompt, include_meta) for r in rows]
        else:
            data = [test_row(r) for r in rows]
        path = write_jsonl(f"{out_root}/{split}.jsonl", data)
        files[split] = {"path": str(path.relative_to(ROOT)) if hasattr(path, "relative_to") else str(path), "rows": len(data)}

    assignment = {r["sample_id"]: {"split": r["split"], "group": r["group_key"], "category": r["category"],
                                   "difficulty": r.get("difficulty")} for r in records}
    summary.update({
        "files": files, "assignment": assignment, "split_info": split_info, "leakage": leak, "stats": stats,
        "coverage": coverage, "shortcut_audit": shortcuts,
        "unapproved_included": sum(1 for r in records if r.get("approval") != "approved"),
    })
    save_json(f"{out_root}/split_manifest.json", summary)
    (out_root / "DATASET_REPORT.md").write_text(render_markdown(summary), encoding="utf-8")
    hard_errors = split_info["errors"] + leak["errors"]
    if release:
        hard_errors += split_info.get("release_errors", []) + coverage["errors"] + shortcuts["errors"]
    if hard_errors and not force:
        raise SystemExit("Build/release errors detected (see split_manifest.json). Re-run with --force only for diagnosis.")
    return summary


def render_markdown(summary):
    s = summary.get("stats", {})
    lines = ["# Dataset build report", "", f"- builder {summary['builder_version']}，生成于 {summary['generated_at']}",
             f"- 样本：{summary['n_records']} 条可用，跳过 {len(summary['skipped'])} 条"]
    if summary.get("allow_unapproved"):
        lines.append(f"- ⚠️ 包含 {summary.get('unapproved_included', 0)} 条**未经人工批准**的样本（--allow-unapproved），只能用于冒烟测试，不得用于训练")
    if not summary["n_records"]:
        reasons = Counter(x["reason"] for x in summary["skipped"])
        lines += ["", "跳过原因：" + ", ".join(f"{k}={v}" for k, v in reasons.items())]
        return "\n".join(lines)
    lines += ["", "## 切分", "", "| Split | 样本 | 模板组 |", "|---|---|---|"]
    for split in SPLITS:
        lines.append(f"| {split} | {s['by_split'].get(split, 0)} | {s['group_counts']['groups_per_split'].get(split, 0)} |")
    lines += ["", "### 类别 × Split", "", "| 类别 | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    cats = sorted({c for v in s["by_split_category"].values() for c in v})
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(str(s["by_split_category"].get(sp, {}).get(c, 0)) for sp in SPLITS) + " |")
    lines += ["", "### 难度 × Split", "", "| 难度 | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    diffs = sorted({d for v in s["by_split_difficulty"].values() for d in v})
    for d in diffs:
        lines.append(f"| {d} | " + " | ".join(str(s["by_split_difficulty"].get(sp, {}).get(d, 0)) for sp in SPLITS) + " |")
    lines += ["", "## 标签分布", "",
              f"- 每样本意图数：{s['intents_per_sample']}",
              f"- 每样本无关事件数：{s['unrelated_per_sample']}",
              f"- 事件关系：{s['relation_counts']}",
              f"- 意图类别：{s['intent_categories']}",
              f"- 状态：{s['status']}；置信度：{s['confidence']}", "", "purpose（类别->目的）：", ""]
    for k, v in s["purpose_by_category"].items():
        lines.append(f"- {k}: {v}")
    L = s["length"]
    lines += ["", "## 长度（估算 token，非分词器计数）", "",
              f"- 输入：p50 {L['est_tokens_input']['p50']} / p90 {L['est_tokens_input']['p90']} / max {L['est_tokens_input']['max']}",
              f"- 目标：p50 {L['est_tokens_target']['p50']} / p90 {L['est_tokens_target']['p90']} / max {L['est_tokens_target']['max']}",
              f"- 合计：p50 {L['est_tokens_total']['p50']} / p90 {L['est_tokens_total']['p90']} / max {L['est_tokens_total']['max']}", ""]
    leak = summary.get("leakage", {})
    lines += ["## 泄漏检查", "", f"- 错误：{len(leak.get('errors', []))}；警告：{len(leak.get('warnings', []))}"]
    for e in leak.get("errors", [])[:20]:
        lines.append(f"- ❌ {e}")
    for w in leak.get("warnings", [])[:20]:
        lines.append(f"- ⚠️ {w}")
    info = leak.get("info", {})
    lines += ["", f"- 仅在测试集出现的 App：{info.get('apps_only_in_holdout')}",
              f"- 仅在训练集出现的 App：{info.get('apps_only_in_train')}", ""]
    split_info = summary.get("split_info", {})
    if split_info.get("errors") or split_info.get("warnings"):
        lines += ["### 固定切分计划"]
        for message in split_info.get("errors", []):
            lines.append(f"- ❌ {message}")
        for message in split_info.get("warnings", []):
            lines.append(f"- ⚠️ {message}")
    coverage = summary.get("coverage")
    if coverage:
        lines += ["", "## Release 覆盖配额", "", f"- 训练指标：{coverage['metrics']}"]
        for error in coverage["errors"]:
            lines.append(f"- {'❌' if summary.get('release') else '⚠️'} {error}")
    shortcuts = summary.get("shortcut_audit")
    if shortcuts:
        lines += ["", "## 表面捷径审计（相关/无关二分类）", "",
                  "| Probe | Accuracy | Balanced Acc | Majority | Train / Test |", "|---|---|---|---|---|"]
        for name, metric in shortcuts.get("probes", {}).items():
            fmt = lambda x: "—" if x is None else f"{x:.3f}"
            lines.append(f"| {name} | {fmt(metric.get('accuracy'))} | {fmt(metric.get('balanced_accuracy'))} | "
                         f"{fmt(metric.get('majority_accuracy'))} | {metric.get('n_train')} / {metric.get('n_eval')} |")
        for error in shortcuts.get("errors", []):
            lines.append(f"- {'❌' if summary.get('release') else '⚠️'} {error}")
        for warning in shortcuts.get("warnings", []):
            lines.append(f"- ⚠️ {warning}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build SFT / test JSONL from approved samples.")
    parser.add_argument("--out", default="data/splits")
    parser.add_argument("--allow-unapproved", action="store_true")
    parser.add_argument("--include-meta", action="store_true")
    parser.add_argument("--force", action="store_true", help="keep files even when leakage errors are found")
    parser.add_argument("--release", action="store_true", help="enforce train coverage and shortcut-audit release gates")
    args = parser.parse_args(argv)
    summary = build(args.out, args.allow_unapproved, args.include_meta, args.force, args.release)
    print(render_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

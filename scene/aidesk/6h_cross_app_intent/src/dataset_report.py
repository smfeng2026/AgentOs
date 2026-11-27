"""Dataset / pilot quality report over data/manifests + artefacts.

Answers, without any model training:
- how many samples exist, in which status / decision / approval state, at what cost,
- how they are distributed over categories, templates, difficulties,
- what the labels look like (relations, purposes, statuses, composition deviations),
- how much the blind annotator and the generator's intended labels agree with the Gold Label
  (computed with the same evaluator used for trained models — intent F1, relation macro-F1, purpose
  over-inference, fact grounding),
- how often the adjudicator had to relabel a program-injected noise event as related (noise pool quality),
- which apps are shortcut risks (always related / always unrelated),
- length statistics for SFT planning.
"""
import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src import manifest as manifest_store
from src.utils import (ROOT, load_json, load_yaml, save_json, write_jsonl, compact_json, now_iso, estimate_tokens,
                       read_text)
from src.evaluator import evaluate, RELATIONS

REPORT_VERSION = "1.1.0"


def _load_sample(m):
    paths = {k: v.get("path") for k, v in (m.get("artefacts") or {}).items()}
    try:
        return {
            "manifest": m,
            "model_input": load_json(paths["model_input"]),
            "gold": load_json(paths["gold_label"]),
            "internal": load_json(paths["internal_labels"]),
            "audit": load_json(paths["audit_report"]),
            "spec": load_json(paths["scenario_spec"]) if paths.get("scenario_spec") else {},
        }
    except Exception:  # noqa: BLE001
        return None


def intended_prediction(internal, spec):
    """Pseudo-prediction from the generator's own labels + the planner's intents (what the pipeline intended)."""
    by_intent = defaultdict(lambda: {"supporting": set(), "related_interest": set(), "contradicting": set()})
    labels = []
    for ev in internal.get("events", []):
        rel, iids = ev.get("relation", "unrelated"), list(ev.get("intent_ids") or [])
        labels.append({"event_id": ev.get("event_id"), "relation": rel, "intent_ids": iids})
        for iid in iids:
            by_intent[iid].setdefault(rel, set()).add(ev.get("event_id"))
    intents = []
    for it in spec.get("intents", []) or []:
        if not isinstance(it, dict):
            continue
        iid = it.get("intent_id")
        sets = by_intent.get(iid, {"supporting": set(), "related_interest": set(), "contradicting": set()})
        intents.append({
            "intent_id": iid, "intent_name": it.get("intent_name"), "category": it.get("category"),
            "purpose": it.get("purpose"), "status": it.get("status"), "confidence_level": it.get("confidence_level"),
            "supporting_event_ids": sorted(sets["supporting"]), "related_interest_event_ids": sorted(sets["related_interest"]),
            "contradicting_event_ids": sorted(sets["contradicting"]),
            "confirmed_facts": [], "inferred_facts": [], "unknown_information": it.get("unknown_information") or [],
            "conflicts": [],
        })
    unrelated = sorted(ev.get("event_id") for ev in internal.get("events", []) if ev.get("relation") == "unrelated")
    return {"intents": intents, "event_labels": labels, "unrelated_event_ids": unrelated,
            "global_observations": {"intent_count": len(intents), "unrelated_event_count": len(unrelated)}}


def _meta(sample):
    m = sample["manifest"]
    task = m.get("task", {})
    return {"category": task.get("scenario_category"), "difficulty": task.get("difficulty"),
            "template_id": task.get("template_id"), "split": task.get("split")}


def _gold_rows(samples):
    return [{"sample_id": s["manifest"]["sample_id"], "model_input": s["model_input"], "gold_label": s["gold"],
             "meta": _meta(s)} for s in samples]


def _dist(values):
    return dict(Counter(values).most_common())


def _pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(q * (len(values) - 1))))], 2)


def _pick(metrics, keys):
    out = {}
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, dict) and "f1" in v:
            v = {"p": v["precision"], "r": v["recall"], "f1": v["f1"], "n": v["support"]}
        out[k] = v
    return out


AGREEMENT_KEYS = ("intent", "intent_count_accuracy", "relation_macro_f1", "relation_accuracy",
                  "purpose_accuracy", "purpose_over_inference_rate", "purpose_under_inference_rate",
                  "purpose_unknown_support", "fact_grounding_precision", "evidence_citation_accuracy")


def build_report():
    cfg = load_yaml("config/dataset_config.yaml")
    manifests = manifest_store.list_manifests()
    samples = [s for s in (_load_sample(m) for m in manifests) if s]
    target_unrelated = int(cfg.get("unrelated_events", 15))

    report = {"version": REPORT_VERSION, "generated_at": now_iso(), "n_manifests": len(manifests),
              "n_loaded": len(samples)}
    if not samples:
        return report

    # ---- overview
    usage_by_agent = defaultdict(lambda: {"calls": 0, "total_tokens": 0})
    for s in samples:
        for agent, u in (s["manifest"].get("usage") or {}).items():
            usage_by_agent[agent]["calls"] += u.get("calls", 0)
            usage_by_agent[agent]["total_tokens"] += u.get("total_tokens", 0)
    n = len(samples)
    elapsed = [s["manifest"].get("elapsed_s") or 0 for s in samples]
    tokens_per_sample = [sum(u.get("total_tokens", 0) for u in (s["manifest"].get("usage") or {}).values()) for s in samples]
    report["overview"] = {
        "n_samples": n,
        "status": _dist(s["manifest"].get("status") for s in samples),
        "decision": _dist(s["manifest"].get("decision") for s in samples),
        "approval": _dist((s["manifest"].get("approval") or {}).get("state", "pending") for s in samples),
        "flags": _dist(f for s in samples for f in s["manifest"].get("flags", [])),
        "attempts": _dist(s["manifest"].get("attempts") for s in samples),
        "validation_error_count": sum(len(s["manifest"].get("validation", {}).get("errors", [])) for s in samples),
        "validation_errors_top": _dist(e.split(":")[-1].strip()[:60] for s in samples
                                       for e in s["manifest"].get("validation", {}).get("errors", [])),
        "elapsed_s": {"p50": _pct(elapsed, .5), "mean": round(statistics.fmean(elapsed), 1), "max": max(elapsed)},
        "tokens_per_sample": {"p50": _pct(tokens_per_sample, .5), "mean": round(statistics.fmean(tokens_per_sample)),
                              "max": max(tokens_per_sample)},
        "tokens_by_agent": {a: {"calls": v["calls"], "total_tokens": v["total_tokens"],
                                "tokens_per_sample": round(v["total_tokens"] / n)} for a, v in usage_by_agent.items()},
        "projected_tokens_for_10k": round(statistics.fmean(tokens_per_sample) * 10000),
        "generator_models": _dist(s["manifest"].get("provenance", {}).get("generator_model") for s in samples),
        "judge_models": _dist(s["manifest"].get("provenance", {}).get("judge_model") for s in samples),
    }

    # ---- coverage
    report["coverage"] = {
        "category": _dist(_meta(s)["category"] for s in samples),
        "template": _dist(_meta(s)["template_id"] for s in samples),
        "difficulty": _dist(_meta(s)["difficulty"] for s in samples),
        "split_assigned_at_generation": _dist(_meta(s)["split"] for s in samples),
    }

    # ---- composition + labels
    intents_per, unrelated_per, relations = Counter(), Counter(), Counter()
    purposes, statuses, confidences, categories = Counter(), Counter(), Counter(), Counter()
    for s in samples:
        gold = s["gold"]
        intents_per[len(gold.get("intents", []) or [])] += 1
        unrelated_per[len(gold.get("unrelated_event_ids", []) or [])] += 1
        for lab in gold.get("event_labels", []) or []:
            if isinstance(lab, dict):
                relations[lab.get("relation")] += 1
        for it in gold.get("intents", []) or []:
            if isinstance(it, dict):
                purposes[f"{it.get('category')}->{it.get('purpose')}"] += 1
                statuses[it.get("status")] += 1
                confidences[it.get("confidence_level")] += 1
                categories[it.get("category")] += 1
    total_rel = sum(relations.values()) or 1
    report["composition"] = {
        "intents_per_sample": dict(sorted(intents_per.items())),
        "unrelated_per_sample": dict(sorted(unrelated_per.items())),
        "target_unrelated": target_unrelated,
        "samples_with_composition_deviation": sum(1 for s in samples if "composition_deviation" in s["manifest"].get("flags", [])),
        "relation_counts": dict(relations), "relation_share": {k: round(v / total_rel, 3) for k, v in relations.items()},
    }
    report["labels"] = {"purpose_by_category": dict(purposes.most_common()), "status": dict(statuses),
                        "confidence": dict(confidences), "intent_category": dict(categories)}

    # ---- agreement (blind / intended vs gold) using the evaluator
    gold_rows = _gold_rows(samples)
    blind_rows = [{"sample_id": s["manifest"]["sample_id"], "prediction": s["audit"].get("blind_annotation")} for s in samples]
    intended_rows = [{"sample_id": s["manifest"]["sample_id"],
                      "prediction": intended_prediction(s["internal"], s["spec"])} for s in samples]
    blind = evaluate(gold_rows, blind_rows)["overall"]
    intended = evaluate(gold_rows, intended_rows)["overall"]
    report["agreement"] = {
        "blind_vs_gold": _pick(blind, AGREEMENT_KEYS),
        "intended_vs_gold": _pick(intended, AGREEMENT_KEYS),
        "blind_vs_gold_by_category": {k: {"n": v["n_samples"], "intent_f1": v["intent"]["f1"],
                                          "relation_macro_f1": v.get("relation_macro_f1"),
                                          "purpose_over_inference": v.get("purpose_over_inference_rate")}
                                      for k, v in evaluate(gold_rows, blind_rows)["slices"]["category"].items()},
    }

    # ---- noise pool quality
    pool_ids, hard_per_sample, relabel_by_pool = Counter(), Counter(), Counter()
    noise_total = noise_relabeled = 0
    for s in samples:
        gold_rel = {x.get("event_id"): x.get("relation") for x in s["gold"].get("event_labels", []) or [] if isinstance(x, dict)}
        hard = 0
        for ev in s["internal"].get("events", []):
            if ev.get("origin") != "noise":
                continue
            noise_total += 1
            pool_ids[ev.get("pool_id")] += 1
            hard += int(bool(ev.get("hard_negative")))
            if gold_rel.get(ev.get("event_id")) not in (None, "unrelated"):
                noise_relabeled += 1
                relabel_by_pool[ev.get("pool_id")] += 1
        hard_per_sample[hard] += 1
    report["noise"] = {
        "pool_versions": _dist(s["manifest"].get("noise", {}).get("pool_version") for s in samples),
        "noise_events_total": noise_total, "noise_relabel_rate": round(noise_relabeled / noise_total, 4) if noise_total else None,
        "relabeled_by_pool_id": dict(relabel_by_pool.most_common(15)),
        "top_pool_ids": dict(pool_ids.most_common(15)),
        "distinct_pool_ids_used": len(pool_ids),
        "hard_negatives_per_sample": dict(sorted(hard_per_sample.items())),
    }

    # ---- length
    system_prompt = read_text("prompts/sft_system.txt")
    in_tok = [estimate_tokens(system_prompt) + estimate_tokens(compact_json(s["model_input"])) for s in samples]
    out_tok = [estimate_tokens(compact_json({k: v for k, v in s["gold"].items() if k != "sample_id"})) for s in samples]
    report["length"] = {"est_tokens_input": {"p50": _pct(in_tok, .5), "p90": _pct(in_tok, .9), "max": max(in_tok)},
                        "est_tokens_target": {"p50": _pct(out_tok, .5), "p90": _pct(out_tok, .9), "max": max(out_tok)}}

    # ---- app role table (shortcut risk)
    app_roles = defaultdict(lambda: [0, 0])
    for s in samples:
        gold_rel = {x.get("event_id"): x.get("relation") for x in s["gold"].get("event_labels", []) or [] if isinstance(x, dict)}
        for ev in s["model_input"].get("events", []):
            rel = gold_rel.get(ev.get("event_id"))
            if rel is None:
                continue
            app_roles[ev.get("app")][0 if rel != "unrelated" else 1] += 1
    risk = sorted(((a, r, u) for a, (r, u) in app_roles.items() if (r + u) >= 10 and (r == 0 or u == 0)),
                  key=lambda x: -(x[1] + x[2]))
    report["apps"] = {"role_counts": {a: {"related": r, "unrelated": u} for a, (r, u) in sorted(app_roles.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))},
                      "shortcut_risk_single_role": [{"app": a, "related": r, "unrelated": u} for a, r, u in risk]}

    # ---- entities / reviewer
    entities = Counter()
    for s in samples:
        ke = s["spec"].get("key_entities") if isinstance(s["spec"].get("key_entities"), dict) else {}
        for k in ("destinations", "products"):
            for v in ke.get(k) or []:
                entities[str(v)] += 1
    sev, itypes, passed = Counter(), Counter(), Counter()
    for s in samples:
        review = s["audit"].get("review") if isinstance(s["audit"].get("review"), dict) else {}
        passed[str(review.get("passed"))] += 1
        for issue in review.get("issues", []) or []:
            if isinstance(issue, dict):
                sev[issue.get("severity")] += 1
                itypes[issue.get("issue_type")] += 1
    report["reviewer"] = {"passed": dict(passed), "issues_by_severity": dict(sev), "issues_by_type": dict(itypes.most_common())}
    report["entities_top"] = dict(entities.most_common(20))

    # ---- per-sample triage table (what a human should look at first)
    rows = []
    for s in samples:
        m, meta = s["manifest"], _meta(s)
        review = s["audit"].get("review") if isinstance(s["audit"].get("review"), dict) else {}
        gold_rel = {x.get("event_id"): x.get("relation") for x in s["gold"].get("event_labels", []) or [] if isinstance(x, dict)}
        blind_rel = {x.get("event_id"): x.get("relation") for x in (s["audit"].get("blind_annotation") or {}).get("event_labels", []) or [] if isinstance(x, dict)}
        disagreements = sorted(e for e in gold_rel if blind_rel.get(e) not in (None, gold_rel[e]))
        n_gold_intents = len(s["gold"].get("intents", []) or [])
        n_blind_intents = len((s["audit"].get("blind_annotation") or {}).get("intents", []) or [])
        sev_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        worst = max((sev_rank.get(str(i.get("severity")).lower(), 0) for i in review.get("issues", []) or [] if isinstance(i, dict)), default=-1)
        rows.append({
            "sample_id": m["sample_id"], "template_id": meta["template_id"], "category": meta["category"],
            "difficulty": meta["difficulty"], "status": m.get("status"), "decision": m.get("decision"),
            "approval": (m.get("approval") or {}).get("state", "pending"), "flags": m.get("flags", []),
            "gold_intents": n_gold_intents, "blind_intents": n_blind_intents,
            "gold_unrelated": len(s["gold"].get("unrelated_event_ids", []) or []),
            "relation_disagreements": len(disagreements), "disagreed_event_ids": disagreements,
            "reviewer_passed": review.get("passed"), "reviewer_worst_severity": worst,
            "attempts": m.get("attempts"), "elapsed_s": m.get("elapsed_s"),
            "tokens": sum(u.get("total_tokens", 0) for u in (m.get("usage") or {}).values()),
        })

    def risk(r):
        return (r["status"] != "HUMAN_APPROVAL_PENDING", -len(r["flags"]), -r["relation_disagreements"],
                -max(r.get("reviewer_worst_severity") or 0, 0), -abs(r["gold_intents"] - r["blind_intents"]))
    rows.sort(key=risk)
    report["samples"] = rows
    def is_clean(r):
        return r["status"] == "HUMAN_APPROVAL_PENDING" and not r["flags"] and r["reviewer_passed"] is True \
            and r["relation_disagreements"] == 0
    report["triage"] = {
        "clean": [r["sample_id"] for r in rows if is_clean(r)],
        "check": [r["sample_id"] for r in rows if r["status"] == "HUMAN_APPROVAL_PENDING" and not is_clean(r)],
        "needs_human_review": [r["sample_id"] for r in rows if r["status"] != "HUMAN_APPROVAL_PENDING"],
    }
    return report


def render_samples_table(report, limit=200):
    rows = report.get("samples", [])[:limit]
    if not rows:
        return "# 样本分诊表\n\n无样本。"
    L = ["# 样本分诊表（按风险排序）", "",
         f"- 可快速复核：{len(report['triage']['clean'])} 条；需仔细核对：{len(report['triage']['check'])} 条；必须人工处理：{len(report['triage']['needs_human_review'])} 条",
         "", "| # | sample_id | 模板 | 难度 | 状态 | 裁决 | 标记 | 意图 gold/盲标 | 无关 | 关系分歧 | 审查 | 尝试 | 秒 | tokens |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | `{r['sample_id']}` | {r['template_id']} | {r['difficulty']} | {r['status']} | {r['decision']} | "
                 f"{','.join(r['flags']) or '-'} | {r['gold_intents']}/{r['blind_intents']} | {r['gold_unrelated']} | "
                 f"{r['relation_disagreements']} | {'pass' if r['reviewer_passed'] else 'fail'} | {r['attempts']} | {r['elapsed_s']} | {r['tokens']} |")
    return "\n".join(L)


def _fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v:.1%}" if pct else (f"{v:.3f}" if isinstance(v, float) else str(v))


def render_markdown(r, title="Dataset / pilot report"):
    if not r.get("n_loaded"):
        return f"# {title}\n\n没有可加载的样本（manifests={r['n_manifests']}）。"
    o = r["overview"]
    L = [f"# {title}", "", f"- 样本：{o['n_samples']}（manifests {r['n_manifests']}）",
         f"- 状态：{o['status']}；裁决：{o['decision']}；批准：{o['approval']}",
         f"- 标记：{o['flags']}；尝试次数：{o['attempts']}",
         f"- 耗时/样本：p50 {o['elapsed_s']['p50']}s，均值 {o['elapsed_s']['mean']}s；token/样本：p50 {o['tokens_per_sample']['p50']}，均值 {o['tokens_per_sample']['mean']}",
         f"- 各 Agent token/样本：{ {a: v['tokens_per_sample'] for a, v in o['tokens_by_agent'].items()} }；按此外推 10,000 条 ≈ {o['projected_tokens_for_10k']:,} tokens",
         f"- 生成模型 {o['generator_models']}；评审模型 {o['judge_models']}", ""]
    if o["validation_error_count"]:
        L += ["## 校验错误（需人工处理）", "", f"- 共 {o['validation_error_count']} 条；Top：" + "; ".join(f"{k} ×{v}" for k, v in o["validation_errors_top"].items()), ""]
    c = r["coverage"]
    L += ["## 覆盖", "", "| 维度 | 分布 |", "|---|---|"]
    for k, v in c.items():
        L.append(f"| {k} | {v} |")
    comp = r["composition"]
    L += ["", "## 组成与标签", "",
          f"- 每样本意图数：{comp['intents_per_sample']}；每样本无关事件数：{comp['unrelated_per_sample']}（目标 {comp['target_unrelated']}，偏离样本 {comp['samples_with_composition_deviation']}）",
          f"- 事件关系：{comp['relation_counts']}（占比 {comp['relation_share']}）",
          f"- purpose：{r['labels']['purpose_by_category']}",
          f"- status：{r['labels']['status']}；confidence：{r['labels']['confidence']}", ""]
    a = r["agreement"]
    def row(name, m):
        i = m.get("intent") or {}
        return (f"| {name} | {_fmt(i.get('f1'))} | {_fmt(m.get('relation_macro_f1'))} | {_fmt(m.get('purpose_accuracy'), True)} | "
                f"{_fmt(m.get('purpose_over_inference_rate'), True)} | {_fmt(m.get('fact_grounding_precision'), True)} |")
    L += ["## 与 Gold 的一致性（同一评测器）", "", "| 标注来源 | Intent F1 | Relation macro-F1 | Purpose Acc | Purpose 过度推断 | Fact Grounding(引用式) |", "|---|---|---|---|---|---|"]
    L.append(row("盲标 vs Gold", a["blind_vs_gold"]))
    L.append(row("生成器预设 vs Gold", a["intended_vs_gold"]))
    L += ["", "盲标 vs Gold 按类别：", "", "| 类别 | n | Intent F1 | Relation macro-F1 | 过度推断 |", "|---|---|---|---|---|"]
    for k, v in a["blind_vs_gold_by_category"].items():
        L.append(f"| {k} | {v['n']} | {_fmt(v['intent_f1'])} | {_fmt(v['relation_macro_f1'])} | {_fmt(v['purpose_over_inference'], True)} |")
    nz = r["noise"]
    L += ["", "## 噪声池", "",
          f"- 版本 {nz['pool_versions']}；使用到的噪声模板 {nz['distinct_pool_ids_used']} 种；每样本难负例数 {nz['hard_negatives_per_sample']}",
          f"- **噪声被裁决器改标为相关：{nz['noise_relabel_rate']}**（{nz['relabeled_by_pool_id']}）",
          f"- Top 噪声模板：{nz['top_pool_ids']}", ""]
    ln = r["length"]
    L += ["## 长度（估算 token）", "",
          f"- 输入 p50/p90/max：{ln['est_tokens_input']['p50']} / {ln['est_tokens_input']['p90']} / {ln['est_tokens_input']['max']}",
          f"- 目标 p50/p90/max：{ln['est_tokens_target']['p50']} / {ln['est_tokens_target']['p90']} / {ln['est_tokens_target']['max']}", ""]
    L += ["## App 角色与捷径风险", "",
          f"- 只以一种角色出现（≥10 条事件）的 App：{[x['app'] for x in r['apps']['shortcut_risk_single_role']]}",
          f"- 角色计数 Top 15：{dict(list(r['apps']['role_counts'].items())[:15])}", ""]
    L += ["## 审查员", "", f"- passed：{r['reviewer']['passed']}；问题严重度：{r['reviewer']['issues_by_severity']}；类型：{r['reviewer']['issues_by_type']}", ""]
    L += ["## 实体 Top", "", f"{r['entities_top']}", ""]
    return "\n".join(L)


def run(out_dir="data/reports"):
    report = build_report()
    out_root = Path(out_dir) if Path(out_dir).is_absolute() else ROOT / out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    save_json(f"{out_root}/dataset_report.json", report)
    md = render_markdown(report)
    (out_root / "DATASET_REPORT.md").write_text(md, encoding="utf-8")
    (out_root / "SAMPLES_TABLE.md").write_text(render_samples_table(report), encoding="utf-8")
    write_jsonl(f"{out_root}/samples_table.jsonl", report.get("samples", []))
    return report, md


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dataset / pilot quality report.")
    parser.add_argument("--out", default="data/reports")
    args = parser.parse_args(argv)
    _, md = run(args.out)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

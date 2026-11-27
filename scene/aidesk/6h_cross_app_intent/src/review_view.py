"""Human review view: everything a reviewer needs to approve / reject one sample, in one screen.

  python -m src.cli review --sample-id c01-s04-1a2b3c4d
  python -m src.cli review --pending --limit 5          # first N samples awaiting approval
  python -m src.cli review --export data/review/review_sheet.md   # all pending samples → one markdown
"""
import argparse

from src import manifest as manifest_store
from src.utils import load_json, now_iso

REL_MARK = {"supporting": "S", "related_interest": "i", "contradicting": "C", "unrelated": "."}
DIFF_REL = {"supporting", "related_interest", "contradicting", "unrelated"}


def _short(value, n=70):
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= n else text[: n - 1] + "…"


def load_sample(sample_id):
    m = manifest_store.load_manifest(sample_id)
    paths = {k: v.get("path") for k, v in (m.get("artefacts") or {}).items()}
    return {
        "manifest": m,
        "model_input": load_json(paths["model_input"]),
        "gold": load_json(paths["gold_label"]),
        "internal": load_json(paths["internal_labels"]),
        "audit": load_json(paths["audit_report"]),
        "spec": load_json(paths["scenario_spec"]) if paths.get("scenario_spec") else {},
    }


def render_review(sample, with_events=True):
    m, gold, audit, spec = sample["manifest"], sample["gold"], sample["audit"], sample["spec"]
    task = m.get("task", {})
    gold_rel = {x.get("event_id"): x for x in gold.get("event_labels", [])}
    blind = audit.get("blind_annotation") or {}
    blind_rel = {x.get("event_id"): x.get("relation") for x in blind.get("event_labels", []) or []}
    internal = {x.get("event_id"): x for x in sample["internal"].get("events", [])}
    L = [f"## {m['sample_id']}", "",
         f"- 模板 `{task.get('template_id')}` · 类别 {task.get('scenario_category')} · 难度 {task.get('difficulty')} · "
         f"窗口 {task.get('window_start')} → {task.get('window_end')}",
         f"- 状态 **{m.get('status')}** · 裁决 {m.get('decision')} · 标记 {m.get('flags')} · 审批 {(m.get('approval') or {}).get('state')}",
         f"- 场景：{spec.get('title')}；实体 {spec.get('key_entities')}", ""]
    if m.get("validation", {}).get("errors") or m.get("validation", {}).get("warnings"):
        L += [f"- 校验：errors={m['validation']['errors']} warnings={m['validation']['warnings']}", ""]
    if with_events:
        L += ["| id | 时间 | App | 动作 | 内容 | 噪声 | Gold | 盲标 | 意图 |", "|---|---|---|---|---|---|---|---|---|"]
        for e in sample["model_input"].get("events", []):
            eid = e["event_id"]
            rel = gold_rel.get(eid, {}).get("relation", "?")
            brel = blind_rel.get(eid, "?")
            origin = internal.get(eid, {})
            noise = "" if origin.get("origin") == "related" else ("难负例" if origin.get("hard_negative") else "噪声")
            intent_ids = ",".join(gold_rel.get(eid, {}).get("intent_ids") or [])
            mark = f"{REL_MARK.get(rel, '?')}"
            bmark = f"{REL_MARK.get(brel, '?')}"
            flag = " ⚠️" if (brel in DIFF_REL and brel != rel) else ""
            L.append(f"| {eid} | {e['timestamp'][11:16]} | {e.get('app')} | {e.get('action')} | "
                     f"{_short(_content_text(e.get('content')), 60)} | {noise} | {mark} | {bmark}{flag} | {intent_ids} |")
        L += ["", "Gold：S=supporting i=related_interest C=contradicting .=unrelated；⚠️ = 盲标与 Gold 不一致（重点核对）", ""]
    L.append("### Gold 意图")
    for i in gold.get("intents", []):
        L.append(f"- **{i.get('intent_name')}** [{i.get('intent_id')}] · {i.get('category')} / {i.get('purpose')} · "
                 f"{i.get('status')} · 置信 {i.get('confidence_level')}")
        L.append(f"  - supporting {i.get('supporting_event_ids')} · related_interest {i.get('related_interest_event_ids')} · "
                 f"contradicting {i.get('contradicting_event_ids')}")
        for f in i.get("confirmed_facts", []):
            L.append(f"  - 已确认：{_short(f.get('fact'), 90)} ← {f.get('source_event_ids')}")
        for f in i.get("inferred_facts", []):
            L.append(f"  - 推断：{_short(f.get('fact'), 90)} ← {f.get('source_event_ids')}")
        if i.get("conflicts"):
            L.append(f"  - 冲突：{i['conflicts']}")
        if i.get("unknown_information"):
            L.append(f"  - 未知：{i['unknown_information']}")
    if len(gold.get("intents", [])) > 1:
        L.append(f"- 全局：{gold.get('global_observations')}")
    L += ["", "### 盲标（只看模型输入）", f"- 意图：{[i.get('intent_name') for i in blind.get('intents', []) or []]}"]
    review = audit.get("review") or {}
    L += ["", f"### 审查员 passed={review.get('passed')}"]
    for issue in review.get("issues", []) or []:
        L.append(f"- [{issue.get('severity')}] {issue.get('issue_type')}：{_short(issue.get('description'), 160)}"
                 + (f"（{issue.get('suggested_fix')}）" if issue.get("suggested_fix") else ""))
    adj = audit.get("adjudication") or {}
    L += ["", f"### 裁决 {audit.get('decision')}", f"{adj.get('rationale')}", "",
          "### 审核清单", "1. 每条 S 事件是否真是决定性证据？搜索/浏览/收藏类是否被错标为 S？",
          "2. 每个噪声事件与所有意图确实无关？（难负例尤其要看）", "3. 意图粒度：是否把一个意图的目的拆成了独立意图，或漏掉了独立证据簇？",
          "4. purpose / status / 置信度是否过度推断？未知项是否其实已知？", "5. 已确认事实都能在引用事件里找到原文依据？", ""]
    return "\n".join(L)


def _content_text(content):
    if isinstance(content, dict):
        return "；".join(f"{k}={v if not isinstance(v, (list, dict)) else '…'}" for k, v in content.items())
    return str(content)


def pending_sample_ids(limit=None):
    ids = []
    for m in manifest_store.list_manifests():
        if (m.get("approval") or {}).get("state", "pending") == "pending" and m.get("artefacts", {}).get("gold_label"):
            ids.append(m["sample_id"])
    ids.sort()
    return ids[:limit] if limit else ids


def export(path, limit=None):
    ids = pending_sample_ids(limit)
    parts = [f"# 人工审核清单 · {now_iso()} · 共 {len(ids)} 条待审", "",
             "批准：`python -m src.cli approve --sample-id <id> --by 名字`；拒绝：`reject ... --note 原因`。", ""]
    for sid in ids:
        try:
            parts.append(render_review(load_sample(sid)))
        except Exception as exc:  # noqa: BLE001
            parts.append(f"## {sid}\n\n无法加载：{exc}\n")
    from src.utils import ROOT
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out, ids


def main(argv=None):
    parser = argparse.ArgumentParser(description="Human review view")
    parser.add_argument("--sample-id")
    parser.add_argument("--pending", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--export", default=None, help="write all pending samples to this markdown file")
    args = parser.parse_args(argv)
    if args.export:
        out, ids = export(args.export, args.limit)
        print(f"wrote {out} ({len(ids)} samples)")
        return 0
    ids = [args.sample_id] if args.sample_id else (pending_sample_ids(args.limit) if args.pending else [])
    for sid in ids:
        print(render_review(load_sample(sid), with_events=not args.no_events))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

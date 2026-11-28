"""Evaluator for cross-app intent-reasoning predictions against Gold Labels.

Prediction rows (JSONL, one per sample) may look like:
  {"sample_id": "...", "prediction": {<gold-shaped object>}}
  {"sample_id": "...", "raw": "<whatever the on-device model printed>"}   ← parsed with extract_json
Gold rows come from data/splits/{test,challenge_test}.jsonl (model_input + gold_label + meta).

Intent names are free text, so "intent P/R/F1" needs an alignment first: predicted and gold intents
are matched one-to-one by greedy overlap (Jaccard) of their evidence event sets. All intent-level
attribute metrics are computed on matched pairs; unmatched predictions are FP, unmatched gold FN.

Fact Grounding Precision here is *citation-based*: a predicted fact counts as grounded when it has a
non-empty fact text, every cited event exists, and every cited event is gold-related to the matched
intent. It does NOT verify that the cited events entail the sentence — use --llm-judge for that.
"""
import argparse
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.utils import ROOT, load_jsonl, write_jsonl, save_json, extract_json, now_iso, read_text

RELATIONS = ("supporting", "related_interest", "contradicting", "unrelated")
RELATED_RELATIONS = RELATIONS[:3]
RELATION_FIELDS = {
    "supporting": "supporting_event_ids",
    "related_interest": "related_interest_event_ids",
    "contradicting": "contradicting_event_ids",
}
ATTRIBUTES = ("category", "purpose", "status", "confidence_level")
FACT_FIELDS = ("confirmed_facts", "inferred_facts")

# First-version targets from the design doc (§9.5) — R&D targets, not measured results.
DESIGN_TARGETS = {
    "intent_macro_f1": 0.85,
    "relation_macro_f1": 0.80,
    "supporting_f1": 0.85,
    "related_interest_f1": 0.75,
    "purpose_accuracy": 0.85,
    "fact_grounding_precision": 0.95,
    "purpose_unknown_recall": 0.90,
}


def _as_list(value):
    return value if isinstance(value, list) else []


def _intents(obj):
    return [i for i in _as_list((obj or {}).get("intents")) if isinstance(i, dict)]


def related_set(intent):
    out = set()
    for field in RELATION_FIELDS.values():
        out.update(x for x in _as_list(intent.get(field)) if isinstance(x, str))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": tp + fn}


def match_intents(gold_intents, pred_intents, threshold=0.4):
    """Greedy one-to-one matching on evidence-set Jaccard ≥ threshold. Returns [(gi, pi, score)]."""
    candidates = []
    for gi, g in enumerate(gold_intents):
        gs = related_set(g)
        for pi, p in enumerate(pred_intents):
            ps = related_set(p)
            union = gs | ps
            score = len(gs & ps) / len(union) if union else 0.0
            if score >= threshold:
                candidates.append((score, gi, pi))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_g, used_p, matches = set(), set(), []
    for score, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, round(score, 3)))
    return matches


def normalise_prediction(row):
    """Return the prediction object for a row, or raise ValueError with the reason."""
    if not isinstance(row, dict):
        raise ValueError("row is not an object")
    pred = row.get("prediction")
    if pred is None and row.get("raw") is not None:
        pred = extract_json(str(row["raw"]))
    if pred is None and "intents" in row and "event_labels" in row:
        pred = row  # bare prediction carrying its own sample_id
    if not isinstance(pred, dict):
        raise ValueError("no prediction object")
    if isinstance(pred.get("gold_label"), dict):
        pred = pred["gold_label"]
    for field in ("intents", "event_labels"):
        if not isinstance(pred.get(field), list):
            raise ValueError(f"prediction field '{field}' must be a list")
    return pred


def _best_overlap(target_set, others):
    best, best_score = None, 0.0
    for idx, other in enumerate(others):
        os_ = related_set(other)
        union = target_set | os_
        score = len(target_set & os_) / len(union) if union else 0.0
        if score > best_score:
            best, best_score = idx, score
    return best, best_score


def evaluate_sample(gold, pred, event_ids, threshold=0.4):
    event_ids = set(event_ids or [])
    g_intents, p_intents = _intents(gold), _intents(pred)
    matches = match_intents(g_intents, p_intents, threshold)
    matched_g = {gi for gi, _, _ in matches}
    matched_p = {pi for _, pi, _ in matches}
    s = {
        "intent_tp": len(matches), "intent_fp": len(p_intents) - len(matches),
        "intent_fn": len(g_intents) - len(matches),
        "intent_count_exact": int(len(p_intents) == len(g_intents)),
        "cat_counts": defaultdict(lambda: [0, 0, 0]),    # category → [tp, fp, fn]
        "attr": {a: [0, 0] for a in ATTRIBUTES},           # [correct, total]
        "purpose_pairs": [],
        "confusion": Counter(), "missing_event_labels": 0, "unknown_event_labels": 0,
        "eti": {rel: [0, 0, 0] for rel in RELATED_RELATIONS},   # [tp, fp, fn]
        "evidence_jaccard": [],
        "citation": {"all": 0, "valid_all": 0, "matched": 0, "in_gold_related": 0},
        "facts": {"matched": 0, "grounded": 0, "unmatched_pred": 0},
        "unknown_ratio": [],
        "multi": {"g": 0, "p": 0, "merge": 0, "split": 0},
        "pairs": [],
    }
    gold_category_of_match = {}
    for gi, pi, score in matches:
        g, p = g_intents[gi], p_intents[pi]
        gold_category_of_match[pi] = gi
        s["cat_counts"][g.get("category") or "unknown"][0] += 1
        for a in ATTRIBUTES:
            s["attr"][a][1] += 1
            s["attr"][a][0] += int(g.get(a) == p.get(a))
        s["purpose_pairs"].append((g.get("purpose"), p.get("purpose")))
        gs_all, ps_all = related_set(g), related_set(p)
        if gs_all | ps_all:
            s["evidence_jaccard"].append(len(gs_all & ps_all) / len(gs_all | ps_all))
        g_related = gs_all
        for rel, field in RELATION_FIELDS.items():
            gs = set(x for x in _as_list(g.get(field)) if x in event_ids)
            ps = set(x for x in _as_list(p.get(field)) if x in event_ids)
            s["eti"][rel][0] += len(gs & ps)
            s["eti"][rel][1] += len(ps - gs)
            s["eti"][rel][2] += len(gs - ps)
        for field in FACT_FIELDS:
            for fact in _as_list(p.get(field)):
                if not isinstance(fact, dict):
                    s["facts"]["matched"] += 1
                    continue
                refs = [r for r in _as_list(fact.get("source_event_ids")) if isinstance(r, str)]
                s["facts"]["matched"] += 1
                grounded = bool(str(fact.get("fact", "")).strip()) and bool(refs) and set(refs) <= event_ids \
                    and set(refs) <= g_related
                s["facts"]["grounded"] += int(grounded)
        s["pairs"].append({"gold": g.get("intent_id"), "pred": p.get("intent_id"), "score": score})
    for gi, g in enumerate(g_intents):
        if gi not in matched_g:
            s["cat_counts"][g.get("category") or "unknown"][2] += 1
    for pi, p in enumerate(p_intents):
        if pi not in matched_p:
            s["cat_counts"][p.get("category") or "unknown"][1] += 1
            for field in FACT_FIELDS:
                s["facts"]["unmatched_pred"] += len(_as_list(p.get(field)))

    # Evidence citation validity over every predicted intent (matched or not).
    for pi, p in enumerate(p_intents):
        g_related = related_set(g_intents[gold_category_of_match[pi]]) if pi in matched_p else None
        for field in FACT_FIELDS:
            for fact in _as_list(p.get(field)):
                if not isinstance(fact, dict):
                    continue
                refs = [r for r in _as_list(fact.get("source_event_ids")) if isinstance(r, str)]
                s["citation"]["all"] += len(refs)
                s["citation"]["valid_all"] += sum(1 for r in refs if r in event_ids)
                if g_related is not None:
                    s["citation"]["matched"] += len(refs)
                    s["citation"]["in_gold_related"] += sum(1 for r in refs if r in g_related)

    # Per-event relation classification (4 classes).
    g_rel = {x.get("event_id"): x.get("relation") for x in _as_list(gold.get("event_labels")) if isinstance(x, dict)}
    p_rel = {x.get("event_id"): x.get("relation") for x in _as_list(pred.get("event_labels")) if isinstance(x, dict)}
    for eid in event_ids:
        truth = g_rel.get(eid)
        guess = p_rel.get(eid)
        if guess is None:
            guess = "unrelated"
            s["missing_event_labels"] += 1
        if guess not in RELATIONS:
            s["unknown_event_labels"] += 1
            guess = "unrelated"
        s["confusion"][(truth, guess)] += 1

    # unknown_information list-size ratio on matched pairs (descriptive; semantic recall needs a judge).
    for gi, pi, _ in matches:
        g_len = len(_as_list(g_intents[gi].get("unknown_information")))
        p_len = len(_as_list(p_intents[pi].get("unknown_information")))
        s["unknown_ratio"].append(p_len / g_len if g_len else (1.0 if p_len == 0 else 0.0))

    # Multi-intent merge / split errors.
    #   merge: an unmatched gold intent overlapped by a prediction that is already matched to another gold intent
    #   split: an unmatched prediction overlapping a matched gold intent, or one gold intent fragmented into ≥ 2
    s["multi"]["g"] += len(g_intents) if len(g_intents) >= 2 else 0   # gold intents inside multi-intent samples
    s["multi"]["p"] += len(p_intents) if len(p_intents) >= 2 else 0   # predictions inside multi-prediction samples
    unmatched_p_by_g = defaultdict(list)
    for pi, p in enumerate(p_intents):
        if pi in matched_p:
            continue
        ps = related_set(p)
        for gi, g in enumerate(g_intents):
            gs = related_set(g)
            if gs & ps:
                unmatched_p_by_g[gi].append(pi)
    for gi, g in enumerate(g_intents):
        if gi in matched_g:
            continue
        gs = related_set(g)
        if any(gs & related_set(p) for pi, p in enumerate(p_intents) if pi in matched_p):
            s["multi"]["merge"] += 1
    for gi, pis in unmatched_p_by_g.items():
        if gi in matched_g:
            s["multi"]["split"] += len(pis)
        elif len(pis) >= 2:
            s["multi"]["split"] += len(pis) - 1
    return s


class Aggregator:
    def __init__(self):
        self.n = 0
        self.intent = [0, 0, 0]
        self.intent_by_category = defaultdict(lambda: [0, 0, 0])
        self.intent_count_exact = 0
        self.attr = {a: [0, 0] for a in ATTRIBUTES}
        self.purpose_pairs = []
        self.confusion = Counter()
        self.missing_event_labels = 0
        self.unknown_event_labels = 0
        self.eti = {rel: [0, 0, 0] for rel in RELATED_RELATIONS}
        self.evidence_jaccard = []
        self.citation = {"all": 0, "valid_all": 0, "matched": 0, "in_gold_related": 0}
        self.facts = {"matched": 0, "grounded": 0, "unmatched_pred": 0}
        self.unknown_ratio = []
        self.multi = {"g": 0, "p": 0, "merge": 0, "split": 0}

    def add(self, s):
        self.n += 1
        for i, k in enumerate(("intent_tp", "intent_fp", "intent_fn")):
            self.intent[i] += s[k]
        for cat, counts in s["cat_counts"].items():
            for i in range(3):
                self.intent_by_category[cat][i] += counts[i]
        self.intent_count_exact += s["intent_count_exact"]
        for a in ATTRIBUTES:
            self.attr[a][0] += s["attr"][a][0]
            self.attr[a][1] += s["attr"][a][1]
        self.purpose_pairs.extend(s["purpose_pairs"])
        self.confusion.update(s["confusion"])
        self.missing_event_labels += s["missing_event_labels"]
        self.unknown_event_labels += s["unknown_event_labels"]
        for rel in RELATED_RELATIONS:
            for i in range(3):
                self.eti[rel][i] += s["eti"][rel][i]
        self.evidence_jaccard.extend(s["evidence_jaccard"])
        for k in self.citation:
            self.citation[k] += s["citation"][k]
        for k in self.facts:
            self.facts[k] += s["facts"][k]
        self.unknown_ratio.extend(s["unknown_ratio"])
        for k in self.multi:
            self.multi[k] += s["multi"][k]

    def metrics(self):
        tp, fp, fn = self.intent
        out = {"n_samples": self.n}
        if self.n == 0:
            return out
        out["intent"] = prf(tp, fp, fn)
        out["intent_macro_f1"] = round(statistics.fmean(prf(*c)["f1"] for c in self.intent_by_category.values()), 4) \
            if self.intent_by_category else None
        out["intent_by_category"] = {cat: prf(*c) for cat, c in sorted(self.intent_by_category.items())}
        out["intent_count_accuracy"] = round(self.intent_count_exact / self.n, 4)
        for a in ATTRIBUTES:
            c, t = self.attr[a]
            out[f"{a}_accuracy"] = round(c / t, 4) if t else None
        pairs = [(g or "unknown", p or "unknown") for g, p in self.purpose_pairs]
        out["purpose_support"] = len(pairs)
        if pairs:
            out["purpose_macro_f1"] = round(_macro_f1(pairs), 4)
            out["purpose_per_class"] = _per_class_prf(pairs)
            unk = [(g, p) for g, p in pairs if g == "unknown"]
            known = [(g, p) for g, p in pairs if g != "unknown"]
            out["purpose_unknown_support"] = len(unk)
            out["purpose_known_support"] = len(known)
            out["purpose_over_inference_rate"] = round(sum(1 for g, p in unk if p != "unknown") / len(unk), 4) if unk else None
            out["purpose_unknown_recall"] = round(sum(1 for g, p in unk if p == "unknown") / len(unk), 4) if unk else None
            out["purpose_under_inference_rate"] = round(sum(1 for g, p in known if p == "unknown") / len(known), 4) if known else None
        rel = {}
        for r in RELATIONS:
            tp_r = self.confusion.get((r, r), 0)
            fp_r = sum(v for (t, p), v in self.confusion.items() if p == r and t != r)
            fn_r = sum(v for (t, p), v in self.confusion.items() if t == r and p != r)
            rel[r] = prf(tp_r, fp_r, fn_r)
        present = [r for r in RELATIONS if rel[r]["support"] or rel[r]["precision"]]
        out["relation_per_class"] = rel
        out["relation_macro_f1"] = round(sum(rel[r]["f1"] for r in present) / len(present), 4) if present else None
        total = sum(self.confusion.values())
        out["relation_accuracy"] = round(sum(v for (t, p), v in self.confusion.items() if t == p) / max(1, total), 4)
        out["relation_confusion"] = {f"{t}->{p}": v for (t, p), v in sorted(self.confusion.items(), key=lambda kv: -kv[1])}
        out["missing_event_labels"] = self.missing_event_labels
        out["unknown_relation_values"] = self.unknown_event_labels
        eti = {rel: prf(*self.eti[rel]) for rel in RELATED_RELATIONS}
        micro = [sum(self.eti[rel][i] for rel in RELATED_RELATIONS) for i in range(3)]
        eti["micro"] = prf(*micro)
        eti["evidence_jaccard_mean"] = round(statistics.fmean(self.evidence_jaccard), 4) if self.evidence_jaccard else None
        out["event_to_intent"] = eti
        c = self.citation
        out["evidence_citation_validity"] = round(c["valid_all"] / c["all"], 4) if c["all"] else None
        out["evidence_citation_accuracy"] = round(c["in_gold_related"] / c["matched"], 4) if c["matched"] else None
        f = self.facts
        out["facts_evaluated"] = f["matched"]
        out["fact_grounding_precision"] = round(f["grounded"] / f["matched"], 4) if f["matched"] else None
        out["unsupported_claim_rate"] = round(1 - f["grounded"] / f["matched"], 4) if f["matched"] else None
        out["facts_in_unmatched_intents"] = f["unmatched_pred"]
        out["unknown_information_count_ratio_mean"] = round(statistics.fmean(self.unknown_ratio), 4) if self.unknown_ratio else None
        m = self.multi
        out["multi_intent"] = {
            "gold_intents": m["g"], "merge_error_rate": round(m["merge"] / m["g"], 4) if m["g"] else None,
            "split_error_rate": round(m["split"] / m["p"], 4) if m["p"] else None,
        }
        return out


def _per_class_prf(pairs):
    labels = sorted({x for pair in pairs for x in pair})
    out = {}
    for lab in labels:
        tp = sum(1 for g, p in pairs if p == lab and g == lab)
        fp = sum(1 for g, p in pairs if p == lab and g != lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        out[lab] = prf(tp, fp, fn)
    return out


def _macro_f1(pairs):
    per = _per_class_prf(pairs)
    return statistics.fmean(v["f1"] for v in per.values())


def evaluate(gold_rows, prediction_rows, threshold=0.4):
    preds = {}
    format_errors = []
    for row in prediction_rows:
        sid = row.get("sample_id") if isinstance(row, dict) else None
        if not sid:
            format_errors.append({"sample_id": None, "reason": "row has no sample_id"})
            continue
        try:
            preds[sid] = normalise_prediction(row)
        except Exception as exc:  # noqa: BLE001
            format_errors.append({"sample_id": sid, "reason": str(exc)[:200]})
    agg_all = Aggregator()
    agg_by = {"category": defaultdict(Aggregator), "difficulty": defaultdict(Aggregator), "template": defaultdict(Aggregator)}
    per_sample = []
    missing_predictions = []
    format_error_sids = {e.get("sample_id") for e in format_errors}
    for gold_row in gold_rows:
        sid = gold_row.get("sample_id")
        meta = gold_row.get("meta", {}) or {}
        gold = gold_row.get("gold_label") or gold_row
        model_input = gold_row.get("model_input") or {}
        event_ids = [e.get("event_id") for e in _as_list(model_input.get("events")) if isinstance(e, dict)]
        pred = preds.get(sid)
        if pred is None:
            if sid not in format_error_sids:      # format errors are counted once, in their own bucket
                missing_predictions.append(sid)
            continue
        s = evaluate_sample(gold, pred, set(event_ids), threshold)
        agg_all.add(s)
        for key, value in (("category", meta.get("category")), ("difficulty", meta.get("difficulty")),
                           ("template", meta.get("template_id"))):
            agg_by[key][value or "unknown"].add(s)
        per_sample.append({
            "sample_id": sid, "intent_tp": s["intent_tp"], "intent_fp": s["intent_fp"], "intent_fn": s["intent_fn"],
            "purpose_pairs": s["purpose_pairs"], "pairs": s["pairs"],
            "fact_grounded": s["facts"]["grounded"], "facts": s["facts"]["matched"],
        })
    return {
        "meta": {
            "generated_at": now_iso(), "match_threshold": threshold,
            "n_gold_rows": len(gold_rows), "n_predictions": len(preds),
            "n_evaluated": agg_all.n, "n_missing_predictions": len(missing_predictions),
            "n_format_errors": len(format_errors),
            "format_valid_rate": round(agg_all.n / max(1, agg_all.n + len(format_errors)), 4),
        },
        "overall": agg_all.metrics(),
        "slices": {key: {k: agg.metrics() for k, agg in sorted(aggs.items(), key=lambda kv: str(kv[0]))}
                   for key, aggs in agg_by.items()},
        "format_errors": format_errors[:200],
        "missing_predictions": missing_predictions[:200],
        "per_sample": per_sample,
        "design_targets": DESIGN_TARGETS,
    }


# ---- optional LLM judge for fact entailment -------------------------------------------------------

def judge_facts(client, gold_rows, prediction_rows, threshold=0.4):
    """Ask a judge model whether each predicted fact (matched intents) is supported by its cited events."""
    events_by_sid = {r.get("sample_id"): {e.get("event_id"): e for e in _as_list((r.get("model_input") or {}).get("events")) if isinstance(e, dict)}
                     for r in gold_rows}
    verdicts = []  # (sample_id, supported: bool)
    for row in prediction_rows:
        sid = row.get("sample_id")
        try:
            pred = normalise_prediction(row)
        except Exception:  # noqa: BLE001
            continue
        gold_row = next((r for r in gold_rows if r.get("sample_id") == sid), None)
        if not gold_row:
            continue
        gold = gold_row.get("gold_label") or gold_row
        events = events_by_sid.get(sid, {})
        p_intents = _intents(pred)
        matches = match_intents(_intents(gold), p_intents, threshold)
        for _, pi, _ in matches:
            facts = [f for field in FACT_FIELDS for f in _as_list(p_intents[pi].get(field)) if isinstance(f, dict)]
            if not facts:
                continue
            items = []
            for f in facts:
                refs = [r for r in _as_list(f.get("source_event_ids")) if isinstance(r, str)]
                items.append({"fact": f.get("fact"), "cited_events": [{"event_id": r, **events.get(r, {})} for r in refs if r in events]})
            if not any(it["cited_events"] for it in items):
                verdicts.extend((sid, False) for _ in items)
                continue
            try:
                answer = client.generate_json(
                    "你是事实核查员。对每条 fact，判断它是否能被 cited_events 中的事件内容直接支持（事件里明示或可直接推出）。"
                    "只输出 JSON：{\"verdicts\": [{\"index\": 0, \"supported\": true|false}]}，index 与输入顺序一致。",
                    {"facts": items}, temperature=0.0)
                flags = {int(v.get("index")): bool(v.get("supported")) for v in _as_list(answer.get("verdicts")) if isinstance(v, dict)}
                verdicts.extend((sid, flags.get(i, False)) for i in range(len(items)))
            except Exception:  # noqa: BLE001
                verdicts.extend((sid, False) for _ in items)
    return verdicts


def render_markdown(result, title="Evaluation report"):
    o = result["overall"]
    m = result["meta"]
    lines = [f"# {title}", "", f"- 评估样本：{o.get('n_samples', 0)}（预测 {m['n_predictions']}，缺失 {m['n_missing_predictions']}，格式错误 {m['n_format_errors']}，格式有效率 {m['format_valid_rate']:.1%}）",
             f"- 意图匹配阈值（证据 Jaccard）：{m['match_threshold']}", ""]
    if not o.get("n_samples"):
        lines.append("没有可评估的样本。")
        return "\n".join(lines)
    intent = o["intent"]
    lines += ["## 意图识别", "", f"- Intent P / R / F1：{intent['precision']:.3f} / {intent['recall']:.3f} / {intent['f1']:.3f}（Gold 意图数 {intent['support']}）",
              f"- 意图数量完全正确率：{o['intent_count_accuracy']:.1%}",
              f"- 类别 / 状态 / 置信度准确率（匹配对）：{o['category_accuracy']} / {o['status_accuracy']} / {o['confidence_level_accuracy']}", ""]
    lines += ["## 事件关系（四分类）", "", "| 关系 | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for r in RELATIONS:
        v = o["relation_per_class"][r]
        lines.append(f"| {r} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    lines += [f"| **macro-F1** | | | **{o['relation_macro_f1']}** | |", f"- 事件关系准确率：{o['relation_accuracy']:.1%}；缺失事件标签：{o['missing_event_labels']}", "",
              "## 事件→意图关联（匹配对）", "", "| 关系 | Precision | Recall | F1 |", "|---|---|---|---|"]
    for r in RELATED_RELATIONS + ("micro",):
        v = o["event_to_intent"][r]
        lines.append(f"| {r} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    lines += [f"- 证据集合 Jaccard 均值：{o['event_to_intent']['evidence_jaccard_mean']}", ""]
    pa = o.get("purpose_accuracy")
    lines += ["## 目的判断（匹配对）", "",
              f"- Purpose Accuracy：{'—' if pa is None else format(pa, '.3f')}；Macro-F1：{o.get('purpose_macro_f1')}",
              f"- **Purpose Over-inference Rate**（Gold=unknown 被判成具体目的）：{o.get('purpose_over_inference_rate')}（unknown 支持数 {o.get('purpose_unknown_support')}）",
              f"- Purpose Under-inference Rate（Gold 已知被判成 unknown）：{o.get('purpose_under_inference_rate')}", "",
              "## 事实与证据", "",
              f"- Evidence Citation Validity（引用的 event_id 存在）：{o.get('evidence_citation_validity')}",
              f"- Evidence Citation Accuracy（引用事件属于该意图的 Gold 相关事件）：{o.get('evidence_citation_accuracy')}",
              f"- Fact Grounding Precision（引用式）：{o.get('fact_grounding_precision')}（评估事实数 {o.get('facts_evaluated')}）",
              f"- Unsupported Claim Rate：{o.get('unsupported_claim_rate')}", "",
              "## 多意图", "", f"- {o.get('multi_intent')}", ""]
    if "fact_grounding_precision_llm" in o:
        lines += [f"- Fact Grounding Precision（LLM 裁判，语义蕴含）：{o['fact_grounding_precision_llm']}（评估 {o.get('facts_llm_evaluated')} 条）", ""]
    lines += ["## 设计目标线（数据集设计 §9.5，研发目标，非实测）", "", "| 指标 | 目标 | 实测 | 达标 |", "|---|---|---|---|"]
    checks = {
        "intent_macro_f1": ("意图识别（样本级 intent F1 代替 macro）", o["intent"]["f1"]),
        "relation_macro_f1": ("事件关系 Macro-F1", o.get("relation_macro_f1")),
        "supporting_f1": ("Supporting Event F1", o["event_to_intent"]["supporting"]["f1"]),
        "related_interest_f1": ("Related Interest F1", o["event_to_intent"]["related_interest"]["f1"]),
        "purpose_accuracy": ("Purpose Accuracy", o.get("purpose_accuracy")),
        "fact_grounding_precision": ("Fact Grounding Precision（引用式）", o.get("fact_grounding_precision")),
        "purpose_unknown_recall": ("Unknown 识别（purpose unknown 召回）", o.get("purpose_unknown_recall")),
    }
    for key, (label, value) in checks.items():
        target = DESIGN_TARGETS[key]
        ok = "—" if value is None else ("✅" if value >= target else "❌")
        lines.append(f"| {label} | ≥ {target:.0%} | {'—' if value is None else f'{value:.1%}'} | {ok} |")
    lines += ["", "## 分片", ""]
    for key, sl in result["slices"].items():
        if not sl:
            continue
        lines += [f"### by {key}", "", "| 片 | n | Intent F1 | Relation macro-F1 | Purpose Acc | Over-inference | Fact Grounding |", "|---|---|---|---|---|---|---|"]
        for name, v in sl.items():
            if not v.get("n_samples"):
                continue
            fmt = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
            lines.append(f"| {name} | {v['n_samples']} | {fmt(v['intent']['f1'])} | {fmt(v.get('relation_macro_f1'))} | {fmt(v.get('purpose_accuracy'))} | {fmt(v.get('purpose_over_inference_rate'))} | {fmt(v.get('fact_grounding_precision'))} |")
        lines.append("")
    return "\n".join(lines)


def run(gold_paths, prediction_paths, out_dir, threshold=0.4, llm_judge=False, title="Evaluation report"):
    gold_rows = []
    for p in gold_paths:
        gold_rows.extend(load_jsonl(p))
    prediction_rows = []
    for p in prediction_paths:
        prediction_rows.extend(load_jsonl(p))
    result = evaluate(gold_rows, prediction_rows, threshold)
    if llm_judge:
        from src.llm_client import OpenAICompatibleClient
        client = OpenAICompatibleClient.from_env("LLM_JUDGE")
        verdicts = judge_facts(client, gold_rows, prediction_rows, threshold)
        if verdicts:
            result["overall"]["fact_grounding_precision_llm"] = round(sum(1 for _, ok in verdicts if ok) / len(verdicts), 4)
            result["overall"]["facts_llm_evaluated"] = len(verdicts)
    from pathlib import Path
    out_root = Path(out_dir) if Path(out_dir).is_absolute() else ROOT / out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    save_json(f"{out_root}/metrics.json", result)
    write_jsonl(f"{out_root}/per_sample.jsonl", result["per_sample"])
    md = render_markdown(result, title)
    (out_root / "EVAL_REPORT.md").write_text(md, encoding="utf-8")
    return result, md


def self_check(gold_rows):
    """Oracle + trivial baselines — used by tests and `--self-check` to validate the evaluator itself."""
    oracle_rows = [{"sample_id": r["sample_id"], "prediction": r.get("gold_label")} for r in gold_rows]
    oracle = evaluate(gold_rows, oracle_rows)["overall"]
    trivial_rows = []
    for r in gold_rows:
        events = _as_list((r.get("model_input") or {}).get("events"))
        trivial_rows.append({"sample_id": r["sample_id"], "prediction": {
            "intents": [], "event_labels": [{"event_id": e.get("event_id"), "relation": "unrelated", "intent_ids": []} for e in events if isinstance(e, dict)],
            "unrelated_event_ids": [e.get("event_id") for e in events if isinstance(e, dict)], "global_observations": {"intent_count": 0, "unrelated_event_count": len(events)}}})
    trivial = evaluate(gold_rows, trivial_rows)["overall"]
    return oracle, trivial


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate intent predictions against gold JSONL.")
    parser.add_argument("--gold", nargs="+", default=["data/splits/test.jsonl", "data/splits/challenge_test.jsonl"])
    parser.add_argument("--predictions", nargs="*", default=[], help="prediction JSONL files (omit with --self-check)")
    parser.add_argument("--out", default="data/eval")
    parser.add_argument("--match-threshold", type=float, default=0.4)
    parser.add_argument("--llm-judge", action="store_true", help="also judge fact entailment with the judge model")
    parser.add_argument("--self-check", action="store_true", help="oracle/trivial baselines instead of --predictions")
    args = parser.parse_args(argv)
    if not args.self_check and not args.predictions:
        parser.error("--predictions is required unless --self-check is given")
    if args.self_check:
        rows = []
        for p in args.gold:
            rows.extend(load_jsonl(p))
        oracle, trivial = self_check(rows)
        out_root = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
        out_root.mkdir(parents=True, exist_ok=True)
        keys = ("intent", "intent_count_accuracy", "relation_macro_f1", "relation_accuracy", "purpose_accuracy",
                "fact_grounding_precision", "evidence_citation_accuracy")
        save_json(f"{out_root}/self_check.json", {"generated_at": now_iso(), "n_rows": len(rows), "gold": args.gold,
                                                  "oracle": {k: oracle.get(k) for k in keys},
                                                  "trivial_all_unrelated": {k: trivial.get(k) for k in keys}})
        lines = ["# Evaluator self-check", "", f"- gold: {args.gold}（{len(rows)} 行）", "",
                 "| 基线 | Intent F1 | Relation macro-F1 | Purpose Acc | Fact Grounding |", "|---|---|---|---|---|"]
        for name, m in (("oracle（gold 当预测，应为满分）", oracle), ("trivial（全部判无关）", trivial)):
            lines.append(f"| {name} | {m.get('intent', {}).get('f1')} | {m.get('relation_macro_f1')} | "
                         f"{m.get('purpose_accuracy')} | {m.get('fact_grounding_precision')} |")
        (out_root / "SELF_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        perfect = oracle["intent"]["f1"] == 1.0 and all(oracle.get(k) == 1.0 for k in (
            "intent_count_accuracy", "relation_macro_f1", "relation_accuracy", "purpose_accuracy", "fact_grounding_precision"))
        return 0 if perfect else 1
    _, md = run(args.gold, args.predictions, args.out, args.match_threshold, args.llm_judge)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

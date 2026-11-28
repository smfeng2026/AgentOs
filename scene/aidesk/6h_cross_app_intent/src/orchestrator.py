"""Pipeline orchestrator (V1.1).

Planner → Generator → validate → noise injection → renumber events → blind Annotator → Reviewer
→ Adjudicator → validate Gold Label → write artefacts + manifest.

V1.1 changes vs the MVP:
- event_ids are renumbered by time after merging related + noise events (no id→label leak),
- noise comes from a seeded, category-aware pool with hard negatives (no fixed 15 strings),
- 15 unrelated is a target, not a hard constraint (adjudicator may relabel noise by real semantics),
- bounded regeneration loop with per-attempt audit trail,
- per-sample manifest (provenance, status, split, approval) + usage/timing per agent.
"""
import random
import time
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.agents import (
    PlannerAgent, GeneratorAgent, AnnotatorAgent, ReviewerAgent, AdjudicatorAgent
)
from src.validators import validate_generated, validate_final
from src.dataset_writer import write_sample
from src import manifest as manifest_store
from src.noise import sample_noise, assign_noise_timestamps, hard_negative_fraction_for
from src.utils import (ROOT, parse_time, now_iso, sha256_file, append_jsonl, stable_hash, load_json)

PIPELINE_VERSION = "1.1.0"
AGENT_NAMES = ("planner", "generator", "annotator", "reviewer", "adjudicator")
PROMPT_FILES = {name: f"prompts/{name}.txt" for name in AGENT_NAMES}
DEFAULT_WINDOW_START_HOURS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
_APP_VOCAB = None


def app_vocabulary():
    global _APP_VOCAB
    if _APP_VOCAB is None:
        _APP_VOCAB = load_json("config/app_vocabulary.json")["apps"]
    return _APP_VOCAB


def _gen_namespace(source_id):
    """Prefix generator ids (g###) so they cannot be confused with final chronological event ids (e###)."""
    if isinstance(source_id, str) and source_id.startswith("e"):
        return "g" + source_id[1:]
    return source_id


def build_model_input(sample_id, window_start, window_end, related_events, noise_events,
                      task="cross_app_intent_reasoning"):
    """Merge related + noise events, sort by time, renumber e001..eNN chronologically.

    Returns (model_input, labeled_events, id_mapping):
    - model_input: clean events only (event_id, timestamp, app, action, content),
    - labeled_events: same order/ids plus origin, generator relation/intent_ids and pool_id,
    - id_mapping: generator event_id → final event_id.
    """
    start, end = parse_time(window_start), parse_time(window_end)
    combined = []
    for seq, e in enumerate(related_events):
        combined.append({"_kind": "related", "_source_id": e.get("event_id"), "_seq": seq, **e})
    for seq, e in enumerate(noise_events):
        combined.append({"_kind": "noise", "_source_id": e.get("pool_id"), "_seq": 100000 + seq, **e})
    combined.sort(key=lambda x: (parse_time(x["timestamp"]), x["_seq"]))

    events, labeled, id_mapping = [], [], {}
    for idx, e in enumerate(combined, start=1):
        new_id = f"e{idx:03d}"
        if e["_kind"] == "related":
            id_mapping[e["_source_id"]] = new_id
        events.append({
            "event_id": new_id,
            "timestamp": e["timestamp"],
            "app": e["app"],
            "action": e["action"],
            "content": e["content"],
        })
        labeled.append({
            "event_id": new_id,
            "origin": e["_kind"],
            "generator_event_id": _gen_namespace(e["_source_id"]) if e["_kind"] == "related" else None,
            "pool_id": e["_source_id"] if e["_kind"] == "noise" else None,
            "hard_negative": bool(e.get("hard_negative", False)),
            "timestamp": e["timestamp"],
            "app": e["app"],
            "action": e["action"],
            "content": e["content"],
            "relation": e.get("relation") if e["_kind"] == "related" else "unrelated",
            "intent_ids": list(e.get("intent_ids") or []) if e["_kind"] == "related" else [],
        })
    hours = (end - start).total_seconds() / 3600
    model_input = {
        "sample_id": sample_id,
        "task": task,
        "window": {"start": window_start, "end": window_end, "duration_hours": round(hours, 2)},
        "history_intents": [],
        "events": events,
    }
    return model_input, labeled, id_mapping


def make_sample_id(task):
    base = str(task.get("template_id") or task.get("scenario_category") or "sample").lower()
    return f"{base}-{uuid.uuid4().hex[:8]}"


def resolve_window(task, cfg, rng):
    tz = ZoneInfo(task.get("timezone") or cfg.get("timezone", "Asia/Shanghai"))
    hours = float(cfg.get("window_hours", 6))
    explicit = task.get("window_start") is not None
    base = parse_time(task["window_start"]) if explicit else datetime.now(tz)
    base = base.astimezone(tz)
    fixed = (explicit and not task.get("jitter_window")) or (not explicit and task.get("fixed_window"))
    if fixed:
        start = base
    else:
        day_offset = rng.randint(0, int(cfg.get("window_day_jitter", 21)))
        hour = rng.choice(cfg.get("window_start_hours", DEFAULT_WINDOW_START_HOURS))
        minute = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        start = (base + timedelta(days=day_offset)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    end = start + timedelta(hours=hours)
    return start.isoformat(), end.isoformat()


def _summarise_usage(recorders):
    usage = {}
    for name, calls in recorders.items():
        usage[name] = {
            "calls": len(calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "total_tokens": sum(c["total_tokens"] for c in calls),
            "latency_s": round(sum(c["latency_s"] for c in calls), 2),
            "models": sorted({c["model"] for c in calls}),
        }
    return usage


def _relation_agreement(blind, gold):
    blind_labels = {x.get("event_id"): x.get("relation") for x in (blind or {}).get("event_labels", [])
                    if isinstance(x, dict)} if isinstance(blind, dict) else {}
    gold_labels = [x for x in (gold or {}).get("event_labels", []) if isinstance(x, dict)]
    if not gold_labels or not blind_labels:
        return None
    same = sum(1 for x in gold_labels if blind_labels.get(x.get("event_id")) == x.get("relation"))
    return round(same / len(gold_labels), 3)


def _preset_agreement(labeled, blind):
    """Relation agreement between the generator preset labels and the blind annotation (reviewer-skip gate)."""
    blind_labels = {x.get("event_id"): x.get("relation") for x in (blind or {}).get("event_labels", [])
                    if isinstance(x, dict)} if isinstance(blind, dict) else {}
    preset = [x for x in (labeled or []) if isinstance(x, dict)]
    if not preset or not blind_labels:
        return None
    same = sum(1 for x in preset if blind_labels.get(x.get("event_id")) == x.get("relation"))
    return round(same / len(preset), 3)


def run_one(gen_llm, judge_llm, task, cfg, taxonomy, noise_pool, sample_id=None, template=None, annotator_llm=None):
    t_start = time.time()
    sample_id = sample_id or make_sample_id(task)
    seed = task.get("seed")
    if seed is None:
        seed = int(stable_hash(sample_id), 16) % (2 ** 32)
    rng = random.Random(seed)
    window_start, window_end = resolve_window(task, cfg, rng)

    total_events = int(cfg.get("total_events", 30))
    related_target = int(cfg.get("related_events", 15))
    noise_count = total_events - related_target
    target_unrelated = int(cfg.get("unrelated_events", 15))
    min_unrelated = int(cfg.get("min_unrelated_events", 10))
    max_unrelated = int(cfg.get("max_unrelated_events", 20))
    max_regen = int(cfg.get("max_regeneration_attempts", 2))
    difficulty = task.get("difficulty") or "D2_multi_app_fusion"

    recorders = {name: [] for name in AGENT_NAMES}
    agents = {
        "planner": PlannerAgent(gen_llm, recorders["planner"]),
        "generator": GeneratorAgent(gen_llm, recorders["generator"]),
        "annotator": AnnotatorAgent(annotator_llm or judge_llm, recorders["annotator"]),
        "reviewer": ReviewerAgent(judge_llm, recorders["reviewer"]),
        "adjudicator": AdjudicatorAgent(judge_llm, recorders["adjudicator"]),
    }
    taxonomy_brief = taxonomy.brief() if taxonomy is not None else None
    vocabulary = app_vocabulary()
    planner_payload = {
        "sample_id": sample_id,
        "scenario_category": task["scenario_category"],
        "subscenario": task.get("subscenario"),
        "template": template,
        "difficulty": difficulty,
        "intent_count": task.get("intent_count", template.get("intent_count", 1) if template else 1),
        "window_start": window_start,
        "window_end": window_end,
        "window_hours": cfg.get("window_hours", 6),
        "total_events": total_events,
        "unrelated_events": target_unrelated,
        "related_events": related_target,
        "app_vocabulary": vocabulary,
        "label_taxonomy": taxonomy_brief,
    }

    attempts, hint = [], None
    spec = gen = model_input = labeled = blind = review = adjudication = gold = None
    validation = {"errors": ["pipeline did not reach adjudication"], "warnings": [], "stats": {}}
    stage, decision = None, None
    for attempt in range(1 + max_regen):
        plan_payload = dict(planner_payload)
        if hint:
            plan_payload["regeneration_hint"] = hint
        spec = agents["planner"].run(plan_payload)
        if not isinstance(spec, dict):
            raise ValueError("planner did not return a JSON object")
        spec["scenario_id"] = sample_id
        spec.setdefault("scenario_category", task["scenario_category"])
        spec["template_id"] = task.get("template_id") or spec.get("template_id")
        spec["difficulty"] = difficulty
        spec_intents = [i for i in spec.get("intents", []) if isinstance(i, dict)]
        intent_categories = [i.get("category") for i in spec_intents if i.get("category")]

        gen = agents["generator"].run(spec, {
            "sample_id": sample_id,
            "window_start": window_start,
            "window_end": window_end,
            "dates_must_be_after": window_end,
            "app_vocabulary": vocabulary,
            "related_event_count": related_target,
            "attempt": attempt,
            "regeneration_hint": hint,
            "label_taxonomy": taxonomy_brief,
        })
        gen_errors = validate_generated(gen, window_start, window_end, expected_related=related_target,
                                        spec=spec, app_vocabulary=set(vocabulary))
        if gen_errors:
            attempts.append({"attempt": attempt, "stage": "generation", "errors": gen_errors})
            stage, decision = "generation", "generation_failed"
            if attempt < max_regen:
                hint = "上一版事件未通过结构校验，请修正：" + "; ".join(gen_errors[:6])
                continue
            break

        key_entities = spec.get("key_entities") if isinstance(spec.get("key_entities"), dict) else {}
        noise_context = {
            "home_city": key_entities.get("home_city"),
            "other_date": (parse_time(window_end) + timedelta(days=rng.choice([12, 20, 26, 35]))).strftime("%m月%d日"),
        }
        noise = sample_noise(
            noise_pool, noise_count, rng, intent_categories, key_entities,
            difficulty=difficulty, context=noise_context,
        )
        noise = assign_noise_timestamps(noise, rng, window_start, window_end,
                                        [e["timestamp"] for e in gen["events"]])
        model_input, labeled, _ = build_model_input(sample_id, window_start, window_end, gen["events"], noise)

        blind = agents["annotator"].run(model_input, taxonomy_brief)
        # Cost saving: skip the adversarial reviewer when the blind annotation already agrees with the
        # generator preset (high relation agreement + same intent count). The adjudicator still runs always.
        skip_threshold = float(cfg.get("review_skip_threshold", 0.9))
        preset_agreement = _preset_agreement(labeled, blind)
        blind_intent_n = len(blind.get("intents", []) or []) if isinstance(blind, dict) else 0
        skip_reviewer = (
            bool(cfg.get("review_skip_enabled", True))
            and preset_agreement is not None and preset_agreement >= skip_threshold
            and blind_intent_n == len(spec_intents)
        )
        if skip_reviewer:
            review = {"sample_id": sample_id, "passed": True, "issues": [], "skipped": True,
                      "summary": f"reviewer skipped: preset-blind agreement {preset_agreement} >= {skip_threshold}"}
        else:
            review = agents["reviewer"].run(spec, labeled, blind)
        adjudication = agents["adjudicator"].run(spec, labeled, model_input, blind, review, taxonomy_brief)
        gold = adjudication.get("gold_label") if isinstance(adjudication, dict) else None
        if not isinstance(gold, dict):
            gold = {}
        gold["sample_id"] = sample_id
        validation = validate_final(
            model_input, gold, taxonomy, expected_total=total_events,
            min_unrelated=min_unrelated, max_unrelated=max_unrelated, target_unrelated=target_unrelated,
            labeled=labeled,
        )
        decision = adjudication.get("decision") if isinstance(adjudication, dict) else None
        stage = "adjudication"
        attempts.append({
            "attempt": attempt,
            "stage": "adjudication",
            "decision": decision,
            "requires_regeneration": adjudication.get("requires_regeneration") if isinstance(adjudication, dict) else None,
            "rationale": adjudication.get("rationale") if isinstance(adjudication, dict) else None,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "noise_pool_ids": [e.get("pool_id") for e in noise],
            "hard_negative_ids": [e.get("pool_id") for e in noise if e.get("hard_negative")],
        })
        if decision == "regenerate" and attempt < max_regen:
            hint = (attempts[-1]["rationale"] or "regenerate") if isinstance(attempts[-1]["rationale"], str) \
                else "regenerate"
            continue
        break

    if stage == "generation":
        status = "NEEDS_HUMAN_REVIEW"
        validation = {"errors": ["generation failed after retries"] + (attempts[-1]["errors"] if attempts else []),
                      "warnings": [], "stats": {}}
    else:
        status = "HUMAN_APPROVAL_PENDING" if not validation["errors"] and decision in ("accept", "revise_label") \
            else "NEEDS_HUMAN_REVIEW"

    stats = validation.get("stats", {})
    flags = []
    if stats and stats.get("unrelated") != target_unrelated:
        flags.append("composition_deviation")
    agreement = _relation_agreement(blind, gold)
    if agreement is not None and agreement < 0.8:
        flags.append("blind_low_agreement")
    if isinstance(blind, dict) and isinstance(gold, dict) and \
            len(blind.get("intents", []) or []) != len(gold.get("intents", []) or []):
        flags.append("blind_intent_count_mismatch")
    if isinstance(review, dict) and review.get("passed") is False:
        severities = [i.get("severity") for i in review.get("issues", []) if isinstance(i, dict)]
        if "critical" in severities:
            flags.append("reviewer_critical_issue")
    if isinstance(review, dict) and review.get("skipped"):
        flags.append("reviewer_skipped")

    internal_labels = {
        "sample_id": sample_id,
        "spec_intents": spec.get("intents", []) if isinstance(spec, dict) else [],
        "events": labeled or [],
    }
    audit = {
        "sample_id": sample_id,
        "pipeline_version": PIPELINE_VERSION,
        "status": status,
        "decision": decision,
        "flags": flags,
        "attempts": attempts,
        "blind_annotation": blind,
        "review": review,
        "adjudication": adjudication,
        "validation": validation,
        "blind_relation_agreement": agreement,
        "preset_relation_agreement": preset_agreement,
        "usage": _summarise_usage(recorders),
        "elapsed_s": round(time.time() - t_start, 2),
    }
    artefacts = write_sample(sample_id, spec, internal_labels, model_input, gold, audit)

    manifest = {
        "sample_id": sample_id,
        "pipeline_version": PIPELINE_VERSION,
        "created_at": now_iso(),
        "task": {
            "scenario_category": task["scenario_category"],
            "template_id": task.get("template_id"),
            "subscenario": task.get("subscenario"),
            "difficulty": difficulty,
            "intent_count": planner_payload["intent_count"],
            "split": task.get("split", "unassigned"),
            "window_start": window_start,
            "window_end": window_end,
            "timezone": task.get("timezone") or cfg.get("timezone"),
            "seed": seed,
        },
        "spec_summary": {
            "title": spec.get("title") or spec.get("scenario_title") if isinstance(spec, dict) else None,
            "intent_names": [i.get("intent_name") for i in spec_intents if isinstance(i, dict)],
            "intent_categories": intent_categories,
            "key_entities": spec.get("key_entities") if isinstance(spec, dict) else None,
        },
        "composition": {
            "total_events": len(model_input["events"]) if model_input else 0,
            "generator_related_events": related_target if stage != "generation" else None,
            "noise_events": noise_count if stage != "generation" else None,
            "hard_negative_noise": len(attempts[-1].get("hard_negative_ids", [])) if attempts and stage != "generation" else None,
            **({"gold_" + k: v for k, v in stats.items()} if stats else {}),
        },
        "noise": {
            "pool_version": noise_pool.get("version"),
            "pool_ids": attempts[-1].get("noise_pool_ids", []) if attempts and stage != "generation" else [],
        },
        "provenance": {
            "generator_model": gen_llm.model,
            "judge_model": judge_llm.model,
            "endpoint_host": gen_llm.host,
            "prompt_sha256": {name: sha256_file(ROOT / path) for name, path in PROMPT_FILES.items()},
            "taxonomy_sha256": sha256_file(ROOT / taxonomy.path) if taxonomy is not None and taxonomy.path else None,
        },
        "decision": decision,
        "status": status,
        "flags": flags,
        "validation": validation,
        "attempts": len(attempts),
        "usage": audit["usage"],
        "elapsed_s": audit["elapsed_s"],
        "approval": {"state": "pending", "by": None, "at": None, "note": None},
        "artefacts": artefacts,
    }
    manifest_store.write_manifest(manifest)
    append_jsonl("logs/generation_logs.jsonl", {
        "sample_id": sample_id, "status": status, "decision": decision, "flags": flags,
        "category": task["scenario_category"], "template_id": task.get("template_id"),
        "attempts": len(attempts), "elapsed_s": audit["elapsed_s"], "at": manifest["created_at"],
    })
    return {
        "sample_id": sample_id,
        "status": status,
        "decision": decision,
        "flags": flags,
        "errors": validation["errors"],
        "attempts": len(attempts),
        "elapsed_s": audit["elapsed_s"],
    }

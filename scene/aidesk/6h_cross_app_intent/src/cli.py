"""Command line entry point.

  python -m src.cli generate  --tasks-file config/pilot_tasks.json --workers 4   # produce samples
  python -m src.cli generate  --template-id C01-S04 --count 3 --difficulty D6_conflict_state_change
  python -m src.cli approve   --sample-id c01-s04-1a2b3c4d --by yourname [--note ...]
  python -m src.cli reject    --sample-id ... --by yourname --note "reason"
  python -m src.cli build     [--allow-unapproved]            # → data/splits/*.jsonl + report
  python -m src.cli evaluate  --predictions pred.jsonl [--llm-judge]
  python -m src.cli report                                   # → data/reports/DATASET_REPORT.md
  python -m src.cli review --pending --limit 5               # human review view, then approve / reject
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv_loader import load_dotenv
from src.llm_client import OpenAICompatibleClient
from src.orchestrator import run_one
from src import manifest as manifest_store
from src.utils import load_yaml, load_json, now_iso, save_json, stable_hash
from src.taxonomy import Taxonomy
from src.noise import load_pool


def load_templates():
    return {t["template_id"]: t for t in load_json("config/templates.json")["templates"]}


def _base_task(args, template=None):
    category = template["category"] if template else args.category
    return {
        "scenario_category": category,
        "template_id": template["template_id"] if template else None,
        "subscenario": template["title"] if template else None,
        "difficulty": (getattr(args, "difficulty", None) or (template["default_difficulty"] if template else None) or "D2_multi_app_fusion"),
        "intent_count": (getattr(args, "intent_count", None) or (template["intent_count"] if template else None) or 1),
        "split": getattr(args, "split", None) or "unassigned",
        "timezone": args.timezone,
        "window_start": args.start,
        "fixed_window": args.fixed_window,
        "jitter_window": args.jitter_window,
    }


def _with_task_ids(tasks, args):
    """Stable identity independent of task-file ordering; run_id allows intentional new batches."""
    occurrences = Counter()
    run_id = getattr(args, "run_id", None) or "default"
    for task in tasks:
        fingerprint = stable_hash(task, 16)
        occurrences[fingerprint] += 1
        task["run_id"] = run_id
        task["task_instance_id"] = stable_hash({"run_id": run_id, "task": task,
                                                 "occurrence": occurrences[fingerprint]}, 16)
    return tasks


def expand_tasks(args, templates):
    tasks = []
    if args.tasks_file:
        for entry in load_json(args.tasks_file):
            if not isinstance(entry, dict) or "template_id" not in entry:
                continue  # "_doc" entries
            template = templates[entry["template_id"]]
            for _ in range(int(entry.get("count", 1))):
                task = _base_task(args, template)
                for key in ("difficulty", "intent_count", "split"):
                    if entry.get(key):
                        task[key] = entry[key]
                tasks.append(task)
        return _with_task_ids(tasks, args)
    template = templates.get(args.template_id) if args.template_id else None
    if args.template_id and template is None:
        raise SystemExit(f"unknown template_id {args.template_id}")
    for _ in range(args.count):
        tasks.append(_base_task(args, template))
    return _with_task_ids(tasks, args)


def cmd_generate(args):
    load_dotenv()
    if os.getenv("MOCK_LLM", "0") == "1":
        raise SystemExit("MOCK_LLM is a placeholder guard. Configure a real API in .env to produce data.")
    cfg = load_yaml("config/dataset_config.yaml")
    taxonomy = Taxonomy.load()
    pool = load_pool()
    templates = load_templates()
    tasks = expand_tasks(args, templates)
    if not tasks:
        raise SystemExit("no tasks (check --tasks-file entries)")
    n_requested = len(tasks)
    existing = manifest_store.list_manifests()
    by_instance = {(m.get("task") or {}).get("task_instance_id"): m for m in existing
                   if (m.get("task") or {}).get("task_instance_id")}
    if args.resume:
        pending = []
        skipped_completed = []
        for task in tasks:
            previous = by_instance.get(task["task_instance_id"])
            completed = previous and (previous.get("run_state") == "completed" or
                                      (previous.get("run_state") is None and previous.get("status") == "HUMAN_APPROVAL_PENDING"))
            if completed:
                skipped_completed.append(previous["sample_id"])
            else:
                pending.append(task)
        tasks = pending
    else:
        collisions = [by_instance[t["task_instance_id"]]["sample_id"] for t in tasks if t["task_instance_id"] in by_instance]
        if collisions:
            raise SystemExit(f"{len(collisions)} task instances already exist; use --resume or a new --run-id")
        skipped_completed = []
    if not tasks:
        summary = {"event": "summary", "n_tasks": n_requested, "n_ok": 0, "n_failed": 0,
                   "n_skipped_completed": len(skipped_completed), "status": {}, "decision": {}, "flags": {},
                   "avg_attempts": 0, "elapsed_s_total": 0, "failed": []}
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    gen_llm = OpenAICompatibleClient.from_env("LLM")
    judge_llm = OpenAICompatibleClient.from_env("LLM_JUDGE")
    # Optional cheaper annotator model (judge tiering). Falls back to judge_llm when LLM_ANNOTATOR_MODEL unset.
    annotator_llm = OpenAICompatibleClient.from_env("LLM_ANNOTATOR") if os.getenv("LLM_ANNOTATOR_MODEL") else None
    print(json.dumps({"event": "start", "n_tasks": len(tasks), "n_requested": n_requested,
                      "n_skipped_completed": len(skipped_completed), "run_id": args.run_id,
                      "workers": args.workers,
                      "generator_model": gen_llm.model, "judge_model": judge_llm.model,
                      "endpoint": gen_llm.host, "noise_pool": pool.get("version")}, ensure_ascii=False), flush=True)

    results, failed = [], []
    if args.workers <= 1:
        for task in tasks:
            try:
                r = run_one(gen_llm, judge_llm, task, cfg, taxonomy, pool,
                            template=templates.get(task.get("template_id")), annotator_llm=annotator_llm)
            except Exception as exc:  # noqa: BLE001
                failed.append({"task": task, "error": str(exc)[:300]})
                print(json.dumps({"event": "failed", "error": str(exc)[:300]}, ensure_ascii=False), flush=True)
                continue
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_one, gen_llm, judge_llm, task, cfg, taxonomy, pool,
                                 None, templates.get(task.get("template_id")), annotator_llm=annotator_llm): task for task in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failed.append({"task": task, "error": str(exc)[:300]})
                    print(json.dumps({"event": "failed", "template": task.get("template_id"),
                                      "error": str(exc)[:300]}, ensure_ascii=False), flush=True)
                    continue
                results.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)

    summary = {
        "event": "summary", "n_tasks": len(tasks), "n_requested": n_requested,
        "n_skipped_completed": len(skipped_completed), "n_ok": len(results), "n_failed": len(failed),
        "status": dict(Counter(r["status"] for r in results)),
        "decision": dict(Counter(str(r["decision"]) for r in results)),
        "flags": dict(Counter(f for r in results for f in r["flags"])),
        "avg_attempts": round(sum(r["attempts"] for r in results) / max(1, len(results)), 2),
        "elapsed_s_total": round(sum(r["elapsed_s"] for r in results), 1),
        "failed": failed,
    }
    save_json(f"logs/batch_{now_iso().replace(':', '').replace('+', '_')}.json",
              {"args": {k: v for k, v in vars(args).items() if k != "func"}, "summary": summary, "results": results})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def cmd_approval(args, state):
    approval = manifest_store.set_approval(args.sample_id, state, args.by, args.note)
    print(json.dumps({"sample_id": args.sample_id, **approval}, ensure_ascii=False))
    return 0


def cmd_build(args):
    from src import dataset_builder
    summary = dataset_builder.build(args.out, args.allow_unapproved, args.include_meta, args.force, args.release)
    print(dataset_builder.render_markdown(summary))
    return 0


def cmd_plan_splits(args):
    from src.split_plan import write_plan
    from pathlib import Path
    from src.utils import ROOT
    path = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    if path.exists() and not args.force:
        raise SystemExit(f"split plan already exists: {args.out}; use --force only before a new dataset version")
    plan = write_plan(args.out, list(load_templates().values()), load_yaml("config/dataset_config.yaml"))
    print(json.dumps({"out": args.out, "version": plan["version"], "groups": len(plan["assignments"])},
                     ensure_ascii=False))
    return 0


def cmd_evaluate(args):
    from src import evaluator
    return evaluator.main(argv=[
        "--gold", *args.gold, "--out", args.out, "--match-threshold", str(args.match_threshold),
        *(["--llm-judge"] if args.llm_judge else []), *(["--self-check"] if args.self_check else ["--predictions", *args.predictions]),
    ])


def cmd_repair(args):
    from src import repair
    return repair.main(["--dry-run"] * int(args.dry_run) + ["--include-approved"] * int(args.include_approved))


def cmd_report(args):
    from src import dataset_report
    _, md = dataset_report.run(args.out)
    print(md)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="src.cli", description="6H-CrossApp-Intent data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="produce samples")
    g.add_argument("--tasks-file", help="JSON list of {template_id, count, difficulty?, intent_count?, split?}")
    g.add_argument("--template-id", help="single template id from config/templates.json")
    g.add_argument("--category", default="travel", help="used when no template_id / tasks-file is given")
    g.add_argument("--count", type=int, default=1)
    g.add_argument("--intent-count", type=int, default=None)
    g.add_argument("--difficulty", default=None, help="D1..D6 difficulty code (see prompts/planner.txt)")
    g.add_argument("--split", default=None, choices=["train", "validation", "test", "challenge"])
    g.add_argument("--start", default=None, help="ISO8601 base window start, e.g. 2026-12-01T09:00:00+08:00")
    g.add_argument("--timezone", default="Asia/Shanghai")
    g.add_argument("--fixed-window", action="store_true", help="use the same window for every sample (no jitter)")
    g.add_argument("--jitter-window", action="store_true", help="jitter even when --start is given")
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--run-id", default="default", help="stable batch namespace; use a new value for an intentional new run")
    g.add_argument("--resume", action="store_true", help="skip completed task instances and retry failed ones")
    g.set_defaults(func=cmd_generate)

    for name, state in (("approve", "approved"), ("reject", "rejected")):
        p = sub.add_parser(name, help=f"mark a sample {state}")
        p.add_argument("--sample-id", required=True)
        p.add_argument("--by", required=True, help="reviewer name")
        p.add_argument("--note", default=None)
        p.set_defaults(func=lambda a, s=state: cmd_approval(a, s))

    b = sub.add_parser("build", help="build train/validation/test JSONL from approved samples")
    b.add_argument("--out", default="data/splits")
    b.add_argument("--allow-unapproved", action="store_true")
    b.add_argument("--include-meta", action="store_true")
    b.add_argument("--force", action="store_true")
    b.add_argument("--release", action="store_true", help="enforce coverage quotas and shortcut-audit gates")
    b.set_defaults(func=cmd_build)

    sp = sub.add_parser("plan-splits", help="write a fixed split plan for every registered template")
    sp.add_argument("--out", default="config/split_plan.json")
    sp.add_argument("--force", action="store_true", help="replace an existing plan (new dataset versions only)")
    sp.set_defaults(func=cmd_plan_splits)

    e = sub.add_parser("evaluate", help="evaluate predictions against test JSONL")
    e.add_argument("--gold", nargs="+", default=["data/splits/test.jsonl", "data/splits/challenge_test.jsonl"])
    e.add_argument("--predictions", nargs="*", default=None, help="prediction JSONL files (omit with --self-check)")
    e.add_argument("--out", default="data/eval")
    e.add_argument("--match-threshold", type=float, default=0.4)
    e.add_argument("--llm-judge", action="store_true")
    e.add_argument("--self-check", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    v = sub.add_parser("review", help="human review view (approve/reject from here)")
    v.add_argument("--sample-id")
    v.add_argument("--pending", action="store_true", help="list samples awaiting approval")
    v.add_argument("--limit", type=int, default=None)
    v.add_argument("--no-events", action="store_true")
    v.add_argument("--export", default=None, help="write all pending samples to a markdown file")
    v.set_defaults(func=lambda a: __import__("src.review_view", fromlist=["main"]).main(
        ["--sample-id", a.sample_id] if a.sample_id else (["--pending"] + (["--limit", str(a.limit)] if a.limit else []))
        + (["--no-events"] if a.no_events else []) + (["--export", a.export] if a.export else [])))

    q = sub.add_parser("repair", help="provenance-preserving repairs (see src/repair.py)")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--include-approved", action="store_true")
    q.set_defaults(func=cmd_repair)

    r = sub.add_parser("report", help="dataset / pilot quality report")
    r.add_argument("--out", default="data/reports")
    r.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

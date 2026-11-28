"""model_eval CLI — snapshot / list / predict / evaluate / compare。

snapshot/list/evaluate/compare 仅需 stdlib + pyyaml；predict 需要 torch 环境（延迟导入）。
评测指标复用 6h_cross_app_intent 的评测器（子进程调用，输出落到本 run 目录），不复制代码。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .const import MODEL_EVAL_ROOT, SIXH_ROOT, STORE_ROOT
from . import runlog, store
from .postprocess import parse_prediction


# ---- helpers -----------------------------------------------------------------------------------------

def load_dotenv():
    p = MODEL_EVAL_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
    hf_home = os.environ.get("HF_HOME")
    if hf_home and not Path(hf_home).is_absolute():
        os.environ["HF_HOME"] = str((MODEL_EVAL_ROOT / hf_home).resolve())


def compact_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_eval_config(name):
    p = MODEL_EVAL_ROOT / "configs" / "eval" / f"{name}.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def dataset_dir(name):
    return MODEL_EVAL_ROOT / "datasets" / name


def load_snapshot(name):
    d = dataset_dir(name)
    meta_p = d / "snapshot.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"数据集快照不存在: {d}（先运行 snapshot）")
    return json.loads(meta_p.read_text(encoding="utf-8"))


def system_prompt(cfg):
    p = (MODEL_EVAL_ROOT / cfg["system_prompt_source"]).resolve()
    text = p.read_text(encoding="utf-8")
    return text, {"source": str(p), "sha256": store.sha256_text(text)}


# ---- snapshot ----------------------------------------------------------------------------------------

def cmd_snapshot(args):
    src = Path(args.source).resolve()
    dst = dataset_dir(args.name)
    if dst.exists() and not args.force:
        print(f"快照已存在: {dst}（--force 覆盖）", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    files = {}
    for fname in args.files:
        s = src / fname
        if not s.exists():
            print(f"缺少源文件: {s}", file=sys.stderr)
            return 1
        shutil.copyfile(s, dst / fname)
        files[fname] = {"sha256": store.sha256_file(dst / fname), "rows": len(read_jsonl(dst / fname))}
    prov = {}
    manifest_p = src / "split_manifest.json"
    if manifest_p.exists():
        m = json.loads(manifest_p.read_text(encoding="utf-8"))
        prov = {k: m.get(k) for k in ("builder_version", "generated_at", "allow_unapproved", "n_records")}
    meta = {"name": args.name, "source": str(src), "copied_at": runlog.now_iso(),
            "provenance": prov, "files": files}
    (dst / "snapshot.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    approval_note = ("⚠ 来源 build 为 allow_unapproved=true：标签尚未人工批准，选型结论只能作参考，"
                     "正式选型前须人工批准或独立重建。" if prov.get("allow_unapproved") else "来源 build 仅含已批准样本。")
    lines = [f"# 快照来源 — {args.name}", "",
             f"- 源目录：`{src}`", f"- 冻结时间：{meta['copied_at']}",
             f"- 来源 build：{json.dumps(prov, ensure_ascii=False)}", "",
             "| 文件 | 行数 | sha256 |", "|---|---|---|"]
    lines += [f"| {f} | {v['rows']} | `{v['sha256']}` |" for f, v in files.items()]
    lines += ["", approval_note, "",
              "本快照为只读副本：工厂侧重跑或重新批准样本不影响本快照；需要新数据时另建快照版本。"]
    (dst / "SOURCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


# ---- list --------------------------------------------------------------------------------------------

def cmd_list(_args):
    runs = []
    for rj in sorted((MODEL_EVAL_ROOT / "runs").glob("*/run.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        runs.append({"run_id": r["run_id"], "status": r.get("status"),
                     "artifact": (r.get("artifact") or {}).get("artifact_id")})
    print(json.dumps({"artifacts": store.list_artifacts(), "runs": runs}, ensure_ascii=False, indent=2))
    return 0


# ---- predict -----------------------------------------------------------------------------------------

def cmd_predict(args):
    resolved = store.resolve_artifact(args.artifact)
    cfg = load_eval_config(args.eval or args.dataset)
    snap = load_snapshot(args.dataset)
    sys_text, prompt_meta = system_prompt(cfg)
    decoding = dict(cfg.get("decoding") or {})
    if args.max_new_tokens:
        decoding["max_new_tokens"] = args.max_new_tokens
    dtype = args.dtype or resolved["base_config"].get("torch_dtype", "bfloat16")
    run_id = runlog.make_run_id(args.artifact, dtype, args.dataset, tag=args.tag)
    payload = {
        "artifact": {"artifact_id": resolved["artifact_id"], "kind": resolved["kind"],
                     "repo": resolved["repo_id"], "revision": resolved["revision"],
                     "manifest_sha256": resolved["manifest_sha256"],
                     "weight_files": resolved["weight_files"]},
        "dataset": {"name": args.dataset, "files": snap["files"]},
        "prompt": prompt_meta, "decoding": decoding,
        "repair": cfg.get("repair") or {}, "match_threshold": cfg.get("match_threshold", 0.4),
        "eval_config": args.eval or args.dataset,
    }
    d = runlog.create_run(run_id, payload)
    pred_path = d / "predictions.jsonl"
    rows = []
    for fname in cfg["gold_files"]:
        rows.extend(read_jsonl(dataset_dir(args.dataset) / fname))
    if args.limit:
        rows = rows[: args.limit]
    try:
        from .runners.hf_runner import HFRunner
        runner = HFRunner(resolved, device=args.device, dtype=dtype, decoding=decoding,
                          allow_download=args.allow_download).load()
    except Exception as exc:
        runlog.finalize_run(d, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        print(f"runner 加载失败: {exc}", file=sys.stderr)
        return 1
    runlog.finalize_run(d, "running", {"runner": runner.info()})
    repair_cfg = cfg.get("repair") or {}
    max_retries = int(repair_cfg.get("max_retries", 1))
    retry_suffix = repair_cfg.get("retry_suffix", "请只输出一个合法 JSON 对象。")
    n_ok_raw = n_ok = 0
    with open(pred_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            sid = row.get("sample_id")
            user = compact_json(row.get("model_input") or {})
            err = None
            try:
                gen = runner.generate(sys_text, user)
                first = parse_prediction(gen["text"])
                final, retry_used = first, False
                if not first["format_ok"] and max_retries > 0:
                    gen2 = runner.generate(sys_text, user, extra_user=retry_suffix)
                    second = parse_prediction(gen2["text"])
                    retry_used = True
                    if second["format_ok"] or not first["prediction"]:
                        final = second
                        gen = {**gen, "text": gen["text"], "retry_raw": gen2["text"],
                               "latency_ms": gen["latency_ms"] + gen2["latency_ms"],
                               "completion_tokens": gen["completion_tokens"] + gen2["completion_tokens"]}
                n_ok_raw += int(first["format_ok"])
                n_ok += int(final["format_ok"])
                rec = {"sample_id": sid, "raw": gen["text"], "prediction": final["prediction"],
                       "format_ok": final["format_ok"], "raw_format_ok": first["format_ok"],
                       "repair_used": final["repair_used"] or retry_used, "method": final["method"],
                       "retry_used": retry_used, "latency_ms": gen["latency_ms"],
                       "prompt_tokens": gen["prompt_tokens"], "completion_tokens": gen["completion_tokens"]}
                if "retry_raw" in gen:
                    rec["retry_raw"] = gen["retry_raw"]
            except Exception as exc:  # 单条失败不终止整批
                err = f"{type(exc).__name__}: {exc}"
                rec = {"sample_id": sid, "raw": None, "prediction": None, "format_ok": False,
                       "raw_format_ok": False, "repair_used": False, "method": "error",
                       "retry_used": False, "error": err}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i + 1}/{len(rows)}] {sid} format_ok={rec['format_ok']}" + (f" ERROR={err}" if err else ""))
    runner.unload()
    runlog.finalize_run(d, "completed", {
        "n_samples": len(rows), "format_ok_raw": n_ok_raw, "format_ok_after_repair": n_ok,
        "predictions": str(pred_path)})
    print(json.dumps({"run_id": run_id, "n": len(rows), "format_ok_raw": n_ok_raw,
                      "format_ok_after_repair": n_ok, "dir": str(d)}, ensure_ascii=False))
    return 0


# ---- evaluate ----------------------------------------------------------------------------------------

def cmd_evaluate(args):
    d = runlog.run_dir(args.run)
    rj = d / "run.json"
    if not rj.exists():
        print(f"run 不存在: {d}", file=sys.stderr)
        return 1
    run = json.loads(rj.read_text(encoding="utf-8"))
    ds = run["dataset"]["name"]
    cfg = load_eval_config(run.get("eval_config") or ds)
    gold = [str(dataset_dir(ds) / f) for f in cfg["gold_files"]]
    pred = d / "predictions.jsonl"
    if not pred.exists():
        print(f"缺少预测文件: {pred}", file=sys.stderr)
        return 1
    py = SIXH_ROOT / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-m", "src.cli", "evaluate", "--gold", *gold,
           "--predictions", str(pred), "--out", str(d / "eval"),
           "--match-threshold", str(run.get("match_threshold", 0.4))]
    proc = subprocess.run(cmd, cwd=str(SIXH_ROOT))
    return proc.returncode


# ---- compare -----------------------------------------------------------------------------------------

COMPARE_COLS = [
    ("n_evaluated", lambda m: m["meta"]["n_evaluated"]),
    ("format_valid", lambda m: m["meta"]["format_valid_rate"]),
    ("intent_f1", lambda m: (m["overall"].get("intent") or {}).get("f1")),
    ("intent_macro_f1", lambda m: m["overall"].get("intent_macro_f1")),
    ("relation_macro_f1", lambda m: m["overall"].get("relation_macro_f1")),
    ("purpose_acc", lambda m: m["overall"].get("purpose_accuracy")),
    ("purpose_over_inf", lambda m: m["overall"].get("purpose_over_inference_rate")),
    ("fact_grounding", lambda m: m["overall"].get("fact_grounding_precision")),
]


def cmd_compare(args):
    entries = []
    for rid in args.runs:
        d = runlog.run_dir(rid)
        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
        mp = d / "eval" / "metrics.json"
        m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
        entry = {"run_id": rid, "artifact": run["artifact"]["artifact_id"],
                 "dataset": run["dataset"]["name"], "evaluated": m is not None}
        if m:
            for name, fn in COMPARE_COLS:
                try:
                    entry[name] = fn(m)
                except (KeyError, TypeError):
                    entry[name] = None
        entries.append(entry)
    day = datetime.now().strftime("%Y-%m-%d")
    out_md = MODEL_EVAL_ROOT / "reports" / f"compare_{day}.md"
    header = ["run", "artifact", "dataset"] + [c[0] for c in COMPARE_COLS]
    lines = [f"# 模型对比 — {day}", "", "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for e in entries:
        cells = [e["run_id"], e["artifact"], e["dataset"]]
        cells += [("未评测" if not e["evaluated"] else str(e.get(name))) for name, _ in COMPARE_COLS]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "指标来源：6h_cross_app_intent 评测器（各 run 的 eval/metrics.json）。",
              "各数据集的审批状态见 model_eval/datasets/<name>/SOURCE.md；未经人工批准的快照结论仅供流程验证。"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n→ {out_md}")
    return 0


# ---- main --------------------------------------------------------------------------------------------

def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(prog="src.cli", description="model_eval — artifact 级评测")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("snapshot", help="冻结 6h 评测集为只读快照")
    s.add_argument("--name", required=True)
    s.add_argument("--source", default=str(SIXH_ROOT / "data" / "splits_pilot"))
    s.add_argument("--files", nargs="+", default=["test.jsonl", "challenge_test.jsonl"])
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_snapshot)

    sub.add_parser("list", help="列出 artifact 与 run").set_defaults(func=cmd_list)

    p = sub.add_parser("predict", help="对快照跑推理（需要 torch 环境）")
    p.add_argument("--artifact", default="qwen3-0.6b__official")
    p.add_argument("--dataset", default="pilot_v1")
    p.add_argument("--eval", default=None, help="评测配置名，默认与 dataset 同名")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=os.environ.get("MODEL_EVAL_DEVICE", "cpu"))
    p.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--tag", default=None, help="run 名后缀，用于同日多次运行（只追加不覆盖）")
    p.add_argument("--allow-download", action="store_true", help="显式允许从 HF 下载权重（须先经用户确认）")
    p.set_defaults(func=cmd_predict)

    e = sub.add_parser("evaluate", help="复用 6h 评测器评估某 run")
    e.add_argument("--run", required=True)
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("compare", help="跨 run 对比表")
    c.add_argument("--runs", nargs="+", required=True)
    c.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
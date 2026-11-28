"""把 6h 已批准样本只读组装为评测快照（6h 零写入）。

复用 6h 的 load_records（含 validate_final 校验）与 test_row（含 canonical_target），
保证与流水线官方导出格式完全一致。用 6h 的 venv 解释器运行（依赖齐全）：
  ../6h_cross_app_intent/.venv/bin/python scripts/freeze_approved.py --name approved_all_v1
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # model_eval
SIXH = ROOT.parent / "6h_cross_app_intent"
sys.path.insert(0, str(SIXH))                          # 注意：本脚本不导入 model_eval.src，避免 src 包名冲突

from src import dataset_builder as db                  # noqa: E402  (6h 的 src)
from src.taxonomy import Taxonomy                      # noqa: E402
from src.utils import load_yaml                        # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml("config/dataset_config.yaml")
    taxonomy = Taxonomy.load()
    records, skipped = db.load_records(cfg, taxonomy)  # 默认仅 approved，且逐条 validate_final
    if not records:
        print("没有可用样本", file=sys.stderr)
        return 1
    for r in records:
        r["split"] = "eval"
        r["group_key"] = db.group_key(r)
        r["category"] = (r["manifest"].get("task") or {}).get("scenario_category") or \
            r["spec"].get("scenario_category") or "unknown"
        r["difficulty"] = (r["manifest"].get("task") or {}).get("difficulty") or r["spec"].get("difficulty")
    rows = [db.test_row(r) for r in records]

    dst = ROOT / "datasets" / args.name
    if dst.exists() and not args.force:
        print(f"快照已存在: {dst}（--force 覆盖）", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / "eval.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    approvals = [(r["manifest"].get("approval") or {}) for r in records]
    prov = {
        "source": str(SIXH / "data"),
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_approved": len(records),
        "skipped": skipped,
        "approvers": dict(Counter(a.get("by") for a in approvals)),
        "approval_range": [min(a.get("at", "") for a in approvals), max(a.get("at", "") for a in approvals)],
        "categories": dict(Counter(r["category"] for r in records)),
        "sample_ids": [r["sample_id"] for r in records],
    }
    meta = {"name": args.name, "files": {"eval.jsonl": {"sha256": sha256_file(out), "rows": len(rows)}},
            "provenance": prov}
    (dst / "snapshot.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# 快照来源 — {args.name}", "",
             f"- 源：`{prov['source']}`（仅 approval.state == approved，逐条通过 validate_final）",
             f"- 冻结时间：{prov['frozen_at']}",
             f"- 样本数：{len(rows)}；审批人：{json.dumps(prov['approvers'], ensure_ascii=False)}",
             f"- 审批时间范围：{prov['approval_range'][0]} ~ {prov['approval_range'][1]}",
             f"- 跳过：{json.dumps(skipped, ensure_ascii=False)}", "",
             "| 文件 | 行数 | sha256 |", "|---|---|---|",
             f"| eval.jsonl | {len(rows)} | `{meta['files']['eval.jsonl']['sha256']}` |", "",
             "类别分布：" + json.dumps(prov["categories"], ensure_ascii=False), "",
             "样本清单：" + ", ".join(prov["sample_ids"]), "",
             "本快照为只读副本；用途为基座模型能力评测。微调后的评测须另建留出集，不得复用本快照训练。"]
    (dst / "SOURCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"name": args.name, "rows": len(rows), "skipped": len(skipped),
                      "sha256": meta["files"]["eval.jsonl"]["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
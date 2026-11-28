"""run 清单：命名、创建、收尾。run.json 钉死权重/数据/prompt 哈希与解码参数。"""
import json
import platform
import re
from datetime import datetime
from pathlib import Path

from .const import MODEL_EVAL_ROOT


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sanitize(text):
    return re.sub(r"[^A-Za-z0-9._\-]", "-", str(text))


def make_run_id(artifact_id, precision, dataset, day=None, tag=None):
    day = day or datetime.now().strftime("%Y-%m-%d")
    rid = f"{day}__{sanitize(artifact_id)}__{sanitize(precision)}__{sanitize(dataset)}"
    return f"{rid}__{sanitize(tag)}" if tag else rid


def run_dir(run_id):
    return MODEL_EVAL_ROOT / "runs" / run_id


def create_run(run_id, payload):
    d = run_dir(run_id)
    if d.exists():
        raise FileExistsError(f"run 已存在（只追加不覆盖）: {d}")
    d.mkdir(parents=True)
    record = {"run_id": run_id, "created_at": now_iso(), "status": "running",
              "hardware": {"platform": platform.platform(), "machine": platform.machine()}, **payload}
    (d / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return d


def finalize_run(d, status, extra=None):
    p = Path(d) / "run.json"
    record = json.loads(p.read_text(encoding="utf-8"))
    record.update({"status": status, "finished_at": now_iso(), **(extra or {})})
    p.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record
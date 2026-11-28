"""下载基座权重到 model_store/base/<base_id>/，钉住上游 commit revision。

仅在用户确认后运行。HF_HOME 由 model_eval/.env 固定在 scene/aidesk/model_store/.hf_cache。
用法：.venv/bin/python scripts/fetch_base.py --base-id qwen3-0.6b
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.cli import load_dotenv  # noqa: E402
from src.const import STORE_ROOT  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--endpoint", default=None, help="覆盖 HF_ENDPOINT（如 https://hf-mirror.com）")
    args = parser.parse_args()
    load_dotenv()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import HfApi, snapshot_download

    up_path = STORE_ROOT / "base" / args.base_id / "upstream.json"
    up = json.loads(up_path.read_text(encoding="utf-8"))
    api = HfApi()
    info = api.model_info(up["repo"])
    revision = info.sha
    print(f"repo={up['repo']} revision={revision}", flush=True)
    path = snapshot_download(
        up["repo"], revision=revision,
        local_dir=str(STORE_ROOT / "base" / args.base_id),
        ignore_patterns=["*.gguf", "*.onnx", "*.mnn", "*.msgpack", "*.h5"],
    )
    files = sorted(p.name for p in Path(path).iterdir() if p.is_file())
    up["revision"] = revision
    up["status"] = "downloaded"
    up["registered_at"] = __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")
    up_path.write_text(json.dumps(up, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"base_id": args.base_id, "revision": revision, "path": path, "files": files},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
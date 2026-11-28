# model_store CLI — 产物注册与校验（stdlib only）
# 用法见 README.md。权重下载/依赖安装不在本 CLI 职责内，须先经用户确认。
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACT_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]*(__[a-z0-9][a-z0-9.\-]*)+$")
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".mnn", ".npz"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_list(_args):
    rows = []
    for up in sorted((ROOT / "base").glob("*/upstream.json")):
        u = load_json(up)
        rows.append({"artifact_id": f"{u['base_id']}__official", "kind": "official",
                     "status": u.get("status", "unknown"), "repo": u.get("repo")})
    for mf in sorted((ROOT / "derived").glob("*/manifest.json")):
        m = load_json(mf)
        rows.append({"artifact_id": m["artifact_id"], "kind": m["kind"],
                     "status": "registered", "parent": m.get("parent_artifact")})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def cmd_verify(args):
    mf = ROOT / "derived" / args.artifact_id / "manifest.json"
    if not mf.exists():
        print(f"manifest not found: {mf}", file=sys.stderr)
        return 1
    m = load_json(mf)
    bad = []
    for entry in m["files"]:
        p = mf.parent / entry["path"]
        if not p.exists():
            bad.append({"path": entry["path"], "reason": "missing"})
        elif sha256_file(p) != entry["sha256"]:
            bad.append({"path": entry["path"], "reason": "sha256 mismatch"})
    print(json.dumps({"artifact_id": m["artifact_id"], "ok": not bad, "problems": bad}, ensure_ascii=False))
    return 0 if not bad else 1


def cmd_register_derived(args):
    if not ARTIFACT_RE.match(args.artifact_id):
        print(f"artifact_id 不符合命名规则: {args.artifact_id}", file=sys.stderr)
        return 1
    adir = ROOT / "derived" / args.artifact_id
    if not adir.is_dir():
        print(f"先把权重文件放入 {adir} 再注册", file=sys.stderr)
        return 1
    if (adir / "manifest.json").exists() and not args.force:
        print("manifest 已存在；--force 覆盖", file=sys.stderr)
        return 1
    base_up = ROOT / "base" / args.base / "upstream.json"
    if not base_up.exists():
        print(f"未知基座: {args.base}", file=sys.stderr)
        return 1
    u = load_json(base_up)
    files = []
    for p in sorted(adir.rglob("*")):
        if p.is_file() and p.suffix in WEIGHT_SUFFIXES:
            files.append({"path": str(p.relative_to(adir)), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    if not files:
        print("目录内没有权重文件（.safetensors/.bin/.gguf/...）", file=sys.stderr)
        return 1
    manifest = {
        "artifact_id": args.artifact_id, "kind": args.kind,
        "base": {"repo": u["repo"], "revision": u.get("revision"), "store_path": f"base/{args.base}"},
        "parent_artifact": args.parent, "training_run": args.training_run,
        "train_data": ({"snapshot": args.train_snapshot, "sha256": args.train_sha256}
                       if args.train_snapshot and args.train_sha256 else None),
        "hyperparams": json.loads(args.hyperparams) if args.hyperparams else {},
        "files": files, "created_at": now_iso(), "notes": args.notes or "",
    }
    (adir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"registered": args.artifact_id, "files": len(files)}, ensure_ascii=False))
    return 0


def cmd_register_base(args):
    up = ROOT / "base" / args.base_id / "upstream.json"
    if not up.exists():
        print(f"未知基座: {args.base_id}", file=sys.stderr)
        return 1
    u = load_json(up)
    u["revision"] = args.revision
    u["status"] = "downloaded"
    u["registered_at"] = now_iso()
    up.write_text(json.dumps(u, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"base_id": args.base_id, "revision": args.revision}, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="model_store.cli", description="权重产物注册与校验")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    v = sub.add_parser("verify")
    v.add_argument("--artifact-id", required=True)
    v.set_defaults(func=cmd_verify)
    r = sub.add_parser("register-derived")
    r.add_argument("--artifact-id", required=True)
    r.add_argument("--kind", required=True, choices=["lora_adapter", "merged", "full_finetune", "quantized"])
    r.add_argument("--base", required=True, help="基座 id，如 qwen3-0.6b")
    r.add_argument("--parent", default=None)
    r.add_argument("--training-run", default=None)
    r.add_argument("--train-snapshot", default=None)
    r.add_argument("--train-sha256", default=None)
    r.add_argument("--hyperparams", default=None, help="JSON 字符串")
    r.add_argument("--notes", default=None)
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=cmd_register_derived)
    b = sub.add_parser("register-base")
    b.add_argument("--base-id", required=True)
    b.add_argument("--revision", required=True)
    b.set_defaults(func=cmd_register_base)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
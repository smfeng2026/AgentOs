"""artifact 解析：唯一入口是 model_store 的注册元数据（manifest / upstream.json）。"""
import hashlib
import json
from pathlib import Path

import yaml

from .const import STORE_ROOT

WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".mnn", ".npz"}
NON_HF_SUFFIXES = {".gguf", ".mnn", ".onnx"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_base_config(base_id):
    p = STORE_ROOT.parent / "model_eval" / "configs" / "bases" / f"{base_id}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"缺少基座配置: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def weight_files(directory):
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in WEIGHT_SUFFIXES)


def list_artifacts():
    rows = []
    for up in sorted((STORE_ROOT / "base").glob("*/upstream.json")):
        u = load_json(up)
        rows.append({"artifact_id": f"{u['base_id']}__official", "kind": "official",
                     "status": u.get("status", "unknown"), "repo": u.get("repo")})
    for mf in sorted((STORE_ROOT / "derived").glob("*/manifest.json")):
        m = load_json(mf)
        rows.append({"artifact_id": m["artifact_id"], "kind": m["kind"],
                     "status": "registered", "parent": m.get("parent_artifact")})
    return rows


def resolve_artifact(artifact_id):
    """返回评测所需的完整解析结果。official → base 目录；其余 → derived manifest。"""
    if artifact_id.endswith("__official"):
        base_id = artifact_id[: -len("__official")]
        up_path = STORE_ROOT / "base" / base_id / "upstream.json"
        if not up_path.exists():
            raise FileNotFoundError(f"未知基座: {base_id}")
        up = load_json(up_path)
        cfg = load_base_config(base_id)
        wdir = STORE_ROOT / "base" / base_id
        files = weight_files(wdir)
        return {
            "artifact_id": artifact_id, "kind": "official", "base_id": base_id,
            "repo_id": up["repo"], "revision": up.get("revision"),
            "weights_dir": str(wdir) if files else None,
            "weight_files": [{"path": str(p.relative_to(wdir)), "sha256": sha256_file(p)} for p in files],
            "manifest": None, "manifest_sha256": None,
            "base_config": cfg, "base_repo_id": up["repo"], "base_weights_dir": str(wdir) if files else None,
        }
    mf_path = STORE_ROOT / "derived" / artifact_id / "manifest.json"
    if not mf_path.exists():
        raise FileNotFoundError(f"artifact 未注册: {artifact_id}（{mf_path}）")
    m = load_json(mf_path)
    for field in ("artifact_id", "kind", "base", "files", "created_at"):
        if field not in m:
            raise ValueError(f"manifest 缺少字段 {field}: {mf_path}")
    if m["artifact_id"] != artifact_id:
        raise ValueError("manifest.artifact_id 与目录名不一致")
    base_id = Path(m["base"]["store_path"]).name
    cfg = load_base_config(base_id)
    wdir = mf_path.parent
    base_wdir = STORE_ROOT / m["base"]["store_path"]
    base_files = weight_files(base_wdir)
    return {
        "artifact_id": artifact_id, "kind": m["kind"], "base_id": base_id,
        "repo_id": m["base"]["repo"], "revision": m["base"].get("revision"),
        "weights_dir": str(wdir),
        "weight_files": m["files"],
        "manifest": m, "manifest_sha256": sha256_file(mf_path),
        "base_config": cfg, "base_repo_id": m["base"]["repo"],
        "base_weights_dir": str(base_wdir) if base_files else None,
    }


def unsupported_formats(resolved):
    """返回 HF runner 无法加载的权重文件（gguf/mnn/onnx 等），供 CLI 提前报错。"""
    return [f["path"] for f in resolved["weight_files"] if Path(f["path"]).suffix in NON_HF_SUFFIXES]
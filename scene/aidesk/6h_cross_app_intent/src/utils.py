import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import yaml

# Project root for every relative path (data/, logs/, config/, prompts/). Tests point it at a temp dir.
ROOT = Path(os.environ.get("DATASET_ROOT") or Path(__file__).resolve().parents[1])


def read_text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def load_json(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(path, obj):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    # atomic replace: a concurrent reader (builder / report / another worker) never sees a half-written file
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


def load_jsonl(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def append_jsonl(path, obj):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_yaml(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def extract_json(text):
    """Parse a JSON object out of a model reply (bare, fenced, or embedded in prose)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Depth-scan from every '{' so trailing prose containing braces cannot break the match.
    for start in [i for i, ch in enumerate(text) if ch == "{"]:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("No JSON object found in model output")


def env(name, default=None):
    return os.getenv(name, default)


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj, n=12):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compact_json(obj):
    """Insertion-ordered compact JSON — used for SFT user/assistant content."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_PUNCT = re.compile(r"[^\w\s\u3400-\u9fff\uf900-\ufaff]", re.U)


def estimate_tokens(text):
    """Rough token estimate for mixed Chinese/English JSON (no tokenizer dependency).

    ~1 token per CJK char, ~1.3 tokens per ASCII word (JSON ids split), ~0.3 per punctuation mark.
    Only used for length statistics — never for billing or training decisions.
    """
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    punct = len(_PUNCT.findall(text))
    digits = sum(1 for ch in text if ch.isspace())
    return int(cjk + 1.3 * words + 0.3 * (punct + digits))


def mask_secret(value, keep=4):
    if not value or len(value) <= keep * 2:
        return "***"
    return f"{value[:keep]}...{value[-keep:]}"


def flatten_entities(key_entities):
    """spec['key_entities'] may be dict-of-lists, a list, or missing → flat list of strings."""
    if not key_entities:
        return []
    if isinstance(key_entities, dict):
        values = []
        for v in key_entities.values():
            if isinstance(v, list):
                values.extend(str(x) for x in v if x)
            elif v:
                values.append(str(v))
        return values
    if isinstance(key_entities, list):
        return [str(x) for x in key_entities if x]
    return [str(key_entities)]

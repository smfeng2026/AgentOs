"""模型输出 → 预测对象。原始格式与修复后格式分开统计，修复手段全部留痕。

extract_json 与 6h src/utils.py 的算法保持一致（围栏剥离 + 字符串感知深度扫描），
额外增加一步保守修复：去除尾随逗号后重扫。不做单引号替换等激进改写。
"""
import json
import re


def extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
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


def _strip_trailing_commas(text):
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _is_prediction(obj):
    return isinstance(obj, dict) and isinstance(obj.get("intents"), list) and isinstance(obj.get("event_labels"), list)


def parse_prediction(raw):
    """返回 {prediction, format_ok, repair_used, method}。format_ok 以评测器最低结构要求为准。"""
    attempts = [
        ("direct", lambda t: json.loads(t)),
        ("extracted", extract_json),
        ("comma_repair", lambda t: extract_json(_strip_trailing_commas(t))),
    ]
    for method, fn in attempts:
        try:
            obj = fn(raw or "")
        except (ValueError, json.JSONDecodeError):
            continue
        return {"prediction": obj, "format_ok": _is_prediction(obj),
                "repair_used": method != "direct", "method": method}
    return {"prediction": None, "format_ok": False, "repair_used": False, "method": "failed"}
"""laya 决策模型测试网站:输入状态文本,提交决策问题,展示校准概率。

用法:
    cd web && python server.py            # http://localhost:8030
    python server.py --port 8031

模型:NandhaKishorM/laya(pip 包 laya)+ convaiinnovations/laya 英文 checkpoint,
低内存加载见 lowmem.py(本机 1.9GB 内存,fp16 assign,峰值 ~650MB)。

多模态:图片 / 音频 / 视频(0.4.30+ 的 vega 引擎,Agent(vegaml=True))按需惰性加载;
首次使用需从 HuggingFace 下载约 2GB(国内可设 https_proxy)。
"""

import argparse
import json
import os
import tempfile
import time

# HF 缓存与模型全部留在仓库内:沙箱/无网环境可离线运行,也避免污染用户目录
_REPO = __import__("pathlib").Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(_REPO / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from lowmem import load_agent

app = FastAPI(title="laya decision-model test bench")
agent = None

# vega 引擎(多模态)惰性加载:仅首次收到 image/audio/video 时触发
_vega_agent = None
_vega_error = None

# 本地已备齐 vega-U + Qwen reader 时优先离线加载(models/vega-U/backbone)
VEGA_LOCAL = str(_REPO / "models" / "vega-U") \
    if (_REPO / "models" / "vega-U" / "engine.safetensors").exists() \
    and (_REPO / "models" / "vega-U" / "backbone").is_dir() else None

# 多语言转写器:本地 whisper-small(已备齐)优先,否则回退 Hub id
ASR_LOCAL = str(_REPO / "models" / "whisper-small") \
    if (_REPO / "models" / "whisper-small" / "model.safetensors").exists() else "openai/whisper-small"


def get_vega():
    """首次调用时加载 vega 引擎;失败则记住原因,前端据此显示提示。"""
    global _vega_agent, _vega_error
    if _vega_agent is not None:
        return _vega_agent
    if _vega_error is not None:
        raise RuntimeError(_vega_error)
    import laya
    try:
        t0 = time.perf_counter()
        _vega_agent = laya.Agent(vegaml=VEGA_LOCAL or True)
        print(f"vega 引擎加载完成({time.perf_counter() - t0:.1f}s)", flush=True)
        return _vega_agent
    except Exception as e:  # noqa: BLE001 - 原样透传给前端提示
        _vega_error = f"{type(e).__name__}: {e}"
        raise RuntimeError(_vega_error) from e


def multimodal_status():
    """不实际加载,只判断 vega 是否可能可用(包已安装)。"""
    try:
        import laya  # noqa: F401
        from laya.vega import is_available
        return is_available()
    except Exception:
        return False


@app.get("/", response_class=HTMLResponse)
def index():
    with open(__file__.replace("server.py", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/info")
def info():
    mm = multimodal_status()
    out = {
        "model": "laya (convaiinnovations/laya, en)",
        "engine": "laya " + __import__("laya").__version__,
        "device": str(agent.device),
        "dtype": str(agent.dtype),
        "questions_per_call": "无限制,单次前向",
        "multimodal": mm,
    }
    if not mm:
        out["multimodal_note"] = "vegaml 包不可用,pip install -U laya 后重启"
    if _vega_agent is not None:
        out["vega"] = "loaded"
        out["multimodal"] = True
    return out


@app.post("/api/predict")
async def predict(
    state: str = Form(""),
    questions: str = Form(...),
    file: UploadFile | None = File(None),
):
    try:
        qs = json.loads(questions)
        if not isinstance(qs, dict) or not qs:
            raise ValueError("questions 必须是非空 JSON 对象")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"questions 不是合法 JSON: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))

    extra = {}
    t0 = time.perf_counter()
    try:
        if file is not None and file.filename:
            data = await file.read()
            suffix = "." + (file.filename.rsplit(".", 1)[-1] or "bin")
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                path = tmp.name
            kind = (file.content_type or "").split("/")[0]
            kwargs = {}
            if kind == "audio":
                kwargs = {"asr": ASR_LOCAL, "language": "zh"}
            result = get_vega().predict(
                **({"image": path} if kind == "image"
                   else {"audio": path} if kind == "audio"
                   else {"video": path}),
                questions=qs, **kwargs)
            if kind == "audio":
                extra["transcript"] = result.get("transcript")
            if kind == "video":
                extra["video_note"] = result.get("video")
        else:
            result = agent.predict(state, qs)
    except RuntimeError as e:
        raise HTTPException(503, f"多模态引擎不可用: {e}")
    except (ValueError, FloatingPointError, KeyError) as e:
        raise HTTPException(400, str(e))
    ms = (time.perf_counter() - t0) * 1000
    return {"result": result, "ms": round(ms, 1),
            "inputs": {"state_chars": len(state)}, **extra}


def main():
    global agent
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8030)
    args = ap.parse_args()

    import uvicorn

    print("加载模型(低内存 fp16)…", flush=True)
    agent = load_agent()
    print(f"模型就绪: http://{args.host}:{args.port}", flush=True)
    if multimodal_status():
        print("多模态(vega)可用:首次上传图片/音频/视频时按需加载(约 2GB,首次需下载)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

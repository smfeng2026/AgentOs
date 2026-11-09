"""laya 决策模型测试网站:输入状态文本,提交决策问题,展示校准概率。

用法:
    cd web && python3 server.py            # http://localhost:8030
    python3 server.py --port 8031

模型:NandhaKishorM/laya(pip 包 laya)+ convaiinnovations/laya 英文 checkpoint,
低内存加载见 lowmem.py(本机 1.9GB 内存,fp16 assign,峰值 ~650MB)。
"""

import argparse
import json
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from lowmem import load_agent

app = FastAPI(title="laya decision-model test bench")
agent = None


@app.get("/", response_class=HTMLResponse)
def index():
    with open(__file__.replace("server.py", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/info")
def info():
    return {
        "model": "laya (convaiinnovations/laya, en)",
        "engine": "laya " + __import__("laya").__version__,
        "device": str(agent.device),
        "dtype": str(agent.dtype),
        "questions_per_call": "无限制,单次前向",
    }


@app.post("/api/predict")
async def predict(
    state: str = Form(""),
    questions: str = Form(...),
):
    try:
        qs = json.loads(questions)
        if not isinstance(qs, dict) or not qs:
            raise ValueError("questions 必须是非空 JSON 对象")
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"questions 不是合法 JSON: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))

    t0 = time.perf_counter()
    try:
        result = agent.predict(state, qs)
    except (ValueError, FloatingPointError, KeyError) as e:
        raise HTTPException(400, str(e))
    ms = (time.perf_counter() - t0) * 1000
    return {"result": result, "ms": round(ms, 1), "inputs": {"state_chars": len(state)}}


def main():
    global agent
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8030)
    args = ap.parse_args()

    import uvicorn

    print("加载模型(低内存 fp16)…", flush=True)
    agent = load_agent()
    print("模型就绪: http://{}:{}".format(args.host, args.port), flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

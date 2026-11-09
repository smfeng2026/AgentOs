# AgentOs
基于鸿蒙框架做的 一些场景 demo 原型

## 目录
- `demo/` 鸿蒙应用原型(noteApp、messageApp)
- `models/` 决策模型
  - `laya-omni/` [ZHEQIUSHUI/laya-omni](https://github.com/ZHEQIUSHUI/laya-omni) 源码 + 权重(formal-v3 fusion)
  - `laya-multilingual/` Laya 基座(mmBERT)
  - `siglip2-base-patch16-256/` SigLIP 2 图像编码器
- `web/` laya-omni 模型测试网站(FastAPI + 网页,`python3 web/server.py`)

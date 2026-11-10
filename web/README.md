# web — laya 决策模型测试网站

输入状态文本(或上传图片/音频/视频),提交 `choice / score / noul` 三类决策问题,展示校准概率。

## 运行

```bash
cd web
python3 server.py               # http://localhost:8030
python3 server.py --port 8031   # 自定义端口
```

模型来自 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya):
pip 包 `laya`(代码,`models/laya` 为完整源码副本)+ `convaiinnovations/laya` 英文
checkpoint(`models/laya-en`)。

## 输入模态

| 标签页 | 引擎 | 说明 |
|---|---|---|
| 文本 | laya(ModernBERT) | 状态文本 + 问题,单次前向出全部答案,英文/中文均可 |
| 图片 | vega(Qwen3.5-0.8B) | 截图、照片等;首次上传时自动加载引擎(需 `models/vega-U/`,见根 README) |
| 音频 | whisper-small + vega | 先转写(中文)再决策;需 `models/whisper-small/` |
| 视频 | whisper + vega | 有声轨走转写,无声抽 4 帧走图片路径 |

`GET /api/info` 的 `multimodal` 字段报告 vega 是否可用。

## 文件

| 文件 | 作用 |
|---|---|
| `server.py` | FastAPI 服务:`/api/info`、`/api/predict`(state 或 file + questions) |
| `index.html` | 测试页:四种模态标签页、拖拽上传、6 个中文示例、概率条形图、置信度条、route 标记、原始 JSON |
| `lowmem.py` | 低内存加载(fp16 + safetensors mmap + assign),1.9GB 无 GPU 机器可用 |

## 问题格式

```json
{
  "q1": {"type": "choice", "instructions": "Which department?", "criteria": ["billing", "technical", "other"]},
  "q2": {"type": "score", "instructions": "How urgent?", "criteria": ["not urgent", "soon", "blocking"]},
  "q3": {"type": "noul", "instructions": "Does the user threaten to cancel?"}
}
```

中文同样支持:指令与候选都可以用中文(实测:中文客服邮件 → 「账务」96%)。

返回 `choice` / `score` / `noul` + `probabilities`(校准概率)+ `answer_confidence`;
vega 路径额外返回 `prediction_set`(conformal 集)、`abstain`、`route`(auto/review)与语音转写文本。

> 文本路径在低配机(1.9GB 无 GPU)上 CPU fp16 可跑,单题约 1-2 秒;多模态路径需要
> 约 4GB 内存与 `torchvision`,Apple Silicon 自动用 MPS 加速。

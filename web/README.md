# web — laya 决策模型测试网站

输入状态文本,提交 `choice / score / noul` 三类决策问题,展示校准概率。

## 运行

```bash
cd web
python3 server.py               # http://localhost:8030
python3 server.py --port 8031   # 自定义端口
```

模型来自 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya):
pip 包 `laya`(代码,`models/laya` 为完整源码副本)+ `convaiinnovations/laya` 英文
checkpoint(`models/laya-en`)。

## 文件

| 文件 | 作用 |
|---|---|
| `server.py` | FastAPI 服务:`/api/info`、`/api/predict`(state + questions) |
| `index.html` | 测试页:状态文本、动态问题编辑、概率条形图、原始 JSON、示例填充 |
| `lowmem.py` | 低内存加载(fp16 + safetensors mmap + assign),1.9GB 无 GPU 机器可用 |

## 问题格式

```json
{
  "q1": {"type": "choice", "instructions": "Which department?", "criteria": ["billing", "technical", "other"]},
  "q2": {"type": "score", "instructions": "How urgent?", "criteria": ["not urgent", "soon", "blocking"]},
  "q3": {"type": "noul", "instructions": "Does the user threaten to cancel?"}
}
```

返回 `choice` / `score` / `noul` + `probabilities`(校准概率)+ `answer_confidence`。

> 本机内存 1.9GB、无 GPU:CPU fp16 推理,单题约 1-2 秒。官方 laya 是纯文本
> 决策模型(多模态属下游 laya-omni / vega 项目),故本站不做图片上传。

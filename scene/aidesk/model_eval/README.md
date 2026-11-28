# model_eval — 端侧模型选型评测

只消费 `model_store` 的权重产物（artifact）与冻结数据集快照，产出可复现的评测 run 与对比报告。
**不产权重、不产训练数据、不写 `6h_cross_app_intent/` 与 `model_store/` 的任何文件。**

## 边界

| 项目 | 所有权 |
|---|---|
| `6h_cross_app_intent/` | 数据工厂：产数据；其评测器被本项目以 CLI 方式复用（指标单一来源） |
| `model_store/` | 权重唯一存放点 + artifact 注册表 |
| `model_finetune/`（未建） | 训练过程；产出注册进 store 后才可评测 |
| `model_eval/` | 评测 run、格式统计、对比报告 |

## 命令（在 model_eval/ 下）

```bash
python3 -m src.cli snapshot --name pilot_v1                 # 冻结 6h 评测集快照（已执行过一次）
python3 -m src.cli list                                     # 列出 store 中的 artifact
python3 -m src.cli predict --artifact qwen3-0.6b__official --dataset pilot_v1   # 需要 torch 环境
python3 -m src.cli evaluate --run <run_id>                  # 复用 6h 评测器，输出到 runs/<run_id>/eval/
python3 -m src.cli compare --runs <r1> <r2>                 # → reports/compare_<date>.md
```

## run 命名与可复现性

`runs/<日期>__<artifact_id>__<精度>__<数据集>/`，只追加不覆盖。`run.json` 钉死：
artifact manifest 与权重文件 sha256、数据集快照 sha256、system prompt 哈希、解码参数、runner/依赖版本、硬件。

`predictions.jsonl` 每行：`{sample_id, raw, prediction, format_ok, repair_used, latency_ms, prompt_tokens, completion_tokens, error}`。
原始格式有效率与修复后得分分开统计（`format_ok` 为修复后；`repair_used` 标记是否动用修复/重试）。

## 环境

- 独立 `.venv`（`requirements.txt`），与 6h 生成环境隔离。**torch 等依赖安装、权重下载须先经用户确认。**
- `.env` 中 `HF_HOME` 指向 `../model_store/.hf_cache`，下载缓存不越出 `scene/aidesk/`。
- 当前骨架阶段：`snapshot / list / evaluate / compare` 可用 stdlib + pyyaml 运行；`predict` 需要 torch。

## 阶段规划

1. 阶段一（当前）：`runners/hf_runner.py` 服务端能力基线，统一精度、Qwen3 non-thinking。
2. 阶段二：微调产物（store derived artifact）同流程评测。
3. 阶段三：`runners/` 增加 mnn / llamacpp 适配器，真机效率 run 复用同一 run 结构。
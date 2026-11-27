# 6H-CrossApp-Intent 数据集生产系统

6 小时窗口、30 条跨 App 事件（15 条与意图相关 + 15 条无关）的端侧意图推理数据集生产线。五个 LLM Agent + 程序化噪声注入与校验：

```
Planner → Generator ─┐
                     ├→ 时间重编号 → 程序注入噪声 → 盲标 Annotator → Reviewer → Adjudicator → 结构/标签校验 → 落盘 + manifest
        （Python）───┘                                                                          ↓
                                                                              HUMAN_APPROVAL_PENDING → 人工批准 → build → train/validation/test
```

> **未经人工批准（`approve`）的样本不会进入训练集。** `build` 默认只吃 `approval.state == approved` 的样本；`HUMAN_APPROVAL_PENDING` 只表示通过了程序校验并等待人工审核。

## 1. 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # requests / jsonschema / python-dateutil / pyyaml
cp .env.example .env                       # 填入 API
```

`.env` 关键项（任何兼容 OpenAI Chat Completions 的网关均可，火山方舟网关已实测）：

```ini
LLM_BASE_URL=...        LLM_API_KEY=...
LLM_MODEL=...           # Planner / Generator（生成模型）
LLM_JUDGE_MODEL=...     # Annotator / Reviewer / Adjudicator（建议与生成模型不同厂商，避免同源偏好）
LLM_TIMEOUT=180  LLM_MAX_RETRIES=3  LLM_RESPONSE_FORMAT=json_object
```

`MOCK_LLM=1` 只会阻止生成（占位保护），不会产出假数据。

## 2. 命令

所有命令在项目根目录执行（`python -m src.cli <子命令>`）。

| 子命令 | 作用 | 典型用法 |
|---|---|---|
| `generate` | 生成样本 | `--tasks-file config/pilot_tasks.json --start 2026-12-01T08:00:00+08:00 --jitter-window --workers 6` |
| | 单模板/单类别 | `--template-id C01-S04 --count 3` 或 `--category travel --difficulty D4_noise_interference --intent-count 2` |
| `approve` / `reject` | 人工审批 | `approve --sample-id c01-s04-1a2b3c4d --by 张三`；`reject ... --note "原因"` |
| `build` | 切分 + 导出 | `build`（只吃 approved）；冒烟用 `build --allow-unapproved --force` |
| `evaluate` | 评测预测 | `evaluate --predictions pred.jsonl`；`evaluate --self-check` |
| `report` | 数据集/试点质量报告 | `report` → `data/reports/DATASET_REPORT.md` + `SAMPLES_TABLE.md`（分诊） |
| `review` | 人工审核视图 | `review --pending --limit 5`；`review --export data/review/review_sheet.md` |
| `repair` | 带溯源的修复 | `repair --dry-run`（噪声池占位符 → home_city） |

生成并发用线程池（`--workers`），每条样本独立落盘，失败不影响其他样本；批次汇总写到 `logs/batch_*.json`。

### 任务文件

`config/pilot_tasks.json` 是 `[{template_id, count, difficulty?, intent_count?, split?}, ...]`；`config/templates.json` 有 80 个模板（`C01-S01…C11-S06` 单意图 11 类 × 6，`C12-M01…M08` 多意图），每个模板带 `evidence_hint / expected_label_hint / default_difficulty`，用于保证场景覆盖和切分隔离。

## 3. 产物

每条样本（`<template>-<8hex>`）：

| 路径 | 内容 | 给谁看 |
|---|---|---|
| `data/scenario_specs/` | Planner 规格（意图、预算、实体、难度） | 内部，**不得进入训练数据** |
| `data/internal_labels/` | 生成器相关事件 + 注入噪声（origin / pool_id / hard_negative） | 内部 |
| `data/model_inputs/` | 端侧模型输入（仅 `sample_id, task, window, history_intents, events[...]`） | 训练输入 / 测试输入 |
| `data/gold_labels/` | Gold Label（intents + event_labels + unrelated_event_ids + global_observations） | SFT 目标 / 评测答案 |
| `data/audit_reports/` | 盲标、审查意见、裁决理由、校验结果 | 审核员 |
| `data/manifests/` | 溯源：任务、组成、噪声池版本、prompt 哈希、模型、token、状态、标记、审批状态 | 构建/报告 |
| `logs/generation_logs.jsonl`、`logs/approval_logs.jsonl` | 生产日志 / 审批日志 | 追溯 |

## 4. 防泄漏设计（为什么不是"随机切分"）

1. **事件 ID 不再泄漏标签**：按时间戳重新编号 `e001…e030`，相关事件与噪声交错（旧实现 e001–e015 恒为相关，随机森林可达 100%）。
2. **噪声是抽样而不是固定 15 条**：`config/noise_pool.json` 72 个槽位模板 + 14 个难负例，按样本意图类别排除（`avoid_categories`）并排除样本实体；难负例比例随难度变化（D4 = 40%），用 `home_city`、窗口后的具体日期填充，不再是"当前所在城市"这类占位文本。
3. **风格统一**：App 名取自封闭词表 `config/app_vocabulary.json`，`action` / `content` 字段一律 snake_case 英文、取值中文——相关事件与噪声在写法上不可区分（校验器强制）。
4. **15/15 是目标不是硬约束**：裁决器可按真实语义把噪声改标为相关（范围 10–20 无关），只打 `composition_deviation` 标记，不算失败。
5. **切分按模板组 + 实体连通分量**：同一模板的样本永远在同一 split；共享目的地/产品的模板合并为同一分量（人群/日期不算，避免过度合并）。测试 split 中难度属于 `challenge_difficulties` 的样本进入 `challenge_test`。
6. **构建时二次校验**：`build` 重新跑 `validate_final`，并做泄漏检查：组重叠、完全/近似重复输入、train↔test 强实体重叠、App 覆盖。有错误时拒绝输出（`--force` 才保留）。

## 5. 数据格式

- `data/splits/train.jsonl`、`validation.jsonl`：SFT 行 `{"messages":[system, user, assistant]}`（`--include-meta` 附 `meta`）。assistant 内容为紧凑 JSON，字段顺序固定：`intents, event_labels, unrelated_event_ids, global_observations`，不含 `sample_id`。
- `data/splits/test.jsonl`、`challenge_test.jsonl`：`{"sample_id", "split", "model_input", "gold_label", "meta"}`。
- `data/splits/split_manifest.json` + `DATASET_REPORT.md`：切分明细、标签分布、长度统计、泄漏检查。

## 6. 评测器（`src/evaluator.py`）

意图名是自由文本，所以预测与 gold 的意图按**证据事件集合 Jaccard ≥ 0.4** 贪心一对一匹配，再算：

- Intent P/R/F1（总体、按类别/难度/模板切片）、意图数量完全正确率、合并/拆分错误率；
- 事件关系 4 类 macro-F1、混淆矩阵、event→intent 归属；
- 类别/purpose/status/confidence 准确率，**purpose 过度推断率 / 不足推断率 / unknown 召回**；
- 证据引用合法率与准确率、事实 grounding（引用式；`--llm-judge` 可选 LLM 蕴含判断）、unknown_information 数量比。

`--self-check` 用 gold 自身作预测（应得满分）和"全部无关"平凡基线，验证评测器本身。

## 7. 质量报告（`src/dataset_report.py`）

`report` 汇总：状态/裁决/审批/标记分布、校验错误、类别/模板/难度覆盖、组成与标签分布、**盲标 vs Gold、生成器预设 vs Gold 的一致性（与评测器同一套指标）**、噪声池使用与"噪声被改标为相关"比例、App 角色与捷径风险、审查员问题、长度与 token 成本（按此外推 10,000 条）。

## 8. 推荐流程（先试点，再放量）

```bash
python -m src.cli generate --tasks-file config/pilot_tasks.json --start 2026-12-01T08:00:00+08:00 --jitter-window --workers 6
python -m src.cli report                                   # 看产出率、一致性、成本、捷径风险
# 逐条看 data/audit_reports/<id>.json 与 gold，然后
python -m src.cli approve --sample-id <id> --by 你的名字
python -m src.cli build                                   # 只用 approved
python -m src.cli evaluate --self-check                   # 评测器自检
```

## 9. 已知局限

1. 噪声池仍是种子库（v2026-11-27.2：91 个基础模板 + 14 个难负例，覆盖词表全部 App；6000 次抽样约 330 种不同内容）。48 条试点够用，10,000 条规模前需扩充到数千变体或引入"按场景生成噪声"。
2. 相关性最终靠裁决器与人工判断，程序只能校验结构与一致性；`NEEDS_HUMAN_REVIEW` 的样本必须人工处理。
3. token/长度为启发式估算（`estimate_tokens`），不是分词器计数。
4. 一条样本串行约 4–5 分钟（5 次 LLM 调用，推理模型，约 5.5 万 token，其中三个评审 Agent 占 3/4 以上）；放量前应先降评审成本（更小的审查模型、盲标与预设一致时跳过审查员）。
5. App 词表是封闭的通用名；若要做"未见 App 泛化"实验，需要在切分时按 App 分组（当前仅报告 App 覆盖）。
6. 网络 unstable：客户端默认直连网关（`LLM_TRUST_ENV=0`）。试点时经系统代理（127.0.0.1:7890）出现 DNS/代理中断，13/48 首轮失败，直连重跑 0 失败。

## 10. 测试

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

覆盖：噪声抽样确定性与排除、事件重编号、生成/最终校验器、构建器审批门与切分隔离、导出格式、评测器（oracle 满分 / 平凡基线 / 过度推断 / 合并拆分）、报告。

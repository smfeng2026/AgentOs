# 6H-CrossApp-Intent 数据集生产系统

6 小时窗口、30 条跨 App 事件（目标 15 条与意图相关 + 15 条无关）的端侧意图推理数据集生产线。五个 LLM Agent + 表面对齐噪声生成、程序化校验与发布审计：

```
Planner → Generator（相关事件 + 表面对齐噪声）→ 校验 → 时间重编号 → 盲标 Annotator → Reviewer → Adjudicator
        → 结构/标签校验 → 落盘 + manifest → HUMAN_APPROVAL_PENDING → 人工批准
        → 固定 split plan → build --release（覆盖配额 + 捷径审计）→ train/validation/test
```

> **未经人工批准（`approve`）的样本不会进入训练集。** `build` 默认只吃 `approval.state == approved` 且生产链已完成的样本；`HUMAN_APPROVAL_PENDING` 只表示通过了程序校验并等待人工审核。`build --release` 还会强制执行正式规模、覆盖和捷径门槛。

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
| | 稳定批次/断点续跑 | `--tasks-file config/production_tasks.json --run-id production-v1 --workers 4 --resume` |
| | 单模板/单类别 | `--template-id C01-S04 --count 3` 或 `--category travel --difficulty D4_noise_interference --intent-count 2` |
| `plan-splits` | 为全部模板生成固定切分计划 | 已生成 `config/split_plan.json`；只有新数据集版本才用 `--force` 重建 |
| `approve` / `reject` | 人工审批 | `approve --sample-id c01-s04-1a2b3c4d --by 张三`；`reject ... --note "原因"` |
| `build` | 切分 + 导出 | `build`（只吃 approved，报告覆盖/捷径）；正式版用 `build --release`；冒烟用 `build --allow-unapproved --force` |
| `evaluate` | 评测预测 | `evaluate --predictions pred.jsonl`；`evaluate --self-check` |
| `report` | 数据集/试点质量报告 | `report` → `data/reports/DATASET_REPORT.md` + `SAMPLES_TABLE.md`（分诊） |
| `review` | 人工审核视图 | `review --pending --limit 5`；`review --export data/review/review_sheet.md` |
| `repair` | 带溯源的修复 | `repair --dry-run`（噪声池占位符 → home_city） |

生成并发用线程池（`--workers`），每条样本独立落盘，失败不影响其他样本；批次汇总写到 `logs/batch_*.json`。

### 任务文件

`config/pilot_tasks.json` 是 `[{template_id, count, difficulty?, intent_count?, split?}, ...]`；`config/templates.json` 有 **74 个模板**（`C01-S01…C11-S06` 单意图 11 类 × 6，`C12-M01…M08` 多意图），每个模板带 `evidence_hint / expected_label_hint / default_difficulty`，用于保证场景覆盖和切分隔离。

`config/production_tasks.json` 是正式放量计划：74 个模板各 24 条，共 1776 条；按当前固定 split plan 为 52 个训练模板组、理论 1248 条 train（拒绝样本需补足）。每个单意图类别选择一个训练模板覆盖 D4。`generate` 为每个任务展开项写入稳定 `task_instance_id`；同一 `--run-id` 不加 `--resume` 重跑会拒绝覆盖，`--resume` 跳过已完成任务并重试失败任务。

### 噪声与表面对齐

默认 `noise_generation.mode=generated_aligned`：Generator 在同一次调用中生成 15 条相关事件和 15 条噪声，至少 60% 噪声需要与不同相关事件共享 `app/action/content keys`，且长度中位数比在 0.65～1.60。程序验证配对、字段风格、重复和结构后才分配时间戳；语义是否真正无关仍由盲标、审查、裁决与人工确认。`mode=pool` 仅保留给旧流程和冒烟测试。

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
2. **噪声与相关事件做表面对齐**：默认由同一次 Generator 调用生成，至少 9 条噪声与相关事件共享 App、动作和 content 字段结构，并保持相近信息量，避免“短文本=噪声”“某 action=噪声”。旧种子池仍可通过 `noise_generation.mode=pool` 用于兼容性测试，但不建议用于正式训练。
3. **风格统一**：App 名取自封闭词表 `config/app_vocabulary.json`，`action` / `content` 字段一律 snake_case 英文、取值中文——相关事件与噪声在写法上不可区分（校验器强制）。
4. **15/15 是目标不是硬约束**：裁决器可按真实语义把噪声改标为相关（范围 10–20 无关），只打 `composition_deviation` 标记，不算失败。
5. **切分按固定模板组计划 + 实体连通分量检查**：`config/split_plan.json` 预先固定全部 74 个模板，新增样本不会触发已有组重排；共享目的地/产品的模板若跨计划 split 会报错。测试 split 中难度属于 `challenge_difficulties` 的样本进入 `challenge_test`。
6. **构建时二次校验与捷径审计**：`build` 重新跑 `validate_final`，并做组重叠、完全/近似重复输入、强实体、App 覆盖检查；另外仅用 content 长度、action、app+action、字段名、位置和小时桶训练简单探针，测量相关/无关二分类捷径。
7. **正式发布门槛**：`build --release` 强制执行 `dataset_config.yaml > release` 中的 train=1200、模板/难度/关系/状态/置信度/purpose 配额和 `shortcut_audit.thresholds`；普通 build 只报告这些错误，便于试点诊断。

## 5. 数据格式

- `data/splits/train.jsonl`、`validation.jsonl`：SFT 行 `{"messages":[system, user, assistant]}`（`--include-meta` 附 `meta`）。system 由 `sft_system.txt` 与统一的 `labeling_policy.txt` 组成；assistant 内容为紧凑 JSON，字段顺序固定：`intents, event_labels, unrelated_event_ids, global_observations`，不含 `sample_id`。
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
# 先用小批验证新噪声和审核口径
python -m src.cli generate --tasks-file config/pilot_tasks.json --run-id pilot-aligned-v1 --start 2026-12-01T08:00:00+08:00 --jitter-window --workers 4 --resume
python -m src.cli report
# 逐条审核后 approve；正式放量沿用固定 split plan
python -m src.cli generate --tasks-file config/production_tasks.json --run-id production-v1 --start 2026-12-01T08:00:00+08:00 --jitter-window --workers 4 --resume
python -m src.cli build --release
python -m src.cli evaluate --self-check
```

> 不要把旧 `data/splits_pilot/` 当作新流程已达标的结果。旧试点未经过表面对齐和 shortcut release gate；正式训练应使用新生成、重新审批的数据版本。

## 9. 已知局限

1. 表面对齐只能由程序检查结构、配对和长度，**不能证明噪声在语义上真的无关**；仍需裁决与人工审核。旧种子噪声池（91 个基础模板 + 14 个难负例）仅作兼容回退，不再建议用于正式训练。
2. 相关性最终靠裁决器与人工判断，程序只能校验结构与一致性；`NEEDS_HUMAN_REVIEW` 的样本必须人工处理。
3. token/长度为启发式估算（`estimate_tokens`），不是分词器计数。
4. 旧试点一条样本串行约 4–5 分钟、约 5.5 万 token；表面对齐噪声会增加 Generator 输出，正式成本必须以新中试的 manifest usage 重新估算。放量前可使用更小的审查模型及 Reviewer skip，但 Adjudicator 和人工审批仍保留。
5. App 词表是封闭的通用名；若要做"未见 App 泛化"实验，需要在切分时按 App 分组（当前仅报告 App 覆盖）。
6. 网络 unstable：客户端默认直连网关（`LLM_TRUST_ENV=0`）。试点时经系统代理（127.0.0.1:7890）出现 DNS/代理中断，13/48 首轮失败，直连重跑 0 失败。

## 10. 测试

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

覆盖：旧噪声池抽样、生成噪声表面对齐、事件重编号、生成/最终校验器、固定切分计划、release 覆盖与捷径门槛、构建器审批门、导出格式、断点任务身份、生成失败持久化、评测器、报告。

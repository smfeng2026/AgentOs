# Dataset build report

- builder 1.1.0，生成于 2026-11-28T10:01:06+08:00
- 样本：48 条可用，跳过 1 条

## 切分

| Split | 样本 | 模板组 |
|---|---|---|
| train | 32 | 16 |
| validation | 4 | 2 |
| test | 6 | 3 |
| challenge_test | 6 | 3 |

### 类别 × Split

| 类别 | train | validation | test | challenge_test |
|---|---|---|---|---|
| entertainment | 2 | 2 | 0 | 0 |
| family | 2 | 0 | 2 | 0 |
| finance_admin | 2 | 2 | 0 | 0 |
| fitness | 4 | 0 | 0 | 0 |
| health_service | 2 | 0 | 0 | 2 |
| learning | 4 | 0 | 0 | 0 |
| local_transport | 2 | 0 | 0 | 2 |
| multi_intent | 2 | 0 | 0 | 2 |
| shopping | 4 | 0 | 0 | 0 |
| social_dining | 2 | 0 | 2 | 0 |
| travel | 2 | 0 | 2 | 0 |
| work | 4 | 0 | 0 | 0 |

### 难度 × Split

| 难度 | train | validation | test | challenge_test |
|---|---|---|---|---|
| D2_multi_app_fusion | 14 | 2 | 6 | 0 |
| D3_weak_evidence | 10 | 2 | 0 | 2 |
| D5_multi_intent | 2 | 0 | 0 | 2 |
| D6_conflict_state_change | 6 | 0 | 0 | 2 |

## 标签分布

- 每样本意图数：{1: 43, 2: 5}
- 每样本无关事件数：{15: 44, 16: 3, 19: 1}
- 事件关系：{'related_interest': 547, 'unrelated': 727, 'supporting': 163, 'contradicting': 3}
- 意图类别：{'travel': 8, 'work': 6, 'shopping': 7, 'social_dining': 4, 'health_service': 4, 'learning': 4, 'entertainment': 4, 'family': 4, 'finance_admin': 4, 'fitness': 4, 'local_transport': 4}
- 状态：{'confirmed': 32, 'cancelled': 6, 'emerging': 13, 'considering': 2}；置信度：{'high': 38, 'low': 15}

purpose（类别->目的）：

- entertainment->fan: 4
- family->family_gathering: 2
- family->renovation: 1
- family->unknown: 1
- finance_admin->banking: 1
- finance_admin->certification: 1
- finance_admin->personal_admin: 1
- finance_admin->unknown: 1
- fitness->training: 2
- fitness->unknown: 2
- health_service->checkup: 2
- health_service->unknown: 2
- learning->exam_prep: 2
- learning->hobby: 1
- learning->unknown: 1
- local_transport->outing: 4
- shopping->household: 1
- shopping->personal_use: 4
- shopping->unknown: 2
- social_dining->socializing: 4
- travel->business: 4
- travel->tourism: 4
- work->meeting: 4
- work->unknown: 2

## 长度（估算 token，非分词器计数）

- 输入：p50 2026 / p90 2199 / max 2268
- 目标：p50 1099 / p90 1304 / max 1551
- 合计：p50 3141 / p90 3469 / max 3779

## 泄漏检查

- 错误：0；警告：0

- 仅在测试集出现的 App：['pharmacy_app']
- 仅在训练集出现的 App：['delivery_app', 'docs_app', 'exam_app', 'furniture_app', 'hospital_app', 'learning_app', 'meeting_app', 'sports_app']

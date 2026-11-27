# Evaluator self-check

- gold: ['data/splits_pilot/test.jsonl', 'data/splits_pilot/challenge_test.jsonl']（12 行）

| 基线 | Intent F1 | Relation macro-F1 | Purpose Acc | Fact Grounding |
|---|---|---|---|---|
| oracle（gold 当预测，应为满分） | 1.0 | 1.0 | 1.0 | 1.0 |
| trivial（全部判无关） | 0.0 | 0.1673 | None | None |

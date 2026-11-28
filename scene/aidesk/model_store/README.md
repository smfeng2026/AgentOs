# model_store — 权重唯一存放点与产物注册表

所有权规则：**6h 产权数据，model_finetune 产权训练过程，model_store 产权重，model_eval 产分数。**
本项目只被写入权重与注册元数据；评测与训练都不得把权重散落在各自目录。

## 布局

```
model_store/
├── base/<base_id>/            # 上游官方权重（下载后落这里），upstream.json 记录 repo/revision
│   ├── mnn-pocket-0.3b/
│   ├── lfm2.5-350m/
│   ├── minicpm4-0.5b/
│   └── qwen3-0.6b/
├── derived/<artifact_id>/     # 衍生产物：微调适配器 / 合并权重 / 量化版
│   ├── manifest.json          # 血缘登记（schema/artifact_manifest.schema.json）
│   └── <权重文件>
├── schema/artifact_manifest.schema.json
└── cli.py                     # list / verify / register-derived / register-base
```

## artifact 命名

`<基座>__<训练集>__<方法>__<变体>`；官方权重特例 `<基座>__official`。例：

- `qwen3-0.6b__official`
- `qwen3-0.6b__sft-pilot-v1__lora-r16`
- `qwen3-0.6b__sft-pilot-v1__lora-r16__merged`
- `qwen3-0.6b__sft-pilot-v1__lora-r16__int4-gguf`（`parent_artifact` 指向 merged）

## 规则

1. 权重文件不进 git（见 .gitignore）；`upstream.json` / `manifest.json` 进 git。
2. 每个 derived 产物必须有 manifest，登记 base revision、训练 run、训练集快照 sha256、超参、逐文件 sha256。
3. 评测只认 artifact_id；`model_eval` 通过 manifest 解析权重路径并钉死哈希。
4. HF 下载缓存固定在 `.hf_cache/`（`HF_HOME`），不越出 `scene/aidesk/`，不进 git。
5. 下载权重、安装依赖必须先经用户确认。

## 命令

```bash
python3 cli.py list                                  # 列出已注册 artifact
python3 cli.py verify --artifact-id <id>             # 逐文件复核 sha256
python3 cli.py register-derived --artifact-id <id> --kind lora_adapter \
    --base qwen3-0.6b --training-run <path> --train-snapshot pilot-v1 --train-sha256 <hash>
python3 cli.py register-base --base-id qwen3-0.6b --revision <commit>   # 权重就位后登记
```

## 权重下载（fresh checkout 必需）

权重不入库（单文件超 GitHub 100MB 限制）。clone 后按 `base/<base_id>/upstream.json` 登记的 repo/revision 重新下载：

```bash
cd ../model_eval
# 依赖安装与下载须先经用户确认；.venv 自建：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_base.py --base-id qwen3-0.6b
.venv/bin/python scripts/fetch_base.py --base-id lfm2.5-350m
.venv/bin/python scripts/fetch_base.py --base-id minicpm4-0.5b
.venv/bin/python scripts/fetch_base.py --base-id mnn-pocket-0.3b
# 国内网络可加镜像：--endpoint https://hf-mirror.com
```

下载脚本钉住上游 commit revision，落盘到 `base/<base_id>/`，缓存固定在本目录 `.hf_cache/`（不入库）。就位后用 `cli.py verify` 复核 sha256。
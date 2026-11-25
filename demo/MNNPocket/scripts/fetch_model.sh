#!/usr/bin/env bash
#
# (Re)downloads the MNN-Pocket-0.3B INT4 package into
# entry/src/main/resources/rawfile/model so the app can bundle it.
#
# The repository ships all model files EXCEPT llm.mnn.weight (163.5 MB, over
# GitHub's 100 MB limit). Run this script after a fresh checkout, or to switch
# model revisions. See SETUP.md for details.
#
# Usage: scripts/fetch_model.sh [hf-endpoint]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${REPO_ROOT}/entry/src/main/resources/rawfile/model"
ENDPOINT="${1:-https://hf-mirror.com}"

mkdir -p "${TARGET}"
cd "${TARGET}"

for file in config.json llm.mnn llm.mnn.weight tokenizer.mtok llm_config.json; do
    echo "==> ${file}"
    curl -fL --retry 3 -O "${ENDPOINT}/taobao-mnn/MNN-Pocket-0.3B-MNN/resolve/main/${file}"
done

# The upstream config.json is already the CPU / 4-thread setup used on phones.
# The only local edit is an explicit max_new_tokens, matching the copy that
# ships inside the app.
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PATCH'
import json

with open('config.json', encoding='utf-8') as handle:
    config = json.load(handle)
config.setdefault('max_new_tokens', 512)
with open('config.json', 'w', encoding='utf-8') as handle:
    json.dump(config, handle, indent=4, ensure_ascii=False)
    handle.write('\n')
PATCH
fi

du -sh "${TARGET}"
ls -la "${TARGET}"

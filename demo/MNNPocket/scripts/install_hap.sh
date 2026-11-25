#!/usr/bin/env bash
#
# Install a (large) HAP onto a connected HarmonyOS device.
#
# `hdc install` streams the whole package and is known to stall mid-transfer
# on big files over USB (observed: 525 MB HAP stuck at 35–86 MB). This script
# splits the HAP into 32 MB chunks, sends each with size verification, then
# reassembles on-device and installs via `bm install -r -p`.
#
# Usage: scripts/install_hap.sh [path-to-hap]
#   default: entry/build/default/outputs/default/entry-default-signed.hap

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HAP="${1:-${REPO_ROOT}/entry/build/default/outputs/default/entry-default-signed.hap}"
HDC="${HDC:-$(find /Applications/DevEco-Studio.app/Contents/sdk -name hdc -type f 2>/dev/null | head -1)}"

if [[ -z "${HDC}" || ! -x "${HDC}" ]]; then
    echo "hdc not found; set HDC=/path/to/hdc" >&2
    exit 1
fi
if [[ ! -f "${HAP}" ]]; then
    echo "HAP not found: ${HAP}" >&2
    exit 1
fi

TARGET="$("${HDC}" list targets | head -1)"
if [[ -z "${TARGET}" ]]; then
    echo "no device connected" >&2
    exit 1
fi
echo "==> device ${TARGET}, hap $(stat -f %z "${HAP}") bytes"

CHUNK_DIR="$(mktemp -d /tmp/hapchunks.XXXXXX)"
RDIR=/data/local/tmp/pocket_chunks
trap 'rm -rf "${CHUNK_DIR}"' EXIT

split -b 32m "${HAP}" "${CHUNK_DIR}/chunk_"
"${HDC}" shell mkdir -p "${RDIR}"

ok=0
total=0
for f in "${CHUNK_DIR}"/chunk_*; do
    total=$((total + 1))
    name="$(basename "$f")"
    want="$(stat -f %z "$f")"
    got=""
    for attempt in 1 2 3; do
        "${HDC}" file send "$f" "${RDIR}/${name}" >/dev/null 2>&1 || true
        got="$("${HDC}" shell stat -c %s "${RDIR}/${name}" 2>/dev/null | tr -d '\r\n ')"
        [[ "${got}" == "${want}" ]] && break
        sleep 1
    done
    if [[ "${got}" != "${want}" ]]; then
        echo "FAIL ${name}: got=${got} want=${want}" >&2
        exit 1
    fi
    ok=$((ok + 1))
done
echo "==> chunks ${ok}/${total} verified"

"${HDC}" shell "cat ${RDIR}/chunk_* > ${RDIR}/full.hap"
remote_size="$("${HDC}" shell stat -c %s "${RDIR}/full.hap" | tr -d '\r\n ')"
local_size="$(stat -f %z "${HAP}")"
if [[ "${remote_size}" != "${local_size}" ]]; then
    echo "FAIL reassembly: ${remote_size} != ${local_size}" >&2
    exit 1
fi

echo "==> bm install"
"${HDC}" shell bm install -r -p "${RDIR}/full.hap"
"${HDC}" shell rm -rf "${RDIR}"
echo "==> done"
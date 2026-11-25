#!/usr/bin/env bash
#
# Builds MNN (LLM enabled) for HarmonyOS arm64 and installs it into the app:
#
#   entry/libs/arm64-v8a/libMNN.so                  prebuilt runtime shipped in the HAP
#   entry/src/main/cpp/third_party/mnn/include/...  headers used by libllmbridge.so
#
# Usage:
#   scripts/build_mnn_ohos.sh [path-to-MNN-source]
#
# Defaults can be overridden with environment variables:
#   MNN_SOURCE   MNN source tree (default: ../model/MNN-Pocket-0.3B/MNN next to the repo)
#   DEVECO_HOME  DevEco Studio install (default: /Applications/DevEco-Studio.app/Contents)
#   CMake        cmake binary (default: the one bundled with the HarmonyOS SDK)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVECO_HOME="${DEVECO_HOME:-/Applications/DevEco-Studio.app/Contents}"
SDK_NATIVE="${DEVECO_HOME}/sdk/default/openharmony/native"
TOOLCHAIN="${SDK_NATIVE}/build/cmake/ohos.toolchain.cmake"
BUILD_DIR="${REPO_ROOT}/.mnn-build/ohos-arm64"

# Resolve the MNN source tree: explicit argument > $MNN_SOURCE > a few usual spots.
if [[ -n "${1:-}" ]]; then
    MNN_SOURCE="$1"
elif [[ -z "${MNN_SOURCE:-}" ]]; then
    MNN_SOURCE=""
    for candidate in \
        "${REPO_ROOT}/../model/MNN-Pocket-0.3B/MNN" \
        "${HOME}/jobs/model/MNN-Pocket-0.3B/MNN" \
        "${REPO_ROOT}/.mnn-src/MNN"; do
        if [[ -f "${candidate}/CMakeLists.txt" ]]; then
            MNN_SOURCE="${candidate}"
            break
        fi
    done
fi

if [[ -z "${MNN_SOURCE}" || ! -f "${MNN_SOURCE}/CMakeLists.txt" ]]; then
    echo "MNN source not found." >&2
    echo "Usage: $0 /path/to/MNN/source   (or set MNN_SOURCE)" >&2
    exit 1
fi
if [[ ! -f "${TOOLCHAIN}" ]]; then
    echo "HarmonyOS toolchain not found: ${TOOLCHAIN}" >&2
    echo "Set DEVECO_HOME to your DevEco Studio Contents directory." >&2
    exit 1
fi

if [[ -n "${CMAKE:-}" ]]; then
    CMAKE_BIN="${CMAKE}"
elif [[ -x "${SDK_NATIVE}/build-tools/cmake/bin/cmake" ]]; then
    CMAKE_BIN="${SDK_NATIVE}/build-tools/cmake/bin/cmake"
elif command -v cmake >/dev/null 2>&1; then
    CMAKE_BIN="$(command -v cmake)"
else
    echo "cmake not found. Install one or set CMAKE=/path/to/cmake." >&2
    exit 1
fi

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo "==> MNN source : ${MNN_SOURCE}"
echo "==> build dir  : ${BUILD_DIR}"
echo "==> toolchain  : ${TOOLCHAIN}"
echo "==> cmake      : ${CMAKE_BIN}"

mkdir -p "${BUILD_DIR}"
"${CMAKE_BIN}" -S "${MNN_SOURCE}" -B "${BUILD_DIR}" -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOHOS_ARCH=arm64-v8a \
    -DOHOS_STL=c++_shared \
    -DOHOS_PLATFORM_LEVEL=12 \
    -DMNN_USE_LOGCAT=ON \
    -DMNN_USE_SSE=OFF \
    -DMNN_ARM82=ON \
    -DMNN_SUPPORT_BF16=ON \
    -DMNN_SEP_BUILD=OFF \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DMNN_BUILD_LLM=ON \
    -DMNN_BUILD_LLM_OMNI=OFF \
    -DMNN_BUILD_CONVERTER=OFF \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_BUILD_BENCHMARK=OFF \
    -DMNN_BUILD_TOOLS=OFF \
    -DNATIVE_LIBRARY_OUTPUT=. \
    -DNATIVE_INCLUDE_OUTPUT=.

echo "==> building libMNN.so (target MNN)"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --target MNN -j "${JOBS}"

echo "==> installing into the app"
mkdir -p "${REPO_ROOT}/entry/libs/arm64-v8a"
cp "${BUILD_DIR}/libMNN.so" "${REPO_ROOT}/entry/libs/arm64-v8a/libMNN.so"
if [[ -x "${SDK_NATIVE}/llvm/bin/llvm-strip" ]]; then
    "${SDK_NATIVE}/llvm/bin/llvm-strip" --strip-unneeded "${REPO_ROOT}/entry/libs/arm64-v8a/libMNN.so"
fi

rm -rf "${REPO_ROOT}/entry/src/main/cpp/third_party/mnn/include"
mkdir -p "${REPO_ROOT}/entry/src/main/cpp/third_party/mnn/include"
cp -R "${MNN_SOURCE}/include/MNN" "${REPO_ROOT}/entry/src/main/cpp/third_party/mnn/include/"
cp -R "${MNN_SOURCE}/transformers/llm/engine/include/llm" \
    "${REPO_ROOT}/entry/src/main/cpp/third_party/mnn/include/"

echo "==> done"
ls -la "${REPO_ROOT}/entry/libs/arm64-v8a/libMNN.so"

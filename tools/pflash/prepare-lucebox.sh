#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: prepare-lucebox.sh <lucebox-checkout> [cuda-arch] [build-dir]

Prepare and build the BF16 Qwen3-0.6B PFlash scorer daemon used by this fork.
The tested Lucebox revision is 99ab4cebd331310adcc37c5ab89e30323b3ddf27.

Examples:
  tools/pflash/prepare-lucebox.sh ../lucebox 86
  tools/pflash/prepare-lucebox.sh ../lucebox 70 ../lucebox/build-pflash-v100
USAGE
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

LUCEBOX=$(realpath "$1")
CUDA_ARCH=${2:-86}
BUILD=${3:-"$LUCEBOX/build-pflash-sm${CUDA_ARCH}"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TESTED_COMMIT=99ab4cebd331310adcc37c5ab89e30323b3ddf27

[[ -d "$LUCEBOX/.git" ]] || { echo "not a Lucebox git checkout: $LUCEBOX" >&2; exit 2; }
HEAD=$(git -C "$LUCEBOX" rev-parse HEAD)
if [[ "$HEAD" != "$TESTED_COMMIT" ]]; then
    echo "warning: tested against Lucebox $TESTED_COMMIT, checkout is $HEAD" >&2
fi

git -C "$LUCEBOX" submodule update --init --recursive

DAEMON="$LUCEBOX/server/test/pflash_daemon.cpp"
if ! grep -q -- '--gpu=' "$DAEMON"; then
    git -C "$LUCEBOX" apply "$SCRIPT_DIR/lucebox-pflash-daemon-gpu.patch"
fi

MMVQ="$LUCEBOX/server/deps/llama.cpp/ggml/src/ggml-cuda/mmvq.cu"
if [[ ! -f "$MMVQ.pflash-original" ]]; then
    if grep -q 'BF16.*PFlash' "$MMVQ"; then
        # Support the name used by the original local experiment; otherwise do
        # not pretend the already-installed stub is the upstream original.
        if [[ -f "$MMVQ.full-for-pflash-bf16" ]]; then
            cp "$MMVQ.full-for-pflash-bf16" "$MMVQ.pflash-original"
        else
            echo "warning: BF16 MMVQ stub is already installed and no original backup was found" >&2
        fi
    else
        cp "$MMVQ" "$MMVQ.pflash-original"
    fi
fi
cp "$SCRIPT_DIR/mmvq-bf16-stub.cu" "$MMVQ"

cmake -S "$LUCEBOX/server" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DDFLASH27B_GPU_BACKEND=cuda \
    -DDFLASH27B_ENABLE_BSA=OFF \
    -DDFLASH27B_SERVER=OFF \
    -DDFLASH27B_TESTS=OFF \
    -DGGML_CUDA=ON
cmake --build "$BUILD" --target pflash_daemon -j "${JOBS:-$(nproc)}"

echo
echo "PFlash daemon: $BUILD/pflash_daemon"
echo "The MMVQ source was saved at: $MMVQ.pflash-original"
echo "This build is intended only for the BF16 Qwen3-0.6B PFlash scorer."

#!/bin/bash
set -euo pipefail
mkdir -p /workspace/llama-pxq4-v100/benches/pxq4-v100/results
cmake -S /workspace/llama-pxq4-v100 -B /models/llama-pxq4-build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_NATIVE=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/workspace/llama-pxq4-v100/benches/pxq4-v100/results/configure.log 2>&1
cmake --build /models/llama-pxq4-build -j12 --target llama-server > /workspace/llama-pxq4-v100/benches/pxq4-v100/results/build.log 2>&1

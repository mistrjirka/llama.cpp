#!/bin/bash
set -euo pipefail
cd /workspace/llama-pxq4-v100
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_70 --use_fast_math -Wno-deprecated-gpu-targets \
    -Iggml/include -Iggml/src -Iggml/src/ggml-cuda benches/pxq4-v100/test_prefill.cu \
    -L/models/llama-pxq4-build/bin -lggml-cuda -lggml-base -lggml -Xlinker -rpath -Xlinker /models/llama-pxq4-build/bin \
    -o /models/llama-pxq4-build/test-pxq4-prefill

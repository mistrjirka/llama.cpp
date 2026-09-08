#!/bin/bash
set -euo pipefail
exec /usr/local/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --force-overwrite=true -o /workspace/llama-pxq4-v100/benches/pxq4-v100/results/decode-profile /models/llama-pxq4-build/bin/llama-server "$@"

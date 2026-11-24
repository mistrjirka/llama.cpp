#!/usr/bin/env bash
set -euo pipefail
export MALLOC_ARENA_MAX=2
# SenseNova-Loop-9B release, original Q6_K. Native container, unchanged weights.
MODEL="$HOME/models/SenseNova-Loop-9B/SenseNova-Loop-9B-native-Q6_K.gguf"
SERVER="$HOME/programy/llama-sensenova-abliterated-build/build-cuda/bin/llama-server"
if [[ ! -x "$SERVER" || ! -s "$MODEL" ]]; then
    echo 'SenseNova normal runtime/model missing' >&2
    exit 1
fi
exec "$SERVER" -m "$MODEL" --mmproj "$HOME/models/SenseNova-Loop-9B/mmproj-SenseNova-Loop-9B-F16.gguf" --jinja --host 0.0.0.0 --port 2345 --no-mmproj-offload --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
    --ctx-size "${SENSENOVA_CTX_SIZE:-110000}" --parallel 1 --gpu-layers 99 --flash-attn on \
    --cache-ram 4096 --ctx-checkpoints 3 --ctx-checkpoint-ram 384 \
    --cache-type-k q8_0 --cache-type-v q8_0 -b 2048 -ub 256 \
    --ctx-checkpoint-turns --prompt-tile 32 \
    --reasoning-format auto --reasoning-budget -1 --n-predict -1 \
    --temp 1.0 --top-p 0.95 --top-k 50 --min-p 0.0 \
    --repeat-penalty 1.0 --presence-penalty 0.0 --frequency-penalty 0.0

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
ROOT=${ROOT:-$(cd -- "$REPO_ROOT/.." && pwd)}
MODEL=${MODEL:-/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf}
SEQ_JSON=${SEQ_JSON:-$ROOT/results/cache100k-sharedstate-ab/seq.json}
SLOT_DIR=${SLOT_DIR:-$ROOT/results/cache100k-sharedstate-ab/slots}
FORK_BIN=${FORK_BIN:-$ROOT/build-rebased/bin/llama-server}
UPSTREAM_BIN=${UPSTREAM_BIN:-$ROOT/build-upstream-current/bin/llama-server}
V100_UUID=${V100_UUID:-GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79}
OUT=${OUT:-$ROOT/results/qwen38-v100-only-ab}
PORT=${PORT:-19521}

mkdir -p "$OUT"

wait_health() {
    local pid=$1
    for _ in $(seq 1 240); do
        curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

run_one() {
    local mode=$1
    local bin=$2
    shift 2
    local dir=$OUT/$mode
    mkdir -p "$dir"

    setsid env CUDA_VISIBLE_DEVICES="$V100_UUID" "$bin" \
        --model "$MODEL" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --parallel 1 \
        --ctx-size 262144 \
        --fit off \
        --gpu-layers 63 \
        --split-mode layer \
        --flash-attn on \
        --batch-size 4096 \
        --ubatch-size 1024 \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --cache-type-k-draft f16 \
        --cache-type-v-draft f16 \
        --spec-type draft-mtp \
        --spec-draft-n-max 2 \
        --no-warmup \
        --perf \
        --slot-save-path "$SLOT_DIR" \
        "$@" >"$dir/server.log" 2>&1 &
    local pid=$!

    cleanup() {
        kill -TERM -- -$pid 2>/dev/null || true
        sleep 1
        kill -KILL -- -$pid 2>/dev/null || true
        wait $pid 2>/dev/null || true
    }

    if ! wait_health "$pid"; then
        tail -100 "$dir/server.log"
        cleanup
        return 1
    fi

    MODE=$mode PORT=$PORT SEQ_JSON=$SEQ_JSON OUT=$OUT python3 -u - <<'PY'
import json
import os
import urllib.request

mode = os.environ["MODE"]
port = os.environ["PORT"]
out = os.environ["OUT"]
seq = json.load(open(os.environ["SEQ_JSON"]))
base = f"http://127.0.0.1:{port}"

def post(path, obj, timeout=1200):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

restored = post("/slots/0?action=restore", {"filename": "cache100k.bin"}, 600)
assert restored.get("n_restored") == 100000, restored
result = post("/completion", {
    "prompt": seq,
    "n_predict": 64,
    "ignore_eos": True,
    "seed": 1234,
    "id_slot": 0,
    "cache_prompt": True,
    "stream": False,
    "return_tokens": True,
})
assert result["timings"]["cache_n"] == 100000, result["timings"]
json.dump(result, open(os.path.join(out, mode, "result.json"), "w"), indent=2)
print(mode, json.dumps(result["timings"]), flush=True)
PY

    cleanup
}

run_one upstream "$UPSTREAM_BIN"
run_one fork "$FORK_BIN" --prefill-reuse 1024 --pipeline-copies 2 --spec-draft-ubatch 1024

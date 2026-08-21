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
OUT=${OUT:-$ROOT/results/qwen38-100k-ab}
PORT=${PORT:-19520}

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

    setsid "$bin" \
        --model "$MODEL" \
        --alias qwen3.8-27b \
        --host 127.0.0.1 \
        --port "$PORT" \
        --ctx-size 262144 \
        --parallel 1 \
        --split-mode layer \
        --fit off \
        --gpu-layers all \
        --tensor-split 64,2 \
        --flash-attn on \
        --batch-size 4096 \
        --ubatch-size 4096 \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --cache-type-k-draft f16 \
        --cache-type-v-draft f16 \
        --cache-ram 0 \
        --ctx-checkpoints 32 \
        --checkpoint-min-step 8192 \
        --spec-type draft-mtp \
        --spec-draft-n-max 2 \
        --jinja \
        --reasoning on \
        --reasoning-preserve \
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

    nvidia-smi --query-gpu=index,name,memory.used,memory.free,power.draw,power.limit --format=csv,noheader >"$dir/idle-gpu.txt"

    MODE=$mode PORT=$PORT SEQ_JSON=$SEQ_JSON python3 -u - <<'PY'
import json
import os
import time
import urllib.request

mode = os.environ["MODE"]
port = os.environ["PORT"]
seq_path = os.environ["SEQ_JSON"]
root = os.getcwd()
base = f"http://127.0.0.1:{port}"
seq = json.load(open(seq_path))

def post(path, obj, timeout=2400):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

restored = post("/slots/0?action=restore", {"filename": "cache100k.bin"}, 600)
assert restored.get("n_restored") == 100000, restored

t0 = time.perf_counter()
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
wall = time.perf_counter() - t0
assert result["timings"]["cache_n"] == 100000, result["timings"]
assert result["timings"]["prompt_n"] == 1000, result["timings"]

out = os.environ.get("OUT") or os.path.join(root, "results", "qwen38-100k-ab")
os.makedirs(os.path.join(out, mode), exist_ok=True)
json.dump(restored, open(os.path.join(out, mode, "restore.json"), "w"), indent=2)
json.dump(result, open(os.path.join(out, mode, "result.json"), "w"), indent=2)
print(mode, "wall", wall, "timings", json.dumps(result["timings"]), flush=True)
PY

    cleanup
}

export OUT
run_one upstream "$UPSTREAM_BIN"
run_one fork "$FORK_BIN" --prefill-reuse 1024 --pipeline-copies 2 --spec-draft-ubatch 1024

python3 - "$OUT" <<'PY'
import json
import os
import sys

out = sys.argv[1]
upstream = json.load(open(os.path.join(out, "upstream", "result.json")))
fork = json.load(open(os.path.join(out, "fork", "result.json")))
u = upstream["timings"]
f = fork["timings"]
summary = {
    "upstream": u,
    "fork": f,
    "pp_gain_pct": 100 * (f["prompt_per_second"] / u["prompt_per_second"] - 1),
    "tg_gain_pct": 100 * (f["predicted_per_second"] / u["predicted_per_second"] - 1),
    "tokens_equal": upstream.get("tokens") == fork.get("tokens"),
    "content_equal": upstream.get("content") == fork.get("content"),
}
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
PY

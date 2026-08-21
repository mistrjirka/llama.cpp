#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
ROOT=${ROOT:-$(cd -- "$REPO_ROOT/.." && pwd)}
MODEL=${MODEL:-/models/GLM-5.2-UD-IQ2_XXS/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf}
SEQ_JSON=${SEQ_JSON:-$ROOT/results/resumed-100k-benchmark-20260821/glm52-10k/seq11k.json}
SLOT_DIR=${SLOT_DIR:-$ROOT/results/resumed-100k-benchmark-20260821/glm52-10k/shared-slot}
ARGS_FILE=${ARGS_FILE:-$ROOT/results/resumed-100k-benchmark-20260821/glm52-10k/fit-dual-ub4096/m768.args}
SERVER_BIN=${SERVER_BIN:-$ROOT/build-upstream-current/bin/llama-server}
PORT=${PORT:-19460}

mkdir -p "$SLOT_DIR"
mapfile -d '' -t FIT_ARGS < <(python3 - "$ARGS_FILE" <<'PY'
import shlex
import sys
for arg in shlex.split(open(sys.argv[1]).read()):
    print(arg, end="\0")
PY
)

setsid "$SERVER_BIN" \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --parallel 1 \
    --fit off \
    "${FIT_ARGS[@]}" \
    --split-mode layer \
    --flash-attn on \
    --batch-size 4096 \
    --ubatch-size 4096 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --threads-batch 32 \
    --no-warmup \
    --perf \
    --slot-save-path "$SLOT_DIR" >"$SLOT_DIR/prepare-glm10k.server.log" 2>&1 &
pid=$!

cleanup() {
    kill -TERM -- -$pid 2>/dev/null || true
    sleep 1
    kill -KILL -- -$pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 360); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { tail -100 "$SLOT_DIR/prepare-glm10k.server.log"; exit 2; }
    sleep 1
done

PORT=$PORT SEQ_JSON=$SEQ_JSON SLOT_DIR=$SLOT_DIR python3 -u - <<'PY'
import json
import os
import urllib.request

port = os.environ["PORT"]
seq = json.load(open(os.environ["SEQ_JSON"]))
slot_dir = os.environ["SLOT_DIR"]
assert len(seq) >= 10000, len(seq)
base = f"http://127.0.0.1:{port}"

def post(path, obj, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

warm = post("/completion", {
    "prompt": seq[:1024],
    "n_predict": 1,
    "ignore_eos": True,
    "seed": 1234,
    "id_slot": 0,
    "cache_prompt": True,
    "stream": False,
    "return_tokens": True,
})
result = post("/completion", {
    "prompt": seq[:10000],
    "n_predict": 1,
    "ignore_eos": True,
    "seed": 1234,
    "id_slot": 0,
    "cache_prompt": True,
    "stream": False,
    "return_tokens": True,
})
assert result.get("tokens_cached") == 10000, result.get("tokens_cached")
saved = post("/slots/0?action=save", {"filename": "cache10k.bin"}, 3600)
assert saved.get("n_saved") == 10000, saved
json.dump(warm, open(os.path.join(slot_dir, "prepare-warm.json"), "w"), indent=2)
json.dump(result, open(os.path.join(slot_dir, "prepare-cache.json"), "w"), indent=2)
json.dump(saved, open(os.path.join(slot_dir, "prepare-cache-save.json"), "w"), indent=2)
print(json.dumps(saved, indent=2))
PY

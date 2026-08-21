#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
ROOT=${ROOT:-$(cd -- "$REPO_ROOT/.." && pwd)}
MODEL=${MODEL:-/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf}
SEQ_JSON=${SEQ_JSON:-$ROOT/results/cache100k-sharedstate-ab/seq.json}
SLOT_DIR=${SLOT_DIR:-$ROOT/results/cache100k-sharedstate-ab/slots}
SERVER_BIN=${SERVER_BIN:-$ROOT/build-upstream-current/bin/llama-server}
PORT=${PORT:-19519}

mkdir -p "$SLOT_DIR"

setsid "$SERVER_BIN" \
    --model "$MODEL" \
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
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --no-warmup \
    --perf \
    --slot-save-path "$SLOT_DIR" >"$SLOT_DIR/prepare-cache.server.log" 2>&1 &
pid=$!

cleanup() {
    kill -TERM -- -$pid 2>/dev/null || true
    sleep 1
    kill -KILL -- -$pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { tail -100 "$SLOT_DIR/prepare-cache.server.log"; exit 2; }
    sleep 1
done

PORT=$PORT SEQ_JSON=$SEQ_JSON SLOT_DIR=$SLOT_DIR python3 -u - <<'PY'
import json
import os
import urllib.request

port = os.environ["PORT"]
seq = json.load(open(os.environ["SEQ_JSON"]))
slot_dir = os.environ["SLOT_DIR"]
assert len(seq) >= 100000, len(seq)
base = f"http://127.0.0.1:{port}"

def post(path, obj, timeout=7200):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

result = post("/completion", {
    "prompt": seq[:100000],
    "n_predict": 1,
    "ignore_eos": True,
    "seed": 1234,
    "id_slot": 0,
    "cache_prompt": True,
    "stream": False,
    "return_tokens": True,
})
assert result.get("tokens_cached") == 100000, result.get("tokens_cached")
saved = post("/slots/0?action=save", {"filename": "cache100k.bin"}, 7200)
assert saved.get("n_saved") == 100000, saved
json.dump(result, open(os.path.join(slot_dir, "prepare-cache.json"), "w"), indent=2)
json.dump(saved, open(os.path.join(slot_dir, "prepare-cache-save.json"), "w"), indent=2)
print(json.dumps(saved, indent=2))
PY

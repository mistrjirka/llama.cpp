#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?mode}
BIN=${2:?llama-server binary}
ARGS_FILE=${3:?llama-fit-params args file}
HW=${4:?dual or v100}
MTP=${5:?0 or 1}
UBATCH=${6:?ubatch}
PORT=${7:?port}
shift 7

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
ROOT=${ROOT:-$(cd -- "$REPO_ROOT/.." && pwd)}
MODEL=${MODEL:-/models/GLM-5.2-UD-IQ2_XXS/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf}
SEQ_JSON=${SEQ_JSON:-$ROOT/results/resumed-100k-benchmark-20260821/glm52-10k/seq11k.json}
SLOT_DIR=${SLOT_DIR:-$ROOT/results/resumed-100k-benchmark-20260821/glm52-10k/shared-slot}
V100_UUID=${V100_UUID:-GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79}
OUT=${OUT:-$ROOT/results/glm52-10k}

mkdir -p "$OUT/$MODE"
mapfile -d '' -t FIT_ARGS < <(python3 - "$ARGS_FILE" <<'PY'
import shlex
import sys
for arg in shlex.split(open(sys.argv[1]).read()):
    print(arg, end="\0")
PY
)

EXTRA=("$@")
if [ "$MTP" = 1 ]; then
    EXTRA+=(--spec-type draft-mtp --spec-draft-n-max 2 --cache-type-k-draft q8_0 --cache-type-v-draft q8_0)
fi

CMD=("$BIN" --model "$MODEL" --host 127.0.0.1 --port "$PORT" --parallel 1 --fit off "${FIT_ARGS[@]}" --split-mode layer --flash-attn on --batch-size 4096 --ubatch-size "$UBATCH" --cache-type-k q8_0 --cache-type-v q8_0 --threads-batch 32 --no-warmup --perf --slot-save-path "$SLOT_DIR" "${EXTRA[@]}")
if [ "$HW" = v100 ]; then
    CUDA_VISIBLE_DEVICES="$V100_UUID" setsid "${CMD[@]}" >"$OUT/$MODE/server.log" 2>&1 &
else
    setsid "${CMD[@]}" >"$OUT/$MODE/server.log" 2>&1 &
fi
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
    kill -0 "$pid" 2>/dev/null || { tail -100 "$OUT/$MODE/server.log"; exit 2; }
    sleep 1
done

MODE=$MODE PORT=$PORT SEQ_JSON=$SEQ_JSON OUT=$OUT python3 -u - <<'PY'
import json
import os
import urllib.request

mode = os.environ["MODE"]
port = os.environ["PORT"]
out = os.environ["OUT"]
seq = json.load(open(os.environ["SEQ_JSON"]))
base = f"http://127.0.0.1:{port}"

def post(path, obj, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def run():
    restored = post("/slots/0?action=restore", {"filename": "cache10k.bin"}, 600)
    assert restored.get("n_restored") == 10000, restored
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
    assert result["timings"]["cache_n"] == 10000, result["timings"]
    assert result["timings"]["prompt_n"] == 1000, result["timings"]
    return result

warm = run()
json.dump(warm, open(os.path.join(out, mode, "warm.json"), "w"), indent=2)
result = run()
json.dump(result, open(os.path.join(out, mode, "result.json"), "w"), indent=2)
print(mode, json.dumps(result["timings"]), flush=True)
PY

#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/llama-mainline-cache
OUTROOT="$ROOT/profiling/completion-overlay-feasibility-2026-08-01/qwen-reactive-protected-oracle-repeats"
MODEL=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
PROMPT="$ROOT/profiling/qwen35-current-mlp-retest-2026-07-31/short-coding-prompt.txt"
TOKENS="$ROOT/profiling/long-coding-cache-2026-07-27/reference.tokens.txt"
LAYER_VALUES="$ROOT/profiling/completion-overlay-feasibility-2026-08-01/qwen-primitive-profile/summary.json"
SCHEDULE="$ROOT/profiling/completion-overlay-feasibility-2026-08-01/qwen-reactive-protected-oracle/completion-flex3.schedule"
GPU_INDEX=${GPU_INDEX:-1}
mkdir -p "$OUTROOT"

wait_for_gpu_idle() {
    local stable=0 deadline=$((SECONDS + 600))
    while ((SECONDS < deadline)); do
        local sample free util
        sample=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
        free=${sample%%,*}; util=${sample##*,}
        if [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] && ((free >= 30000 && util <= 5)); then
            stable=$((stable + 1))
            if ((stable >= 3)); then sleep 3; return 0; fi
        else
            stable=0
        fi
        sleep 2
    done
    return 1
}

common_env=(
    CUDA_VISIBLE_DEVICES=0
    GGML_CUDA_DISABLE_GRAPHS=1
    "GGML_COMPLETION_FORCE_TOKENS=$TOKENS"
    GGML_COMPLETION_BENCH_WARMUP_TOKENS=0
    GGML_COMPLETION_BENCH_MEASURE_TOKENS=512
    GGML_EXPERT_CACHE_MIB=auto
    GGML_EXPERT_CACHE_RESERVE_MIB=2560
    GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto
    GGML_MOE_DYNAMIC_THRESHOLD=2
    GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10
    GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=1
    GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=16
    GGML_MOE_DYNAMIC_MAX_UPLOAD_MIB_PER_TOKEN=64
    GGML_MOE_DYNAMIC_PREFILL_OBSERVE=1
    GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=18
    GGML_MOE_DYNAMIC_WARM_START_TOTAL=4096
    GGML_MOE_DYNAMIC_WARM_START_EXCLUDE_FLEX=1
    GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0
    GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL=0
    GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_PER_LAYER=0
    GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_TOTAL=0
    GGML_MOE_DYNAMIC_FIXED_TOPOLOGY=1
    GGML_MOE_DYNAMIC_FIXED_FULL_GPU_BYPASS=0
    GGML_MOE_DYNAMIC_ORACLE_FULL_GPU_BYPASS=0
    GGML_MOE_DYNAMIC_SEGMENTED_LRU=1
    GGML_MOE_DYNAMIC_PROTECTED_HITS=2
    GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1
    GGML_MOE_DYNAMIC_SKIP_COLD_BRANCH=1
    GGML_MOE_DYNAMIC_ASYNC_PROMOTION=1
    GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD=1
    GGML_MOE_DYNAMIC_VICTIM_POLICY=decayed_lfu
    GGML_MOE_DYNAMIC_PREDICTIVE_FLEX_SLOTS=3
    GGML_MOE_DYNAMIC_PREDICTIVE_SHARED_SLOTS=0
    GGML_MOE_DYNAMIC_PREDICTIVE_RELEASE_AFTER_TARGET=1
    GGML_MOE_DYNAMIC_PREDICTIVE_VICTIM_POLICY=unused_lfu
    GGML_MOE_DYNAMIC_GLOBAL_SCHEDULER=0
    GGML_MOE_DYNAMIC_SHARED_ADMISSION_BUDGET=0
    GGML_MOE_DYNAMIC_GLOBAL_TRANSFER_PENALTY_WEIGHT=0
    GGML_MOE_DYNAMIC_GLOBAL_PREDICTIVE_MIN_VALUE_US=0
    GGML_MOE_DYNAMIC_ONLINE_MLP_UPLOAD=1
    GGML_MOE_DYNAMIC_ONLINE_MLP_MAX_ADMISSIONS_PER_TOKEN=16
    GGML_MOE_DYNAMIC_ONLINE_MLP_MAX_UPLOAD_MIB_PER_TOKEN=64
    GGML_MOE_DYNAMIC_ONLINE_MLP_EXPECTED_LAYER_US=1125
    GGML_MOE_DYNAMIC_ONLINE_MLP_EXPECTED_UPLOAD_GIB_S=6.0
    GGML_MOE_DYNAMIC_ONLINE_MLP_UPLOAD_OVERHEAD_US=250
    GGML_MOE_DYNAMIC_ONLINE_MLP_DEADLINE_CHECK=1
    GGML_MOE_DYNAMIC_ONLINE_MLP_DEADLINE_MARGIN_US=250
)

run_one() {
    local arm=$1 trace_path=$2 schedule_path=$3
    local out="$OUTROOT/$arm"
    mkdir -p "$out"; rm -f "$out"/*
    wait_for_gpu_idle
    echo "### $(date -Is) arm=$arm"

    local extra_env=()
    if [[ -n "$trace_path" ]]; then
        extra_env+=("GGML_MOE_DYNAMIC_TRACE=$trace_path" GGML_MOE_DYNAMIC_TRACE_COMPACT=1 GGML_MOE_DYNAMIC_TRACE_FLUSH_EVERY=256)
    fi
    if [[ -n "$schedule_path" ]]; then
        extra_env+=("GGML_MOE_DYNAMIC_ORACLE_SCHEDULE=$schedule_path")
    fi

    local start_ns end_ns rc=0
    start_ns=$(date +%s%N)
    env -i HOME="$HOME" PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        "${common_env[@]}" "${extra_env[@]}" \
        timeout 1800s "$ROOT/build-v100/bin/llama-completion" \
            -m "$MODEL" -f "$PROMPT" -n 512 -c 4096 -b 512 -ub 128 -fa on -dev CUDA0 \
            -t 40 -tb 48 --poll 50 --fit off -ngl 99 --cpu-moe \
            -s 1 --temp 0 --ignore-eos --single-turn --no-conversation \
            --no-display-prompt --simple-io -v \
            >"$out/stdout.txt" 2>"$out/stderr.txt" || rc=$?
    end_ns=$(date +%s%N)

    python3 - "$arm" "$rc" "$start_ns" "$end_ns" "$out" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

arm, rc, start_ns, end_ns, out = sys.argv[1:]
out = Path(out)
text = (out / "stderr.txt").read_text(errors="replace")
data = (out / "stdout.txt").read_bytes()

def last(pattern):
    matches = re.findall(pattern, text, re.MULTILINE)
    return matches[-1] if matches else None

marker = last(r"benchmark decode measurement complete after (\d+) tokens; elapsed = ([0-9.]+) ms, ([0-9.]+) tokens per second")
result = {
    "arm": arm,
    "exit": int(rc),
    "wall_s": (int(end_ns) - int(start_ns)) / 1e9,
    "sha256": hashlib.sha256(data).hexdigest(),
    "tokens": int(marker[0]) if marker else None,
    "elapsed_s": float(marker[1]) / 1000.0 if marker else None,
    "tps": float(marker[2]) if marker else None,
    "cache": last(r"^(?:.*? )?(moe-dynamic-cache:.*)$"),
    "coverage": last(r"^(?:.*? )?(moe-dynamic-coverage:.*)$"),
    "worker": last(r"^(?:.*? )?(moe-promotion-worker:.*)$"),
    "oracle": last(r"^(?:.*? )?(moe-oracle-summary:.*)$"),
    "predictive": last(r"^(?:.*? )?(moe-predictive-usefulness-summary:.*)$"),
    "global": last(r"^(?:.*? )?(moe-global-scheduler-summary:.*)$"),
}
(out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if result["exit"] != 0 or result["tokens"] != 512 or result["tps"] is None:
    raise SystemExit(1)
PY
}

# Balanced profile-off order: A B B A B A.
run_one control-r1 "" ""
run_one oracle-r1 "" "$SCHEDULE"
run_one oracle-r2 "" "$SCHEDULE"
run_one control-r2 "" ""
run_one oracle-r3 "" "$SCHEDULE"
run_one control-r3 "" ""

python3 - "$OUTROOT" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(path.read_text()) for path in sorted(root.glob('*-r*/summary.json'))]

def metric(line, pattern):
    if not line:
        return None
    match = re.search(pattern, line)
    return float(match.group(1)) if match else None

def describe(values):
    return {
        'runs': values,
        'mean': statistics.fmean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values),
    }

arms = {}
for arm in ('control', 'oracle'):
    selected = sorted((row for row in rows if row['arm'].startswith(arm+'-')), key=lambda row: row['arm'])
    arms[arm] = {
        'throughput': describe([row['tps'] for row in selected]),
        'full_gpu_layers': describe([
            metric(row['coverage'], r'full-gpu-expert-layers=(\d+)/') for row in selected
        ]),
        'cpu_dependent_layers_per_token': describe([
            metric(row['coverage'], r'mean-cpu-dependent-layers=([0-9.]+)') for row in selected
        ]),
        'worker_mib': describe([
            metric(row['worker'], r'bytes=([0-9.]+) MiB') for row in selected
        ]),
        'rows': selected,
    }

hashes = {row['sha256'] for row in rows}
control_mean = arms['control']['throughput']['mean']
oracle_mean = arms['oracle']['throughput']['mean']
result = {
    'arms': arms,
    'hash_matches': len(hashes) == 1,
    'oracle_vs_control_percent_by_mean': (oracle_mean/control_mean - 1)*100,
    'oracle_vs_control_percent_by_median': (
        arms['oracle']['throughput']['median']/arms['control']['throughput']['median'] - 1
    )*100,
    'full_gpu_layer_gain_mean': (
        arms['oracle']['full_gpu_layers']['mean'] - arms['control']['full_gpu_layers']['mean']
    ),
    'cpu_dependent_layers_removed_per_token_mean': (
        arms['control']['cpu_dependent_layers_per_token']['mean'] -
        arms['oracle']['cpu_dependent_layers_per_token']['mean']
    ),
    'additional_worker_mib_mean': (
        arms['oracle']['worker_mib']['mean'] - arms['control']['worker_mib']['mean']
    ),
}
(root/'results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
if not result['hash_matches']:
    raise SystemExit('forced-token output hash mismatch')
PY

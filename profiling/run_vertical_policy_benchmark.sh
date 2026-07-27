#!/usr/bin/env bash
set -u

MODEL=${MODEL:-/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf}
PROMPT_FILE=${PROMPT_FILE:-profiling/coding-long-prompt.txt}
OUTDIR=${OUTDIR:-profiling/vertical-policy-benchmark}
CONTEXT=${CONTEXT:-65536}
WARMUP_TOKENS=${WARMUP_TOKENS:-256}
MEASURE_TOKENS=${MEASURE_TOKENS:-256}
CASE=${1:-baseline}
TRACE=${TRACE:-0}
TOTAL_TOKENS=$((WARMUP_TOKENS + MEASURE_TOKENS))

mkdir -p "$OUTDIR"
out="$OUTDIR/$CASE.out"
err="$OUTDIR/$CASE.err"
gpu="$OUTDIR/$CASE.gpu.tsv"
trace="$PWD/$OUTDIR/$CASE.jsonl"
summary="$OUTDIR/$CASE.summary.json"
rm -f "$out" "$err" "$gpu" "$trace" "$summary"

monitor_gpu() {
    while true; do
        printf '%s\t' "$(date +%s.%N)"
        nvidia-smi -i "${NVIDIA_SMI_GPU_INDEX:-1}" \
            --query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
            --format=csv,noheader,nounits | tr -d ' ' | tr ',' '\t'
        sleep 0.2
    done
}
monitor_gpu > "$gpu" 2>/dev/null &
monitor_pid=$!

common_env=(
    CUDA_VISIBLE_DEVICES=0
    "GGML_COMPLETION_BENCH_WARMUP_TOKENS=$WARMUP_TOKENS"
)
if [ "${CUDA_GRAPHS:-0}" != 1 ]; then
    common_env+=(GGML_CUDA_DISABLE_GRAPHS=1)
fi
policy_env=()
case "$CASE" in
    baseline)
        ;;
    legacy)
        policy_env=(
            GGML_EXPERT_CACHE_MIB=auto
            "GGML_EXPERT_CACHE_RESERVE_MIB=${CACHE_RESERVE_MIB:-2048}"
            GGML_EXPERT_CACHE_PROFILE=1
            GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto
            GGML_MOE_DYNAMIC_THRESHOLD=2
            GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=8
            GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=8
            GGML_MOE_DYNAMIC_PREFILL_OBSERVE=0
            GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=0
            GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0
            GGML_MOE_DYNAMIC_SEGMENTED_LRU=0
            GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1
        )
        ;;
    evolved|evolved-trace|evolved-noexec|evolved-gpu-only|evolved-gpu-map|evolved-gpu-map-rebuild)
        policy_env=(
            GGML_EXPERT_CACHE_MIB=auto
            "GGML_EXPERT_CACHE_RESERVE_MIB=${CACHE_RESERVE_MIB:-2048}"
            GGML_EXPERT_CACHE_PROFILE=1
            GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto
            GGML_MOE_DYNAMIC_THRESHOLD=2
            GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=8
            GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=32
            GGML_MOE_DYNAMIC_PREFILL_OBSERVE=1
            GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=16
            GGML_MOE_DYNAMIC_WARM_START_TOTAL=512
            GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=1
            GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL=16
            GGML_MOE_DYNAMIC_SEGMENTED_LRU=1
            GGML_MOE_DYNAMIC_PROTECTED_HITS=2
            GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1
        )
        if [ "$CASE" = evolved-noexec ]; then
            policy_env+=(GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=9)
        fi
        if [ "$CASE" = evolved-gpu-only ]; then
            policy_env+=(GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=1 GGML_MOE_DYNAMIC_FORCE_GPU_ONLY=1)
        fi
        if [ "$CASE" = evolved-gpu-map ]; then
            policy_env+=(
                GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=1
                GGML_MOE_DYNAMIC_FORCE_GPU_ONLY=1
                GGML_MOE_DYNAMIC_GPU_ROUTE_MAP=1
                GGML_MOE_DYNAMIC_GPU_MAP_WARM_START_PER_LAYER=16
                GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=0
                GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0
            )
        fi
        if [ "$CASE" = evolved-gpu-map-rebuild ]; then
            policy_env+=(
                GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=1
                GGML_COMPLETION_GPU_ROUTE_MAP_AFTER_WARMUP=1
            )
        fi
        if [ "$TRACE" = 1 ] || [ "$CASE" = evolved-trace ]; then
            policy_env+=("GGML_MOE_DYNAMIC_TRACE=$trace" GGML_MOE_DYNAMIC_TRACE_FLUSH_EVERY=1024)
        fi
        ;;
    *)
        echo "unknown case: $CASE" >&2
        kill "$monitor_pid" 2>/dev/null || true
        exit 2
        ;;
esac

start_ns=$(date +%s%N)
rc=0
env \
    -u GGML_EXPERT_CACHE_MIB -u GGML_EXPERT_CACHE_RESERVE_MIB -u GGML_EXPERT_CACHE_PROFILE \
    -u GGML_MOE_DYNAMIC_SPLIT_SLOTS -u GGML_MOE_DYNAMIC_THRESHOLD \
    -u GGML_MOE_DYNAMIC_MIN_HOT_ROUTES -u GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN \
    -u GGML_MOE_DYNAMIC_PREFILL_OBSERVE -u GGML_MOE_DYNAMIC_WARM_START_PER_LAYER \
    -u GGML_MOE_DYNAMIC_WARM_START_TOTAL -u GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER \
    -u GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL -u GGML_MOE_DYNAMIC_SEGMENTED_LRU \
    -u GGML_MOE_DYNAMIC_PROTECTED_HITS -u GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH \
    -u GGML_MOE_DYNAMIC_TRACE -u GGML_MOE_DYNAMIC_TRACE_FLUSH_EVERY \
    "${common_env[@]}" "${policy_env[@]}" \
    timeout 10800s build-v100/bin/llama-completion \
    -m "$MODEL" -f "$PROMPT_FILE" -n "$TOTAL_TOKENS" -c "$CONTEXT" \
    -b 2048 -ub 512 -fa on -dev CUDA0 -t 40 -tb 48 \
    -fit on -fitt 8192 -fitc 512 -s 1 --temp 0 \
    --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err" || rc=$?
end_ns=$(date +%s%N)

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

python3 - "$CASE" "$rc" "$start_ns" "$end_ns" "$err" "$out" "$gpu" "$summary" <<'PY'
import hashlib
import json
import re
import statistics
import sys

case, rc, start_ns, end_ns, err_path, out_path, gpu_path, summary_path = sys.argv[1:]
text = open(err_path, errors='replace').read()

def last(pattern, default=None):
    values = re.findall(pattern, text)
    return values[-1] if values else default

samples = []
for line in open(gpu_path, errors='replace'):
    fields = line.strip().split('\t')
    if len(fields) != 6:
        continue
    try:
        samples.append(tuple(map(float, fields)))
    except ValueError:
        pass

result = {
    'case': case,
    'exit': int(rc),
    'wall_s': (int(end_ns) - int(start_ns)) / 1e9,
    'sha256': hashlib.sha256(open(out_path, 'rb').read()).hexdigest(),
    'prompt_tokens': last(r'prompt eval time\s*=.*?/\s*(\d+) tokens'),
    'prompt_tps': last(r'prompt eval time\s*=.*?([0-9.]+) tokens per second'),
    'measured_runs': last(r'(?<!prompt )eval time\s*=.*?/\s*(\d+) runs'),
    'measured_tps': last(r'(?<!prompt )eval time\s*=.*?([0-9.]+) tokens per second'),
    'cache': last(r'(moe-dynamic-cache:.*)'),
    'coverage': last(r'(moe-dynamic-coverage:.*)'),
    'worker': last(r'(moe-promotion-worker: jobs=.*)'),
    'transfer': last(r'(expert-cache-transfer-profile:.*)'),
    'submit': last(r'(expert-cache-submit-profile:.*)'),
    'gpu_samples': len(samples),
    'gpu_util_mean': statistics.mean(row[3] for row in samples) if samples else None,
    'gpu_util_median': statistics.median(row[3] for row in samples) if samples else None,
    'gpu_below_25_pct': 100 * sum(row[3] < 25 for row in samples) / len(samples) if samples else None,
    'min_free_mib': min((row[2] for row in samples), default=None),
    'max_used_mib': max((row[1] for row in samples), default=None),
    'mean_power_w': statistics.mean(row[5] for row in samples) if samples else None,
}
open(summary_path, 'w').write(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY

exit "$rc"

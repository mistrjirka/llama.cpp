#!/usr/bin/env bash
set -u

MODEL=${MODEL:-/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf}
PROMPT_FILE=${PROMPT_FILE:-profiling/coding-long-prompt.txt}
OUTDIR=${OUTDIR:-profiling/long-coding-profile}
TOKENS=${TOKENS:-256}
CONTEXT=${CONTEXT:-65536}
CASE=${1:-baseline}
TRACE=${TRACE:-0}

mkdir -p "$OUTDIR"
out="$OUTDIR/$CASE.out"
err="$OUTDIR/$CASE.err"
time_log="$OUTDIR/$CASE.time"
gpu_log="$OUTDIR/$CASE.gpu.tsv"
trace_log="$PWD/$OUTDIR/$CASE.jsonl"
rm -f "$out" "$err" "$time_log" "$gpu_log" "$trace_log"

monitor_gpu() {
    while true; do
        printf '%s\t' "$(date +%s.%N)"
        nvidia-smi -i "${NVIDIA_SMI_GPU_INDEX:-1}" --query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem \
            --format=csv,noheader,nounits | tr -d ' ' | tr ',' '\t'
        sleep 0.2
    done
}
monitor_gpu > "$gpu_log" 2>/dev/null &
monitor_pid=$!

case_env=()
case "$CASE" in
    baseline)
        ;;
    dynamic-full8|dynamic-full8-trace)
        case_env=(
            GGML_EXPERT_CACHE_MIB=auto
            GGML_EXPERT_CACHE_RESERVE_MIB=2048
            GGML_EXPERT_CACHE_PROFILE=1
            GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto
            GGML_MOE_DYNAMIC_THRESHOLD=2
            GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=8
            GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=8
            GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1
        )
        if [ "$TRACE" = 1 ] || [ "$CASE" = dynamic-full8-trace ]; then
            case_env+=("GGML_MOE_DYNAMIC_TRACE=$trace_log")
        fi
        ;;
    dynamic-partial4)
        case_env=(
            GGML_EXPERT_CACHE_MIB=auto
            GGML_EXPERT_CACHE_RESERVE_MIB=2048
            GGML_EXPERT_CACHE_PROFILE=1
            GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto
            GGML_MOE_DYNAMIC_THRESHOLD=2
            GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=4
            GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=8
            GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1
        )
        ;;
    *)
        echo "unknown case: $CASE" >&2
        kill "$monitor_pid" 2>/dev/null || true
        exit 2
        ;;
esac

start_ns=$(date +%s%N)
rc=0
env -u GGML_EXPERT_CACHE_MIB -u GGML_EXPERT_CACHE_RESERVE_MIB \
    -u GGML_EXPERT_CACHE_PROFILE -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED \
    -u GGML_MOE_DYNAMIC_SPLIT_SLOTS -u GGML_MOE_DYNAMIC_SPLIT_LAYERS \
    -u GGML_MOE_DYNAMIC_THRESHOLD -u GGML_MOE_DYNAMIC_MIN_HOT_ROUTES \
    -u GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN -u GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH \
    -u GGML_MOE_DYNAMIC_PROFILE_COMPUTE -u GGML_MOE_DYNAMIC_TRACE \
    CUDA_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 "${case_env[@]}" \
    timeout 7200s build-v100/bin/llama-completion \
    -m "$MODEL" -f "$PROMPT_FILE" -n "$TOKENS" -c "$CONTEXT" \
    -b 2048 -ub 512 -fa on -dev CUDA0 -t 40 -tb 48 \
    -fit on -fitt 8192 -fitc 512 -s 1 --temp 0 \
    --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err" || rc=$?
end_ns=$(date +%s%N)

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

python3 - "$CASE" "$rc" "$start_ns" "$end_ns" "$err" "$out" "$gpu_log" <<'PY'
import hashlib
import json
import re
import statistics
import sys

case, rc, start_ns, end_ns, err_path, out_path, gpu_path = sys.argv[1:]
text = open(err_path, errors='replace').read()

def last(pattern, default=None):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else default

samples = []
for line in open(gpu_path, errors='replace'):
    fields = line.strip().split('\t')
    if len(fields) != 8:
        continue
    try:
        ts, used, free, gpu, mem, power, sm, memclk = map(float, fields)
    except ValueError:
        continue
    if used >= 1024:
        samples.append((ts, used, free, gpu, mem, power, sm, memclk))

summary = {
    'case': case,
    'exit': int(rc),
    'wall_s': (int(end_ns) - int(start_ns)) / 1e9,
    'sha256': hashlib.sha256(open(out_path, 'rb').read()).hexdigest(),
    'prompt_tokens': last(r'prompt eval time\s*=.*?/\s*(\d+) tokens'),
    'prompt_tps': last(r'prompt eval time\s*=.*?([0-9.]+) tokens per second'),
    'decode_runs': last(r'(?<!prompt )eval time\s*=.*?/\s*(\d+) runs'),
    'decode_tps': last(r'(?<!prompt )eval time\s*=.*?([0-9.]+) tokens per second'),
    'allocated_mib': last(r'expert-cache: entries=\d+ allocated=([0-9.]+) MiB'),
    'cache_summary': last(r'(moe-dynamic-cache:.*)'),
    'coverage_summary': last(r'(moe-dynamic-coverage:.*)'),
    'transfer_summary': last(r'(expert-cache-transfer-profile:.*)'),
    'submit_summary': last(r'(expert-cache-submit-profile:.*)'),
    'worker_summary': last(r'(moe-promotion-worker: jobs=.*)'),
    'profile_summary': last(r'(expert-cache-profile:.*)'),
    'gpu_samples': len(samples),
    'gpu_util_mean': statistics.mean(s[3] for s in samples) if samples else None,
    'gpu_util_median': statistics.median(s[3] for s in samples) if samples else None,
    'gpu_zero_pct': 100.0 * sum(s[3] == 0 for s in samples) / len(samples) if samples else None,
    'gpu_below_25_pct': 100.0 * sum(s[3] < 25 for s in samples) / len(samples) if samples else None,
    'min_free_mib': min((s[2] for s in samples), default=None),
    'max_used_mib': max((s[1] for s in samples), default=None),
    'mean_power_w': statistics.mean(s[5] for s in samples) if samples else None,
}
print(json.dumps(summary, indent=2))
open(out_path + '.summary.json', 'w').write(json.dumps(summary, indent=2) + '\n')
PY

exit "$rc"

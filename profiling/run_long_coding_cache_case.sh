#!/usr/bin/env bash
set -u

MODEL=${MODEL:-/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf}
PROMPT_FILE=${PROMPT_FILE:-profiling/coding-long-prompt.txt}
OUTDIR=${OUTDIR:-profiling/long-coding-cache-2026-07-27}
CASE=${1:-horizontal-reference}
REP=${2:-1}
TOTAL_TOKENS=${TOTAL_TOKENS:-2048}
WARMUP_TOKENS=${WARMUP_TOKENS:-512}
CONTEXT=${CONTEXT:-65536}
CACHE_RESERVE_MIB=${CACHE_RESERVE_MIB:-8192}
MIN_HOT_ROUTES=${MIN_HOT_ROUTES:-8}
FORCE_TOKENS=${FORCE_TOKENS:-$PWD/$OUTDIR/reference.tokens.txt}
MEASURE_TOKENS=$((TOTAL_TOKENS - WARMUP_TOKENS))

if [ "$MEASURE_TOKENS" -le 0 ]; then
    echo "TOTAL_TOKENS must exceed WARMUP_TOKENS" >&2
    exit 2
fi

mkdir -p "$OUTDIR"
stem="$CASE-r$REP"
out="$OUTDIR/$stem.out"
err="$OUTDIR/$stem.err"
gpu="$OUTDIR/$stem.gpu.tsv"
summary="$OUTDIR/$stem.summary.json"
trace="$PWD/$OUTDIR/$stem.jsonl"
rm -f "$out" "$err" "$gpu" "$summary" "$trace"

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
    GGML_CUDA_DISABLE_GRAPHS=1
    "GGML_COMPLETION_BENCH_WARMUP_TOKENS=$WARMUP_TOKENS"
)
case_env=()
model_args=()
case "$CASE" in
    horizontal-reference)
        model_args=(-fit on -fitt 8192 -fitc 512)
        case_env=("GGML_COMPLETION_TOKEN_DUMP=$PWD/$OUTDIR/reference.tokens.txt")
        ;;
    horizontal-replay)
        if [ ! -s "$FORCE_TOKENS" ]; then
            echo "missing forced token file: $FORCE_TOKENS" >&2
            kill "$monitor_pid" 2>/dev/null || true
            exit 2
        fi
        model_args=(-fit on -fitt 8192 -fitc 512)
        case_env=("GGML_COMPLETION_FORCE_TOKENS=$FORCE_TOKENS")
        ;;
    dynamic|dynamic-trace|dynamic-unprofiled)
        if [ ! -s "$FORCE_TOKENS" ]; then
            echo "missing forced token file: $FORCE_TOKENS" >&2
            kill "$monitor_pid" 2>/dev/null || true
            exit 2
        fi
        model_args=(--fit off -ngl 99 --cpu-moe)
        case_env=(
            "GGML_COMPLETION_FORCE_TOKENS=$FORCE_TOKENS"
            GGML_EXPERT_CACHE_MIB=auto
            "GGML_EXPERT_CACHE_RESERVE_MIB=$CACHE_RESERVE_MIB"
            GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto
            GGML_MOE_DYNAMIC_THRESHOLD=2
            "GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=$MIN_HOT_ROUTES"
            GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=32
            GGML_MOE_DYNAMIC_PREFILL_OBSERVE=1
            GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=16
            GGML_MOE_DYNAMIC_WARM_START_TOTAL=512
            GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0
            GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_PER_LAYER=1
            GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_TOTAL=16
            GGML_MOE_DYNAMIC_CROSS_LAYER_MIN_OBSERVATIONS=8
            GGML_MOE_DYNAMIC_SEGMENTED_LRU=1
            GGML_MOE_DYNAMIC_PROTECTED_HITS=2
            GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1
            GGML_MOE_DYNAMIC_SKIP_COLD_BRANCH=1
            GGML_MOE_DYNAMIC_ASYNC_PROMOTION=1
            GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD=1
        )
        if [ "$CASE" != dynamic-unprofiled ]; then
            case_env+=(GGML_EXPERT_CACHE_PROFILE=1)
        fi
        if [ "$CASE" = dynamic-trace ]; then
            case_env+=(
                "GGML_MOE_DYNAMIC_TRACE=$trace"
                GGML_MOE_DYNAMIC_TRACE_COMPACT=1
                GGML_MOE_DYNAMIC_TRACE_FLUSH_EVERY=1024
            )
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
    -u GGML_COMPLETION_TOKEN_DUMP -u GGML_COMPLETION_FORCE_TOKENS -u GGML_COMPLETION_LOGITS_DUMP \
    -u GGML_EXPERT_CACHE_MIB -u GGML_EXPERT_CACHE_RESERVE_MIB -u GGML_EXPERT_CACHE_PROFILE \
    -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED \
    -u GGML_MOE_DYNAMIC_SPLIT_SLOTS -u GGML_MOE_DYNAMIC_SPLIT_LAYERS \
    -u GGML_MOE_DYNAMIC_THRESHOLD -u GGML_MOE_DYNAMIC_MIN_HOT_ROUTES \
    -u GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN -u GGML_MOE_DYNAMIC_PREFILL_OBSERVE \
    -u GGML_MOE_DYNAMIC_WARM_START_PER_LAYER -u GGML_MOE_DYNAMIC_WARM_START_TOTAL \
    -u GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER -u GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL \
    -u GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_PER_LAYER -u GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_TOTAL \
    -u GGML_MOE_DYNAMIC_CROSS_LAYER_MIN_OBSERVATIONS -u GGML_MOE_DYNAMIC_CROSS_LAYER_MIN_SCORE \
    -u GGML_MOE_DYNAMIC_CROSS_LAYER_DECAY_INTERVAL -u GGML_MOE_DYNAMIC_CROSS_LAYER_INTERACTION \
    -u GGML_MOE_DYNAMIC_SEGMENTED_LRU -u GGML_MOE_DYNAMIC_PROTECTED_HITS \
    -u GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH -u GGML_MOE_DYNAMIC_SKIP_COLD_BRANCH \
    -u GGML_MOE_DYNAMIC_ASYNC_PROMOTION -u GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD \
    -u GGML_MOE_DYNAMIC_VERIFY_PROMOTIONS_PER_SHAPE \
    -u GGML_MOE_DYNAMIC_TRACE -u GGML_MOE_DYNAMIC_TRACE_COMPACT -u GGML_MOE_DYNAMIC_TRACE_FLUSH_EVERY \
    -u GGML_MOE_DYNAMIC_STATIC_MAP -u GGML_MOE_DYNAMIC_FORCE_GPU_ONLY \
    -u GGML_MOE_DYNAMIC_EXACT_CPU_FALLBACK -u GGML_MOE_SKELETON_ONLY \
    "${common_env[@]}" "${case_env[@]}" \
    timeout 21600s build-v100/bin/llama-completion \
        -m "$MODEL" -f "$PROMPT_FILE" -n "$TOTAL_TOKENS" -c "$CONTEXT" \
        -b 2048 -ub 512 -fa on -dev CUDA0 -t 40 -tb 48 \
        "${model_args[@]}" --seed 1 --temp 0 --ignore-eos \
        --single-turn --no-conversation --simple-io --no-display-prompt -v \
        > "$out" 2> "$err" || rc=$?
end_ns=$(date +%s%N)

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

python3 - "$CASE" "$REP" "$rc" "$start_ns" "$end_ns" "$err" "$out" "$gpu" "$summary" "$WARMUP_TOKENS" "$MEASURE_TOKENS" "$TOTAL_TOKENS" "$FORCE_TOKENS" <<'PY'
import hashlib, json, re, statistics, sys
case, rep, rc, start_ns, end_ns, err_path, out_path, gpu_path, summary_path, warmup, measured, total, force_path = sys.argv[1:]
text = open(err_path, errors='replace').read()

def last(pattern, default=None):
    values = re.findall(pattern, text)
    return values[-1] if values else default

def integer(pattern, default=0):
    value = last(pattern)
    return int(value) if value is not None else default

samples=[]
for line in open(gpu_path, errors='replace'):
    fields=line.strip().split('\t')
    if len(fields) != 6:
        continue
    try:
        samples.append(tuple(map(float, fields)))
    except ValueError:
        pass

total_ms = last(r'total time\s*=\s*([0-9.]+) ms')
result = {
    'case': case,
    'rep': int(rep),
    'exit': int(rc),
    'wall_s': (int(end_ns)-int(start_ns))/1e9,
    'sha256': hashlib.sha256(open(out_path,'rb').read()).hexdigest(),
    'warmup_tokens': int(warmup),
    'measured_tokens': int(measured),
    'requested_tokens': int(total),
    'forced_token_lines': sum(1 for _ in open(force_path)) if case != 'horizontal-reference' else None,
    'measured_time_ms': float(total_ms) if total_ms else None,
    'measured_tps': int(measured)*1000.0/float(total_ms) if total_ms else None,
    'prompt_tokens': integer(r'prompt eval time\s*=.*?/\s*(\d+) tokens'),
    'prompt_tps': last(r'prompt eval time\s*=.*?([0-9.]+) tokens per second'),
    'gpu_samples': len(samples),
    'gpu_util_mean': statistics.fmean(x[3] for x in samples) if samples else None,
    'gpu_util_median': statistics.median(x[3] for x in samples) if samples else None,
    'min_free_mib': min((x[2] for x in samples), default=None),
    'max_used_mib': max((x[1] for x in samples), default=None),
    'mean_power_w': statistics.fmean(x[5] for x in samples) if samples else None,
    'graph_splits': last(r'sched_reserve: graph splits\s*=\s*([^\n]+)'),
    'model_buffers': re.findall(r'load_tensors:\s+(CPU_Mapped|CUDA_Host|CUDA0) model buffer size =\s+([0-9.]+) MiB', text),
    'worker': last(r'(moe-promotion-worker: jobs=.*)'),
    'cache': last(r'(moe-dynamic-cache:.*)'),
    'coverage': last(r'(moe-dynamic-coverage:.*)'),
    'cache_profile': last(r'(expert-cache-profile:.*)'),
    'transfer_profile': last(r'(expert-cache-transfer-profile:.*)'),
    'submit_profile': last(r'(expert-cache-submit-profile:.*)'),
    'trace_bytes': None,
}
trace_path = out_path.rsplit('.', 1)[0] + '.jsonl'
try:
    import os
    result['trace_bytes'] = os.path.getsize(trace_path)
except OSError:
    pass
assertions=[]
def check(name, ok, detail): assertions.append({'name':name,'ok':bool(ok),'detail':detail})
check('process_exit', int(rc)==0, rc)
check('warmup_marker', f'benchmark decode warmup complete after {warmup} tokens' in text, warmup)
check('measured_total_present', total_ms is not None, total_ms)
if case == 'horizontal-reference':
    reference_path = out_path.rsplit('/', 1)[0] + '/reference.tokens.txt'
    reference_lines = sum(1 for _ in open(reference_path))
    result['reference_token_lines'] = reference_lines
    check('reference_token_count', reference_lines == int(total), reference_lines)
else:
    check('forced_token_count', result['forced_token_lines'] >= int(total), result['forced_token_lines'])
if case.startswith('dynamic'):
    check('worker_summary_present', result['worker'] is not None, result['worker'])
    check('coverage_present', result['coverage'] is not None, result['coverage'])
result['assertions']=assertions
result['assertions_passed']=all(item['ok'] for item in assertions)
with open(summary_path,'w') as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
if not result['assertions_passed']:
    raise SystemExit(3)
PY
summary_rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$summary_rc"

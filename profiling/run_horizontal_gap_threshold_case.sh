#!/usr/bin/env bash
set -u

MODEL=${MODEL:-/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf}
PROMPT_FILE=${PROMPT_FILE:-profiling/real-chat-prompt.txt}
FORCE_TOKENS=${FORCE_TOKENS:-$PWD/profiling/dynamic-real-chat-ab-2026-07-26/reference.tokens.txt}
OUTDIR=${OUTDIR:-profiling/horizontal-gap-analysis-2026-07-27/threshold-sweep}
MIN_HOT_ROUTES=${1:-8}
REP=${2:-1}
PROFILE=${PROFILE:-0}
CACHE_RESERVE_MIB=${CACHE_RESERVE_MIB:-8192}
WARMUP_TOKENS=${WARMUP_TOKENS:-64}
MEASURE_TOKENS=${MEASURE_TOKENS:-256}
TOTAL_TOKENS=$((WARMUP_TOKENS + MEASURE_TOKENS))
profile_env=()
if [ "$PROFILE" != 0 ]; then
    profile_env=(GGML_EXPERT_CACHE_PROFILE=1)
fi

mkdir -p "$OUTDIR"
stem="hot${MIN_HOT_ROUTES}-profile${PROFILE}-reserve${CACHE_RESERVE_MIB}-r${REP}"
out="$OUTDIR/$stem.out"
err="$OUTDIR/$stem.err"
summary="$OUTDIR/$stem.summary.json"
rm -f "$out" "$err" "$summary"

start_ns=$(date +%s%N)
rc=0
env \
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
    -u GGML_MOE_DYNAMIC_VERIFY_PROMOTIONS_PER_SHAPE -u GGML_MOE_DYNAMIC_TRACE \
    -u GGML_MOE_DYNAMIC_STATIC_MAP -u GGML_MOE_DYNAMIC_FORCE_GPU_ONLY \
    -u GGML_MOE_DYNAMIC_EXACT_CPU_FALLBACK -u GGML_MOE_SKELETON_ONLY \
    CUDA_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 \
    GGML_COMPLETION_BENCH_WARMUP_TOKENS="$WARMUP_TOKENS" \
    GGML_COMPLETION_FORCE_TOKENS="$FORCE_TOKENS" \
    GGML_EXPERT_CACHE_MIB=auto "GGML_EXPERT_CACHE_RESERVE_MIB=$CACHE_RESERVE_MIB" \
    "${profile_env[@]}" GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto \
    GGML_MOE_DYNAMIC_THRESHOLD=2 GGML_MOE_DYNAMIC_MIN_HOT_ROUTES="$MIN_HOT_ROUTES" \
    GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=32 GGML_MOE_DYNAMIC_PREFILL_OBSERVE=1 \
    GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=16 GGML_MOE_DYNAMIC_WARM_START_TOTAL=512 \
    GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0 \
    GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_PER_LAYER=1 \
    GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_TOTAL=16 \
    GGML_MOE_DYNAMIC_CROSS_LAYER_MIN_OBSERVATIONS=8 \
    GGML_MOE_DYNAMIC_SEGMENTED_LRU=1 GGML_MOE_DYNAMIC_PROTECTED_HITS=2 \
    GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1 GGML_MOE_DYNAMIC_SKIP_COLD_BRANCH=1 \
    GGML_MOE_DYNAMIC_ASYNC_PROMOTION=1 GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD=1 \
    timeout 10800s build-v100/bin/llama-completion \
        -m "$MODEL" -f "$PROMPT_FILE" -n "$TOTAL_TOKENS" -c 8192 \
        -b 2048 -ub 512 -fa on -dev CUDA0 -t 40 -tb 48 \
        --fit off -ngl 99 --cpu-moe --seed 1 --temp 0 --conversation --single-turn \
        --simple-io --no-display-prompt \
        -sys 'You are a senior systems engineer. Give detailed, technically rigorous, practical answers.' \
        -v > "$out" 2> "$err" || rc=$?
end_ns=$(date +%s%N)

python3 - "$MIN_HOT_ROUTES" "$PROFILE" "$CACHE_RESERVE_MIB" "$REP" "$rc" "$start_ns" "$end_ns" "$err" "$out" "$summary" "$WARMUP_TOKENS" "$MEASURE_TOKENS" <<'PY'
import hashlib, json, re, sys
threshold, profile, reserve, rep, rc, start_ns, end_ns, err_path, out_path, summary_path, warmup, measured = sys.argv[1:]
text = open(err_path, errors='replace').read()

def last(pattern, default=None):
    values = re.findall(pattern, text)
    return values[-1] if values else default

total_ms = last(r'total time\s*=\s*([0-9.]+) ms')
result = {
    'min_hot_routes': int(threshold),
    'profile': int(profile),
    'reserve_mib': int(reserve),
    'rep': int(rep),
    'exit': int(rc),
    'wall_s': (int(end_ns) - int(start_ns)) / 1e9,
    'sha256': hashlib.sha256(open(out_path, 'rb').read()).hexdigest(),
    'warmup_tokens': int(warmup),
    'measured_tokens': int(measured),
    'measured_time_ms': float(total_ms) if total_ms else None,
    'measured_tps': int(measured) * 1000.0 / float(total_ms) if total_ms else None,
    'graph_splits': last(r'sched_reserve: graph splits\s*=\s*([^\n]+)'),
    'worker': last(r'(moe-promotion-worker: jobs=.*)'),
    'cache': last(r'(moe-dynamic-cache:.*)'),
    'coverage': last(r'(moe-dynamic-coverage:.*)'),
    'cache_profile': last(r'(expert-cache-profile:.*)'),
    'transfer_profile': last(r'(expert-cache-transfer-profile:.*)'),
    'submit_profile': last(r'(expert-cache-submit-profile:.*)'),
}
assertions = [
    {'name': 'process_exit', 'ok': int(rc) == 0, 'detail': int(rc)},
    {'name': 'warmup_marker', 'ok': f'benchmark decode warmup complete after {warmup} tokens' in text, 'detail': int(warmup)},
    {'name': 'measured_total_present', 'ok': total_ms is not None, 'detail': total_ms},
    {'name': 'coverage_present', 'ok': result['coverage'] is not None, 'detail': result['coverage']},
]
result['assertions'] = assertions
result['assertions_passed'] = all(item['ok'] for item in assertions)
with open(summary_path, 'w') as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
if not result['assertions_passed']:
    raise SystemExit(3)
PY
summary_rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$summary_rc"

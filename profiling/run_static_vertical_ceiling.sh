#!/usr/bin/env bash
set -u

MODEL=${MODEL:-/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf}
PROMPT_FILE=${PROMPT_FILE:-profiling/coding-sanity-prompt.txt}
STATIC_MAP=${STATIC_MAP:-$PWD/profiling/static-vertical-ceiling-2026-07-26/static-frequency-map.txt}
OUTDIR=${OUTDIR:-profiling/static-vertical-ceiling-2026-07-26/matrix}
CASE=${1:-baseline}
REP=${2:-1}
CONTEXT=${CONTEXT:-4096}
WARMUP_TOKENS=${WARMUP_TOKENS:-64}
MEASURE_TOKENS=${MEASURE_TOKENS:-256}
CACHE_RESERVE_MIB=${CACHE_RESERVE_MIB:-8192}
TOTAL_TOKENS=$((WARMUP_TOKENS + MEASURE_TOKENS))

mkdir -p "$OUTDIR"
stem="$CASE-r$REP"
out="$OUTDIR/$stem.out"
err="$OUTDIR/$stem.err"
gpu="$OUTDIR/$stem.gpu.tsv"
summary="$OUTDIR/$stem.summary.json"
rm -f "$out" "$err" "$gpu" "$summary"

monitor_gpu() {
    while true; do
        printf '%s\t' "$(date +%s.%N)"
        nvidia-smi -i "${NVIDIA_SMI_GPU_INDEX:-1}" --query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
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
    GGML_MOE_DYNAMIC_TRACE_GRAPH_BUILD=1
)
case_env=()
model_args=()
case "$CASE" in
    baseline)
        model_args=(-fit on -fitt 8192 -fitc 512)
        ;;
    vertical)
        model_args=(--fit off -ngl 99 --cpu-moe)
        case_env=(
            GGML_EXPERT_CACHE_MIB=auto
            "GGML_EXPERT_CACHE_RESERVE_MIB=$CACHE_RESERVE_MIB"
            GGML_EXPERT_CACHE_PROFILE=1
            GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto
            GGML_MOE_DYNAMIC_MIN_COMPLETE_COVERAGE=0
            GGML_MOE_DYNAMIC_FORCE_GPU_ONLY=1
            GGML_MOE_DYNAMIC_GPU_ROUTE_MAP=1
            "GGML_MOE_DYNAMIC_STATIC_MAP=$STATIC_MAP"
            GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=0
            GGML_MOE_DYNAMIC_PREFILL_OBSERVE=0
            GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=0
            GGML_MOE_DYNAMIC_WARM_START_TOTAL=0
            GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0
            GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL=0
            GGML_MOE_DYNAMIC_ASYNC_PROMOTION=1
        )
        ;;
    skeleton)
        model_args=(--fit off -ngl 99 --cpu-moe)
        case_env=(GGML_MOE_SKELETON_ONLY=1)
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
    -u GGML_MOE_DYNAMIC_SPLIT_SLOTS -u GGML_MOE_DYNAMIC_FORCE_GPU_ONLY \
    -u GGML_MOE_DYNAMIC_GPU_ROUTE_MAP -u GGML_MOE_DYNAMIC_STATIC_MAP \
    -u GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN -u GGML_MOE_DYNAMIC_PREFILL_OBSERVE \
    -u GGML_MOE_DYNAMIC_WARM_START_PER_LAYER -u GGML_MOE_DYNAMIC_WARM_START_TOTAL \
    -u GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER -u GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL \
    -u GGML_MOE_SKELETON_ONLY \
    "${common_env[@]}" "${case_env[@]}" \
    timeout 10800s build-v100/bin/llama-completion \
        -m "$MODEL" -f "$PROMPT_FILE" -n "$TOTAL_TOKENS" -c "$CONTEXT" \
        -b 2048 -ub 512 -fa on -dev CUDA0 -t 40 -tb 48 \
        "${model_args[@]}" -s 1 --temp 0 \
        --single-turn --no-conversation --no-display-prompt --simple-io -v \
        > "$out" 2> "$err" || rc=$?
end_ns=$(date +%s%N)

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

python3 - "$CASE" "$REP" "$rc" "$start_ns" "$end_ns" "$err" "$out" "$gpu" "$summary" "$MEASURE_TOKENS" <<'PY'
import hashlib, json, re, statistics, sys
case, rep, rc, start_ns, end_ns, err_path, out_path, gpu_path, summary_path, expected_runs = sys.argv[1:]
text = open(err_path, errors='replace').read()

def last(pattern, default=None):
    vals = re.findall(pattern, text)
    return vals[-1] if vals else default

def integer(pattern, default=0):
    v = last(pattern)
    return int(v) if v is not None else default

samples=[]
for line in open(gpu_path, errors='replace'):
    fields=line.strip().split('\t')
    if len(fields)!=6: continue
    try: samples.append(tuple(map(float, fields)))
    except ValueError: pass

transfer = last(r'(expert-cache-transfer-profile:.*)')
profile = last(r'(expert-cache-profile:.*)')
cache = last(r'(moe-dynamic-cache:.*)')
alloc = last(r'expert-cache: entries=(\d+) allocated=([0-9.]+) MiB')
measured_runs = integer(r'(?<!prompt )eval time\s*=.*?/\s*(\d+) runs')
result={
    'case':case, 'rep':int(rep), 'exit':int(rc),
    'wall_s':(int(end_ns)-int(start_ns))/1e9,
    'sha256':hashlib.sha256(open(out_path,'rb').read()).hexdigest(),
    'prompt_tokens':integer(r'prompt eval time\s*=.*?/\s*(\d+) tokens'),
    'prompt_tps':last(r'prompt eval time\s*=.*?([0-9.]+) tokens per second'),
    'measured_runs':measured_runs,
    'measured_tps':last(r'(?<!prompt )eval time\s*=.*?([0-9.]+) tokens per second'),
    'graph_nodes':last(r'sched_reserve: graph nodes\s*=\s*(\d+).*?,\s*(\d+)'),
    'graph_splits':last(r'sched_reserve: graph splits\s*=\s*(\d+).*?,\s*(\d+)'),
    'cache':cache, 'profile':profile, 'transfer':transfer,
    'cache_entries':int(alloc[0]) if alloc else None,
    'cache_allocated_mib':float(alloc[1]) if alloc else None,
    'static_map_count':len(re.findall(r'moe-static-frozen-map:', text)),
    'decode_static_assertions':len(re.findall(r'tokens=1 feature=1 active=1 split-requested=1 common-supported=1 split-supported=1 force-gpu-only=1 gpu-route-map=1 static-map=1', text)),
    'decode_skeleton_assertions':len(re.findall(r'moe-skeleton-only-build: layer=\d+ tokens=1', text)),
    'cuda_oom':bool(re.search(r'cudaMalloc failed|out of memory', text, re.I)),
    'gpu_samples':len(samples),
    'gpu_util_mean':statistics.mean(x[3] for x in samples) if samples else None,
    'gpu_util_median':statistics.median(x[3] for x in samples) if samples else None,
    'min_free_mib':min((x[2] for x in samples), default=None),
    'max_used_mib':max((x[1] for x in samples), default=None),
    'mean_power_w':statistics.mean(x[5] for x in samples) if samples else None,
}
assertions=[]
def check(name, ok, detail): assertions.append({'name':name,'ok':bool(ok),'detail':detail})
check('process_exit', int(rc)==0, rc)
check('measured_token_count', measured_runs==int(expected_runs), measured_runs)
if case=='vertical':
    ready=integer(r'moe-dynamic-cache:.*?ready-slots=(\d+)')
    copying=integer(r'moe-dynamic-cache:.*?copying-slots=(\d+)')
    dynamic_groups=integer(r'expert-cache-profile:.*?dynamic-groups=(\d+)')
    dynamic_bytes=last(r'expert-cache-profile:.*?dynamic-bytes=([0-9.]+) MiB')
    cold_calls=integer(r'expert-cache-profile:.*?cold-compute-calls=(\d+)')
    decision_calls=integer(r'expert-cache-profile:.*?decision-sync-calls=(\d+)')
    check('all_cache_components_attached', result['cache_entries']==144, result['cache_entries'])
    check('all_layers_static_mapped', result['static_map_count']==48, result['static_map_count'])
    check('decode_graph_static_route_map', result['decode_static_assertions']>=48, result['decode_static_assertions'])
    check('ready_slots_nonzero', ready>0, ready)
    check('copying_slots_zero', copying==0, copying)
    check('measured_dynamic_copy_groups_zero', dynamic_groups==0, dynamic_groups)
    check('measured_dynamic_copy_bytes_zero', dynamic_bytes is not None and float(dynamic_bytes)==0.0, dynamic_bytes)
    check('cpu_cold_compute_zero', cold_calls==0, cold_calls)
    check('cpu_route_decision_sync_zero', decision_calls==0, decision_calls)
    check('no_cuda_oom', not result['cuda_oom'], result['cuda_oom'])
if case=='skeleton':
    check('decode_graph_skips_moe_ffn', result['decode_skeleton_assertions']>=48, result['decode_skeleton_assertions'])
result['assertions']=assertions
result['assertions_passed']=all(x['ok'] for x in assertions)
with open(summary_path,'w') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
if not result['assertions_passed']:
    sys.exit(3)
PY
summary_rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$summary_rc"

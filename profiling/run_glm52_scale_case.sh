#!/usr/bin/env bash
set -u

CASE=${1:-horizontal}
REP=${2:-1}
MODEL=${MODEL:-/models/GLM-5.2-UD-IQ2_XXS/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf}
PROMPT_FILE=${PROMPT_FILE:-profiling/glm52-scale-2026-07-27/prompt.txt}
OUTDIR=${OUTDIR:-profiling/glm52-scale-2026-07-27}
CONTEXT=${CONTEXT:-2048}
WARMUP_TOKENS=${WARMUP_TOKENS:-16}
MEASURE_TOKENS=${MEASURE_TOKENS:-32}
TOTAL_TOKENS=$((WARMUP_TOKENS + MEASURE_TOKENS))
# Empirically optimal on the 32 GiB V100 for this GLM workload: 3 GiB leaves
# ~1.36 GiB measured headroom while fitting one more expert in most layers.
CACHE_RESERVE_MIB=${CACHE_RESERVE_MIB:-3072}
VERTICAL_FIT_TARGET_MIB=${VERTICAL_FIT_TARGET_MIB:-16384}
MIN_HOT_ROUTES=${MIN_HOT_ROUTES:-4}
THREADS=${THREADS:-40}
BATCH_THREADS=${BATCH_THREADS:-48}
POLL=${POLL:-50}
CPU_RANGE=${CPU_RANGE:-}
CPU_STRICT=${CPU_STRICT:-0}
BATCH_CPU_RANGE=${BATCH_CPU_RANGE:-}
BATCH_CPU_STRICT=${BATCH_CPU_STRICT:-$CPU_STRICT}
TRACE_GRAPH_BUILD=${TRACE_GRAPH_BUILD:-1}
CUDA_GRAPHS=${CUDA_GRAPHS:-0}
VERBOSE=${VERBOSE:-1}
PROFILE=${PROFILE:-0}
TRACE_PATH=${TRACE_PATH:-}
TRACE_COMPACT=${TRACE_COMPACT:-0}
CROSS_LAYER_PREFETCH_PER_LAYER=${CROSS_LAYER_PREFETCH_PER_LAYER:-0}
CROSS_LAYER_PREFETCH_TOTAL=${CROSS_LAYER_PREFETCH_TOTAL:-0}
CROSS_LAYER_MIN_OBSERVATIONS=${CROSS_LAYER_MIN_OBSERVATIONS:-1}
CROSS_LAYER_GLOBAL_RANK=${CROSS_LAYER_GLOBAL_RANK:-0}
CROSS_LAYER_GLOBAL_RANK_OVERSUBSCRIBE=${CROSS_LAYER_GLOBAL_RANK_OVERSUBSCRIBE:-4.0}
CROSS_LAYER_DISTANCE2_WEIGHT=${CROSS_LAYER_DISTANCE2_WEIGHT:-1.0}
DYNAMIC_FIXED_TOPOLOGY=${DYNAMIC_FIXED_TOPOLOGY:-0}
DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=${DYNAMIC_MAX_ADMISSIONS_PER_TOKEN:-16}
DYNAMIC_WARM_START_PER_LAYER=${DYNAMIC_WARM_START_PER_LAYER:-8}
DYNAMIC_WARM_START_TOTAL=${DYNAMIC_WARM_START_TOTAL:-608}
DYNAMIC_ASYNC_PROMOTION=${DYNAMIC_ASYNC_PROMOTION:-1}
URGENT_PREDICT_UPLOAD=${URGENT_PREDICT_UPLOAD:-1}
STATIC_MAP=${STATIC_MAP:-$PWD/$OUTDIR/static-frequency-map.txt}
FORCE_TOKENS=${FORCE_TOKENS:-$PWD/$OUTDIR/reference.tokens.txt}

mkdir -p "$OUTDIR"
stem="$CASE-r$REP"
out="$OUTDIR/$stem.out"
err="$OUTDIR/$stem.err"
gpu="$OUTDIR/$stem.gpu.tsv"
summary="$OUTDIR/$stem.summary.json"
rm -f "$out" "$err" "$gpu" "$summary"
if [ "$CASE" = horizontal ]; then
    rm -f "$FORCE_TOKENS"
fi

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
    "GGML_COMPLETION_BENCH_MEASURE_TOKENS=$MEASURE_TOKENS"
)
if [ "$CUDA_GRAPHS" = 0 ]; then
    common_env+=(GGML_CUDA_DISABLE_GRAPHS=1)
fi
if [ "$PROFILE" != 0 ]; then
    common_env+=("GGML_EXPERT_CACHE_PROFILE=$PROFILE")
fi
case_env=()
model_args=()
case "$CASE" in
    horizontal)
        common_env+=("GGML_COMPLETION_TOKEN_DUMP=$FORCE_TOKENS")
        model_args=(-fit on -fitt 4096 -fitc "$CONTEXT")
        ;;
    vertical|vertical-nocache|vertical-free)
        if [ "$CASE" != vertical-free ]; then
            if [ ! -s "$FORCE_TOKENS" ]; then
                echo "missing forced token file: $FORCE_TOKENS" >&2
                kill "$monitor_pid" 2>/dev/null || true
                exit 2
            fi
            common_env+=("GGML_COMPLETION_FORCE_TOKENS=$FORCE_TOKENS")
        fi
        model_args=(-fit on -fitt "$VERTICAL_FIT_TARGET_MIB" -fitc "$CONTEXT" --cpu-moe)
        if [ "$CASE" = vertical ] || [ "$CASE" = vertical-free ]; then
            case_env=(
                GGML_EXPERT_CACHE_MIB=auto
                "GGML_EXPERT_CACHE_RESERVE_MIB=$CACHE_RESERVE_MIB"
                GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto
                GGML_MOE_DYNAMIC_THRESHOLD=2
                "GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=$MIN_HOT_ROUTES"
                "GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=$DYNAMIC_MAX_ADMISSIONS_PER_TOKEN"
                GGML_MOE_DYNAMIC_PREFILL_OBSERVE=1
                "GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=$DYNAMIC_WARM_START_PER_LAYER"
                "GGML_MOE_DYNAMIC_WARM_START_TOTAL=$DYNAMIC_WARM_START_TOTAL"
                GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0
                GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL=0
                "GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_PER_LAYER=$CROSS_LAYER_PREFETCH_PER_LAYER"
                "GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_TOTAL=$CROSS_LAYER_PREFETCH_TOTAL"
                "GGML_MOE_DYNAMIC_CROSS_LAYER_MIN_OBSERVATIONS=$CROSS_LAYER_MIN_OBSERVATIONS"
                "GGML_MOE_DYNAMIC_CROSS_LAYER_GLOBAL_RANK=$CROSS_LAYER_GLOBAL_RANK"
                "GGML_MOE_DYNAMIC_CROSS_LAYER_GLOBAL_RANK_OVERSUBSCRIBE=$CROSS_LAYER_GLOBAL_RANK_OVERSUBSCRIBE"
                "GGML_MOE_DYNAMIC_CROSS_LAYER_DISTANCE2_WEIGHT=$CROSS_LAYER_DISTANCE2_WEIGHT"
                "GGML_MOE_DYNAMIC_FIXED_TOPOLOGY=$DYNAMIC_FIXED_TOPOLOGY"
                GGML_MOE_DYNAMIC_SEGMENTED_LRU=1
                GGML_MOE_DYNAMIC_PROTECTED_HITS=2
                GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1
                GGML_MOE_DYNAMIC_SKIP_COLD_BRANCH=1
                "GGML_MOE_DYNAMIC_ASYNC_PROMOTION=$DYNAMIC_ASYNC_PROMOTION"
                "GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD=$URGENT_PREDICT_UPLOAD"
                "GGML_MOE_DYNAMIC_TRACE_GRAPH_BUILD=$TRACE_GRAPH_BUILD"
            )
        fi
        ;;
    vertical-static)
        if [ ! -s "$FORCE_TOKENS" ] || [ ! -s "$STATIC_MAP" ]; then
            echo "missing forced token file or static map: $FORCE_TOKENS $STATIC_MAP" >&2
            kill "$monitor_pid" 2>/dev/null || true
            exit 2
        fi
        common_env+=("GGML_COMPLETION_FORCE_TOKENS=$FORCE_TOKENS")
        model_args=(-fit on -fitt "$VERTICAL_FIT_TARGET_MIB" -fitc "$CONTEXT" --cpu-moe)
        case_env=(
            GGML_EXPERT_CACHE_MIB=auto
            "GGML_EXPERT_CACHE_RESERVE_MIB=$CACHE_RESERVE_MIB"
            GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto
            GGML_MOE_DYNAMIC_MIN_COMPLETE_COVERAGE=0
            GGML_MOE_DYNAMIC_GPU_ROUTE_MAP=1
            GGML_MOE_DYNAMIC_EXACT_CPU_FALLBACK=1
            "GGML_MOE_DYNAMIC_STATIC_MAP=$STATIC_MAP"
            GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=0
            GGML_MOE_DYNAMIC_PREFILL_OBSERVE=0
            GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=0
            GGML_MOE_DYNAMIC_WARM_START_TOTAL=0
            GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0
            GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL=0
            GGML_MOE_DYNAMIC_ASYNC_PROMOTION=1
            "GGML_MOE_DYNAMIC_TRACE_GRAPH_BUILD=$TRACE_GRAPH_BUILD"
        )
        ;;
    *)
        echo "unknown case: $CASE" >&2
        kill "$monitor_pid" 2>/dev/null || true
        exit 2
        ;;
esac

if [ -n "$TRACE_PATH" ]; then
    rm -f "$TRACE_PATH"
    case_env+=("GGML_MOE_DYNAMIC_TRACE=$TRACE_PATH")
    case_env+=("GGML_MOE_DYNAMIC_TRACE_COMPACT=$TRACE_COMPACT")
fi

cpu_args=(--poll "$POLL")
if [ -n "$CPU_RANGE" ]; then
    cpu_args+=(-Cr "$CPU_RANGE" --cpu-strict "$CPU_STRICT")
fi
if [ -n "$BATCH_CPU_RANGE" ]; then
    cpu_args+=(-Crb "$BATCH_CPU_RANGE" --cpu-strict-batch "$BATCH_CPU_STRICT")
fi
verbosity_args=()
if [ "$VERBOSE" != 0 ]; then
    verbosity_args=(-v)
fi

start_ns=$(date +%s%N)
rc=0
env \
    -u GGML_CUDA_DISABLE_GRAPHS \
    -u GGML_EXPERT_CACHE_MIB -u GGML_EXPERT_CACHE_RESERVE_MIB -u GGML_EXPERT_CACHE_PROFILE \
    -u GGML_MOE_DYNAMIC_SPLIT_SLOTS -u GGML_MOE_DYNAMIC_THRESHOLD \
    -u GGML_MOE_DYNAMIC_MIN_HOT_ROUTES -u GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN \
    -u GGML_MOE_DYNAMIC_PREFILL_OBSERVE -u GGML_MOE_DYNAMIC_WARM_START_PER_LAYER \
    -u GGML_MOE_DYNAMIC_WARM_START_TOTAL -u GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER \
    -u GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL -u GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_PER_LAYER \
    -u GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_TOTAL -u GGML_MOE_DYNAMIC_CROSS_LAYER_MIN_OBSERVATIONS \
    -u GGML_MOE_DYNAMIC_CROSS_LAYER_GLOBAL_RANK -u GGML_MOE_DYNAMIC_CROSS_LAYER_GLOBAL_RANK_OVERSUBSCRIBE \
    -u GGML_MOE_DYNAMIC_CROSS_LAYER_DISTANCE2_WEIGHT -u GGML_MOE_DYNAMIC_FIXED_TOPOLOGY \
    -u GGML_MOE_DYNAMIC_SEGMENTED_LRU \
    -u GGML_MOE_DYNAMIC_PROTECTED_HITS -u GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH \
    -u GGML_MOE_DYNAMIC_SKIP_COLD_BRANCH -u GGML_MOE_DYNAMIC_ASYNC_PROMOTION \
    -u GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD -u GGML_MOE_DYNAMIC_TRACE \
    -u GGML_MOE_DYNAMIC_TRACE_COMPACT \
    -u GGML_MOE_DYNAMIC_STATIC_MAP -u GGML_MOE_DYNAMIC_GPU_ROUTE_MAP \
    -u GGML_MOE_DYNAMIC_EXACT_CPU_FALLBACK -u GGML_MOE_DYNAMIC_MIN_COMPLETE_COVERAGE \
    "${common_env[@]}" "${case_env[@]}" \
    timeout 21600s build-v100/bin/llama-completion \
        -m "$MODEL" -f "$PROMPT_FILE" -n "$TOTAL_TOKENS" -c "$CONTEXT" \
        -b 512 -ub 128 -fa on -dev CUDA0 -t "$THREADS" -tb "$BATCH_THREADS" \
        "${cpu_args[@]}" "${model_args[@]}" -s 1 --temp 0 --ignore-eos \
        --single-turn --no-conversation --no-display-prompt --simple-io "${verbosity_args[@]}" \
        > "$out" 2> "$err" || rc=$?
end_ns=$(date +%s%N)

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

python3 - "$CASE" "$REP" "$rc" "$start_ns" "$end_ns" "$err" "$out" "$gpu" "$summary" "$MEASURE_TOKENS" "$CONTEXT" "$VERTICAL_FIT_TARGET_MIB" "$CACHE_RESERVE_MIB" "$MIN_HOT_ROUTES" "$THREADS" "$BATCH_THREADS" "$POLL" "$CPU_RANGE" "$CPU_STRICT" "$BATCH_CPU_RANGE" "$BATCH_CPU_STRICT" "$TRACE_GRAPH_BUILD" "$CUDA_GRAPHS" "$VERBOSE" "$CROSS_LAYER_PREFETCH_PER_LAYER" "$CROSS_LAYER_PREFETCH_TOTAL" "$CROSS_LAYER_MIN_OBSERVATIONS" "$CROSS_LAYER_GLOBAL_RANK" "$CROSS_LAYER_GLOBAL_RANK_OVERSUBSCRIBE" "$CROSS_LAYER_DISTANCE2_WEIGHT" "$DYNAMIC_FIXED_TOPOLOGY" "$DYNAMIC_MAX_ADMISSIONS_PER_TOKEN" "$DYNAMIC_WARM_START_PER_LAYER" "$DYNAMIC_WARM_START_TOTAL" "$DYNAMIC_ASYNC_PROMOTION" "$URGENT_PREDICT_UPLOAD" <<'PY'
import hashlib, json, re, statistics, sys
(case, rep, rc, start_ns, end_ns, err_path, out_path, gpu_path, summary_path,
 expected_runs, context, vertical_fit_target, cache_reserve, min_hot_routes,
 threads, batch_threads, poll, cpu_range, cpu_strict,
 batch_cpu_range, batch_cpu_strict, trace_graph_build, cuda_graphs, verbose,
 cross_per_layer, cross_total, cross_min_observations, cross_global_rank,
 cross_global_rank_oversubscribe, cross_distance2_weight,
 dynamic_fixed_topology, dynamic_max_admissions_per_token,
 dynamic_warm_start_per_layer, dynamic_warm_start_total,
 dynamic_async_promotion, urgent_predict_upload) = sys.argv[1:]
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
    if len(fields)!=6:
        continue
    try:
        samples.append(tuple(map(float, fields)))
    except ValueError:
        pass

measured_runs = integer(r'(?<!prompt )eval time\s*=.*?/\s*(\d+) runs')
reported_tps = last(r'(?<!prompt )eval time\s*=.*?([0-9.]+) tokens per second')
benchmark_marker = last(
    r'benchmark decode measurement complete after (\d+) tokens; '
    r'elapsed = ([0-9.]+) ms, ([0-9.]+) tokens per second')
marker_runs = int(benchmark_marker[0]) if benchmark_marker else 0
marker_elapsed_s = float(benchmark_marker[1]) / 1000.0 if benchmark_marker else None
marker_tps = float(benchmark_marker[2]) if benchmark_marker else None

# `llama_perf_context_reset()` currently loses decode accounting for some
# CPU-MoE graph layouts. Derive a second measurement from verbose monotonic
# timestamps so the fallback is explicit rather than silently reporting zero.
def timestamp_seconds(stamp):
    fields = [int(x) for x in stamp.split('.')]
    if len(fields) != 4:
        return None
    minutes, seconds, milliseconds, microseconds = fields
    return minutes * 60.0 + seconds + milliseconds / 1000.0 + microseconds / 1e6

warmup = re.search(
    r'(?m)^(\d+\.\d+\.\d+\.\d+) I llama_completion: benchmark decode warmup complete',
    text)
timestamp_runs = 0
timestamp_elapsed_s = None
timestamp_tps = None
if warmup:
    tail = text[warmup.end():]
    completions = re.findall(r'(?m)^(\d+\.\d+\.\d+\.\d+) D n_past = \d+', tail)
    timestamp_runs = len(completions)
    if completions:
        begin = timestamp_seconds(warmup.group(1))
        end = timestamp_seconds(completions[-1])
        if begin is not None and end is not None and end > begin:
            timestamp_elapsed_s = end - begin
            timestamp_tps = timestamp_runs / timestamp_elapsed_s

if marker_tps is not None and marker_runs == int(expected_runs):
    effective_tps = marker_tps
    effective_runs = marker_runs
elif reported_tps is not None and measured_runs == int(expected_runs):
    effective_tps = float(reported_tps)
    effective_runs = measured_runs
else:
    effective_tps = timestamp_tps
    effective_runs = timestamp_runs
allocation = last(r'expert-cache: entries=(\d+) allocated=([0-9.]+) MiB')
result = {
    'case': case,
    'rep': int(rep),
    'exit': int(rc),
    'wall_s': (int(end_ns)-int(start_ns))/1e9,
    'sha256': hashlib.sha256(open(out_path,'rb').read()).hexdigest(),
    'context': int(context),
    'vertical_fit_target_mib': int(vertical_fit_target),
    'cache_reserve_mib': int(cache_reserve),
    'min_hot_routes': int(min_hot_routes),
    'threads': int(threads),
    'batch_threads': int(batch_threads),
    'poll': int(poll),
    'cpu_range': cpu_range or None,
    'cpu_strict': int(cpu_strict),
    'batch_cpu_range': batch_cpu_range or None,
    'batch_cpu_strict': int(batch_cpu_strict),
    'trace_graph_build': int(trace_graph_build),
    'cuda_graphs': int(cuda_graphs),
    'verbose': int(verbose),
    'cross_layer_prefetch_per_layer': int(cross_per_layer),
    'cross_layer_prefetch_total': int(cross_total),
    'cross_layer_min_observations': int(cross_min_observations),
    'cross_layer_global_rank': int(cross_global_rank),
    'cross_layer_global_rank_oversubscribe': float(cross_global_rank_oversubscribe),
    'cross_layer_distance2_weight': float(cross_distance2_weight),
    'dynamic_fixed_topology': int(dynamic_fixed_topology),
    'dynamic_max_admissions_per_token': int(dynamic_max_admissions_per_token),
    'dynamic_warm_start_per_layer': int(dynamic_warm_start_per_layer),
    'dynamic_warm_start_total': int(dynamic_warm_start_total),
    'dynamic_async_promotion': int(dynamic_async_promotion),
    'urgent_predict_upload': int(urgent_predict_upload),
    'prompt_tokens': integer(r'prompt eval time\s*=.*?/\s*(\d+) tokens'),
    'prompt_tps': last(r'prompt eval time\s*=.*?([0-9.]+) tokens per second'),
    'measured_runs': measured_runs,
    'reported_measured_tps': float(reported_tps) if reported_tps is not None else None,
    'marker_measured_runs': marker_runs,
    'marker_measured_s': marker_elapsed_s,
    'marker_measured_tps': marker_tps,
    'timestamp_measured_runs': timestamp_runs,
    'timestamp_measured_s': timestamp_elapsed_s,
    'timestamp_measured_tps': timestamp_tps,
    'effective_measured_runs': effective_runs,
    'effective_measured_tps': effective_tps,
    'measured_time_ms': marker_elapsed_s * 1000.0 if marker_elapsed_s is not None else
        last(r'total time\s*=\s*([0-9.]+) ms'),
    'graph_nodes': last(r'sched_reserve: graph nodes\s*=\s*([^\n]+)'),
    'graph_splits': last(r'sched_reserve: graph splits\s*=\s*([^\n]+)'),
    'cuda_model_buffer_mib': last(r'CUDA0 model buffer size\s*=\s*([0-9.]+) MiB'),
    'cpu_model_buffer_mib': last(r'CPU(?:_Mapped)? model buffer size\s*=\s*([0-9.]+) MiB'),
    'cache_entries': int(allocation[0]) if allocation else None,
    'cache_allocated_mib': float(allocation[1]) if allocation else None,
    'cache_line': last(r'(moe-dynamic-cache:.*)'),
    'coverage_line': last(r'(moe-dynamic-coverage:.*)'),
    'worker_line': last(r'(moe-promotion-worker:.*)'),
    'cuda_oom': bool(re.search(r'cudaMalloc failed|out of memory', text, re.I)),
    'gpu_samples': len(samples),
    'gpu_util_mean': statistics.mean(x[3] for x in samples) if samples else None,
    'gpu_util_median': statistics.median(x[3] for x in samples) if samples else None,
    'min_free_mib': min((x[2] for x in samples), default=None),
    'max_used_mib': max((x[1] for x in samples), default=None),
    'mean_power_w': statistics.mean(x[5] for x in samples) if samples else None,
}
assertions = [
    {'name':'process_exit','ok':int(rc)==0,'detail':int(rc)},
    {'name':'measured_token_count','ok':effective_runs==int(expected_runs),'detail':{'marker':marker_runs,'reported':measured_runs,'timestamp':timestamp_runs}},
    {'name':'no_cuda_oom','ok':not result['cuda_oom'],'detail':result['cuda_oom']},
]
if case in ('vertical', 'vertical-free', 'vertical-static') and int(verbose) != 0:
    assertions.extend([
        {'name':'cache_allocated','ok':result['cache_entries'] is not None and result['cache_entries']>0,'detail':result['cache_entries']},
        {'name':'coverage_present','ok':result['coverage_line'] is not None,'detail':result['coverage_line']},
    ])
result['assertions'] = assertions
result['assertions_passed'] = all(x['ok'] for x in assertions)
with open(summary_path,'w') as f:
    json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
if not result['assertions_passed']:
    raise SystemExit(3)
PY
summary_rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$summary_rc"

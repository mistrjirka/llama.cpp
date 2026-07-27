#!/usr/bin/env bash
set -u

MODEL=${MODEL:-/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf}
PROMPT_FILE=${PROMPT_FILE:-profiling/real-chat-prompt.txt}
FORCE_TOKENS=${FORCE_TOKENS:-$PWD/profiling/dynamic-real-chat-ab-2026-07-26/reference.tokens.txt}
OUTDIR=${OUTDIR:-profiling/urgent-predict-upload-ab-2026-07-27}
CASE=${1:-deferred}
REP=${2:-1}
TOTAL_TOKENS=${TOTAL_TOKENS:-512}
WARMUP_TOKENS=${WARMUP_TOKENS:-64}
CONTEXT=${CONTEXT:-8192}

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
        nvidia-smi -i "${NVIDIA_SMI_GPU_INDEX:-1}" \
            --query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
            --format=csv,noheader,nounits | tr -d ' ' | tr ',' '\t'
        sleep 0.2
    done
}
monitor_gpu > "$gpu" 2>/dev/null &
monitor_pid=$!

case_env=()
case "$CASE" in
    deferred)
        case_env=(GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD=0)
        ;;
    urgent)
        case_env=(GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD=1)
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
    -u GGML_MOE_DYNAMIC_STATIC_MAP -u GGML_MOE_DYNAMIC_FORCE_GPU_ONLY \
    -u GGML_MOE_DYNAMIC_EXACT_CPU_FALLBACK -u GGML_MOE_SKELETON_ONLY \
    -u GGML_MOE_DYNAMIC_TRACE \
    CUDA_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 \
    "GGML_COMPLETION_BENCH_WARMUP_TOKENS=$WARMUP_TOKENS" \
    GGML_EXPERT_CACHE_MIB=auto GGML_EXPERT_CACHE_RESERVE_MIB=8192 GGML_EXPERT_CACHE_PROFILE=1 \
    GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto GGML_MOE_DYNAMIC_THRESHOLD=2 \
    GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=8 GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=32 \
    GGML_MOE_DYNAMIC_PREFILL_OBSERVE=1 GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=16 \
    GGML_MOE_DYNAMIC_WARM_START_TOTAL=512 GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0 \
    GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_PER_LAYER=1 \
    GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_TOTAL=16 \
    GGML_MOE_DYNAMIC_CROSS_LAYER_MIN_OBSERVATIONS=8 \
    GGML_MOE_DYNAMIC_SEGMENTED_LRU=1 GGML_MOE_DYNAMIC_PROTECTED_HITS=2 \
    GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1 GGML_MOE_DYNAMIC_SKIP_COLD_BRANCH=1 \
    GGML_MOE_DYNAMIC_ASYNC_PROMOTION=1 \
    GGML_COMPLETION_FORCE_TOKENS="$FORCE_TOKENS" \
    "${case_env[@]}" \
    timeout 3600s build-v100/bin/llama-completion \
        -m "$MODEL" --fit off -ngl 99 --cpu-moe \
        -c "$CONTEXT" -b 2048 -ub 512 -fa on -t 40 -tb 48 \
        --seed 1 --temp 0 --conversation --single-turn --simple-io --no-display-prompt \
        -sys 'You are a senior systems engineer. Give detailed, technically rigorous, practical answers.' \
        -f "$PROMPT_FILE" -n "$TOTAL_TOKENS" -v \
        > "$out" 2> "$err" || rc=$?
end_ns=$(date +%s%N)

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

python3 - "$CASE" "$REP" "$rc" "$start_ns" "$end_ns" "$err" "$out" "$gpu" "$summary" <<'PY'
import hashlib,json,re,statistics,sys
case,rep,rc,start_ns,end_ns,err_path,out_path,gpu_path,summary_path=sys.argv[1:]
text=open(err_path,errors='replace').read()
def last(pattern,default=None):
    values=re.findall(pattern,text)
    return values[-1] if values else default
def integer(pattern,default=0):
    value=last(pattern)
    return int(value) if value is not None else default
samples=[]
for line in open(gpu_path,errors='replace'):
    fields=line.strip().split('\t')
    if len(fields)==6:
        try:samples.append(tuple(map(float,fields)))
        except ValueError:pass
worker=last(r'moe-promotion-worker: jobs=(\d+) batches=(\d+) urgent-jobs=(\d+) urgent-batches=(\d+) bytes=([0-9.]+) MiB staging-bytes=([0-9.]+) MiB staging-copy=([0-9.]+) ms host-issue=([0-9.]+) ms sync=([0-9.]+) ms queue-delay=([0-9.]+) ms max-depth=(\d+)')
coverage=last(r'moe-dynamic-coverage: full-gpu-expert-layers=(\d+)/(\d+) cpu-dependent-layers=(\d+) full-gpu-expert-path-tokens=(\d+)/(\d+) mean-cpu-dependent-layers=([0-9.]+)')
cache=last(r'moe-dynamic-cache: layers=(\d+) admissions=(\d+) resident-hits=(\d+) resident-misses=(\d+) resident-hit-rate=([0-9.]+) executed-gpu-routes=(\d+) cpu-routes=(\d+) execution-hit-rate=([0-9.]+) ready-slots=(\d+) copying-slots=(\d+) split-enabled=(\d+)/(\d+)')
total_ms_value=last(r'total time\s*=\s*([0-9.]+) ms')
measured_runs=512-64
measured_tps=(measured_runs*1000/float(total_ms_value)) if total_ms_value is not None else None
result={
 'case':case,'rep':int(rep),'exit':int(rc),'wall_s':(int(end_ns)-int(start_ns))/1e9,
 'sha256':hashlib.sha256(open(out_path,'rb').read()).hexdigest(),
 'measured_runs':measured_runs,
 'measured_time_ms':float(total_ms_value) if total_ms_value is not None else None,
 'measured_tps':measured_tps,
 'worker':worker,'coverage':coverage,'cache':cache,
 'gpu_samples':len(samples),
 'gpu_util_mean':statistics.mean(x[3] for x in samples) if samples else None,
 'gpu_util_median':statistics.median(x[3] for x in samples) if samples else None,
 'min_free_mib':min((x[2] for x in samples),default=None),
 'max_used_mib':max((x[1] for x in samples),default=None),
 'mean_power_w':statistics.mean(x[5] for x in samples) if samples else None,
}
assertions=[]
def check(name,ok,detail):assertions.append({'name':name,'ok':bool(ok),'detail':detail})
check('process_exit',int(rc)==0,rc)
check('warmup_reset_applied','benchmark decode warmup complete after 64 tokens' in text,measured_runs)
check('measured_total_present',total_ms_value is not None,total_ms_value)
check('worker_summary_present',worker is not None,worker)
check('coverage_present',coverage is not None,coverage)
if case=='urgent' and worker:
    check('urgent_jobs_present',int(worker[2])>0,worker[2])
if case=='deferred' and worker:
    check('urgent_jobs_absent',int(worker[2])==0,worker[2])
result['assertions']=assertions
result['assertions_passed']=all(x['ok'] for x in assertions)
with open(summary_path,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
if not result['assertions_passed']:sys.exit(3)
PY
summary_rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$summary_rc"

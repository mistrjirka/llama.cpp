#!/usr/bin/env bash
set -euo pipefail
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=staged-worker-repeats
mkdir -p "$outdir"
printf 'mode\trep\texit\tprompt_tps\tdecode_tps\tadmissions\thit_rate\tupload_mib\tstage_mib\tstage_ms\tissue_ms\tsync_ms\tqueue_ms\tmax_depth\tsha256\n' > "$outdir/results.tsv"
main_cpus='0-22,24-46'
run_case() {
  local mode=$1 rep=$2
  local err="$outdir/${mode}-r${rep}.err" out="$outdir/${mode}-r${rep}.out"
  rm -f "$err" "$out"
  local -a env_args=(
    -u GGML_EXPERT_CACHE_MIB
    -u GGML_EXPERT_CACHE_VMM
    -u GGML_EXPERT_CACHE_MANAGED
    -u GGML_MOE_STATIC_SPLIT_SLOTS
    -u GGML_MOE_STATIC_SPLIT_MAP
    -u GGML_MOE_DYNAMIC_SPLIT_SLOTS
    -u GGML_MOE_DYNAMIC_SPLIT_LAYERS
    -u GGML_MOE_DYNAMIC_THRESHOLD
    -u GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN
    -u GGML_MOE_DYNAMIC_ASYNC_PROMOTION
    -u GGML_MOE_DYNAMIC_STAGING_MIB
    -u GGML_MOE_DYNAMIC_STAGING_SLOTS
    -u GGML_MOE_DYNAMIC_WORKER_CPU
    GGML_CUDA_DISABLE_GRAPHS=1
    CUDA_VISIBLE_DEVICES=0
  )
  case "$mode" in
    noadmit)
      env_args+=(GGML_EXPERT_CACHE_MIB=64 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_SPLIT_LAYERS=28 GGML_MOE_DYNAMIC_THRESHOLD=999 GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=0 GGML_MOE_DYNAMIC_ASYNC_PROMOTION=0)
      ;;
    fallback)
      env_args+=(GGML_EXPERT_CACHE_MIB=64 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_SPLIT_LAYERS=28 GGML_MOE_DYNAMIC_THRESHOLD=2.0 GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10 GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=1 GGML_MOE_DYNAMIC_ASYNC_PROMOTION=0 GGML_EXPERT_CACHE_PROFILE=1)
      ;;
    direct-worker)
      env_args+=(GGML_EXPERT_CACHE_MIB=64 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_SPLIT_LAYERS=28 GGML_MOE_DYNAMIC_THRESHOLD=2.0 GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10 GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=1 GGML_MOE_DYNAMIC_ASYNC_PROMOTION=1 GGML_MOE_DYNAMIC_STAGING_MIB=1 GGML_MOE_DYNAMIC_STAGING_SLOTS=24 GGML_MOE_DYNAMIC_WORKER_CPU=47 GGML_EXPERT_CACHE_PROFILE=1)
      ;;
    staged-worker)
      env_args+=(GGML_EXPERT_CACHE_MIB=64 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_SPLIT_LAYERS=28 GGML_MOE_DYNAMIC_THRESHOLD=2.0 GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10 GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=1 GGML_MOE_DYNAMIC_ASYNC_PROMOTION=1 GGML_MOE_DYNAMIC_STAGING_MIB=96 GGML_MOE_DYNAMIC_STAGING_SLOTS=24 GGML_MOE_DYNAMIC_WORKER_CPU=47 GGML_EXPERT_CACHE_PROFILE=1)
      ;;
  esac
  rc=0
  env "${env_args[@]}" taskset -c "$main_cpus" timeout 1800s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 128 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt 1024 -fitc 512 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err" || rc=$?
  python3 - "$mode" "$rep" "$rc" "$err" "$out" <<'PY' | tee -a "$outdir/results.tsv"
import hashlib,re,sys
mode,rep,rc,err,out=sys.argv[1:]
text=open(err,errors='replace').read()
def tps(label):
    m=re.findall(rf'{label}\s*=.*?\(.*?([0-9.]+) tokens per second\)',text)
    return m[-1] if m else ''
dyn=re.findall(r'moe-dynamic-cache:.*?admissions=(\d+).*?hit-rate=([0-9.]+)',text)
adm,hit=dyn[-1] if dyn else ('0','0')
cache=re.findall(r'expert-cache: entries=(\d+) allocated=([0-9.]+) MiB.*?uploaded=([0-9.]+) MiB',text)
entries,alloc,uploaded=cache[-1] if cache else ('0','0','0')
worker=re.findall(r'moe-promotion-worker: jobs=\d+ batches=\d+ bytes=[0-9.]+ MiB staging-bytes=([0-9.]+) MiB staging-copy=([0-9.]+) ms host-issue=([0-9.]+) ms sync=([0-9.]+) ms queue-delay=([0-9.]+) ms max-depth=(\d+)',text)
stage_mib,stage_ms,issue_ms,sync_ms,queue_ms,max_depth=worker[-1] if worker else ('0','0','0','0','0','0')
print('\t'.join([mode,rep,rc,tps('prompt eval time'),tps('eval time'),adm,hit,uploaded,stage_mib,stage_ms,issue_ms,sync_ms,queue_ms,max_depth,hashlib.sha256(open(out,'rb').read()).hexdigest()]))
PY
}
for rep in 1 2 3; do
  run_case baseline "$rep"
  run_case noadmit "$rep"
  run_case fallback "$rep"
  run_case direct-worker "$rep"
  run_case staged-worker "$rep"
done
python3 - <<'PY'
import csv,statistics
rows=list(csv.DictReader(open('staged-worker-repeats/results.tsv'),delimiter='\t'))
for mode in ('baseline','noadmit','fallback','direct-worker','staged-worker'):
    values=[float(row['decode_tps']) for row in rows if row['mode']==mode]
    print(mode,'mean',f'{statistics.fmean(values):.4f}','stdev',f'{statistics.stdev(values):.4f}','values',values)
PY

#!/usr/bin/env bash
set -euo pipefail
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=no-mask-memset-ab
mkdir -p "$outdir"
printf 'mode\trep\texit\tprompt_tps\tdecode_tps\tsha256\n' > "$outdir/results.tsv"
run_case() {
  local mode=$1 rep=$2
  local err="$outdir/${mode}-r${rep}.err" out="$outdir/${mode}-r${rep}.out"
  local fit=1024
  local -a env_args=(
    -u GGML_EXPERT_CACHE_MIB -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED
    -u GGML_MOE_STATIC_SPLIT_SLOTS -u GGML_MOE_STATIC_SPLIT_MAP
    -u GGML_MOE_DYNAMIC_SPLIT_SLOTS -u GGML_MOE_DYNAMIC_SPLIT_LAYERS
    -u GGML_MOE_DYNAMIC_THRESHOLD -u GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN
    -u GGML_MOE_DYNAMIC_ASYNC_PROMOTION
    GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0
  )
  case "$mode" in
    noadmit-layer28)
      env_args+=(GGML_EXPERT_CACHE_MIB=64 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_SPLIT_LAYERS=28 GGML_MOE_DYNAMIC_THRESHOLD=999 GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=0 GGML_MOE_DYNAMIC_ASYNC_PROMOTION=0)
      ;;
    baseline-all)
      fit=8192
      ;;
    noadmit-all)
      fit=8192
      env_args+=(GGML_EXPERT_CACHE_MIB=8192 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_THRESHOLD=999 GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=0 GGML_MOE_DYNAMIC_ASYNC_PROMOTION=0)
      ;;
  esac
  env "${env_args[@]}" timeout 1800s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 128 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt "$fit" -fitc 512 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err"
  python3 - "$mode" "$rep" "$err" "$out" <<'PY' | tee -a "$outdir/results.tsv"
import hashlib,re,sys
mode,rep,err,out=sys.argv[1:]
text=open(err,errors='replace').read()
def tps(label):
 m=re.findall(rf'{label}\s*=.*?\(.*?([0-9.]+) tokens per second\)',text)
 return m[-1] if m else ''
print('\t'.join([mode,rep,'0',tps('prompt eval time'),tps('eval time'),hashlib.sha256(open(out,'rb').read()).hexdigest()]))
PY
}
for rep in 1 2; do
  run_case baseline-layer28 "$rep"
  run_case noadmit-layer28 "$rep"
  run_case baseline-all "$rep"
  run_case noadmit-all "$rep"
done
python3 - <<'PY'
import csv,statistics
rows=list(csv.DictReader(open('no-mask-memset-ab/results.tsv'),delimiter='\t'))
for mode in ('baseline-layer28','noadmit-layer28','baseline-all','noadmit-all'):
 vals=[float(r['decode_tps']) for r in rows if r['mode']==mode]
 print(mode,statistics.fmean(vals),vals)
PY

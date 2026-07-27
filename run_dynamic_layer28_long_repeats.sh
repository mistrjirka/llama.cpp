#!/usr/bin/env bash
set -u
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and comprehensive unit tests. Return only compilable code.'
outdir=dynamic-layer28-long-repeats
mkdir -p "$outdir"
printf 'mode\trep\tdecode_tps\tprompt_tps\tadmissions\thit_rate\tupload_mib\tsha256\n' > "$outdir/results.tsv"
run_case() {
  local mode=$1 rep=$2
  local err="$outdir/${mode}-r${rep}.err" out="$outdir/${mode}-r${rep}.out"
  rm -f "$err" "$out"
  local -a env_args=(-u GGML_EXPERT_CACHE_MIB -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED -u GGML_MOE_STATIC_SPLIT_SLOTS -u GGML_MOE_STATIC_SPLIT_MAP -u GGML_MOE_DYNAMIC_SPLIT_SLOTS -u GGML_MOE_DYNAMIC_SPLIT_LAYERS GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0)
  if [[ "$mode" == dynamic ]]; then
    env_args+=(GGML_EXPERT_CACHE_MIB=64 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_SPLIT_LAYERS=28 GGML_MOE_DYNAMIC_THRESHOLD=2.0 GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10 GGML_EXPERT_CACHE_PROFILE=1)
  fi
  env "${env_args[@]}" timeout 2400s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 512 -c 1024 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt 1024 -fitc 1024 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err"
  python3 - "$mode" "$rep" "$err" "$out" <<'PY' | tee -a "$outdir/results.tsv"
import hashlib,re,sys
mode,rep,err,out=sys.argv[1:]
text=open(err,errors='replace').read()
def tps(label):
    m=re.findall(rf'{label}\s*=.*?\(.*?([0-9.]+) tokens per second\)',text)
    return m[-1] if m else ''
dyn=re.findall(r'moe-dynamic-cache:.*?admissions=(\d+).*?hit-rate=([0-9.]+)',text)
adm,hit=dyn[-1] if dyn else ('0','0')
cache=re.findall(r'expert-cache: entries=(\d+) allocated=([0-9.]+) MiB.*?uploaded=([0-9.]+) MiB',text)
entries,alloc,uploaded=cache[-1] if cache else ('0','0','0')
data=open(out,'rb').read()
print('\t'.join([mode,rep,tps('eval time'),tps('prompt eval time'),adm,hit,uploaded,hashlib.sha256(data).hexdigest()]))
PY
}
for rep in 1 2 3; do
  run_case baseline "$rep"
  run_case dynamic "$rep"
done
python3 - <<'PY'
import csv,statistics
rows=list(csv.DictReader(open('dynamic-layer28-long-repeats/results.tsv'),delimiter='\t'))
for mode in ('baseline','dynamic'):
 vals=[float(r['decode_tps']) for r in rows if r['mode']==mode]
 print(mode,'decode_mean',f'{statistics.fmean(vals):.4f}','stdev',f'{statistics.stdev(vals):.4f}','values',vals)
base=statistics.fmean(float(r['decode_tps']) for r in rows if r['mode']=='baseline')
dyn=statistics.fmean(float(r['decode_tps']) for r in rows if r['mode']=='dynamic')
print('relative',f'{(dyn/base-1)*100:.3f}%')
PY

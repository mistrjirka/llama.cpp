#!/usr/bin/env bash
set -u
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=dynamic-layer28-slots
mkdir -p "$outdir"
printf 'name\tslots\tdecode_tps\tprompt_tps\tcache_mib\tadmissions\thit_rate\tupload_mib\tsha256\n' > "$outdir/results.tsv"
run_case() {
  local name=$1 slots=$2 budget=$3
  local err="$outdir/$name.err" out="$outdir/$name.out"
  rm -f "$err" "$out"
  local -a env_args=(-u GGML_EXPERT_CACHE_MIB -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED -u GGML_MOE_STATIC_SPLIT_SLOTS -u GGML_MOE_STATIC_SPLIT_MAP -u GGML_MOE_DYNAMIC_SPLIT_SLOTS -u GGML_MOE_DYNAMIC_SPLIT_LAYERS GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0)
  if (( slots > 0 )); then
    env_args+=(GGML_EXPERT_CACHE_MIB="$budget" GGML_MOE_DYNAMIC_SPLIT_SLOTS="$slots" GGML_MOE_DYNAMIC_SPLIT_LAYERS=28 GGML_MOE_DYNAMIC_THRESHOLD=2.0 GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10 GGML_EXPERT_CACHE_PROFILE=1)
  fi
  env "${env_args[@]}" timeout 1800s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 128 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt 1024 -fitc 512 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err"
  python3 - "$name" "$slots" "$err" "$out" <<'PY' | tee -a "$outdir/results.tsv"
import hashlib,re,sys
name,slots,err,out=sys.argv[1:]
text=open(err,errors='replace').read()
def tps(label):
    m=re.findall(rf'{label}\s*=.*?\(.*?([0-9.]+) tokens per second\)',text)
    return m[-1] if m else ''
cache=re.findall(r'expert-cache: entries=(\d+) allocated=([0-9.]+) MiB.*?uploaded=([0-9.]+) MiB',text)
entries,alloc,uploaded=cache[-1] if cache else ('0','0','0')
dyn=re.findall(r'moe-dynamic-cache:.*?admissions=(\d+).*?hit-rate=([0-9.]+)',text)
adm,hit=dyn[-1] if dyn else ('0','0')
data=open(out,'rb').read()
print('\t'.join([name,slots,tps('eval time'),tps('prompt eval time'),alloc,adm,hit,uploaded,hashlib.sha256(data).hexdigest()]))
PY
  echo completed "$name"
}
run_case baseline-a 0 0
run_case slots4 4 32
run_case slots8 8 64
run_case slots16 16 96
run_case slots32 32 160
run_case baseline-b 0 0
cat "$outdir/results.tsv"

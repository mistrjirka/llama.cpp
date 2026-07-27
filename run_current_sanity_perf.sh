#!/usr/bin/env bash
set -u
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=current-sanity-perf
mkdir -p "$outdir"
printf 'name\tfit_mib\texit\tprompt_tps\tdecode_tps\tcache_mib\tlayers\tadmissions\thit_rate\tdynamic_mib\tdynamic_issue_ms\ttarget_sync_ms\tsha256\n' > "$outdir/results.tsv"
run_case() {
  local name=$1 fit=$2 slots=$3 layers=$4 threshold=$5
  local err="$outdir/$name.err" out="$outdir/$name.out"
  rm -f "$err" "$out"
  local -a env_args=(
    -u GGML_EXPERT_CACHE_MIB -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED
    -u GGML_MOE_STATIC_SPLIT_SLOTS -u GGML_MOE_STATIC_SPLIT_MAP
    -u GGML_MOE_DYNAMIC_SPLIT_SLOTS -u GGML_MOE_DYNAMIC_SPLIT_LAYERS
    -u GGML_MOE_DYNAMIC_THRESHOLD -u GGML_EXPERT_CACHE_PROFILE
    GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0
  )
  if (( slots > 0 )); then
    env_args+=(GGML_EXPERT_CACHE_MIB=8192 GGML_MOE_DYNAMIC_SPLIT_SLOTS="$slots" GGML_MOE_DYNAMIC_THRESHOLD="$threshold" GGML_EXPERT_CACHE_PROFILE=1)
    if [[ "$layers" != all ]]; then env_args+=(GGML_MOE_DYNAMIC_SPLIT_LAYERS="$layers"); fi
  fi
  rc=0
  env "${env_args[@]}" timeout 1800s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 128 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt "$fit" -fitc 512 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err" || rc=$?
  python3 - "$name" "$fit" "$rc" "$err" "$out" <<'PY' | tee -a "$outdir/results.tsv"
import hashlib,re,sys
name,fit,rc,err,out=sys.argv[1:]
text=open(err,errors='replace').read()
def tps(label):
    m=re.findall(rf'{label}\s*=.*?\(.*?([0-9.]+) tokens per second\)',text)
    return m[-1] if m else ''
cache=re.findall(r'expert-cache: entries=(\d+) allocated=([0-9.]+) MiB',text)
entries,cache_mib=cache[-1] if cache else ('0','0')
dyn=re.findall(r'moe-dynamic-cache: layers=(\d+).*?admissions=(\d+).*?hit-rate=([0-9.]+)',text)
layers,adm,hit=dyn[-1] if dyn else ('0','0','0')
profile=re.findall(r'expert-cache-profile:.*?dynamic-bytes=([0-9.]+) MiB dynamic-issue=([0-9.]+) ms.*?dynamic-target-sync=([0-9.]+) ms',text)
dmib,dissue,dsync=profile[-1] if profile else ('0','0','0')
data=open(out,'rb').read()
print('\t'.join([name,fit,rc,tps('prompt eval time'),tps('eval time'),cache_mib,layers,adm,hit,dmib,dissue,dsync,hashlib.sha256(data).hexdigest()]))
PY
  echo completed "$name"
}
run_case baseline8g-a 8192 0 all 999
run_case noadmit-all8 8192 8 all 999
run_case dynamic-all8 8192 8 all 2.0
run_case dynamic-all32 8192 32 all 2.0
run_case baseline8g-b 8192 0 all 999
run_case baseline1g 1024 0 all 999
run_case noadmit-layer28 1024 8 28 999
run_case dynamic-layer28 1024 8 28 2.0
run_case baseline1g-b 1024 0 all 999
cat "$outdir/results.tsv"

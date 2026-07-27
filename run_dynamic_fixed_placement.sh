#!/usr/bin/env bash
set -u
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=dynamic-fixed-placement
mkdir -p "$outdir"
printf 'mode\tslots\texit\tprompt_tps\tdecode_tps\tpeak_mib\tcache_mib\tlayers\tadmissions\thit_rate\tupload_mib\tsha256\n' > "$outdir/results.tsv"
run_case() {
  local mode=$1 slots=$2
  local name="${mode}-s${slots}"
  local err="$outdir/$name.err" out="$outdir/$name.out" gpu="$outdir/$name.gpu"
  rm -f "$err" "$out" "$gpu"
  local -a env_args=(
    -u GGML_EXPERT_CACHE_MIB
    -u GGML_EXPERT_CACHE_VMM
    -u GGML_EXPERT_CACHE_MANAGED
    -u GGML_MOE_STATIC_SPLIT_SLOTS
    -u GGML_MOE_STATIC_SPLIT_MAP
    -u GGML_MOE_DYNAMIC_SPLIT_SLOTS
    -u GGML_MOE_DYNAMIC_SPLIT_LAYERS
    GGML_CUDA_DISABLE_GRAPHS=1
    CUDA_VISIBLE_DEVICES=0
  )
  if [[ "$mode" == dynamic ]]; then
    env_args+=(
      GGML_EXPERT_CACHE_MIB=8192
      GGML_MOE_DYNAMIC_SPLIT_SLOTS="$slots"
      GGML_MOE_DYNAMIC_THRESHOLD=2.0
      GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10
      GGML_EXPERT_CACHE_PROFILE=1
    )
  fi
  (while :; do nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 ~ /V100/ {print $2}' >> "$gpu"; sleep 0.2; done) & mon=$!
  local rc=0
  env "${env_args[@]}" timeout 1800s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 128 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt 8192 -fitc 512 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err" || rc=$?
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  python3 - "$mode" "$slots" "$rc" "$err" "$gpu" "$out" <<'PY' | tee -a "$outdir/results.tsv"
import hashlib,re,sys
mode,slots,rc,err,gpu,out=sys.argv[1:]
text=open(err,errors='replace').read()
def tps(label):
    m=re.findall(rf'{label}\s*=.*?\(.*?([0-9.]+) tokens per second\)',text)
    return m[-1] if m else ''
cache=re.findall(r'expert-cache: entries=(\d+) allocated=([0-9.]+) MiB.*?uploaded=([0-9.]+) MiB',text)
entries,alloc,uploaded=cache[-1] if cache else ('0','0','0')
dyn=re.findall(r'moe-dynamic-cache: layers=(\d+).*?admissions=(\d+).*?hit-rate=([0-9.]+)',text)
layers,adm,hit=dyn[-1] if dyn else ('0','0','0')
vals=[int(float(x.strip())) for x in open(gpu) if x.strip()]
data=open(out,'rb').read()
print('\t'.join([mode,slots,rc,tps('prompt eval time'),tps('eval time'),str(max(vals) if vals else ''),alloc,layers,adm,hit,uploaded,hashlib.sha256(data).hexdigest()]))
PY
  echo completed "$name"
}
run_case baseline 0
for slots in 8 16 32 64; do
  run_case dynamic "$slots"
done
run_case baseline 0
cat "$outdir/results.tsv"

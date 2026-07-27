#!/usr/bin/env bash
set -u
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=dynamic-layer-subsets
mkdir -p "$outdir"
printf 'name\tlayers\texit\twall_s\tprompt_tps\tdecode_tps\tpeak_mib\tcache_mib\tadmissions\thit_rate\tready_slots\tupload_mib\tsha256\n' > "$outdir/results.tsv"
for spec in \
  'top4|44,30,28,35|256' \
  'top8|44,30,28,35,47,29,37,46|384' \
  'top12|44,30,28,35,47,29,37,46,43,27,25,34|512'; do
  IFS='|' read -r name layers budget <<< "$spec"
  err="$outdir/$name.err"; out="$outdir/$name.out"; gpu="$outdir/$name.gpu"
  rm -f "$err" "$out" "$gpu"
  (while :; do nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 ~ /V100/ {print $2}' >> "$gpu"; sleep 0.2; done) & mon=$!
  start=$(date +%s.%N); rc=0
  env -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED -u GGML_MOE_STATIC_SPLIT_SLOTS -u GGML_MOE_STATIC_SPLIT_MAP \
    GGML_EXPERT_CACHE_MIB="$budget" GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_SPLIT_LAYERS="$layers" GGML_EXPERT_CACHE_PROFILE=1 \
    GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0 timeout 1800s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 128 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt 1024 -fitc 512 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err" || rc=$?
  end=$(date +%s.%N)
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  python3 - "$name" "$layers" "$rc" "$start" "$end" "$err" "$gpu" "$out" <<'PY' | tee -a "$outdir/results.tsv"
import hashlib,re,sys
name,layers,rc,start,end,err,gpu,out=sys.argv[1:]
text=open(err,errors='replace').read()
def tps(label):
    m=re.findall(rf'{label}\s*=.*?\(.*?([0-9.]+) tokens per second\)',text)
    return m[-1] if m else ''
cache=re.findall(r'expert-cache: entries=(\d+) allocated=([0-9.]+) MiB.*?uploaded=([0-9.]+) MiB',text)
entries,alloc,uploaded=cache[-1] if cache else ('0','0','0')
dyn=re.findall(r'moe-dynamic-cache:.*?admissions=(\d+).*?hit-rate=([0-9.]+).*?ready-slots=(\d+)',text)
adm,hit,ready=dyn[-1] if dyn else ('0','0','0')
vals=[int(float(x.strip())) for x in open(gpu) if x.strip()]
data=open(out,'rb').read()
print('\t'.join([name,layers,rc,f'{float(end)-float(start):.3f}',tps('prompt eval time'),tps('eval time'),str(max(vals) if vals else ''),alloc,adm,hit,ready,uploaded,hashlib.sha256(data).hexdigest()]))
PY
  echo completed "$name"
done
cat "$outdir/results.tsv"

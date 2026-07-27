#!/usr/bin/env bash
set -u
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=dynamic-fixed-placement-no-admit
mkdir -p "$outdir"
printf 'slots\texit\tprompt_tps\tdecode_tps\tcache_mib\tlayers\tadmissions\thit_rate\tupload_mib\tsha256\n' > "$outdir/results.tsv"
for slots in 8 16 32 64; do
  err="$outdir/s${slots}.err"; out="$outdir/s${slots}.out"
  rm -f "$err" "$out"
  rc=0
  env -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED -u GGML_MOE_STATIC_SPLIT_SLOTS -u GGML_MOE_STATIC_SPLIT_MAP -u GGML_MOE_DYNAMIC_SPLIT_LAYERS \
    GGML_EXPERT_CACHE_MIB=8192 GGML_MOE_DYNAMIC_SPLIT_SLOTS="$slots" GGML_MOE_DYNAMIC_THRESHOLD=999 GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10 GGML_EXPERT_CACHE_PROFILE=1 \
    GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0 timeout 1800s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 128 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt 8192 -fitc 512 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err" || rc=$?
  python3 - "$slots" "$rc" "$err" "$out" <<'PY' | tee -a "$outdir/results.tsv"
import hashlib,re,sys
slots,rc,err,out=sys.argv[1:]
text=open(err,errors='replace').read()
def tps(label):
    m=re.findall(rf'{label}\s*=.*?\(.*?([0-9.]+) tokens per second\)',text)
    return m[-1] if m else ''
cache=re.findall(r'expert-cache: entries=(\d+) allocated=([0-9.]+) MiB.*?uploaded=([0-9.]+) MiB',text)
entries,alloc,uploaded=cache[-1] if cache else ('0','0','0')
dyn=re.findall(r'moe-dynamic-cache: layers=(\d+).*?admissions=(\d+).*?hit-rate=([0-9.]+)',text)
layers,adm,hit=dyn[-1] if dyn else ('0','0','0')
print('\t'.join([slots,rc,tps('prompt eval time'),tps('eval time'),alloc,layers,adm,hit,uploaded,hashlib.sha256(open(out,'rb').read()).hexdigest()]))
PY
done
cat "$outdir/results.tsv"

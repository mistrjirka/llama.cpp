#!/usr/bin/env bash
set -u
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=cap8-thresholds
mkdir -p "$outdir"
printf 'threshold\tdecode_tps\tprompt_tps\tadmissions\thit_rate\tdynamic_mib\tdynamic_issue_ms\ttarget_sync_ms\tsha256\n' > "$outdir/results.tsv"
for threshold in 2 3 4 5; do
  err="$outdir/t${threshold}.err"; out="$outdir/t${threshold}.out"
  rm -f "$err" "$out"
  env -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED -u GGML_MOE_STATIC_SPLIT_SLOTS -u GGML_MOE_STATIC_SPLIT_MAP -u GGML_MOE_DYNAMIC_SPLIT_LAYERS \
    GGML_EXPERT_CACHE_MIB=8192 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_THRESHOLD="$threshold" \
    GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10 GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=8 GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=1 GGML_EXPERT_CACHE_PROFILE=1 \
    GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0 timeout 1800s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 128 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt 8192 -fitc 512 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err"
  python3 - "$threshold" "$err" "$out" <<'PY' | tee -a "$outdir/results.tsv"
import hashlib,re,sys
threshold,err,out=sys.argv[1:]
text=open(err,errors='replace').read()
def tps(label):
 m=re.findall(rf'{label}\s*=.*?\(.*?([0-9.]+) tokens per second\)',text)
 return m[-1] if m else ''
dyn=re.findall(r'moe-dynamic-cache:.*?admissions=(\d+).*?hit-rate=([0-9.]+)',text)
adm,hit=dyn[-1] if dyn else ('0','0')
prof=re.findall(r'expert-cache-profile:.*?dynamic-bytes=([0-9.]+) MiB dynamic-issue=([0-9.]+) ms.*?dynamic-target-sync=([0-9.]+) ms',text)
dmib,dissue,dsync=prof[-1] if prof else ('0','0','0')
print('\t'.join([threshold,tps('eval time'),tps('prompt eval time'),adm,hit,dmib,dissue,dsync,hashlib.sha256(open(out,'rb').read()).hexdigest()]))
PY
done
cat "$outdir/results.tsv"

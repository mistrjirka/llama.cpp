#!/usr/bin/env bash
set -u
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=dynamic-admission-caps-v2
mkdir -p "$outdir"
printf 'name\tcap\texit\tprompt_tps\tdecode_tps\tpeak_mib\tadmissions\thit_rate\tready_slots\tupload_mib\tdynamic_issue_ms\ttarget_sync_ms\tsha256\n' > "$outdir/results.tsv"
run_case() {
  local name=$1 cap=$2
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
    -u GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN
    GGML_CUDA_DISABLE_GRAPHS=1
    CUDA_VISIBLE_DEVICES=0
  )
  if [[ "$name" != baseline && "$name" != baseline2 ]]; then
    env_args+=(
      GGML_EXPERT_CACHE_MIB=8192
      GGML_MOE_DYNAMIC_SPLIT_SLOTS=8
      GGML_MOE_DYNAMIC_THRESHOLD=2.0
      GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10
      GGML_EXPERT_CACHE_PROFILE=1
    )
    if [[ "$cap" == unlimited ]]; then
      env_args+=(GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=-1)
    else
      env_args+=(GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN="$cap")
    fi
  fi
  (while :; do nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 ~ /V100/ {print $2}' >> "$gpu"; sleep 0.2; done) & mon=$!
  local rc=0
  env "${env_args[@]}" timeout 1800s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 128 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt 8192 -fitc 512 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$out" 2> "$err" || rc=$?
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  python3 - "$name" "$cap" "$rc" "$err" "$gpu" "$out" <<'PY' | tee -a "$outdir/results.tsv"
import hashlib,re,sys
name,cap,rc,err,gpu,out=sys.argv[1:]
text=open(err,errors='replace').read()
def tps(label):
    m=re.findall(rf'{label}\s*=.*?\(.*?([0-9.]+) tokens per second\)',text)
    return m[-1] if m else ''
dyn=re.findall(r'moe-dynamic-cache:.*?admissions=(\d+).*?hit-rate=([0-9.]+).*?ready-slots=(\d+)',text)
adm,hit,ready=dyn[-1] if dyn else ('0','0','0')
cache=re.findall(r'expert-cache: entries=(\d+) allocated=([0-9.]+) MiB.*?uploaded=([0-9.]+) MiB',text)
entries,alloc,uploaded=cache[-1] if cache else ('0','0','0')
prof=re.findall(r'expert-cache-profile:.*?dynamic-issue=([0-9.]+) ms.*?dynamic-target-sync-calls=\d+ dynamic-target-sync=([0-9.]+) ms',text)
dissue,dsync=prof[-1] if prof else ('0','0')
vals=[int(float(x.strip())) for x in open(gpu) if x.strip()]
print('\t'.join([name,cap,rc,tps('prompt eval time'),tps('eval time'),str(max(vals) if vals else ''),adm,hit,ready,uploaded,dissue,dsync,hashlib.sha256(open(out,'rb').read()).hexdigest()]))
PY
}
run_case baseline none
run_case unlimited unlimited
run_case cap8 8
run_case cap4 4
run_case cap2 2
run_case baseline2 none
cat "$outdir/results.tsv"

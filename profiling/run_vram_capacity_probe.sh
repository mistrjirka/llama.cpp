#!/usr/bin/env bash
set -u
MODEL=${MODEL:-/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf}
OUTDIR=${OUTDIR:-profiling/vram-capacity-probe-2026-07-27}
RESERVE_MIB=${1:-1024}
TOKENS=${TOKENS:-24}
CONTEXT=${CONTEXT:-65536}
mkdir -p "$OUTDIR"
stem="reserve${RESERVE_MIB}"
out="$OUTDIR/$stem.out"
err="$OUTDIR/$stem.err"
gpu="$OUTDIR/$stem.gpu.tsv"
summary="$OUTDIR/$stem.summary.json"
rm -f "$out" "$err" "$gpu" "$summary"

monitor_gpu() {
    while true; do
        printf '%s\t' "$(date +%s.%N)"
        nvidia-smi -i "${NVIDIA_SMI_GPU_INDEX:-1}" \
          --query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory \
          --format=csv,noheader,nounits | tr -d ' ' | tr ',' '\t'
        sleep 0.1
    done
}
monitor_gpu > "$gpu" 2>/dev/null & monitor_pid=$!
rc=0
env \
  -u GGML_MOE_DYNAMIC_STATIC_MAP -u GGML_MOE_DYNAMIC_FORCE_GPU_ONLY \
  -u GGML_MOE_DYNAMIC_EXACT_CPU_FALLBACK -u GGML_MOE_SKELETON_ONLY \
  CUDA_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=1 \
  GGML_EXPERT_CACHE_MIB=auto "GGML_EXPERT_CACHE_RESERVE_MIB=$RESERVE_MIB" \
  GGML_EXPERT_CACHE_PROFILE=1 GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto \
  GGML_MOE_DYNAMIC_THRESHOLD=2 GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=8 \
  GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=32 GGML_MOE_DYNAMIC_PREFILL_OBSERVE=1 \
  GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=16 GGML_MOE_DYNAMIC_WARM_START_TOTAL=512 \
  GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0 \
  GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_PER_LAYER=1 \
  GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_TOTAL=16 \
  GGML_MOE_DYNAMIC_CROSS_LAYER_MIN_OBSERVATIONS=8 \
  GGML_MOE_DYNAMIC_SEGMENTED_LRU=1 GGML_MOE_DYNAMIC_PROTECTED_HITS=2 \
  GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH=1 GGML_MOE_DYNAMIC_SKIP_COLD_BRANCH=1 \
  GGML_MOE_DYNAMIC_ASYNC_PROMOTION=1 GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD=1 \
  timeout 3600s build-v100/bin/llama-completion \
    -m "$MODEL" --fit off -ngl 99 --cpu-moe -c "$CONTEXT" -b 2048 -ub 512 \
    -fa on -dev CUDA0 -t 40 -tb 48 --seed 1 --temp 0 --ignore-eos \
    --single-turn --no-conversation --simple-io --no-display-prompt \
    -p 'Write a compact C function that sums an integer array, then explain one edge case.' \
    -n "$TOKENS" -v > "$out" 2> "$err" || rc=$?
kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true

python3 - "$RESERVE_MIB" "$rc" "$err" "$gpu" "$summary" <<'PY'
import json,re,statistics,sys
reserve,rc,err_path,gpu_path,summary_path=sys.argv[1:]
text=open(err_path,errors='replace').read()
rows=[]
for line in open(gpu_path,errors='replace'):
 f=line.strip().split('\t')
 if len(f)==5:
  try: rows.append(tuple(map(float,f)))
  except ValueError: pass

def last(pattern):
 x=re.findall(pattern,text); return x[-1] if x else None
plans=re.findall(r'moe-dynamic-vram-plan: budget=([0-9.]+) MiB selected-layers=(\d+) per-layer=([0-9.]+) MiB expert-bundle=([0-9.]+) MiB slots=(\d+)/(\d+) planned=([0-9.]+) MiB',text)
allocs=re.findall(r'expert-cache: allocated ([0-9.]+) MiB for (.*?) on .*? \(total ([0-9.]+) / ([0-9.]+) MiB\)',text)
result={
 'reserve_mib':int(reserve),'exit':int(rc),'samples':len(rows),
 'peak_used_mib':max((r[1] for r in rows),default=None),
 'min_free_mib':min((r[2] for r in rows),default=None),
 'gpu_util_mean':statistics.fmean(r[3] for r in rows) if rows else None,
 'plans':[{'budget_mib':float(a),'layers':int(b),'per_layer_mib':float(c),'bundle_mib':float(d),'slots':int(e),'experts':int(f),'planned_mib':float(g)} for a,b,c,d,e,f,g in plans],
 'allocation_count':len(allocs),
 'allocated_mib':float(allocs[-1][2]) if allocs else 0.0,
 'budget_mib':float(allocs[-1][3]) if allocs else None,
 'allocation_failed':'failed to allocate' in text or 'out of memory' in text.lower(),
 'worker':last(r'(moe-promotion-worker: jobs=.*)'),
 'cache':last(r'(moe-dynamic-cache:.*)'),
 'coverage':last(r'(moe-dynamic-coverage:.*)'),
}
result['valid_decode_cache_probe']=result['exit']==0 and result['allocation_count']>0 and result['allocated_mib']>0
open(summary_path,'w').write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not result['valid_decode_cache_probe']: raise SystemExit(3)
PY
summary_rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$summary_rc"

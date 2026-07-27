#!/usr/bin/env bash
set -euo pipefail
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=logit-sanity
mkdir -p "$outdir"
common=(
  -m "$model" -p "$prompt" -n 16 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0
  -t 40 -tb 48 -fit on -fitt 8192 -fitc 512
  -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io
)
run_case() {
  local name=$1; shift
  rm -f "$outdir/$name.out" "$outdir/$name.err" "$outdir/$name.logits"
  env \
    -u GGML_EXPERT_CACHE_VMM \
    -u GGML_EXPERT_CACHE_MANAGED \
    -u GGML_EXPERT_CACHE_MIB \
    -u GGML_MOE_STATIC_SPLIT_SLOTS \
    -u GGML_MOE_STATIC_SPLIT_MAP \
    -u GGML_MOE_DYNAMIC_SPLIT_SLOTS \
    -u GGML_MOE_DYNAMIC_SPLIT_LAYERS \
    -u GGML_MOE_DYNAMIC_THRESHOLD \
    -u GGML_COMPLETION_FORCE_TOKENS \
    GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0 \
    GGML_COMPLETION_LOGITS_DUMP="$PWD/$outdir/$name.logits" \
    "$@" \
    timeout 1800s build-v100/bin/llama-completion "${common[@]}" \
    > "$outdir/$name.out" 2> "$outdir/$name.err"
  printf '%s sha=%s logits_bytes=%s\n' "$name" "$(sha256sum "$outdir/$name.out" | awk '{print $1}')" "$(wc -c < "$outdir/$name.logits")"
}

rm -f "$outdir/baseline.tokens"
env \
  -u GGML_EXPERT_CACHE_VMM \
  -u GGML_EXPERT_CACHE_MANAGED \
  -u GGML_EXPERT_CACHE_MIB \
  -u GGML_MOE_STATIC_SPLIT_SLOTS \
  -u GGML_MOE_STATIC_SPLIT_MAP \
  -u GGML_MOE_DYNAMIC_SPLIT_SLOTS \
  -u GGML_MOE_DYNAMIC_SPLIT_LAYERS \
  GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0 \
  GGML_COMPLETION_LOGITS_DUMP="$PWD/$outdir/baseline.logits" \
  GGML_COMPLETION_TOKEN_DUMP="$PWD/$outdir/baseline.tokens" \
  timeout 1800s build-v100/bin/llama-completion "${common[@]}" \
  > "$outdir/baseline.out" 2> "$outdir/baseline.err"
printf 'baseline sha=%s tokens=%s logits_bytes=%s\n' \
  "$(sha256sum "$outdir/baseline.out" | awk '{print $1}')" \
  "$(wc -l < "$outdir/baseline.tokens")" \
  "$(wc -c < "$outdir/baseline.logits")"

force=(GGML_COMPLETION_FORCE_TOKENS="$PWD/$outdir/baseline.tokens")
run_case noadmit-all \
  GGML_EXPERT_CACHE_MIB=8192 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_THRESHOLD=999 \
  "${force[@]}"
run_case noadmit-layer28 \
  GGML_EXPERT_CACHE_MIB=64 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_SPLIT_LAYERS=28 GGML_MOE_DYNAMIC_THRESHOLD=999 \
  "${force[@]}"
run_case dynamic-layer28 \
  GGML_EXPERT_CACHE_MIB=64 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_SPLIT_LAYERS=28 GGML_MOE_DYNAMIC_THRESHOLD=2.0 \
  "${force[@]}"

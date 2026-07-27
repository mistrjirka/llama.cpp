#!/usr/bin/env bash
set -euo pipefail
model=/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
prompt='Implement a bounded thread-safe LRU cache in C++20 using std::list and std::unordered_map. Include get, put, erase, clear, O(1) operations, and concise unit tests. Return only compilable code.'
outdir=warm-latency-trace
mkdir -p "$outdir"
for spec in 'noadmit 999' 'dynamic 2.0'; do
  read -r name threshold <<< "$spec"
  rm -f "$outdir/$name.jsonl" "$outdir/$name.err" "$outdir/$name.out"
  env -u GGML_EXPERT_CACHE_VMM -u GGML_EXPERT_CACHE_MANAGED -u GGML_MOE_STATIC_SPLIT_SLOTS -u GGML_MOE_STATIC_SPLIT_MAP \
    GGML_EXPERT_CACHE_MIB=64 GGML_MOE_DYNAMIC_SPLIT_SLOTS=8 GGML_MOE_DYNAMIC_SPLIT_LAYERS=28 \
    GGML_MOE_DYNAMIC_THRESHOLD="$threshold" GGML_MOE_DYNAMIC_ADMISSION_RATIO=1.10 GGML_EXPERT_CACHE_PROFILE=1 \
    GGML_MOE_DYNAMIC_TRACE="$PWD/$outdir/$name.jsonl" GGML_CUDA_DISABLE_GRAPHS=1 CUDA_VISIBLE_DEVICES=0 \
    timeout 1800s build-v100/bin/llama-completion \
    -m "$model" -p "$prompt" -n 128 -c 512 -b 2048 -ub 512 -fa on -dev CUDA0 \
    -t 40 -tb 48 -fit on -fitt 1024 -fitc 512 \
    -s 1 --temp 0 --single-turn --no-conversation --no-display-prompt --simple-io -v \
    > "$outdir/$name.out" 2> "$outdir/$name.err"
  echo "=== $name ==="
  grep -E 'moe-dynamic-cache|expert-cache-profile|prompt eval time|eval time|total time' "$outdir/$name.err" | tail -n 10
done

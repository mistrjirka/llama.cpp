#!/usr/bin/env bash
set -euo pipefail
R=/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907
B=$R/candidate-bin
V100=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/Tesla V100/ {print $2}')
export CUDA_VISIBLE_DEVICES="$V100" LD_LIBRARY_PATH="$B"
export GGML_CUDA_VOLTA_Q8_MULTI=1 GGML_CUDA_VOLTA_Q8_MULTI_PACK=4
for VARIANT in old refined; do
 unset GGML_CUDA_VOLTA_Q8_REFINED
 if [ "$VARIANT" = refined ]; then export GGML_CUDA_VOLTA_Q8_REFINED=1; fi
 "$B/test-backend-ops" test -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=2,nr23=\[8,1\]' > "$R/correct-$VARIANT.txt" 2>&1
 grep -q '16/16 tests passed' "$R/correct-$VARIANT.txt"
 ncu --target-processes all --clock-control none --kernel-name-base demangled --kernel-name 'regex:ggml_q8.*partial_kernel' --launch-count 1 --section LaunchStats --section Occupancy --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum --force-overwrite -o "$R/ncu-$VARIANT" "$B/test-backend-ops" test -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=2,nr23=\[8,1\],kv=4096,nb=16,' > "$R/ncu-$VARIANT.log" 2>&1
 ncu --import "$R/ncu-$VARIANT.ncu-rep" --page details > "$R/ncu-$VARIANT.txt"
 echo "$VARIANT";grep -E 'Registers Per|Occupancy|local_op' "$R/ncu-$VARIANT.txt"
done
# Bracket the old/refined paths with standard attention. No profiler for timings.
for VARIANT in standard old refined refined old standard; do
 unset GGML_CUDA_VOLTA_Q8_MULTI GGML_CUDA_VOLTA_Q8_REFINED
 if [ "$VARIANT" != standard ]; then export GGML_CUDA_VOLTA_Q8_MULTI=1; fi
 if [ "$VARIANT" = refined ]; then export GGML_CUDA_VOLTA_Q8_REFINED=1; fi
 N=$(find "$R" -maxdepth 1 -name 'micro-*.txt' | wc -l)
 "$B/test-backend-ops" perf -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=2,nr23=\[8,1\]' > "$R/micro-$N-$VARIANT.txt" 2>&1
 echo "$VARIANT";grep 'FLASH_ATTN_EXT' "$R/micro-$N-$VARIANT.txt" | tail -6
done

#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79
export OPENBLAS_NUM_THREADS=1
D=/models/llama-pxq4-build/divergence-0908
V=benches/pxq4-v100/validation/divergence-0908
M=/models/PXA-Fusion4-35B-GGUF/PXA-Fusion4-35B-PXQ4.gguf
PY=/workspace/.venv-gguf/bin/python
unset GGML_CUDA_CUBLAS_COMPUTE_TYPE GGML_CUDA_PXQ4_PREFILL GGML_CUDA_PXQ4_NATIVE
for mode in integer mmvf; do
    if [ "$mode" = integer ]; then F=0; else F=1; fi
    GGML_CUDA_PXQ4_MMV_F32=$F "$D/teacher-port" "$M" "$V/tokens.json" "$D/port-decode-only-$mode" 1 1 > "$V/port-decode-only-$mode.log" 2>&1
    "$PY" benches/pxq4-v100/validation/analyze_logits.py "$D/pxa-decode-only" "$D/port-decode-only-$mode" --output "$V/pxa-pure-decode-vs-$mode.json"
    PXQ_TRACE_POS=0 PXQ_TRACE_MATCH='^l_out-[0-9]+$|result_output' GGML_CUDA_PXQ4_MMV_F32=$F "$D/teacher-port" "$M" "$V/trace-tokens.json" "$D/port-coarse-$mode" 256 1 > "$V/port-coarse-$mode.log" 2>&1
    "$PY" "$V/analyze_traces.py" "$D/pxa-coarse-trace.trace.json" "$D/port-coarse-$mode.trace.json" --output "$V/coarse-pxa-vs-$mode.json" > "$V/coarse-pxa-vs-$mode.txt"
done
for tool in memcheck racecheck initcheck synccheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 97 "$D/test-mmvf" compact > "$V/mmvf-$tool.log" 2>&1
    tail -3 "$V/mmvf-$tool.log"
done

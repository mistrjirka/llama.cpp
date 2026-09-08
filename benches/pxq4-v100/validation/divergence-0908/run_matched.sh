#!/usr/bin/env bash
# Run with the Development Sandbox gpu:all lock, with no competing GPU inference.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
V="$PWD/benches/pxq4-v100/validation/divergence-0908"
B=${PXQ_BUILD:-/models/llama-pxq4-build}
PXA=${PXA_REPO:-/workspace/pxa}
PB=${PXA_BUILD:-/models/pxa-fullprobs-build}
O=${PXQ_COMPARE_OUTPUT:-$B/divergence-recheck}
MODEL=${PXQ_MODEL:-/models/PXA-Fusion4-35B-GGUF/PXA-Fusion4-35B-PXQ4.gguf}
PY=${PXQ_PYTHON:-/workspace/.venv-gguf/bin/python}
export CUDA_VISIBLE_DEVICES=${PXQ_GPU:-GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79}
export OPENBLAS_NUM_THREADS=1
unset PXA_REFERENCE PXA_ENHANCE PXA_PXQ_MMVQ PXA_PXQ_MMVQ_FUGSPLIT PXQ_SHADOW_MODES PXQ_SHADOW_NATIVE
unset PXQ_TRACE_DIR PXQ_TRACE_POS GGML_CUDA_CUBLAS_COMPUTE_TYPE
mkdir -p "$O"
grep -qx 'GGML_CUDA:BOOL=ON' "$PB/CMakeCache.txt" || { echo 'PXA CUDA build required' >&2; exit 2; }
cmake --build "$B" -j12 --target llama-server > "$O/build-port.log" 2>&1
cmake --build "$PB" -j12 --target llama-server > "$O/build-pxa.log" 2>&1
c++ -O2 -std=c++17 -Iinclude -Iggml/include -Ivendor "$V/teacher_logits.cpp" -L"$B/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$B/bin" -o "$O/teacher-port"
c++ -O2 -std=c++17 -DPXA_HEADER -I"$PXA/include" -I"$PXA/ggml/include" -I"$PXA/vendor" "$V/teacher_logits.cpp" -L"$PB/src" -lllama -L"$PB/ggml/src" -lggml -Wl,-rpath,"$PB/src:$PB/ggml/src" -o "$O/teacher-pxa"
for mode in pxa integer direct-f32; do
    EXE="$O/teacher-port"; [[ $mode == pxa ]] && EXE="$O/teacher-pxa"
    export GGML_CUDA_PXQ4_NATIVE=1 GGML_CUDA_PXQ4_PREFILL=1 GGML_CUDA_PXQ4_MMV_F32=0
    [[ $mode != direct-f32 ]] || export GGML_CUDA_PXQ4_MMV_F32=1
    "$EXE" "$MODEL" "$V/tokens.json" "$O/$mode" 256 1 > "$O/$mode.log" 2>&1
    if [[ ${PXQ_VALIDATE_LONG:-0} == 1 ]]; then
        "$EXE" "$MODEL" "$V/long-independent.json" "$O/long-$mode" 2048 1 > "$O/long-$mode.log" 2>&1
    fi
done
for mode in integer direct-f32; do
    "$PY" "$V/../analyze_logits.py" "$O/pxa" "$O/$mode" --output "$O/pxa-vs-$mode.json"
    if [[ ${PXQ_VALIDATE_LONG:-0} == 1 ]]; then
        "$PY" "$V/../analyze_logits.py" "$O/long-pxa" "$O/long-$mode" --output "$O/long-pxa-vs-$mode.json"
    fi
done

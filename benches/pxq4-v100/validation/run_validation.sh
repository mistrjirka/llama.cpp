#!/usr/bin/env bash
# Run under the Development Sandbox gpu:all resource lock.
# Build/logit artifacts go to the existing build directory, not into model files.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT=$PWD
V="$ROOT/benches/pxq4-v100/validation"
B=${PXQ_BUILD:-/models/llama-pxq4-build}
PY=${PXQ_PYTHON:-/workspace/.venv-gguf/bin/python}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
MODEL=${PXQ_MODEL:-/models/PXA-Fusion4-35B-GGUF/PXA-Fusion4-35B-PXQ4.gguf}
export CUDA_VISIBLE_DEVICES=${PXQ_GPU:-GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79}
export OPENBLAS_NUM_THREADS=1
unset GGML_CUDA_PXQ4_NATIVE GGML_CUDA_PXQ4_PREFILL GGML_CUDA_PXQ4_COALESCED GGML_CUDA_PXQ4_RPB GGML_CUDA_PXQ4_WARP_ROWS GGML_CUDA_PXQ4_VDR
[[ -f "$B/CMakeCache.txt" ]] || { echo 'Configure first with benches/pxq4-v100/run_build.sh' >&2; exit 2; }
cmake --build "$B" -j12 --target llama-server test-gguf test-quantize-fns > "$V/recheck-build.log" 2>&1
build_cuda() {
    "$NVCC" -O2 -std=c++17 -arch=sm_70 -lineinfo -Wno-deprecated-gpu-targets \
        -Iggml/include -Iggml/src -Iggml/src/ggml-cuda "$1" \
        -L"$B/bin" -lggml-cuda -lggml-base -lggml -Xlinker -rpath -Xlinker "$B/bin" -o "$B/$2"
}
build_cuda "$V/test_independent.cu" test-pxq4-independent
build_cuda "$V/test_prefill_edges.cu" test-pxq4-edges
build_cuda benches/pxq4-v100/test_prefill.cu test-pxq4-prefill
for test in dispatch f32; do
    c++ -O2 -std=c++17 -Iggml/include "$V/test_$test.cpp" -L"$B/bin" \
        -lggml-cuda -lggml-base -lggml -Wl,-rpath,"$B/bin" -o "$B/test-pxq4-$test"
done
for test in independent prefill edges dispatch f32; do
    "$B/test-pxq4-$test" > "$V/recheck-$test.log" 2>&1
done
"$B/test-pxq4-edges" duplicate > "$V/recheck-duplicates.log" 2>&1
for tool in memcheck racecheck initcheck synccheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 94 "$B/test-pxq4-independent" compact > "$V/recheck-decode-$tool.log" 2>&1
    compute-sanitizer --tool "$tool" --error-exitcode 94 "$B/test-pxq4-edges" duplicate > "$V/recheck-prefill-$tool.log" 2>&1
done
"$B/bin/test-gguf" > "$V/recheck-gguf.log" 2>&1
"$B/bin/test-quantize-fns" > "$V/recheck-quantization.log" 2>&1
c++ -O2 -std=c++17 -Iinclude -Iggml/include -Ivendor "$V/teacher_logits.cpp" \
    -L"$B/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$B/bin" -o "$B/teacher-logits"
"$PY" "$V/make_fixtures.py"
O="$B/validation-recheck"
mkdir -p "$O"
for mode in reference native native-batched; do
    case "$mode" in
        reference) N=0; P=0; STEP=1;;
        native) N=1; P=1; STEP=1;;
        native-batched) N=1; P=1; STEP=32;;
    esac
    GGML_CUDA_PXQ4_NATIVE=$N GGML_CUDA_PXQ4_PREFILL=$P "$B/teacher-logits" "$MODEL" "$V/fixtures.json" "$O/$mode" 256 "$STEP" > "$V/recheck-logits-$mode.log" 2>&1
done
"$PY" "$V/analyze_logits.py" "$O/reference" "$O/native" --output "$V/recheck-native-vs-reference.json"
"$PY" "$V/analyze_logits.py" "$O/reference" "$O/native-batched" --output "$V/recheck-batched-vs-reference.json"
# Optional long test reuses the pinned code-prefix fixture already in this workspace.
if [[ ${PXQ_VALIDATE_LONG:-0} == 1 ]]; then
    [[ -f "$V/fixture-long.json" ]] || { echo 'Long fixture missing; see make_fixtures.py and VALIDATION.md' >&2; exit 3; }
    "$PY" - "$V/fixture-long.json" "$O/fixture-long.json" "$O/shared-code100k.bin" <<'PY'
import json,pathlib,sys
x=json.loads(pathlib.Path(sys.argv[1]).read_text());x[0]['cache_file']=sys.argv[3]
pathlib.Path(sys.argv[2]).write_text(json.dumps(x))
PY
    GGML_CUDA_PXQ4_NATIVE=1 GGML_CUDA_PXQ4_PREFILL=1 "$B/teacher-logits" "$MODEL" "$O/fixture-long.json" "$O/long-native" 2048 1 > "$V/recheck-long-native.log" 2>&1
    GGML_CUDA_PXQ4_NATIVE=0 GGML_CUDA_PXQ4_PREFILL=0 "$B/teacher-logits" "$MODEL" "$O/fixture-long.json" "$O/long-reference" 2048 1 > "$V/recheck-long-reference.log" 2>&1
    "$PY" "$V/analyze_logits.py" "$O/long-reference" "$O/long-native" --output "$V/recheck-long-comparison.json"
fi
printf 'Kernel, sanitizer, dispatch and GGUF checks passed. Inspect model-distribution metrics separately.\n'

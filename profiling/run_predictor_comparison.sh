#!/usr/bin/env bash
set -euo pipefail

TRACE=${1:-profiling/glm52-scale-2026-07-27/routes-long512.jsonl}
OUTDIR=${2:-profiling/predictor-comparison}
PYTHON_BIN=${PYTHON_BIN:-/workspace/.venv-predictor/bin/python}
MAX_STEPS=${MAX_STEPS:-}
BIN_SIZE=${BIN_SIZE:-64}
WARMUP=${WARMUP:-256}
RUN_HISTORY_LIBRARY=${RUN_HISTORY_LIBRARY:-0}

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN=$(command -v python3)
fi
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy
PY
then
    echo "NumPy is required. Set PYTHON_BIN to a Python environment containing numpy." >&2
    exit 2
fi
if [[ ! -s "$TRACE" ]]; then
    echo "missing route trace: $TRACE" >&2
    exit 2
fi

mkdir -p "$OUTDIR"
max_args=()
if [[ -n "$MAX_STEPS" ]]; then
    max_args=(--max-steps "$MAX_STEPS")
fi

"$PYTHON_BIN" profiling/compare_route_predictors.py \
    "$TRACE" \
    --bin-size "$BIN_SIZE" \
    --budgets 8,16,32 \
    "${max_args[@]}" \
    --output "$OUTDIR/predictor-comparison.json"

"$PYTHON_BIN" profiling/analyze_route_horizons.py \
    "$TRACE" \
    --warmup "$WARMUP" \
    --horizons 1,2,3,4,6,8,12,16 \
    --budgets 8,16,32 \
    "${max_args[@]}" \
    --output "$OUTDIR/route-horizons.json"

"$PYTHON_BIN" profiling/analyze_per_layer_predictor_behavior.py \
    "$TRACE" \
    --bin-size "$BIN_SIZE" \
    "${max_args[@]}" \
    --output "$OUTDIR/per-layer-predictor-behavior.json"

if [[ "$RUN_HISTORY_LIBRARY" != 0 ]]; then
    "$PYTHON_BIN" profiling/analyze_indexed_history_library.py \
        "$TRACE" \
        --histories 1,2,3 \
        --group-sizes 16,32,64 \
        --capacities 32,128 \
        --provider-policy useful \
        "${max_args[@]}" \
        --output "$OUTDIR/indexed-history-library.json"
fi

if command -v g++ >/dev/null 2>&1; then
    g++ -O3 -march=native -std=c++17 \
        profiling/moe_route_predictor_compare_microbench.cpp \
        -o "$OUTDIR/moe-route-predictor-microbench"
    "$OUTDIR/moe-route-predictor-microbench" \
        | tee "$OUTDIR/moe-route-predictor-microbench.txt"
fi

cat <<EOF
Predictor comparison complete:
  $OUTDIR/predictor-comparison.json
  $OUTDIR/route-horizons.json
EOF
if [[ "$RUN_HISTORY_LIBRARY" != 0 ]]; then
    echo "  $OUTDIR/indexed-history-library.json"
fi

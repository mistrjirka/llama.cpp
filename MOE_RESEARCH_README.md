# MoE GPU-residency research orientation

**Last updated:** 1 August 2026
**Primary hardware:** Tesla V100-SXM2-32GB
**Primary objective:** Beat clean upstream `llama.cpp` on route-identical large-MoE coding workloads under the same total VRAM budget.

## Current conclusion

Clean upstream static placement remains the system to beat.

The current CPU/GPU split-cache architecture is functional and can improve route coverage, but it does not convert that coverage into a reliable throughput win. A repeated perfect-future-route Qwen oracle removed 6.45 CPU-dependent layers/token yet tied the reactive cache and remained about 42.5% slower than clean upstream.

The scheduler-level selective-copy-cache pivot was also tested. It is not applicable to current single-GPU Qwen CUDA execution because overflow expert weights are read directly from `CUDA_Host`; there are no per-token expert-slice staging copies for the scheduler to cache.

The recommended next architecture is a kernel-level mixed-residency `MUL_MAT_ID` path that resolves selected experts from either a compact GPU cache or the existing `CUDA_Host` tensor while preserving one ordinary CUDA graph.

## Read these first

1. [`vertical_expert_loading_handoff_with_completion_overlay_2026-07-31(2).md`](vertical_expert_loading_handoff_with_completion_overlay_2026-07-31(2).md) — original theory, implementation history, retained results, and completion-overlay proposal.
2. [`profiling/completion-overlay-feasibility-2026-08-01/ACTIVE_ORACLE_REPORT.md`](profiling/completion-overlay-feasibility-2026-08-01/ACTIVE_ORACLE_REPORT.md) — active perfect-route experiments and the stop decision for the current split architecture.
3. [`profiling/selective-copy-cache-2026-08-01/README.md`](profiling/selective-copy-cache-2026-08-01/README.md) — selective-copy pivot, applicability tests, and degradation-bug analysis.

## Highest-confidence findings

### Clean disabled path

A real patched-runtime regression was found and fixed. Dynamic scheduler graph/name inspection ran even when no cache was active.

```text
Qwen before fix:
clean upstream       ≈ 33.31 tok/s
patched-disabled     ≈ 24.91 tok/s
regression           = -25.22%

Qwen after fix:
clean mean           = 32.8500 tok/s
fixed-disabled mean  = 32.8177 tok/s
difference           = -0.10%
```

Any future feature must preserve an exact upstream-equivalent disabled path.

### Active perfect-route completion oracle

Balanced three-run Qwen result:

| Arm | Mean | Median |
|---|---:|---:|
| Reactive control | **19.0066 tok/s** | 18.8388 |
| Reactive + perfect oracle | 18.8763 tok/s | **18.8647** |

The oracle increased fully covered layers but added more transfer cost than CPU work removed. This is a tie, not a win.

### Model-specific outcome

| Model | Current recommendation |
|---|---|
| Qwen3.5-122B-A10B | Continue only with a new single-graph/kernel-level architecture |
| GLM-5.2 | Park on V100; bundles are too large and routes too incomplete |
| Laguna-S-2.1 | Keep as a correctness/generalization test, not a performance target |

## Bugs and limitations discovered

### Fixed: disabled scheduler overhead

Dynamic bookkeeping polluted the ordinary scheduler path even when caching was disabled.

### Fixed: predictive suffix overlapped warm residents

Warm start filled slots intended as a protected predictive suffix. Oracle packets then displaced useful residents. The opt-in fix is:

```text
GGML_MOE_DYNAMIC_WARM_START_EXCLUDE_FLEX=1
```

### Identified: decode cache can allocate without decode use

Cache buffers may be allocated during graph construction or prompt processing even when measured decode later has zero eligible inputs, zero hits, and zero misses. Decode-only caches should allocate lazily after observing an eligible decode operation.

### Architectural: fixed CPU fallback remains scheduled

The dynamic fixed topology learns exact route residency only after a CPU snapshot segment executes. Cheaply removing the following CPU cold segment requires a graph or kernel redesign. Two bypass prototypes were slower and remain disabled.

### Structural: no upstream selective-copy path for Qwen CUDA

On the tested single-GPU CUDA configuration:

```text
normal overflow -> CUDA_Host -> CUDA reads weights directly
--no-host       -> CPU buffer -> MoE operation runs on CPU
```

No scheduler-managed selected-slice staging path exists between them.

## Benchmark discipline

All performance claims must use:

- Identical model and quantization
- Identical prompt and context
- Forced identical output tokens
- Matching forced-output SHA-256
- Equal total VRAM constraints
- Clean upstream as the primary baseline
- Patched-disabled as a secondary integrity control
- Balanced alternating run order
- All individual measurements, mean, median and variance

A forced-token hash proves the same forced sequence was consumed. It does not prove identical logits, hidden states or router decisions.

## Directory map

```text
profiling/
├── completion-overlay-feasibility-2026-08-01/
│   ├── ACTIVE_ORACLE_REPORT.md
│   ├── qwen-reactive-protected-oracle-repeats/results.json
│   └── active-oracle and baseline scripts/results
├── selective-copy-cache-2026-08-01/
│   ├── README.md
│   ├── run_qwen_equal_vram_sweep.sh
│   └── compact result summaries
├── qwen35-current-mlp-retest-2026-07-31/
└── coding-cache-retest-2026-07-30/
```

## Recommended next experiment

Implement a one-layer Qwen CUDA prototype that preserves the ordinary `MUL_MAT_ID` graph and resolves each selected expert from:

1. A compact device-resident expert cache when present, or
2. The existing `CUDA_Host` source when absent.

First use exact fixed residency, not prediction. Measure one routed layer with and without the cache, including fill cost and lookup overhead. Continue only after the primitive demonstrates positive net value.

Do not return to predictor tuning until the perfect-residency kernel primitive wins.

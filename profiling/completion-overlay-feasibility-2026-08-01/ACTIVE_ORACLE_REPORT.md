# Active Qwen completion-oracle report

**Date:** 1 August 2026
**Hardware:** Tesla V100-SXM2-32GB
**Model:** Qwen3.5-122B-A10B UD-Q3_K_XL
**Workload:** Route-identical 512-token coding continuation

## Verdict

A perfect noncausal route oracle substantially increased fully GPU-covered MoE layers but did not improve throughput. The current CPU/GPU split architecture failed its end-to-end feasibility gate.

Balanced run order was `A B B A B A`, with three runs per arm.

| Arm | Runs, tok/s | Mean | Median |
|---|---|---:|---:|
| Reactive control | 19.4795, 18.8388, 18.7016 | **19.0066** | 18.8388 |
| Reactive + perfect oracle | 18.8048, 18.8647, 18.9594 | 18.8763 | **18.8647** |

The oracle was 0.69% slower by means and 0.14% faster by medians. This is a tie.

Clean upstream Qwen remained approximately 32.85–33.10 tok/s, so the perfect oracle was about 42.5% slower than the target baseline.

## Mechanism result

| Metric | Reactive control | Perfect oracle |
|---|---:|---:|
| Fully GPU-covered layers/run | 5,599.7 | **8,896.3** |
| CPU-dependent layers/token | 37.0418 | **30.5903** |
| Worker traffic/run | 22,872.3 MiB | **45,717.1 MiB** |

The oracle removed 6.45 CPU-dependent layers/token but added 44.71 MiB/token of worker traffic and about 4.48 ms/token of staging-copy time.

Measured median cold-branch work removed was only about:

```text
6.45 layers/token × 0.3272 ms/layer ≈ 2.11 ms/token
```

Transfer cost exceeded the CPU work removed.

## Bugs discovered during the oracle work

### Disabled scheduler pollution — fixed

The patched scheduler performed dynamic graph/name inspection even when no expert cache was attached. This caused a 25.22% Qwen regression. An exact upstream-equivalent early path restored Qwen to within 0.10% of clean upstream.

### Predictive suffix overlap — fixed

Warm start filled the nominal predictive suffix. Oracle packets therefore evicted useful warm residents. The protected-suffix control is:

```text
GGML_MOE_DYNAMIC_WARM_START_EXCLUDE_FLEX=1
```

With a frozen protected core, exact completion packets improved throughput by 8.19% relative to the matching inactive layout. This proved the overlap bug was real.

### Fixed cold fallback — architectural limitation

The fixed topology contains a CPU route-snapshot prefix followed by CPU cold-expert compute. Exact residency is known only after the snapshot. Two bypass prototypes were tested:

| Prototype | Result |
|---|---:|
| Synchronize snapshot, then skip cold suffix | -3.31% |
| Oracle-prevalidated asynchronous prefix | -2.28% |

Both changed transfer/route timing and reduced useful completion coverage. They remain disabled.

## Decision

Stop work on predictor tuning for the current split architecture. Perfect future routes already failed the end-to-end gate.

Do not prioritize:

- Larger predictor MLPs
- More route-history features
- Additional eviction policies
- More aggressive completion traffic
- GLM or Laguna completion packets on this V100

The next research direction must preserve the ordinary single CUDA graph and avoid scheduler-level CPU/GPU expert splitting.

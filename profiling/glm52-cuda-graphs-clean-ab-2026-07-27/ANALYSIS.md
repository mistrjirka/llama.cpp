# GLM-5.2 frozen exact path: CUDA graphs — 2026-07-27

## Decision

Keep CUDA graphs disabled for the current frozen exact GPU-hit plus CPU-miss graph.

CUDA graph capture is functional and reused across decode, but four clean crossed pairs showed no repeatable throughput improvement. The mixed CPU/GPU path still has 152 scheduler splits, so GPU launch capture does not remove the CPU fallback islands or scheduler boundaries that dominate token latency.

## Measurement correction

The original runner required verbose logging to derive decode time from per-token `n_past` timestamps. With CUDA graphs enabled, verbose mode emitted approximately 3572 `CUDA Graph id ... reused` diagnostics per run, confounding the comparison.

`llama-completion` now accepts `GGML_COMPLETION_BENCH_MEASURE_TOKENS` and emits one coarse info-level completion marker:

```text
benchmark decode measurement complete after N tokens; elapsed = ... ms, ... tokens per second
```

The runner prefers this marker, so production measurements can use `VERBOSE=0` and avoid per-token/per-split debug I/O.

Marker validation with verbose mode enabled:

- marker: 5.317797 seconds, 6.017529 tok/s;
- old timestamp method: 5.317713 seconds, 6.017624 tok/s;
- elapsed difference: 0.084 ms over 32 measured tokens.

The coarse marker is therefore consistent with the established timestamp method while working without verbose logs.

## Clean crossed A/B

All accepted runs used:

- frozen static expert map;
- 3072 MiB cache reserve;
- 40 decode threads and 48 batch threads;
- graph-build diagnostics off;
- verbose logging off;
- 16 warmup plus 32 measured tokens;
- identical forced-token stream.

All eight runs produced the exact accepted output SHA-256:

`e165b05c0568da0b099159b3045190e4f33ffbd97698d0a9881b982ead61f514`

| CUDA graphs | Run 2 | Run 3 | Run 4 | Run 5 | Mean | Median | Std. dev. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Enabled | 5.9382 | 5.8879 | 5.9040 | 5.8846 | 5.9036 | 5.8959 | 0.0245 |
| Disabled | 6.1856 | 5.8609 | 5.8345 | 5.7606 | 5.9104 | 5.8477 | 0.1883 |

Mean difference, enabled relative to disabled: **-0.11%**.

The disabled arm had greater host-side run variation, but neither the mean nor median supports a CUDA-graph speedup. A separate verbose enabled run confirmed the accepted 152-split topology and exact output parity.

## Memory

CUDA graphs increased peak measured V100 usage by approximately 36 MiB:

- enabled: 31167 MiB peak used, 1328 MiB minimum free;
- disabled: 31131 MiB peak used, 1364 MiB minimum free.

This remains above the 512 MiB safety gate, but there is no throughput benefit to justify the additional memory.

## Excluded initialization failure

An initial quiet graph-enabled attempt failed during the 14.1 GiB model-buffer allocation while the V100 was already near 31 GiB used by another transient allocation. The process exited before generation and its result is excluded from the A/B. Subsequent runs began with the V100 fully free and completed normally.

## Why capture is neutral

The CUDA backend creates and reuses split-local graphs, but the end-to-end graph is not an all-GPU decode graph. Each token still crosses many CPU/GPU scheduler boundaries and executes exact CPU cold experts. Capturing kernels inside GPU islands cannot eliminate:

- CPU cold-branch matrix work;
- CPU/GPU branch merge dependencies;
- 152 scheduler submissions and boundaries;
- synchronization needed before downstream layers consume the merged result.

CUDA graphs should be revisited only after the exact CPU fallback islands are reduced or replaced by a stable GPU spill path.

## Runner controls

`profiling/run_glm52_scale_case.sh` now records and supports:

- `CUDA_GRAPHS=0|1`;
- `VERBOSE=0|1`;
- the coarse benchmark marker as the preferred throughput source.

The established default remains `CUDA_GRAPHS=0`.

## Next priority

The next useful optimization must reduce CPU cold work or scheduler boundaries. Small CUDA launch optimizations, host affinity changes, logging changes, and CUDA graph capture have all been neutral at the current topology.

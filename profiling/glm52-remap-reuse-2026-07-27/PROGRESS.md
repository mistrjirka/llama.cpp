# GLM-5.2 route-remap reuse result — 2026-07-27

## Repository state

Work is in `/workspace/llama-mainline-cache` on branch `expert-cache-mainline`.

The established performance baseline is committed in:

- `a4700633c perf: tune GLM static vertical cache reserve`

Baseline configuration and result:

- frozen static vertical expert map;
- 3072 MiB cache reserve;
- 5.9630 mean decode tok/s over three runs;
- 152 decode graph splits at batch size 1;
- 1364 MiB minimum measured free VRAM;
- deterministic output SHA-256 `e165b05c0568da0b099159b3045190e4f33ffbd97698d0a9881b982ead61f514`.

## Experiment

The tested optimization removed two canonical-expert-to-cache-slot remap kernel launches per routed layer:

- gate refreshed the per-layer CUDA mapped-ID buffer;
- up reused the gate mapping;
- down reused the same mapping.

The implementation temporarily modified:

- `ggml/include/ggml.h`;
- `ggml/src/ggml.c`;
- `ggml/src/ggml-cuda/ggml-cuda.cu`;
- `src/llama-graph.cpp`.

## Validation

Build:

```sh
cmake --build build-v100 --target llama-completion test-moe-split-backends -j 16
```

Result: exit code 0. Only expected CUDA architecture and pre-existing test warnings were emitted.

Focused test:

```sh
build-v100/bin/test-moe-split-backends
```

Result: exit code 0.

Exact GLM benchmark repetitions:

| Run | Decode tok/s | Decode splits | Minimum free VRAM | Output hash |
|---|---:|---:|---:|---|
| 1 | 6.0247 | 152 | 1364 MiB | exact match |
| 2 | 5.9185 | 152 | 1364 MiB | exact match |
| 3 | 6.0457 | 152 | 1364 MiB | exact match |

Mean: 5.9963 tok/s. Sample standard deviation: 0.0682 tok/s.

Established baseline mean: 5.9630 tok/s. Sample standard deviation: 0.0866 tok/s.

Observed difference: +0.56%.

## Decision

Reject the optimization as not demonstrated.

The measured difference is smaller than ordinary run-to-run variation. The remap kernels are probably hidden behind the substantially larger GPU/CPU expert work and overlap with the CPU cold branch. Retaining a shared mutable mapped-ID buffer between independent graph nodes would add ordering assumptions without a meaningful measured benefit.

The four source edits were reverted. The compact benchmark summaries are retained as negative-result evidence. Because the build was produced before the source revert, rebuild the affected targets before the next benchmark.

## Next priority

Investigate static initialization and cold-path costs rather than tiny remap kernels:

1. bypass or simplify the promotion-worker queue for frozen static-map initialization;
2. measure whether initialization can upload complete expert bundles directly without 4011 queued component jobs;
3. preserve the established 152-split decode topology and 5.9630 tok/s baseline;
4. then profile CPU cold-branch compute, which is more likely to be on the decode critical path.

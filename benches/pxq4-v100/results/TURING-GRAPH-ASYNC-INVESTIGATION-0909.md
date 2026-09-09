# RTX 2080 Ti PXQ graph/async investigation — 2026-09-09

## Production baseline

This investigation starts from `v100-optimized` at `8ab7d7bdd475`, after the validated
no-extra-VRAM Turing PXQ prefill work was merged.  The large persistent decoded-weight
cache and the 192 MiB dequant/GEMM double-buffer remain separate experiments.

Primary test: Ornith-1.5-9B PXQ4, RTX 2080 Ti 22 GB (sm75), `-b 2048 -ub 512 -fa 1`.

## What the CUDA graph showed

Qwen3.5 recurrent layers fan out from `attn_norm` into four independent projections:

- qkv_mixed: 8192 x 4096
- z: 4096 x 4096
- alpha: 32 x 4096
- beta: 32 x 4096

The same model also has 32 dense FFN `gate`/`up` projection pairs.  Full-attention
layers have the normal three-way Q/K/V fan-out.

llama.cpp already has a CUDA concurrent-stream graph optimizer, but its generic discovery
is intentionally limited to 3-way `attn_norm` fan-out and `ggml_nrows(node) <= 1`.
That keeps the mechanism effectively decode-oriented on this model.

## Async experiments

### Generic prompt Q/K/V concurrency: reject

Removing the `nrows <= 1` discovery guard made all eight full-attention layers actually
launch Q/K/V on three CUDA streams at P4096.  It was slower:

- P512: about -0.35%
- P4096: 2843.5 -> 2804.9 tok/s, -1.36%

The guard therefore protects a real Turing resource-contention problem; it is not just a
missing optimization.

### Whole-node FFN gate/up concurrency: neutral

A standalone two-branch PXQ microbenchmark saved about 1.4% for the pair, but extending
the graph optimizer to FFN gate/up was effectively neutral end-to-end (-0.06% in the
matched P4096 comparison).

### GDN recurrent-branch || z: reject

A long two-stream region from `attn_norm` through the recurrent/GDN branch while `z`
runs independently was slower when isolated:

- P512: -0.89%
- P4096: -0.52%
- P8192: -0.73%

### Strict whole-node qkv || z: reject

After adding scheduler allocation dependencies, the intended 24 recurrent two-stream
regions really launched.  P4096 was still about 1.0% slower.  Nsight showed why: although
there was 68.3 ms of branch overlap, 24.9 ms was the qkv FP16->FP32 conversion colliding
with the z GEMM.  Total GEMM and conversion time increased enough to erase the overlap.

### Fine-grained staged overlap: valid, but small in the real graph

A standalone staged pair using the real PXQ dequant and conversion kernels does much better:

1. convert the shared F32 activation to F16 once;
2. dequantize weights on separate streams;
3. run only the tensor-core GEMMs concurrently;
4. join;
5. serialize FP16->FP32 output conversion.

Microbench results:

- GDN qkv+z: 1.1124 -> 0.8833 ms, +25.9% pair-level
- FFN gate+up: 2.0436 -> 1.7461 ms, +17.0% pair-level

Integrated staged fusion really launches the side-stream work, but Turing contention still
reduces the end-to-end result:

- GDN staged only: roughly +0.3% P4096
- FFN staged only: roughly +0.5% P4096
- GDN+FFN staged: roughly +0.7% P4096

This is not attractive enough to merge compared with the simpler activation-reuse change.

## Main useful result: reuse the converted activation

Every large PXQ->cuBLAS prompt matmul converts its F32 activation to F16.  Qwen3.5 often
feeds the same activation into multiple serial projections.  A small persistent F16 scratch buffer can safely reuse that exact conversion while the graph remains serial.
The buffer is allocated to the actual activation size (about 4 MiB for this model at ubatch 512),
with a 32 MiB default ceiling rather than a fixed 32 MiB allocation.

The implementation is physical-sm75 + PXQ + non-view contiguous F32 input + N>=256 only,
and is disabled automatically when the CUDA graph has concurrent-stream events. The cache key
is the logical source tensor and is reset at each graph compute. `GGML_CUDA_TURING_SRC1_REUSE=0`
disables it; `GGML_CUDA_TURING_SRC1_REUSE_MB` controls the allocation ceiling (32 MiB default).

At P4096 Nsight counts changed from:

- F32->F16 conversions: 1984 -> 1024
- conversion kernel time: 71.08 -> 43.56 ms
- PXQ dequant: 293.41 -> 292.84 ms
- GEMM: 642.69 -> 645.01 ms
- F16->F32: 75.90 -> 75.85 ms

The profiled wall time fell 1551.18 -> 1522.73 ms (~1.87%).  Normal benchmarks are affected
by TU102 clock/temperature drift, but all prompt-size sweeps are positive:

| test | off | reuse | gain |
|---|---:|---:|---:|
| P512 | 2805.1 | 2860.1 | +1.96% |
| P1024 | 2870.9 | 2920.9 | +1.74% |
| P2048 | 2871.6 | 2910.9 | +1.37% |
| P8192 | 2783.8 | 2820.9 | +1.33% |
| TG128 | 82.29 | 82.32 | +0.04% |

Cold/no-warmup also improved:

- P512: 1901.4 -> 1916.0 tok/s (+0.77%)
- P4096: 2676.5 -> 2698.2 tok/s (+0.81%)

Cross-tier P2048 was positive for PXQ2/PXQ3/PXQ4-HQ/PXQ6 (+0.37%, +0.78%, +0.75%, +0.63%
respectively).  The production candidate is intentionally PXQ-only; ordinary Q4/Q5 paths
are left unchanged.


V100 regression after the sm75-only hardening:

- targeted RMS_NORM / SSM_CONV / CONCAT suite: 273/273 passed on CUDA0;
- P2048 and TG128 differences versus the merged sm70 baseline were within ~0.2% noise.

## Correctness and safety

Full-vocabulary teacher-forced reuse-off vs reuse-on comparison over 512 positions:

- top-1 agreement: 512/512
- KL / TV / logit RMSE / max logit error: all exactly 0
- PPL ratio: 1.0

Compute Sanitizer memcheck on the active P512 path:

- ERROR SUMMARY: 0 errors
- LEAK SUMMARY: 0 bytes leaked

The reuse path changes neither the dequantized weights nor the FP16 activation values; it
only avoids repeating the same F32->F16 conversion for later consumers of the same tensor.

## Conclusion

On sm75, broad kernel concurrency is usually the wrong lever for this graph.  The tensor-core
GEMMs, dequantizers and bandwidth-bound conversion/elementwise kernels contend too strongly.
The useful graph-level optimization is instead to identify repeated work exposed by fan-out
and reuse it.  Activation conversion reuse is exact, small-memory, prompt-only, and materially
simpler than the async alternatives.

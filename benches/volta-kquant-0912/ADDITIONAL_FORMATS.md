# Additional Volta quant formats — continuation notes

Date: 2026-09-12

## Scope

Continue the generic SM70/V100 quantization study beyond Q2_K–Q6_K, retaining only formats with real downloaded-model validation and clear end-to-end benefit. Synthetic-only wins are not sufficient for integration.

## Integrated baseline

The validated Q2_K/Q3_K/Q4_K/Q5_K/Q6_K conversion work was integrated locally into `v100-optimized` as:

```text
67fb9991 cuda: optimize Volta K-quant conversion
```

The branch is local-only unless separately pushed.

## Recursive model/format inventory

A recursive scan of `/models` found 29 substantial GGUF files (>100 MiB), plus the small metadata shard of Qwen3.8-Flash-Next. Tensor types were inspected from GGUF metadata rather than inferred from filenames.

Important non-K-format coverage found locally includes:

- Qwen3.8-Flash-Next UD-IQ4_XS: ~45.7 GiB IQ4_NL, ~31.6 GiB IQ3_S, ~1.6 GiB IQ4_XS across shards.
- Laguna-S-2.1 UD-IQ2_M: ~17.8 GiB IQ2_XXS, ~12.9 GiB IQ3_XXS, plus IQ2_S/IQ4_XS.
- Gemma-4-26B-A4B: ~5.1 GiB Q5_1.
- PXA-Fusion4 / PXQ format-test models: ~3.7 GiB aggregate MXFP4 plus PXQ1/2/3/4/4HQ/6 coverage.

### Formats deliberately not pursued in this conversion pass

The available IQ2/IQ3/IQ4 validation tensors in Laguna and Qwen Flash Next are overwhelmingly routed `ffn_*_exps` MoE expert tensors (plus a very large embedding tensor in Qwen). They do not provide a clean validation model for the dense standalone `convert.cu` dequantize-to-FP16 path. Do not claim an IQ conversion optimization from synthetic tests alone.

Gemma-4-26B-A4B's 29 Q5_1 tensors are all `ffn_down_exps.weight`, so Q5_1 has the same problem: the locally available real model validates the MoE/MMQ path rather than dense conversion.

PXQ already has dedicated native Volta prefill/decode kernels in this branch and is a separate optimization family. It is not being reworked as part of the generic conversion pass.

No substantial TQ validation model was found in `/models`.

## MXFP4 — retained candidate

### Why MXFP4 was selected

PXA-Fusion4-35B-PXQ4 contains MXFP4 in real dense paths, including token embeddings, attention gates/QKV/output, SSM projections and shared experts. This makes it suitable for validating the generic `convert.cu` MXFP4-to-FP16 path.

Baseline Nsight Systems profile on the V100, PXA-Fusion4 p1024:

- `dequantize_block_mxfp4<half>`: 11.543 ms across 290 launches.
- Share of measured prompt GPU kernel time: ~3.3%.

### Rejected grouping-only experiment

Simply placing multiple unchanged stock 32-thread superblocks in one CTA was not good enough. At representative sizes it ranged from regressions to only ~3% improvement. This was rejected.

### Packed/remapped implementation

The retained experiment remaps one warp as eight four-lane groups, one group per 32-value `block_mxfp4`. Each group reads adjacent quant bytes and each lane emits two aligned 64-bit stores (four low-nibble and four high-nibble FP16 results). FP32 arithmetic and final FP16 rounding are unchanged.

Private rollback/sweep control:

```text
GGML_CUDA_VOLTA_MXFP4_PACKED=0  stock
GGML_CUDA_VOLTA_MXFP4_PACKED=1  packed 32-thread CTA
GGML_CUDA_VOLTA_MXFP4_PACKED=2  packed 128-thread CTA
GGML_CUDA_VOLTA_MXFP4_PACKED=3  packed 256-thread CTA (retained default)
```

Dispatch is restricted to FP16 destination, exact SM70, and at least 128 MXFP4 superblocks. Other paths remain stock.

### Conversion sweep

Final fresh sweep, microseconds per conversion:

| superblocks | stock | packed32 | packed128 | packed256 | packed256 change |
|---:|---:|---:|---:|---:|---:|
| 32 | 2.644 | 2.587 | 2.548 | 2.568 | +2.95% |
| 64 | 2.585 | 2.599 | 2.560 | 2.560 | +0.96% |
| 128 | 2.568 | 2.527 | 2.525 | 2.517 | +2.03% |
| 256 | 2.818 | 2.636 | 2.556 | 2.568 | +9.73% |
| 512 | 3.281 | 3.056 | 2.556 | 2.595 | +26.44% |
| 2,048 | 6.531 | 5.394 | 3.228 | 3.174 | +105.74% |
| 4,096 | 10.711 | 8.444 | 4.092 | 4.227 | +153.39% |
| 32,768 | 69.371 | 52.695 | 30.781 | 30.495 | +127.48% |
| 65,536 | 134.794 | 102.113 | 59.331 | 58.875 | +128.95% |

A 128-superblock production guard is conservative; smaller sizes are not required for the retained real-model benefit.

### Correctness

The standalone checker compares every FP16 output bit against stock for odd/boundary/large superblock counts through 65,536 superblocks (16,777,216 output values) and packed CTA sizes 32/128/256.

- bit mismatches: zero
- Compute Sanitizer memcheck: zero errors

### Real-model performance

PXA-Fusion4-35B-PXQ4 initial ABBA, stock vs packed-256:

- p1024: +0.77%
- p4096: +1.22%
- p16384: +0.84%

Because stock clocks drifted in that run, a stricter six-process interleaved p4096 test was run (`A B A B A B`, five samples/process):

- stock: 2676.676 tok/s, 1530.292 ms
- packed-256: 2716.971 tok/s, 1507.582 ms
- change: **+1.505%**, **-22.711 ms**

## Direct regression matrix versus untouched V100 branch

To check for regressions from the combined new quant changes, the current tree was compared directly against the existing clean V100 baseline worktree at `eae5d0ec` (`/workspace/llama-muse-v100-baseline`). Both sides used the same V100 UUID, Q8_0 K/V cache, p4096, b4096, ub1024, FA on, five timing samples/process, and A-B-B-A process order.

| validation model | old V100 tok/s | current tok/s | change | time change |
|---|---:|---:|---:|---:|
| Ornith-1.5-9B Q2_K | 3026.086 | 3233.579 | **+6.857%** | -86.875 ms |
| Ornith-1.5-9B Q3_K_L | 2983.681 | 3179.414 | **+6.560%** | -84.512 ms |
| Muse-Glimmer Q4_K_XL | 937.962 | 1056.500 | **+12.638%** | -489.988 ms |
| Qwen3.8-27B Q5_K_XL (Q5_K + Q6_K) | 938.499 | 999.671 | **+6.518%** | -267.068 ms |
| PXA-Fusion4 PXQ4 + dense MXFP4 | 2654.124 | 2696.877 | **+1.611%** | -24.473 ms |
| Ornith PXQ4-HQ control | 3201.648 | 3200.539 | -0.035% | +0.447 ms |

The unrelated PXQ4-HQ control is effectively unchanged and its -0.035% delta is far below run-to-run variation. No tested format regressed materially versus the pre-change V100 branch.

The final current-tree MXFP4 path is default-on only for exact SM70 FP16 conversion with at least 128 superblocks; `GGML_CUDA_VOLTA_MXFP4_PACKED=0` restores stock behavior.

The changed `convert.cu` translation unit compiled successfully in the combined SM70+SM75 build and contains both `sm_70` and `sm_75` cubins. Together with the SM70 full build, bitwise checker, sanitizer, real-model ABBA and direct baseline matrix, MXFP4 is accepted for integration.

## Artifacts

```text
/workspace/gguf-format-inventory.json
/workspace/volta-extra-quant-0912/pxa-mxfp4-stock-kernels.txt
/workspace/volta-extra-quant-0912/mxfp4-final-sweep.txt
/workspace/volta-extra-quant-0912/mxfp4-final-bitwise.txt
/workspace/volta-extra-quant-0912/mxfp4-final-memcheck.txt
/workspace/volta-extra-quant-0912/mxfp4-abba/summary.json
/workspace/volta-extra-quant-0912/mxfp4-abab4k/summary.json
/workspace/volta-extra-quant-0912/regression-vs-v100-base/summary.json
```

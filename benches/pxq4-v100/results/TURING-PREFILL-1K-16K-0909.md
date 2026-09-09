# Turing PXQ prefill optimization, 1k-16k (2026-09-09)

## Scope

Target: RTX 2080 Ti / sm75 prompt processing with PXQ, with decode/TG and V100 behavior preserved.
The main measurements use q8_0 KV, Flash Attention, batch 4096 and the same Ornith-1.5-9B source
across PXQ and normal GGUF quants. Fusion4-35B PXQ1/PXQU is used as a larger MoE/no-op control.

## Main finding: use 4096-token ubatches

Nsight Systems on PXQ4 P8192 showed why larger ubatches matter. Moving `ubatch=1024 -> 4096`
reduced repeated work approximately fourfold:

- PXQ dequant kernel time: 293.44 ms -> 73.24 ms (-75.0%)
- PXQ dequant launches: 1472 -> 368
- CUDA kernel launches: 11654 -> 2954 (-74.7%)
- F32->F16 activation conversions: 1984 -> 496
- profiled PP: 2998.8 -> 3159.9 tok/s

Going beyond 4096 was counterproductive. On the 9B PXQ2/PXQ4 tests, 6144 and 8192-token
ubatches were slower at both P8192 and P16384. Fusion4-35B showed the same broad direction from
1024 to 4096, so 4096 is a practical cross-model sm75 sweet spot rather than an Ornith-only setting.

## Accepted exact optimization: PXQ SwiGLU fusion

For dense PXQ FFNs on Turing, the normal F16 cuBLAS path already writes each gate/up GEMM to an
F16 temporary and then expands it to F32. SwiGLU subsequently reads those F32 values. The new
fusion keeps the existing F16 gate/up temporaries, converts their values to float inside the SwiGLU
kernel, and writes the final F32 result. It also converts the shared activation to F16 once for the
pair. This removes redundant output conversion/storage without changing the numerical contract.

The path is guarded to physical sm75, PXQ weights, dense `MUL_MAT` gate/up pairs sharing the same
F32 activation, standard SwiGLU, contiguous tensors, serial graph execution, and N >= 256. It is
default-on; `GGML_CUDA_TURING_PXQ_SWIGLU_FUSION=0` disables it. Decode/TG does not enter the path.

Cross-tier P1k-P16k results, fusion off -> on, tok/s:

| tier | P1024 | P2048 | P4096 | P8192 | P16384 |
|---|---:|---:|---:|---:|---:|
| PXQ2 | 2841.8 -> 2905.5 (+2.24%) | 3420.6 -> 3509.9 (+2.61%) | 3444.2 -> 3531.2 (+2.52%) | 3447.8 -> 3525.3 (+2.25%) | 3291.5 -> 3366.9 (+2.29%) |
| PXQ3 | 2775.8 -> 2858.4 (+2.98%) | 3337.0 -> 3450.3 (+3.40%) | 3354.6 -> 3471.7 (+3.49%) | 3356.6 -> 3468.3 (+3.33%) | 3220.8 -> 3327.6 (+3.32%) |
| PXQ4 | 2754.2 -> 2836.3 (+2.98%) | 3308.8 -> 3437.5 (+3.89%) | 3347.9 -> 3463.9 (+3.46%) | 3344.8 -> 3463.3 (+3.54%) | 3211.8 -> 3320.2 (+3.37%) |
| PXQ4-HQ | 2734.5 -> 2828.1 (+3.42%) | 3298.7 -> 3421.5 (+3.72%) | 3325.5 -> 3456.5 (+3.94%) | 3331.5 -> 3451.7 (+3.61%) | 3199.2 -> 3308.7 (+3.42%) |
| PXQ6 | 2695.7 -> 2768.0 (+2.68%) | 3252.1 -> 3378.6 (+3.89%) | 3306.3 -> 3425.1 (+3.59%) | 3305.9 -> 3423.0 (+3.54%) | 3179.9 -> 3284.4 (+3.29%) |

Fusion4-35B no-op controls were within noise: PXQ1 +0.23%/+0.06% at P4096/P8192 and PXQU
-0.12%/+0.15%. Its expert `MUL_MAT_ID` graph does not match the dense fusion guard.

Full-vocabulary teacher-forced comparison over 128 positions gave byte-identical raw logits:

- top-1 agreement: 128/128
- KL / TV / logit RMSE / max logit error: exactly 0
- PPL ratio: 1.0
- SHA-256 of off/on F32 logit files: identical

## Final PXQ versus similarly-sized normal quants

With the accepted fusion default-on and `ubatch=4096`, the same-model matched-size comparison is:

| pair | P1024 | P2048 | P4096 | P8192 | P16384 |
|---|---:|---:|---:|---:|---:|
| PXQ2 vs Q2_K | +4.26% | +5.88% | +5.47% | +6.18% | +5.78% |
| PXQ3 vs Q3_K_L | +5.10% | +6.03% | +5.16% | +5.17% | +5.00% |
| PXQ4 vs Q4_K_M | +2.72% | +4.29% | +3.53% | +4.55% | +4.59% |
| PXQ6 vs Q5_K_M | -0.22% | +1.99% | +2.63% | +3.32% | +3.59% |

The PXQ4/Q4_K_M files are essentially the same size (~5.65/5.63 GB). PXQ3 is slightly smaller
than Q3_K_L; PXQ2/PXQ6 are close to their matched peers.

## Experiments rejected

- 64x128 direct PXQ->WMMA: ~133 registers/thread and very poor occupancy; ~724 tok/s at P1024.
- 64x64 direct PXQ->WMMA: register use fell to ~72/thread but remained roughly 3x slower than
  exact dequant + cuBLAS. Occupancy was not the fundamental problem.
- direct PXQ4 signed-int8/MMQ prototype: generic MMQ configuration used 255 registers/thread,
  ~62 KiB shared memory and 25% occupancy; ~1.2k tok/s at P4096. Smaller row tiles hit launch/
  resource constraints. Not merged.
- persistent decoded-weight cache: after fixing model ownership and CUDA graph-capture safety it was
  byte-exact, but with the already-optimal ubatch=4096 a 4 GiB cache gave only +0.23% at P8192 and
  +0.40% at P16384 while TG measured ~1.2% lower. Not merged.
- cuBLAS prefill-reuse tiling on sm75: consistently slower; not merged.

## Practical configuration

For sm75 PXQ prompt-heavy workloads, prefer `--batch-size 4096 --ubatch-size 4096` when VRAM permits.
Keep the existing sm75 cuBLAS crossover and activation-conversion reuse. The SwiGLU fusion is
intended to remain default-on on sm75; use `GGML_CUDA_TURING_PXQ_SWIGLU_FUSION=0` only for A/B or
troubleshooting. Do not enable the rejected persistent weight cache or direct WMMA/MMQ experiments.

## Final integration validation

The refactored production candidate was rebuilt in the normal dual-architecture SM70+SM75 build.

- full-vocabulary fusion-off/on raw logits were byte-identical over 128 scored positions (KL/TV/RMSE/max error all 0; identical SHA-256);
- TG128 measured 81.7180 -> 81.6285 tok/s (-0.11%, neutral);
- the same toggle on V100 measured 3385.88 -> 3367.94 PP/s (-0.53%, noise); the fusion code is physically guarded to SM75 and cannot execute on SM70;
- Compute Sanitizer memcheck on an active P512 SM75 fusion path reported **0 errors and 0 leaked bytes**;
- true mixed V100 + RTX 2080 Ti **layer split** loaded and ran PXQ4 successfully; the two-sample performance check was noisy but showed no regression signal;
- production Qwen3.8-27B tensor split (`CUDA1/CUDA0`, 4:5) passed after fixing the optional GDN index-input split-state check;
- PXQ true tensor split currently fails earlier in the generic meta-buffer packing path and is tracked as a separate limitation, not part of this fusion.

The production recommendation is therefore: use layer split for mixed-GPU PXQ today, and do not infer PXQ tensor-parallel support from the Qwen tensor-split validation.

# PXQ family CUDA expansion — 2026-09-09

Branch: `exp/pxq-all-v100`, based on `v100-optimized` commit `474397df7`.
This report covers the runtime/inference port. PXQ quantization/export itself remains in the PXA quantizer.

## Implemented wire formats

The fork now understands the complete PXA slab family used by PXQU:

| Tensor type | GGUF wire id | payload bytes / 32 weights | row metadata | nominal tier |
|---|---:|---:|---:|---:|
| PXQ1 | 248 | 5 | +2 B/row | ~1.26 bpw |
| PXQ2 | 254 | 9 | +2 B/row | ~2.27 bpw |
| PXQ3 | 255 | 13 | +2 B/row | ~3.27 bpw |
| PXQ4 | 252 | 17 | +2 B/row | ~4.27 bpw |
| PXQ4-HQ | 253 | 18 | +2 B/row | ~4.52 bpw |
| PXQ6 | 256 | 21 | +2 B/row | ~5.27 bpw |

`PXQ_UNIVERSAL`/PXQU is a model-level mixture of those tensor codecs rather than a separate tensor codec. The loader, GGUF Python reader, CUDA fallback, native decode, fused gate/up and grouped expert prefill all accept mixed PXQ tensor types.

## V100 performance versus PXA

One Tesla V100-SXM2-32GB, one GPU only, FP16 KV, FA on, batch 2048, ubatch 512. `llama-bench`, PP512 and TG128. PXA is patched CUDA build `896c189`. Results below are short-context throughput and are not directly comparable across rows with different model artifacts.

### Ornith 1.5 9B

PXQ4, PXQ4-HQ and PXQ6 are the user's Hugging Face artifacts. PXQ2/PXQ3 were locally quantized from the hash-verified BF16 source solely to validate those codecs before corresponding uploads exist.

| Quant | fork PP | PXA PP | fork/PXA | fork TG | PXA TG | fork/PXA |
|---|---:|---:|---:|---:|---:|---:|
| PXQ2 | 2846.47 | 2946.68 | 96.6% | 90.65 | 94.19 | 96.2% |
| PXQ3 | 2799.86 | 2931.64 | 95.5% | 72.22 | 77.76 | 92.9% |
| PXQ4 | 2717.88 | 2887.36 | 94.1% | 98.65 | 104.91 | 94.0% |
| PXQ4-HQ | 2754.13 | 2930.32 | 94.0% | 94.35 | 94.48 | 99.9% |
| PXQ6 | 2730.40 | 2875.78 | 94.9% | 67.45 | 72.16 | 93.5% |

The first generic PXQ6 implementation was only 376.8 PP / 11.3 TG. Coalescing the exact full-matrix dequant stores raised PP to ~2780, and a 256-thread panel decode raised TG to ~68.4 while preserving the stored floating codebook. The final table reflects those repaired paths.

### MoE format/PXQU stress models

These are intentionally double-lossy format-validation artifacts derived from Fusion4 PXQ4. They must **not** be used as model-quality comparisons.

| Artifact | fork PP | PXA PP | fork/PXA | fork TG | PXA TG | fork/PXA |
|---|---:|---:|---:|---:|---:|---:|
| PXQ1 35B format test | 2453.34 | 2550.81 | 96.2% | 117.71 | 127.76 | 92.1% |
| mixed PXQU 35B | 1886.12 | 1600.59 | **117.8%** | 102.61 | 66.56 | **154.2%** |

The mixed PXQU model deliberately cycles PXQ1/PXQ2/PXQ3/PXQ4/PXQ6 independently across gate/up/down tensors in every MoE layer. It therefore exercises mixed-type fused gate/up dispatch, not merely multiple uniform files.

## Numerical validation

Independent exact dequant-to-cuBLAS fallback remains available with `GGML_CUDA_PXQ_NATIVE=0` and is used as the local stored-weight reference. On the final 16-position/full-vocabulary probes:

| Path | top-1 vs exact | mean KL vs exact | note |
|---|---:|---:|---|
| PXQ2 panel decode | 100% | 4.17e-5 | closer to exact than PXA on this probe |
| PXQ3 panel decode | 100% | 1.56e-5 | closer to exact than PXA on this probe |
| PXQ6 panel decode | 100% | 9.71e-6 | closer to exact than PXA on this probe |
| PXQ4-HQ fast DP4A | 100% | 4.20e-4 | PXA-like fast arithmetic; 94.7 TG/s |
| PXQ4-HQ floating control | 100% | 1.32e-5 | higher fidelity; ~71.6 TG/s |
| mixed PXQU final | 100% | 1.25e-3 | deliberately extreme mixed/double-lossy file |

PXQ4-HQ fast versus PXA itself is 100% top-1 on the same probe, mean KL `3.51e-4`.
The final mixed PXQU path is also 100% top-1 versus PXA on the probe (mean KL `1.77e-3`).
PXQ1 on its deliberately double-lossy stress artifact is noisier: 15/16 top-1 versus exact, and PXA shows the same 15/16 top-1 behavior.

Compute Sanitizer memcheck with CUDA graphs disabled completed on both the uniform PXQ4-HQ fast path and the mixed PXQU model with **0 errors and 0 leaked bytes**.

## Execution strategy

- PXQ4 keeps the previously validated V100-specific coalesced Q8_1/DP4A kernel and grouped Volta WMMA prefill.
- PXQ4-HQ has a dedicated PXQ4-style DP4A kernel adapted to its two scale bytes/row and 128-byte code offset. A floating-book control remains available.
- PXQ1/PXQ2/PXQ3/PXQ6 default to a 256-thread direct panel decode that reads FP32 activations and the original stored floating codebooks. A generic Q8_1/s8 path is retained as an A/B control but is not the default.
- Dense prefill defaults to exact coalesced dequant-to-F16 + cuBLAS on V100. The experimental direct dense WMMA path remains opt-in because it was slower.
- Routed MoE prefill uses the shared grouped WMMA path on V100.
- PXQU fused gate/up accepts different PXQ codecs and physical strides for the two tensors after validating each slab layout independently.

## Diagnostic controls

- `GGML_CUDA_PXQ_NATIVE=0`: force exact dequant/cuBLAS fallback for all PXQ tiers.
- `GGML_CUDA_PXQ_FLOAT_DECODE=0`: select the lower-fidelity generic Q8_1/s8 path for PXQ1/2/3/6.
- `GGML_CUDA_PXQ4HQ_MMV_F32=1`: use floating-book PXQ4-HQ decode instead of the fast DP4A path.
- `GGML_CUDA_PXQ_DENSE_PREFILL=1`: opt into the experimental dense direct-WMMA path (default off after V100 A/B).
- `GGML_CUDA_PXQ_PREFILL=0`: disable shared native routed-MoE prefill for non-PXQ4 tiers.

## Regression note

A broad stock CUDA `MUL_MAT`/`MUL_MAT_ID` sweep produced one Q5_1 tolerance outlier once (`ERR=0.000590638` at a `0.0005` threshold). The exact stock operator immediately passed when isolated and then passed 20/20 repeated runs. It does not touch PXQ dispatch. A final post-HQ full sweep is recorded separately in `pxq-all-backend-matmul-final2.log`.

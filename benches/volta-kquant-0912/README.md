# Volta K-quant optimization study

Date: 2026-09-12

This work is now scoped as a general CUDA SM70 / V100 quantization-path study rather than a Muse-Glimmer-specific optimization. Muse Glimmer remains the first production validation model because its Q4_K-heavy dense path exposed the opportunity clearly. Model-specific attention tuning is deliberately deferred until the quantization study is complete.

## Goal

Improve prompt-processing throughput and TTFT for dense K-quant weights on V100 without changing dequantized FP16 values, hurting token generation, or regressing shorter prompts.

Primary mechanism under study: improve the standalone quantized-weight -> FP16 conversion used before cuBLAS by fixing poorly coalesced output access and increasing independent work per CTA. Register pressure and occupancy are constraints/diagnostics, not optimization objectives by themselves.

## Worktree

- worktree: `/workspace/volta-quant-exp`
- branch: `perf/volta-kquant`
- base: `eae5d0ec`
- target: Tesla V100-SXM2-32GB, UUID `GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79`
- Q8_0 K/V cache for model-level validation
- no commit or push without explicit approval

## Q4_K: proven reference case

The Muse experiment established the transferable optimization method. Packed/coalesced Q4_K -> FP16 stores plus four quant blocks per 128-thread CTA changed the large conversion kernel from 1.170 ms to 0.452 ms in Nsight Compute, raised measured DRAM throughput from 389.7 to 751.6 GB/s, and reduced LG-throttle strongly.

On Muse Glimmer 30B at restored 100k + 1k append, the Q4_K conversion change alone measured:

| path | PP tok/s | wall TTFT | TG tok/s |
|---|---:|---:|---:|
| stock | 544.400 | 1896.10 ms | 28.972 |
| packed Q4_K | 579.295 | 1785.64 ms | 29.004 |
| change | +6.41% | -5.83% | noise |

The later Muse Flash-Attention staging result is separate from this quantization study.

## Q5_K: current target

Q5_K has the same broad issue: stock threads emit two adjacent FP16 values at two regions separated by 32 elements. The experimental kernel packs each adjacent FP16 pair into one 32-bit store and groups multiple 64-thread Q5_K blocks into a CTA.

Bitwise checker covers block counts `1,2,3,4,5,7,8,9,97,348160`, CTA sizes 64/128/256, including 89,128,960 outputs in the largest case. Current result: zero bit mismatches.

Fresh V100 conversion-only sweep:

| Q5_K blocks | stock 64T | packed 64T | packed 128T | packed 256T |
|---:|---:|---:|---:|---:|
| 20,480 | 32.031 us | 31.698 us | 21.985 us | 21.985 us |
| 122,880 | 177.249 us | 176.343 us | 117.729 us | 116.669 us |
| 348,160 | 501.617 us | 497.295 us | 329.042 us | **325.222 us** |

At the large shape, packed-256 improves conversion throughput by 54.24% over stock. The gain mostly comes from grouping/concurrency plus the packed stores; packed-64 alone is only about 1% faster.

Same-size Nsight Compute attribution at 348,160 Q5_K blocks:

| metric | stock 64T | packed 256T |
|---|---:|---:|
| profiled duration | 588.13 us | 324.45 us |
| memory throughput | 527.32 GB/s | 735.24 GB/s |
| DRAM peak utilization | 58.71% | 81.57% |
| registers/thread | 29 | 23 |
| achieved occupancy | 26.47% | 84.41% |
| eligible warps/scheduler | 0.46 | 2.51 |
| cycles with no eligible warp | 68.75% | 45.26% |
| excessive global sectors | 40% | 3% |

The profile supports the same mechanism as Q4_K: coalescing removes most wasted sectors and grouping enough independent blocks per CTA dramatically improves latency hiding. The register reduction is useful but is not sufficient by itself to explain the gain.

Primary Qwen3.8-27B `UD-Q5_K_XL` production-style test, restored 100k + 1k append + TG64, ABBA order with six retained requests per variant:

| path | PP tok/s | prompt time | TG tok/s |
|---|---:|---:|---:|
| stock | 431.956 | 2315.13 ms | 16.408 |
| packed-256 | 437.682 | 2284.77 ms | 16.381 |
| change | **+1.33%** | **-1.31% (-30.37 ms)** | -0.16% |

All retained outputs have the same token hash. The TG difference is small enough to treat as run noise unless a larger decode test reproduces it.

Final short-context ABBA gate, six timing samples per side:

| prompt | stock | packed-256 | change |
|---:|---:|---:|---:|
| 1,024 | 958.083 tok/s | 991.382 | **+3.48%** |
| 4,096 | 1055.263 | 1060.273 | **+0.47%** |
| 16,384 | 969.159 | 978.280 | **+0.94%** |

No tested prompt length regressed. A second, isolated Q5_K ABBA on Ornith-1.5-9B `Q5_K_M` confirms that the result is not Qwen-specific:

| prompt | stock | packed-256 | change |
|---:|---:|---:|---:|
| 1,024 | 3015.958 tok/s | 3178.768 | **+5.40%** |
| 4,096 | 3365.737 | 3414.435 | **+1.45%** |
| 16,384 | 3143.523 | 3182.098 | **+1.23%** |

Together with the 100k+1k gain, bitwise checker, zero-error Compute Sanitizer run, and same-size Nsight attribution, Q5_K packed-256 is a validated Volta candidate. The selector remains experimental until the multi-quant study is cleaned up as a whole.

## Q2_K: grouped-stock candidate

Q2_K benefits even more from grouping unchanged 64-thread stock blocks into a 256-thread CTA. This preserves the stock arithmetic and output mapping exactly.

Conversion-only sweep:

| Q2_K blocks | stock 64T | grouped 128T | grouped 256T |
|---:|---:|---:|---:|
| 20,480 | 37.345 us | 20.936 us (+78.38%) | **19.144 us (+95.08%)** |
| 122,880 | 209.567 us | 107.315 us (+95.28%) | **100.475 us (+108.58%)** |
| 348,160 | 530.760 us | 283.648 us (+87.12%) | **279.910 us (+89.62%)** |

Bitwise comparison against stock passed at block counts `1,2,3,4,5,7,8,9,97,348160` for both 128- and 256-thread CTAs; Compute Sanitizer memcheck reports zero errors.

Nsight on the large grouped kernel measures 735.69 GB/s, 82.05% DRAM utilization and 88.57% achieved occupancy. The stock profile was only 19.68% occupied and spent 72.22% of scheduler cycles with no eligible warp. Excessive global sectors remain about 6%, confirming this is primarily a latency-hiding/CTA-granularity win.

Real-model ABBA on `/models/ornith9b-normal-quants-0909/Ornith-1.5-9B-Q2_K.gguf`, Q8_0 K/V cache:

| prompt | stock | grouped-256 | change |
|---:|---:|---:|---:|
| 1,024 | 3011.288 tok/s | 3163.850 | **+5.07%** |
| 4,096 | 3373.005 | 3408.977 | **+1.07%** |
| 16,384 | 3171.820 | 3191.277 | **+0.61%** |

No tested prompt length regressed.

## Q6_K: grouped-stock candidate

Q6_K does not need a new arithmetic mapping to get a large kernel-level gain. The candidate simply runs multiple independent stock 64-thread Q6_K blocks in one 256-thread CTA, preserving the exact stock helper.

Conversion-only sweep:

| Q6_K blocks | stock 64T | grouped 128T | grouped 256T |
|---:|---:|---:|---:|
| 20,480 | 31.616 us | 22.779 us (+38.80%) | **22.513 us (+40.44%)** |
| 122,880 | 176.323 us | 122.363 us (+44.10%) | **120.822 us (+45.94%)** |
| 348,160 | 522.824 us | 341.484 us (+53.10%) | **336.988 us (+55.15%)** |

Bitwise comparison against stock passed for block counts `1,2,3,4,5,7,8,9,97,348160` at both 128 and 256 threads, including 89,128,960 outputs in the largest case. Compute Sanitizer memcheck reports zero errors.

The optimization survives the complete Q6_K -> FP16 -> cuBLAS path on five 1000-token Qwen-sized matrices:

| matrix (m x k) | stock | grouped-256 | runtime change |
|---|---:|---:|---:|
| 17408 x 5120 | 3152.08 us | 2921.43 us | **-7.32%** |
| 5120 x 17408 | 3134.69 us | 2878.20 us | **-8.18%** |
| 6144 x 5120 | 1202.43 us | 1118.68 us | **-6.96%** |
| 5120 x 6144 | 1121.15 us | 1046.12 us | **-6.69%** |
| 1024 x 5120 | 294.18 us | 279.38 us | **-5.03%** |

Nsight on the large grouped kernel measures 744.95 GB/s, 83.63% DRAM utilization and 88.33% achieved occupancy. The stock profile was only 20.29% occupied with 75.47% of scheduler cycles having no eligible warp. The uncoalesced-sector fraction remains roughly 15%, so this win is primarily latency hiding and CTA scheduling rather than a new memory layout.

A real-model isolated Q6_K ABBA used Ornith-1.5-9B `Q5_K_M`, whose dense Q6_K tensors include 16 `ffn_down`, 12 `attn_qkv`, 4 `attn_v`, and `output.weight` tensors. All other new K-quant selectors were forced off:

| prompt | stock | grouped-256 | change |
|---:|---:|---:|---:|
| 1,024 | 3027.302 tok/s | 3043.027 | **+0.52%** |
| 4,096 | 3358.688 | 3362.427 | **+0.11%** |
| 16,384 | 3134.127 | 3139.192 | **+0.16%** |

The model-wide gain is naturally smaller because Q6_K is only a subset of this mixed quant, but no prompt length regressed. An earlier Ornith-35B `AD-Q6_K-Q5_K` probe was discarded: all Q6_K tensors in that file are routed `ffn_down_exps` MoE weights, so it is not a valid test of the dense dequantize-to-FP16 path optimized here.

## Q3_K: packed-load/store candidate

Q3_K was the exception to simple CTA grouping: stock already reaches about 90% occupancy, but Nsight reported roughly 60% excessive global sectors, with only about 4.3/32 bytes used per load sector and 8/32 bytes per store sector.

The retained kernel keeps the stock thread/output mapping, reads each thread's four q/hmask bytes as aligned 16-bit pairs, preserves the stock FP32 arithmetic and `__float2half_rn` rounding, and emits all four FP16 outputs with one aligned 64-bit store. `block_q3_K` is 110 bytes, so 32-bit input loads were deliberately avoided because odd blocks are only 2-byte aligned.

Conversion-only sweep:

| Q3_K blocks | stock | packed 64T | packed 128T | packed 256T |
|---:|---:|---:|---:|---:|
| 20,480 | 42.849 us | 38.257 | 24.945 | **24.489 (+74.97%)** |
| 122,880 | 232.028 us | 196.946 | 124.017 | **121.912 (+90.32%)** |
| 348,160 | 631.562 us | 568.064 | 349.215 | **334.694 (+88.70%)** |

The SASS contains a single `STG.E.64.SYS` for the output. Nsight reports only 1.37% excessive sectors for packed-256 versus about 60% for stock, 86.35% achieved occupancy, and 2.08 eligible warps/scheduler. A separate warp-remapped experiment was correct but slower at the largest shape, so it was removed.

Bitwise comparison passed for block counts `1,2,3,4,5,7,8,9,97,348160` at 64/128/256 threads, including 89,128,960 outputs in the largest case. Compute Sanitizer memcheck reports zero errors.

Real-model ABBA on `/models/ornith9b-normal-quants-0909/Ornith-1.5-9B-Q3_K_L.gguf`, Q8_0 K/V cache:

| prompt | stock | packed-256 | change |
|---:|---:|---:|---:|
| 1,024 | 2989.452 tok/s | 3135.157 | **+4.87%** |
| 4,096 | 3343.160 | 3376.083 | **+0.98%** |
| 16,384 | 3159.045 | 3187.593 | **+0.90%** |

No tested prompt length regressed.

## Small-work guard

A dedicated sweep showed launch overhead dominates for tiny conversions. Some winners fluctuate a few percent slower below 128 blocks, while all five are faster at 512 blocks and strongly faster at 2048 blocks:

| quant | change at 512 blocks | change at 2048 blocks |
|---|---:|---:|
| Q2_K | +2.93% | +54.48% |
| Q3_K | +17.29% | +64.15% |
| Q4_K | +16.41% | +118.27% |
| Q5_K | +2.31% | +55.03% |
| Q6_K | +0.85% | +66.48% |

Production dispatch therefore keeps the stock path below 512 quant blocks. Model weight matrices exercised here are comfortably above that threshold.

## Why the formats differ

The initial screening grouped independent stock 64-thread blocks without changing their arithmetic:

| quant | grouped 128T | grouped 256T | diagnosis |
|---|---:|---:|---|
| Q2_K | +97.11% | +109.00% | strongly latency/concurrency limited |
| Q3_K | -5.75% | -0.79% | grouping alone fails; memory layout was the issue |
| Q5_K | +13.72% | +17.48% | grouping helps; packed stores add more |
| Q6_K | +46.86% | +48.68% | strongly latency/concurrency limited |

Stock Nsight measurements support this split: Q2_K and Q6_K had only about 20% achieved occupancy and spent over 70% of scheduler cycles with no eligible warp, while Q3_K already had about 90% occupancy but roughly 60% excessive global sectors.

## Combined integration gate

With all five optimized K-quant paths enabled together, Qwen3.8-27B `UD-Q5_K_XL` at restored 100k + 1k append + TG64 measured:

| path | PP tok/s | prompt time | TG tok/s |
|---|---:|---:|---:|
| all stock | 431.838 | 2315.80 ms | 16.412 |
| all optimized | 441.216 | 2266.47 ms | 16.379 |
| change | **+2.17%** | **-2.13% (-49.33 ms)** | -0.20% |

All twelve retained outputs had the same token hash. Live `nvidia-smi` verification during the run showed the V100 UUID `GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79` at 97% utilization while the RTX 2080 Ti remained idle. A separate 128-token decode ABBA measured 27.6422 tok/s stock versus 27.6438 tok/s optimized, a **+0.006%** difference. The earlier -0.20% TG delta was therefore run noise; no decode regression was reproduced.

Combined short-context ABBA, six timing samples per side:

| prompt | all stock | all optimized | change |
|---:|---:|---:|---:|
| 1,024 | 952.221 tok/s | 1014.488 | **+6.54%** |
| 4,096 | 1046.418 | 1066.397 | **+1.91%** |
| 16,384 | 968.395 | 982.560 | **+1.46%** |

No tested short prompt regressed. The combined gain is larger than Q5_K alone at all three lengths, showing that the additional K-quant paths contribute in the mixed `UD-Q5_K_XL` model rather than merely passing isolated microbenchmarks.

## Integration state

- Q2_K: grouped-256 default on SM70 FP16 for `nb >= 512`.
- Q3_K: packed-256 default on SM70 FP16 for `nb >= 512`.
- Q4_K: packed-128 default on SM70 FP16 for `nb >= 512`; selector renamed from Muse-specific to `GGML_CUDA_VOLTA_Q4K_PACKED`.
- Q5_K: packed-256 default on SM70 FP16 for `nb >= 512`.
- Q6_K: grouped-256 default on SM70 FP16 for `nb >= 512`.
- Other architectures, destination types, small conversions, and explicit mode `0` retain stock behavior.
- The retained kernels are bitwise-equivalent to stock in the dedicated checkers; Q2/Q3/Q5/Q6 sanitizer runs report zero errors, and Q4 had already passed the same gates in the Muse study.
- Model-specific Muse Flash Attention optimization remains separate and resumes only after the combined K-quant regression gate is complete.


## Final build gate

The cleaned quantization diff builds successfully for a combined `CMAKE_CUDA_ARCHITECTURES=70;75` CUDA target. This verifies that the exact-SM70 dispatch remains compile-safe in the normal V100+Turing build configuration.

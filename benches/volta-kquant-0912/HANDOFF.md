# Volta K-quant conversion optimization handoff

Date: 2026-09-12

## Scope

This work reframes the Muse Q4_K conversion result as a generic SM70/V100 K-quant optimization pass. It covers the dense dequantize-to-FP16 path for Q2_K, Q3_K, Q4_K, Q5_K and Q6_K. Muse-specific Flash Attention tuning is intentionally separate.

This does **not** cover IQ/TQ formats; those use different dequantizers and should be treated as a separate study if desired.

## Repository state

- Worktree: `/workspace/volta-quant-exp`
- Branch: `perf/volta-kquant`
- Base: `eae5d0ec20767a68217738ec7fbdb08da35eec8b`
- Target GPU: Tesla V100-SXM2-32GB, UUID `GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79`
- RTX 2080 Ti UUID `GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29` remained idle during measured V100 runs.
- Q8_0 K/V cache was used for model-level gates.
- No commit or push was made.
- After cleanup, the only tracked source change is `ggml/src/ggml-cuda/convert.cu`.
- Benchmark/checker scripts, summaries and profiler artifacts are under `benches/volta-kquant-0912/` and remain untracked unless explicitly selected for integration.

## Production dispatch

All optimized paths are limited to:

- FP16 dequantization destination
- exact SM70 / Volta
- at least 512 quant blocks

Stock behavior remains for other architectures, destination types and small work.

Per-format rollback/sweep variables remain available on the private branch:

- `GGML_CUDA_VOLTA_Q2K_GROUPED`
- `GGML_CUDA_VOLTA_Q3K_PACKED`
- `GGML_CUDA_VOLTA_Q4K_PACKED`
- `GGML_CUDA_VOLTA_Q5K_PACKED`
- `GGML_CUDA_VOLTA_Q6K_GROUPED`

Mode `0` restores stock. Retained defaults are Q2=256T, Q3=256T, Q4=128T, Q5=256T, Q6=256T.

## Retained implementations

### Q2_K

Group four unchanged stock 64-thread blocks into one 256-thread CTA. Arithmetic and output mapping are unchanged.

Large conversion sweep: 530.760 us -> 279.910 us. Nsight: 735.69 GB/s, 82.05% DRAM utilization, 88.57% achieved occupancy.

Ornith-1.5-9B Q2_K model ABBA:

| prompt | stock | optimized | change |
|---:|---:|---:|---:|
| 1,024 | 3011.288 | 3163.850 | +5.07% |
| 4,096 | 3373.005 | 3408.977 | +1.07% |
| 16,384 | 3171.820 | 3191.277 | +0.61% |

### Q3_K

Grouping alone failed because stock already had high occupancy. The retained kernel keeps stock arithmetic/thread mapping, loads q/hmask as aligned 16-bit pairs and packs each thread's four FP16 outputs into one aligned 64-bit store.

Large conversion sweep: 631.562 us -> 334.694 us. Excessive global sectors drop from about 60% to 1.37%; packed-256 reaches 2.08 eligible warps/scheduler.

Ornith-1.5-9B Q3_K_L model ABBA:

| prompt | stock | optimized | change |
|---:|---:|---:|---:|
| 1,024 | 2989.452 | 3135.157 | +4.87% |
| 4,096 | 3343.160 | 3376.083 | +0.98% |
| 16,384 | 3159.045 | 3187.593 | +0.90% |

### Q4_K

Muse-discovered packed path generalized to `GGML_CUDA_VOLTA_Q4K_PACKED`: 32-bit quant loads, packed FP16 stores, four independent blocks in a 128-thread CTA.

Large Muse conversion profile: 1.170 ms -> 0.452 ms, 389.74 -> 751.56 GB/s, LG-throttle 44.86 -> 14.20 cycles/instruction.

Muse 100k restored + 1k append, Q4-only change:

- PP: 544.400 -> 579.295 tok/s, +6.41%
- wall TTFT: 1896.10 -> 1785.64 ms, -5.83%
- TG: unchanged within noise
- token hash unchanged

### Q5_K

Packed 16-bit q/hi-bit loads, adjacent FP16 results packed into aligned 32-bit stores, four independent 64-thread blocks per 256-thread CTA.

Large conversion: 501.617 us -> 325.222 us. Nsight: 527.32 -> 735.24 GB/s, excessive sectors 40% -> 3%, achieved occupancy 26.47% -> 84.41%.

Qwen3.8-27B UD-Q5_K_XL 100k restored + 1k append:

- PP: 431.956 -> 437.682 tok/s, +1.33%
- prompt time: 2315.13 -> 2284.77 ms, -30.37 ms
- output hash unchanged

Qwen short gate: +3.48%, +0.47%, +0.94% at 1k/4k/16k.

Independent Ornith-1.5-9B Q5_K_M isolated Q5 gate:

- 1k: +5.40%
- 4k: +1.45%
- 16k: +1.23%

### Q6_K

Group four unchanged stock 64-thread blocks into one 256-thread CTA. No arithmetic or output remap is required.

Large conversion: 522.824 us -> 336.988 us. Nsight: 744.95 GB/s, 83.63% DRAM utilization, 88.33% achieved occupancy.

Full Q6_K -> FP16 -> cuBLAS on 1000-token Qwen-sized matrices improved runtime by 5.03-8.18%.

Real-model isolated Q6 gate used dense Q6 tensors inside Ornith-1.5-9B Q5_K_M:

- 1k: +0.52%
- 4k: +0.11%
- 16k: +0.16%

The smaller model-wide effect is expected because Q6 is only a subset of that mixed quant. A cancelled Ornith-35B probe must not be used as evidence: all Q6 tensors there are routed `ffn_down_exps` MoE weights, so that file does not exercise the dense conversion path reliably.

## Small-work guard

The dedicated sweep showed the optimized launches can lose for very small conversions, while every retained path was faster at 512 blocks and strongly faster at 2048 blocks:

| quant | change at 512 blocks | change at 2048 blocks |
|---|---:|---:|
| Q2_K | +2.93% | +54.48% |
| Q3_K | +17.29% | +64.15% |
| Q4_K | +16.41% | +118.27% |
| Q5_K | +2.31% | +55.03% |
| Q6_K | +0.85% | +66.48% |

The production threshold is therefore 512 blocks.

## Combined integration gate

All five optimized paths enabled together on Qwen3.8-27B UD-Q5_K_XL, restored 100k + 1k append + TG64:

| path | PP tok/s | prompt time | TG tok/s |
|---|---:|---:|---:|
| stock | 431.838 | 2315.80 ms | 16.412 |
| optimized | 441.216 | 2266.47 ms | 16.379 |
| change | +2.17% | -2.13% (-49.33 ms) | -0.20% |

All twelve retained outputs had the same token hash. A dedicated decode ABBA then measured 27.6422 tok/s stock versus 27.6438 tok/s optimized (+0.006%), proving the -0.20% TG difference above was run noise.

Combined short-context ABBA:

- 1k: +6.54%
- 4k: +1.91%
- 16k: +1.46%

No tested prompt length regressed.

## Correctness

Dedicated bitwise checkers compare optimized FP16 output bits against stock for odd, boundary and very large block counts. Retained Q2/Q3/Q4/Q5/Q6 candidates report zero mismatches. Q2/Q3/Q5/Q6 Compute Sanitizer memcheck runs report zero errors; Q4 passed the same sanitizer gate in the Muse study.

The cleaned SM70 production build compiles and `git diff --check` passes. Alignment assumptions for Q3/Q4/Q5 vectorized accesses are now encoded with compile-time assertions in `convert.cu`.

## Cleanup/integration guidance

- Keep the K-quant conversion work independent of Muse Flash Attention so either can be reverted separately.
- Do not restore the model-specific `test-backend-ops` performance cases; exact-shape benchmarks live under `benches/volta-kquant-0912/`.
- Keep mode `0` rollback knobs on the private fork unless there is a reason to remove them later.
- Do not change dequant arithmetic/order during cleanup; the bitwise equivalence depends on preserving it.
- Do not commit or push without explicit approval.

## Next step

The SM70+SM75 CUDA compile gate passed. This K-quant pass is ready for review/integration. The next phase is Muse-specific optimization, primarily the D128/GQA16 Flash Attention search described in `/workspace/muse-glimmer-exp/benches/muse-glimmer-0912/HANDOFF.md`.

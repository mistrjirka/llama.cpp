# Upstream sync and regression check — 2026-09-11

`v100-optimized` was benchmarked after merging upstream llama.cpp through `5bda51bfb` and rebuilding for SM70 + SM75. The pre-sync fork is `a7caa68b8`; the benchmarked merge commit is `a74cedb93`. After these measurements, upstream advanced by one four-line `ggml/src/ggml-backend.cpp` fix (`43f3dda62`); that commit was merged as `d183ec0f9` and the server was rebuilt. It does not touch CUDA kernels or server dispatch, so the tables below remain pinned to the exact measured `5bda51bfb` baseline rather than relabeling unmeasured results.

The regression checks below use the same model, saved KV state, batch settings, GPU placement, and request shape on both sides. Each process receives an unreported warm request; the tables use retained samples. Qwen uses `Qwen3.8-27B-UD-Q5_K_XL.gguf` with `q8_0` K/V and MTP disabled.

## Sync regression: pre-sync fork vs merged fork

| Workload | Pre-sync PP | Post-sync PP | PP change | Pre-sync TG | Post-sync TG | TTFT change |
|---|---:|---:|---:|---:|---:|---:|
| V100, 100k cached + 1k | 431.14 | 430.03 | -0.26% | 16.39 | 16.34 | +0.35% |
| RTX 2080 Ti, 65,536 cached + 1k | 491.47 | 490.97 | -0.10% | 17.27 | 17.26 | -0.10% |
| V100 + RTX 2080 Ti, 100k cached + 1k | 689.84 | 685.72 | -0.60% | 26.40 | 26.32 | +1.01% |
| Ornith V100, 100k cached + 1k, `MMQ=moe` | 808.01 | 805.56 | -0.30% | 56.97 | 56.82 | +0.23% |

The mixed-GPU result had more process-to-process drift than the single-GPU tests. In the immediately following current-upstream comparison the merged fork measured 689.91–690.98 PP/s in the two retained process arms, matching the pre-sync ~690 PP/s envelope. No material performance regression was found.

## Current upstream vs synced fork

Current upstream is `5bda51bfb`.

| Workload | Upstream PP | Synced fork PP | PP gain | Upstream TTFT | Synced fork TTFT | TTFT reduction |
|---|---:|---:|---:|---:|---:|---:|
| V100, 100k cached + 1k | 298.24 | 430.87 | **+44.47%** | 3.404 s | 2.370 s | **-30.37%** |
| RTX 2080 Ti, 65,536 cached + 1k | 384.91 | 496.44 | **+28.98%** | 2.627 s | 2.048 s | **-22.03%** |
| V100 + RTX 2080 Ti, 100k cached + 1k | 405.41 | 690.44 | **+70.31%** | 2.524 s | 1.503 s | **-40.44%** |
| Ornith V100, 100k cached + 1k | 534.97 | 803.26 | **+50.15%** | 1.933 s | 1.306 s | **-32.44%** |

The Qwen dual-GPU 64-token decode result is 23.96 tok/s upstream vs 26.47 tok/s on the fork (+10.45%). End-to-end time for the 1k append plus 64 generated tokens is 5.160 s upstream vs 3.891 s on the fork (-24.59%).

The Ornith comparison uses upstream's normal dispatch and the fork's recommended `GGML_CUDA_VOLTA_FORCE_MMQ=moe` setting. Decode is essentially unchanged: 56.97 tok/s upstream vs 56.92 tok/s on the fork. The gain is in prompt processing / TTFT.

The standalone RTX 2080 Ti uses a 67,584-token context with 65,536 cached tokens plus a 1,000-token append. The 22 GB card had about 463 MiB free at this allocation. A 100k Q8 KV context with this Q5_K_XL model does not fit fully resident on the card.

## MMQ policy

### Dense Qwen: selector is neutral, global forcing is expensive

A mirrored normal / `MMQ=moe` / `MMQ=moe` / normal run on one V100 gave:

| Qwen V100, 100k + 1k | PP | TG | TTFT |
|---|---:|---:|---:|
| Normal dispatch | 429.86 | 16.36 | 2.387 s |
| `GGML_CUDA_VOLTA_FORCE_MMQ=moe` | 429.17 | 16.35 | 2.376 s |

The selector changes PP by -0.16% and TG by -0.09%, which is measurement noise and agrees with the implementation: it only forces MMQ when `n_experts > 0`.

A separate same-session comparison with a binary compiled using global `GGML_CUDA_FORCE_MMQ=ON` measured 295.70 PP/s versus 434.24 PP/s with normal dispatch (-31.90%). TTFT rose from 2.354 s to 3.432 s (+45.79%).

### Ornith: selective MoE forcing is better

| Ornith V100, 100k + 1k | PP | TG | TTFT |
|---|---:|---:|---:|
| `GGML_CUDA_VOLTA_FORCE_MMQ=moe` | **804.23** | 56.93 | **1.290 s** |
| Global `GGML_CUDA_FORCE_MMQ=ON` | 776.86 | **57.05** | 1.348 s |

Global forcing gives ~0.22% more decode throughput but loses 3.40% prompt throughput and increases TTFT by 4.50%. The recommended serving configuration is therefore the normal SM70/SM75 build with `GGML_CUDA_VOLTA_FORCE_MMQ=moe` set for V100-containing launchers.

## Automatic Qwen batch defaults

The server now applies the measured Qwen3.8-27B batch/ubatch values only when `-b/-ub` are left at their defaults. Startup probes against the merged build confirmed:

| Detected hardware | Automatic batch / ubatch |
|---|---:|
| V100, 131072 ctx | `4096 / 4096` |
| RTX 2080 Ti, 32768 or 67584 ctx | `4096 / 2048` |
| V100 + RTX 2080 Ti, 409600 ctx | `4096 / 2048` |

An explicit `--batch-size 2048 --ubatch-size 1024` probe did not trigger the automatic override. The detector keys on GGUF metadata (`qwen35`, `Qwen3.8-27B`) and the selected CUDA device descriptions; other models, GPU topologies, and context sizes keep upstream defaults. `LLAMA_V100_AUTO_BATCH=0` disables the feature.

The values come from the 16k prompt sweep: V100 `ubatch=1024/2048/4096` measured 876.49 / 923.31 / 952.18 tok/s; RTX 2080 Ti measured 890.95 / 923.63 / 930.62 tok/s. On the 22 GB card, 2048 is the default because 4096 gains only ~0.8% while consuming more compute-buffer VRAM. The mixed 400k profile uses 4096/2048; 4096 ubatch had less than ~1 GiB of free RTX VRAM before additional server state.

## Correctness and build checks

- The merged SM70+SM75 build completed successfully for `llama-server`, `test-backend-ops`, `test-recurrent-state-rollback`, and `test-mtp-draft-budget`.
- `test-mtp-draft-budget` passed all 48 boundary cases plus the default `-1` override case.
- The exact Qwen long-Q8 attention geometry `D=256`, `GQA=6`, `KV=101120`, `Q=1000`, Q8 K/V passes the CPU-reference test 1/1 on both V100 and RTX 2080 Ti after the merge.
- The previous-recurrent-snapshot subtest passes on both the pre-sync and post-sync Ornith builds. A separate older dirty-context rollback subtest fails on both trees with the same mismatch, so it is a pre-existing fork test failure rather than a regression introduced by this upstream merge.

## Retained results

The JSON files in this directory retain per-process measurements, exact arguments and relevant environment settings. `run_sync_bench.py` is the benchmark harness used for the Qwen and MMQ comparisons. Server logs are local diagnostics and are not required to reproduce the summary tables.

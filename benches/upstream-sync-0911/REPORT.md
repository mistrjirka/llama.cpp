# Upstream sync and regression check — 2026-09-11

`v100-optimized` is synced with upstream llama.cpp through `8172e6577`. The runtime benchmark baseline is `43f3dda62`; `8172e6577` adds only six lines to `tools/server/tests/unit/test_completion.py`, so CUDA, server runtime, model and scheduling source are unchanged. The pre-sync fork is `a7caa68b8`; the runtime upstream merge is `d183ec0f9`, the Qwen batch-default commit is `720798061`, and the final test-only upstream merge is `1f81621be`.

The regression checks below use the same model, saved KV state, batch settings, GPU placement, and request shape on both sides. Each process receives an unreported warm request; the tables use retained samples. Qwen uses `Qwen3.8-27B-UD-Q5_K_XL.gguf` with `q8_0` K/V and MTP disabled.

## Sync regression: pre-sync fork vs merged fork

| Workload | Pre-sync PP | Post-sync PP | PP change | Pre-sync TG | Post-sync TG | TTFT change |
|---|---:|---:|---:|---:|---:|---:|
| V100, 100k cached + 1k | 431.14 | 430.03 | -0.26% | 16.39 | 16.34 | +0.35% |
| RTX 2080 Ti, 65,536 cached + 1k | 491.47 | 490.97 | -0.10% | 17.27 | 17.26 | -0.10% |
| V100 + RTX 2080 Ti, 100k cached + 1k | 689.84 | 685.72 | -0.60% | 26.40 | 26.32 | +1.01% |
| Ornith V100, 100k cached + 1k, `MMQ=moe` | 808.01 | 805.56 | -0.30% | 56.97 | 56.82 | +0.23% |

The mixed-GPU result had more process-to-process drift than the single-GPU tests. In the immediately following current-upstream comparison the merged fork measured 689.91–690.98 PP/s in the two retained process arms, matching the pre-sync ~690 PP/s envelope. No material performance regression was found.

## Upstream runtime vs synced fork

The measured upstream runtime is `43f3dda62`. The branch also contains current upstream `8172e6577`, whose only additional change is the Python unit-test edit described above. The V100 row was rerun after the final server/defaults changes with a rebuilt fork binary.

| Workload | Upstream PP | Synced fork PP | PP gain | Upstream TTFT | Synced fork TTFT | TTFT reduction |
|---|---:|---:|---:|---:|---:|---:|
| V100, 100k cached + 1k | 297.69 | 429.66 | **+44.33%** | 3.411 s | 2.380 s | **-30.23%** |
| RTX 2080 Ti, 65,536 cached + 1k | 382.30 | 494.92 | **+29.46%** | 2.656 s | 2.058 s | **-22.53%** |
| V100 + RTX 2080 Ti, 100k cached + 1k | 408.08 | 690.96 | **+69.32%** | 2.505 s | 1.505 s | **-39.93%** |
| Ornith V100, 100k cached + 1k | 539.11 | 803.18 | **+48.98%** | 1.911 s | 1.299 s | **-32.06%** |

The Qwen dual-GPU 64-token decode result is 23.97 tok/s upstream vs 26.50 tok/s on the fork (+10.56%). End-to-end time for the 1k append plus 64 generated tokens is 5.141 s upstream vs 3.889 s on the fork (-24.35%).

The Ornith comparison uses upstream's normal dispatch and the fork's recommended `GGML_CUDA_VOLTA_FORCE_MMQ=moe` setting. Decode is essentially unchanged: 57.09 tok/s upstream vs 56.87 tok/s on the fork. The gain is in prompt processing / TTFT.

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

## Ornith MTP3 sync regression

The production draft is the Shisa 12K KL-distilled `mtp-shisa-ornith15-all-Q5_0.gguf`. The regression uses the real dual-GPU production geometry: Ornith AD-Q6_K-Q5_K target, MTP3, Q8 target/draft KV, `GGML_CUDA_VOLTA_FORCE_MMQ=moe`, four 100k cached histories and 128 generated tokens per agent. Mirrored pre/post/post/pre process arms exclude one warm turn per process.

| Metric | Pre-sync | Post-sync | Change |
|---|---:|---:|---:|
| Aggregate generated throughput | 89.35 tok/s | 89.46 tok/s | **+0.13%** |
| Mean per-agent TG | 27.85 tok/s | 27.90 tok/s | **+0.19%** |
| Draft acceptance | 60.0% | 60.6% | +0.6 pp |
| Four-agent wall time | 5.734 s | 5.725 s | **-0.16%** |

No MTP regression was found.

## Automatic Qwen batch defaults

The server applies measured Qwen3.8-27B batch/ubatch values only on the two tested single-GPU types and only when the corresponding values were not explicitly supplied. Startup probes against the merged build confirmed:

| Detected hardware | Automatic batch / ubatch |
|---|---:|
| V100 | `4096 / 4096` |
| RTX 2080 Ti | `4096 / 2048` |
| V100 + RTX 2080 Ti | no automatic override |

CLI, environment and INI-config probes all preserved explicit batch/ubatch values; `LLAMA_V100_AUTO_BATCH=0` also disabled the override. Mixed V100+RTX 2080 Ti placement receives no automatic batch override. The detector keys on GGUF metadata (`qwen35`, `Qwen3.8-27B`) and selected CUDA device descriptions. Other models and GPU topologies keep upstream defaults.

The values come from the 16k prompt sweep: V100 `ubatch=1024/2048/4096` measured 876.49 / 923.31 / 952.18 tok/s; RTX 2080 Ti measured 890.95 / 923.63 / 930.62 tok/s. On the 22 GB card, 2048 is the default because 4096 gains only ~0.8% while consuming more compute-buffer VRAM. The mixed 400k profile uses 4096/2048; 4096 ubatch had less than ~1 GiB of free RTX VRAM before additional server state.

## Current cold prompt processing

These Qwen measurements use upstream `43f3dda62`, default runtime paths, Q8 K/V and matched batch/ubatch settings.

| Hardware | 1k upstream | 1k fork | Gain | 16k upstream | 16k fork | Gain |
|---|---:|---:|---:|---:|---:|---:|
| V100 | 859.13 | 904.11 | **+5.24%** | 871.78 | 961.59 | **+10.30%** |
| RTX 2080 Ti | 670.74 | 925.19 | **+37.94%** | 642.02 | 929.52 | **+44.78%** |
| V100 + RTX 2080 Ti | 963.61 | 1088.98 | **+13.01%** | 1025.85 | 1242.23 | **+21.09%** |

## Correctness and build checks

- The merged SM70+SM75 build completed successfully for `llama-server`, `test-backend-ops`, `test-recurrent-state-rollback`, and `test-mtp-draft-budget`.
- `test-mtp-draft-budget` passed all 48 boundary cases plus the default `-1` override case.
- The exact Qwen long-Q8 attention geometry `D=256`, `GQA=6`, `KV=101120`, `Q=1000`, Q8 K/V passes the CPU-reference test 1/1 on both V100 and RTX 2080 Ti after the merge.
- The previous-recurrent-snapshot subtest passes on both the pre-sync and post-sync Ornith builds. A separate older dirty-context rollback subtest fails on both trees with the same mismatch, so it is a pre-existing fork test failure rather than a regression introduced by this upstream merge.

## Retained results

The JSON files in this directory retain per-process measurements, exact arguments and relevant environment settings. `cold-current/` contains the refreshed cold-prompt cells and `mtp-final-sync-all.json` contains the mirrored production MTP3 regression. `run_sync_bench.py` is the benchmark harness used for the long-context Qwen and MMQ comparisons. Server logs are local diagnostics and are not required to reproduce the summary tables.

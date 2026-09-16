# Upstream comparison: September 12, 2026

Server rows restore the same validated prefix, append 1,000 tokens, then generate 128 tokens. Gemma and Muse use `llama-bench -d 100000 -p 1000 -n 128` so the 100k state is outside the timed append. For V100 MoE rows, upstream is given its faster available global FORCE_MMQ build while the fork uses its more selective `GGML_CUDA_VOLTA_FORCE_MMQ=moe` policy; dense models stay on normal dispatch. Full commands, A-B-B-A controls, hashes, standard deviations, logs, and sync validation are in [`benches/upstream-sync-0912/REPORT.md`](../../benches/upstream-sync-0912/REPORT.md).

| Workload | Upstream PP | `v100-optimized` PP | PP gain | TTFT reduction | TG gain |
|---|---:|---:|---:|---:|---:|
| Qwen3.8 27B · V100 · 100k+1k | 298.47 | **440.08** | **+47.45%** | **-31.61%** | +0.08% |
| Qwen3.8 27B · RTX 2080 Ti · 65k+1k | 383.73 | **493.53** | **+28.61%** | **-21.67%** | +0.05% |
| Qwen3.8 27B · V100+RTX · 100k+1k | 405.93 | **695.92** | **+71.44%** | **-40.71%** | **+10.49%** |
| Ornith 1.5 35B-A3B · V100 · 100k+1k | 696.70 | **976.39** | **+40.15%** | **-27.75%** | +0.62% |
| Ornith 1.5 35B-A3B · RTX 2080 Ti · 65k+1k | 1317.78 | **1560.49** | **+18.42%** | **-14.92%** | +0.94% |
| Ornith 1.5 35B-A3B · V100+RTX · 100k+1k | 1208.72 | **1474.71** | **+22.01%** | **-17.19%** | +1.14% |
| Gemma 4 31B · V100 · 100k depth + 1k | 201.67 | **282.56** | **+40.11%** | — | -0.15% |
| Gemma 4 26B-A4B · V100 · 100k depth + 1k | 691.68 | **952.20** | **+37.66%** | — | +0.55% |
| Muse Glimmer 30B · V100 · 100k depth + 1k | 550.54 | **600.86** | **+9.14%** | — | -0.02% |

The Qwen V100/dual and all Ornith rows reproduced the same token hashes between upstream and current. Qwen RTX was stable within each arm but produced different upstream/current sequences; the performance comparison does not depend on token identity.


[Back to the README](../../README.md)

# Upstream sync and headline benchmark refresh — 2026-09-12

## Source state

- Current upstream: `3057bb66c86c46d5781e50e85462a760ba7d1feb` (`ui : add cache (#28802)`).
- Previous upstream sync point: `8172e6577ac2b35de1ec1e5d1c0aaad6c4a2129f`.
- Upstream merge on `v100-optimized`: `59c40658`.
- Generic Volta K-quant conversion: `67fb9991`.
- Volta MXFP4 conversion: `28f08591`.
- Muse D128/GQA16 FlashAttention specialization: `0e805fd1`.
- Benchmark target: Tesla V100-SXM2-32GB (SM70), RTX 2080 Ti 22 GB (SM75), and the matched mixed-GPU configurations documented below.
- Q8_0 target K/V cache and FlashAttention are used throughout the server headline matrix.

The upstream sync added 17 upstream commits. The only overlapping NVIDIA CUDA file was `ggml/src/ggml-cuda/mmq.cuh`; upstream added AMD GCN configuration dispatch and the merge preserved the existing Volta MMQ path. `tests/CMakeLists.txt` had one small conflict: upstream's new build-only handling for `test-backend-ops` was retained while the fork's `test-ssm-conv-snapshot` registration was preserved.

Post-sync validation:

- normal SM70 build: pass
- normal SM70+SM75 build: pass
- FORCE_MMQ SM70+SM75 build: pass
- targeted Muse D128/GQA16/Q8 FlashAttention CPU-reference test: pass
- `test-ssm-conv-snapshot`: 108/108 cases pass across V100, RTX 2080 Ti and CPU
- pre-sync vs post-sync V100 Qwen p4096 check: -0.25% PP / +9.2 ms, within process drift; no material sync regression

## Fresh cached-server headline matrix

The base server matrix restores the exact same validated prefix, appends 1,000 tokens, and generates 128 tokens in the same request. It uses A-B-B-A process ordering (upstream-current-current-upstream), one warm request per process, and two retained requests per process. The final V100 Ornith headline row was then replaced by a dedicated policy-fair A-B-C-C-B-A check comparing upstream normal, upstream global FORCE_MMQ, and current selective `MMQ=moe`; the fastest valid upstream policy is used in the headline.

The prefix bytes in every saved state were checked against the prompt sequence before the run. GPU placement was also checked live: V100-only and RTX-only rows used only the requested GPU; mixed rows used both GPUs.

| Workload | Upstream PP | Current PP | PP gain | Upstream TTFT | Current TTFT | TTFT reduction | Upstream TG | Current TG | TG gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8 27B / V100 / 100k+1k | 298.47 | **440.08** | **+47.45%** | 3.409 s | **2.331 s** | **-31.61%** | 16.33 | 16.34 | +0.08% |
| Qwen3.8 27B / RTX 2080 Ti / 65k+1k | 383.73 | **493.53** | **+28.61%** | 2.645 s | **2.072 s** | **-21.67%** | 17.34 | 17.35 | +0.05% |
| Qwen3.8 27B / V100+RTX / 100k+1k | 405.93 | **695.92** | **+71.44%** | 2.513 s | **1.490 s** | **-40.71%** | 23.97 | **26.48** | **+10.49%** |
| Ornith 1.5 35B-A3B / V100 / 100k+1k | 696.70 | **976.39** | **+40.15%** | 1.490 s | **1.076 s** | **-27.75%** | 56.84 | **57.19** | +0.62% |
| Ornith 1.5 35B-A3B / RTX 2080 Ti / 65k+1k | 1317.78 | **1560.49** | **+18.42%** | 0.806 s | **0.686 s** | **-14.92%** | 59.39 | **59.95** | +0.94% |
| Ornith 1.5 35B-A3B / V100+RTX / 100k+1k | 1208.72 | **1474.71** | **+22.01%** | 0.883 s | **0.731 s** | **-17.19%** | 67.29 | **68.06** | +1.14% |

Qwen V100, Qwen dual, Ornith V100, Ornith RTX, and Ornith dual produced identical token hashes across upstream and current. The Qwen RTX row produced stable per-arm hashes but different upstream/current token sequences; throughput claims do not rely on token identity.

### Server configurations

- Qwen V100: `UD-Q5_K_XL`, native 131072 context, b4096/ub4096, normal build.
- Qwen RTX: `UD-Q5_K_XL`, 67584 context, b4096/ub2048, normal build.
- Qwen dual: production 409600 YaRN context, RTX:V100 tensor split 4:5, b4096/ub2048, internal CUDA all-reduce.
- Ornith V100: AD-Q6_K/Q5_K, b2048/ub512; upstream uses its faster global FORCE_MMQ build, while current uses the normal build + selective `GGML_CUDA_VOLTA_FORCE_MMQ=moe`. A dedicated policy check measured upstream normal 542.97 PP/s, upstream FORCE_MMQ 696.70 PP/s, current global FORCE_MMQ 943.65 PP/s, and current selective 976.39 PP/s. The selective fork policy is ~3.5% faster than globally forcing MMQ.
- Ornith RTX: AD-Q5_K/Q4_K, b4096/ub512, global FORCE_MMQ build on both arms.
- Ornith dual: AD-Q6_K/Q5_K, tensor split 1:1, b2048/ub1024, internal CUDA all-reduce, global FORCE_MMQ build on both arms.

## Gemma and Muse depth-mode rows

Gemma 4 and Muse Glimmer use `llama-bench -d 100000 -p 1000 -n 128`. The 100k state is constructed outside the timed region and then reused for the timed 1k append / 128-token generation samples. This is important for hybrid-cache models: current upstream can report a restored server slot while still recomputing the hybrid cache, so a saved-slot server comparison would not isolate the 1k append fairly.

Fresh depth-mode results:

| Model | Upstream PP | Current PP | PP gain | Upstream TG | Current TG | TG gain |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 31B | 201.67 | **282.56** | **+40.11%** | 15.67 | 15.64 | -0.15% |
| Gemma 4 26B-A4B | 691.68 | **952.20** | **+37.66%** | 53.12 | **53.41** | +0.55% |
| Muse Glimmer 30B | 550.54 | **600.86** | **+9.14%** | 29.41 | 29.41 | -0.02% |

Gemma 31B and Muse use normal dispatch on both arms. Gemma 26B-A4B is a routed MoE model, so a follow-up policy check compared upstream normal, upstream global FORCE_MMQ, and current selective `MMQ=moe`: **688.29 / 691.68 / 952.20 PP/s** respectively. The headline therefore gives upstream its faster global-FORCE result and the fork its intended selective policy. Original depth-mode runs are in `headline-gemma-summary.json`; the MMQ policy check is in `mmq-fairness/gemma26.json`.

## Artifacts

- `headline_server_bench.py`
- `headline_gemma_bench.py`
- `headline-server-summary.json` and `headline-gemma-summary.json` retain the original fresh matrix
- `headline-final-summary.json` is the canonical README/chart source after applying the MMQ fairness checks
- per-profile server JSON files and server logs in this directory
- MMQ fairness checks: `mmq-fairness/ornith-v100.json` and `mmq-fairness/gemma26.json`
- pre-sync/post-sync check: `/workspace/volta-extra-quant-0912/upstream-sync-3057bb66/summary.json`

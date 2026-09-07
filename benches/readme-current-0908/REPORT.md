# Current upstream vs `v100-optimized` benchmark refresh — 8 September 2026

## What is being compared

The fork was first synced through upstream llama.cpp `67672dc5b`, then benchmarked as `c886bc606`. At benchmark time `git rev-list --left-right --count v100-optimized...upstream/master` was `69 0`: the fork contained all upstream commits and was 69 fork commits ahead. The direct comparison therefore uses the exact upstream parent of the current fork rather than an older vanilla snapshot.

Hardware: Tesla V100-SXM2 32 GB at a 200 W limit and RTX 2080 Ti 22 GB at a 250 W limit. No profiler ran during timings and the GPU resource was locked against concurrent benchmark jobs. GPU clocks were not manually locked.

## V100: 100k cached Qwen3.8-27B append

One V100 only; Qwen3.8-27B `UD-Q5_K_XL`; q8_0 K/V; FlashAttention on; MTP off; 100,000 restored tokens. Every measured request restores the same state, submits the same token history, verifies `cache_n=100000`, and processes only the requested suffix. A/B/B/A process ordering is upstream/fork/fork/upstream with three retained measurements for every suffix in every process, giving six measurements per side. The generated control token is exact across all runs.

| appended prompt | upstream `67672dc5b` | `v100-optimized` `c886bc606` | change |
|---:|---:|---:|---:|
| +128 | 157.43 PP/s | **195.97 PP/s** | **+24.48%** |
| +256 | 215.96 PP/s | **297.95 PP/s** | **+37.97%** |
| +1,000 | 299.86 PP/s | **431.85 PP/s** | **+44.02%** |

At +1,000 tokens the prompt phase is 3335.01 ms upstream versus 2315.63 ms on the fork, saving about 1.019 s.

## Single-GPU `llama-bench`

Qwen3.8-27B `UD-Q5_K_XL`; q8_0 K/V; FlashAttention on; batch/ubatch 2048/512; split mode none; only the named GPU selected. Each process performs three internal benchmark repetitions. Process order is upstream/fork/fork/upstream; the reported number is the mean of the two process-level `avg_ts` values per side.

### Tesla V100-SXM2 32 GB

| test | upstream | fork | change |
|---|---:|---:|---:|
| pp512 | 773.84 tok/s | **797.81 tok/s** | **+3.10%** |
| pp2048 | 775.32 tok/s | **800.47 tok/s** | **+3.24%** |
| tg128 | 27.58 tok/s | **27.68 tok/s** | +0.36% |

### RTX 2080 Ti 22 GB

| test | upstream | fork | change |
|---|---:|---:|---:|
| pp512 | 660.28 tok/s | **680.34 tok/s** | **+3.04%** |
| pp2048 | 660.80 tok/s | **682.19 tok/s** | **+3.24%** |
| tg128 | 24.67 tok/s | **24.74 tok/s** | +0.28% |

## Four-agent Ornith, 100k cached per agent, MTP3

Both GPUs are used. Target is Ornith-1.5-35B-A3B `AD-Q6_K-Q5_K`, draft is Shisa Q5_0 MTP, target and draft KV are q8_0, and each agent starts from a real 100k C++ history before appending 38/42/44/41 tokens and generating 128 tokens. The fixed numerator for aggregate PP+TG is therefore 677 tokens per complete turn; aggregate generated uses 512 tokens.

The strict comparison disables fork-only prefix sharing, deferred-MTP prompt handling, separate draft ubatch and custom pipeline-copy selection. Upstream cannot natively restore the fork's `.draft`/`.spec` companions, so the upstream server contains only the documented restore-only setup shim in `upstream-restore-only-shim.diff`. The shim runs before timing and changes no completion/decode path.

| engine / mode | whole turn | aggregate PP+TG | aggregate generated | mean TG / agent |
|---|---:|---:|---:|---:|
| upstream `67672dc5b` | 10.701 s | 63.32 tok/s | 47.89 tok/s | 14.80 tok/s |
| `v100-optimized`, common-denominator | **7.766 s** | **87.18 tok/s** | **65.94 tok/s** | **19.88 tok/s** |
| `v100-optimized`, normal optimized serving | **5.601 s** | **121.02 tok/s** | **91.53 tok/s** | **28.38 tok/s** |

The strict fork result is **+37.69% aggregate PP+TG** with a **27.43% shorter complete turn**. Normal serving is intentionally separate: its 121.02 tok/s includes fork-only serving features and is a practical deployment measurement, not a pure kernel A/B.

The mirrored strict test retains eight measurements per side after warmups. The normal-serving arm retains four measurements after warmup. Raw per-run wall times, acceptance counts, output hashes, exact argv, and process-level benchmark rows are retained in this directory.

## Files

- `llama-bench-summary.json` / `llama-bench-raw.json`: V100 and RTX single-GPU tests.
- `v100-100k-summary.json` / `v100-100k-raw.json`: exact-token 100k cached V100 append test.
- `ornith-fair-summary.json` / `ornith-fair-raw.json`: strict and normal four-agent Ornith results.
- `run-*.py`: benchmark harnesses.
- `upstream-restore-only-shim.diff`: the only upstream source modification used for the Ornith setup path.
- `manifest.json`: commits, binary hashes, hardware limits, and cache hash.

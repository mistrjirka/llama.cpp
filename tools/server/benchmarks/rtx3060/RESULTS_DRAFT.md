# RTX 3060 hub experiment results

**Draft, 2026-11-24.** Completed measurements are below. Final launch settings, long-context validation, soak results, and the running deployment are not yet confirmed in this report.

## Current decision

Keep native MTP at depth 2, CUDA graphs, lazy recurrent updates, and immutable checkpoint sharing. Draft-only Q4 KV is the preferred memory candidate. Keep target KV at Q8_0 and the current 256-token microbatch unless final tests support a change. Leave adaptive MTP, pinned state staging, and runtime compression disabled.

The original setup already used MTP2. The new measured improvements are checkpoint sharing on a branching workload and smaller draft KV. Do not describe the MTP-versus-disabled comparison as a new speed improvement.

## Test conditions and evidence

- Hardware: RTX 3060 12GB, Intel N95. The measured GPU link is PCIe 3.0 x1.
- Model: SenseNova-Loop-9B native Q6_K weights, 42 recurrent and 14 attention layers. Target KV remains Q8_0. Configured context is 110,000, allocated as 110,080 tokens.
- Speed suite: four prompts, two repetitions, temperature 0, seed 42, up to 192 output tokens, prompt reuse disabled. It includes code, JSON, explanation, and a 250-row catalog. This catalog test is not the separate 100K-token test.
- Generation rate is generated tokens divided by recorded decode time. Total request time includes prompt processing. Results are workload-specific.
- Speed/cache PSS and GPU use were sampled every 500 ms. PSS means process RAM with shared pages apportioned. The fork test also records the kernel's peak RSS after resetting its peak counter.
- Data: `/home/jirka/.cache/rtx3060-experiments-20261124`. Read each variant's `results.json`, `config.json`, and `server.log`. The early `speed-summary.json` contains only four variants and is incomplete.
- Reproduction and summarization: [run_experiments.py](run_experiments.py), [summarize.py](summarize.py). No new inference was run to prepare this report.

## Speed and VRAM decisions

The repeated MTP2 controls measured 40.60-40.75 generation tokens/s and 11,255 MiB peak GPU use.

| Change | Measured result | Decision |
| --- | --- | --- |
| MTP depth 0 / 1 / 2 / 3 / 4 | 23.75 / 35.98 / 40.63 / 38.47 / 35.55 tok/s | Keep depth 2. It is about 71% faster than disabled MTP on this suite. |
| Draft KV Q8_0 -> Q4_0 | 40.45 tok/s. GPU peak 11,255 -> 11,147 MiB, saving 108 MiB. All eight outputs match the MTP2 reference hashes. | Preferred candidate. Target KV and weights are unchanged. Full final validation pending. |
| Adaptive MTP, maximum 2 / 4 | 38.54 / 37.26 tok/s. No VRAM reduction at maximum 2. | Reject for the current workload. |
| Disable CUDA graphs | 39.20 versus 40.65 tok/s in the paired control. Saves 26 MiB GPU and about 26 MiB sampled PSS. | Keep graphs enabled. |
| Enable extra graph optimization | 40.61 tok/s, same GPU peak | No demonstrated benefit. Keep the default. |
| Graph chunk size 128 | 40.68 tok/s, saves 8 MiB GPU | No material benefit established. Keep the default. |
| CPU threads 2 | 40.70 tok/s, same GPU peak | No material speed or RAM benefit. Keep the existing setting. |
| Microbatch 128 | 40.74 tok/s. Saves 62 MiB GPU and about 57 MiB sampled PSS. Prompt time rises 19.19 -> 20.01 s. | Alternative if memory is more important. Keep 256 for now. |
| Microbatch 512 | 40.09 tok/s, costs 184 MiB GPU. Prompt time falls to 18.63 s. | Reject for the memory goal. |
| Microbatch 64 | 40.16 tok/s. Saves 106 MiB GPU, but total time rises 56.84 -> 61.26 s. | Reject. |
| Draft Q4 plus microbatch 128 | 40.28 tok/s, 11,085 MiB GPU. Total time 58.04 s versus 56.84 s control. | Extra 62 MiB saving over draft Q4 alone, with a prompt-speed cost. Not the preferred default. |
| Disable lazy recurrent updates | 40.14 tok/s, costs 158 MiB GPU. Prompt processing is about 1.3 s faster on this suite. | Keep lazy updates for the memory goal. |
| No warmup / logical batch 512 | Final tests underway | Pending. |

Some changes produce different greedy output text even though repeated runs are stable. Microbatch changes differ from the MTP2 reference on the catalog prompt. Disabling lazy updates differs on explanation and catalog prompts. MTP depths 0, 1, and 4 also differ on some prompts. Draft Q4, adaptive MTP, graph changes, and two CPU threads match the reference hashes on the recorded cases. These checks are not a broad quality evaluation or proof of universal output identity.

## Checkpoint sharing: measured RAM saving

Checkpoint sharing reuses one immutable recurrent-state object when multiple cached branches retain it. Request and eviction metadata still retain their original semantics.

| Fork workload, 14 requests | Original binary | Sharing off | Sharing on |
| --- | ---: | ---: | ---: |
| Kernel peak RSS, MiB | 5,243.14 | 5,241.27 | 4,794.57 |
| Sampled peak PSS, MiB | 5,157.44 | 5,155.50 | 4,708.83 |
| Idle PSS, MiB | 4,654.55 | 4,653.37 | 4,382.60 |
| Total request seconds | 62.07 | 62.00 | 60.95 |
| Parked checkpoint references / unique objects | Not logged | 10 / 10 | 10 / 8 |
| Maximum unique checkpoint payload saving, MiB | Not logged | 0 | 175.91 |

**Keep sharing.** Against sharing-off, the fork trace saves 446.70 MiB of peak RSS and 270.77 MiB of idle PSS. The observed process difference includes temporary allocation and allocator effects. It is not the same quantity as the 175.91 MiB retained-payload saving.

All variants reused 92,526 prompt tokens and returned the same 14 forced JSON responses. This validates the tested cache and ownership paths, not semantic model quality. The independent-prompt cache trace retained eight distinct checkpoints and showed no sharing benefit. Savings therefore depend on branch ancestry.

Whole-object content interning was not implemented. Recreated checkpoints can have different `id_task` metadata, which affects eviction. Sharing payloads across those wrappers would need a separate measured design.

Sources: `fork-summary.json`, `fork/{original,sharing_off,sharing_on}/results.json`, and the corresponding server logs.

## What the cache limit means

`--cache-ram 4096` is a budget for parked cache entries. It is not a 4 GiB limit on the whole process. Model mappings, active checkpoints, scratch buffers, and other process allocations add RAM use.

- The existing target Q8 cache uses 30,464 bytes per token, or about 35,246 tokens per GiB of KV payload. That density has not changed.
- A full 110,080-token target KV payload is 3.123 GiB. Checkpoints, metadata, and the draft add separate storage.
- One full recurrent checkpoint is about 87.94 MiB. It captures the recurrent state across the model and does not grow one record per past token.
- The CPU image projector also holds 875.61 MiB of weights and reserves 248.10 MiB of compute scratch. Reserved scratch is not necessarily resident before an image request. The final vision test will measure its actual peak and idle RAM. No projector lifecycle change was made.
- Existing block deduplication remains useful. In the 22-request independent-prompt cache trace, peak logical cache reached 2,921.0 MiB while peak accounted storage was 1,891.4 MiB. All recorded entries stayed within the 4,096 MiB cache budget.
- There is no single guaranteed number of parked conversations per GiB. Their lengths, shared prefixes, checkpoint counts, and branch ancestry determine it.

## Swap failure found and repaired

The hub advertised a 16,000 MiB logical zram swap device but capped its compressed resident allocation at 3,000 MiB. The cap was reached while logical swap slots remained free. The recorded failed-write counter reached 47,732,577, with kernel swap-write errors.

The user applied the repair at 21:32 CET on 2026-11-24. The live and persistent resident cap now read zero, meaning no separate resident cap. The failed-write counter did not increase during the initial ten-second check. The comparison cache and fork suites used `memory.swap.max=0` and recorded zero process swap.

This establishes the observed swap-write failure, not the cause of every historical llama.cpp crash. Final sustained-workload confirmation remains part of the soak test. See [hub-zram-recovery.md](../cache-research/hub-zram-recovery.md) for the captured evidence and kernel sources.

## PCIe and pinned staging

A standalone copy benchmark restored every byte exactly. Median bandwidth:

| Transfer | Pageable | Pinned |
| --- | ---: | ---: |
| 4 MiB CPU -> GPU | 0.776 GB/s | 0.787 GB/s |
| 4 MiB GPU -> CPU | 0.804 GB/s | 0.822 GB/s |
| 64 MiB CPU -> GPU | 0.786 GB/s | 0.788 GB/s |
| 64 MiB GPU -> CPU | 0.821 GB/s | 0.823 GB/s |

Pinned 4 MiB staging has a small copy-level benefit. It did not show a clear end-to-end cache benefit: sharing-off took 77.29 s and pinned-only 77.26 s. Sharing-on took 76.75 s and sharing-plus-pinned 76.67 s. Leave `LLAMA_STATE_STREAM_PINNED` off for deployment.

At 0.787 GB/s, a full 3.123 GiB target KV upload takes at least 4.26 s. This is a calculated transfer lower bound, not a measured whole-cache restoration time. It makes frequent full CPU-cache reloads expensive on this hub. Pinned memory cannot remove the PCIe x1 limit.

Sources: `transfer-results.json`, `cache-summary.json`, and [TRANSFER.md](TRANSFER.md).

## Exact compression: tested offline, not deployed

The screen used 25.125 MiB of parsed F32 recurrent tensors and 76.144 MiB of Q8 tensors from three saved captures. All 120 combinations restored every byte. Times include preprocessing and reversal, but exclude file I/O, allocation, and the final comparison.

| Payload and method | Saved bytes | Encode / decode ms per MiB | Decision |
| --- | ---: | ---: | --- |
| F32, raw Zstd1 | 6.16% | 1.510 / 0.858 | Baseline |
| F32, byte shuffle + Zstd1 | 14.25% | 1.547 / 0.715 | Keep as a research candidate |
| F32, exponent separation + Zstd1 | 14.30% | 1.572 / 0.685 | Keep as a research candidate |
| F32, integer XOR + byte shuffle + Zstd1 | 13.37% | 1.514 / 0.691 | No advantage over simpler choices |
| F32, scalar bitshuffle + Zstd1 | 12.59% | 3.902 / 4.587 | Reject this prototype |
| Q8, raw Zstd1 | 3.31% | 1.357 / 0.832 | Too little saving for the hot cache |
| Q8, scale separation + Zstd1 | 4.68% | 1.601 / 0.988 | Reject for the current hot cache |
| Q8, added zigzag | 4.68% | 1.610 / 0.988 | No material extra saving |

These rows use up to 4 MiB blocks. F32 tensors limited their actual blocks to 2 MiB. At the current 128 KiB block size, exponent separation saved only 7.59%. Applying the 14.30% sample ratio to one 87.94 MiB checkpoint would save about 12.58 MiB and cost about 138 ms to encode and 60 ms to decode. That is an estimate, not a whole-checkpoint measurement.

Peak screening RSS was 47.60 MiB, with 24.66 MiB peak counted scratch and codec contexts. No runtime codec is enabled. CPU decompression before upload saves RAM but sends the same bytes across PCIe. Active VRAM savings require a separate compressed representation and suitable GPU kernels.

Full results: [exact-compression-rtx3060.md](../cache-research/exact-compression-rtx3060.md), with the adjacent raw and aggregate JSON.

## Paper decisions

| Proposal | Status on this implementation |
| --- | --- |
| [ZipNN](https://arxiv.org/html/2411.05239v1) exponent separation | The idea was screened on actual hub snapshots. Its paper measured model weights, so local results above determine the decision. |
| [SplitZip](https://arxiv.org/html/2605.01708v2) / [GAAC](https://arxiv.org/html/2610.38210v2) | Exact BF16 methods. Their formats do not directly compress Q8_0 integer blocks. GAAC's approximately 11.78 resident bits/value exceed this target's 8.5 bits/value. No port. |
| [CacheGen](https://arxiv.org/html/2310.07240v3) | Full method includes lossy quantization. Only exact entropy/preprocessing ideas were considered here. |
| [ReplaySSM](https://arxiv.org/html/2610.13778v1) | Existing lazy recurrent code already overlaps part of the rollback idea. No full ReplaySSM implementation or claimed paper-level speedup. |
| [VeriCache](https://arxiv.org/html/2605.17613v1) | Requires full target verification and substantial transfers. No port. The measured x1 link is a major constraint. |
| KIVI, KVQuant, KVzip/KVzap, MiniCache, [DAMP](https://arxiv.org/html/2608.27513v1) | Target quantization, pruning, merging, or state rounding changes computation. Defer as separate quality-changing projects. |
| Full LMCache pipeline integration | Not implemented. Only bounded pinned staging was isolated and tested. |

See [COMPRESSION_RESEARCH.md](COMPRESSION_RESEARCH.md) and [rtx3060-paper-review.md](../cache-research/rtx3060-paper-review.md) for primary sources, paper hardware, and applicability limits.

## Completion fields

| Item | Status to fill before final report |
| --- | --- |
| Final no-warmup / logical-batch comparison | Pending |
| Approximately 100K-token request | Pending: tokens, answer check, peak RAM/VRAM, latency |
| Repeated-cache soak | Pending: requests, retained entries, memory plateau, swap-write delta |
| Vision regression | Pending if included in final gate |
| Final launcher settings and tested commit | Pending. Current experiment source is `04df977`, with exact-compression evidence integrated as `1968e3c`. |
| Source push | Root reports commit `3f8ed81` pushed and verified. Later changes may supersede it. |
| Running hub deployment | Pending: service, binary/commit, MTP depth, draft/target types |

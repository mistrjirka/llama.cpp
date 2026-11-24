# RTX 3060 hub experiment results

**Draft, 2026-11-24.** Final settings keep microbatch 256 and change only the draft KV to Q4. The original-versus-final 100K comparison is complete. Final 8K/32K, full-cache soak, vision, and deployment checks remain pending.

## Current decision

Select native MTP2, draft Q4_0 KV, target Q8_0 KV, logical batch 2048, and microbatch 256. Keep the 110,000-token configured context, 4,096 MiB cache budget, three checkpoints with a 384 MiB checkpoint cap, CUDA graphs, lazy recurrent updates, immutable checkpoint sharing, and startup warmup. The clean release excludes the adaptive-MTP and pinned-staging runtime experiments. Runtime compression is not enabled.

Draft Q4 alone saves **108 MiB of measured GPU use**, with short-suite decode at 40.45 tok/s versus 40.62-40.75 tok/s for repeated controls. It does not establish a steady CPU RAM reduction. The separate branching workload saves **446.70 MiB of peak RSS** from checkpoint sharing. These are different memory measurements and workloads.

Reject microbatch 128: its additional 62 MiB GPU saving costs about 5.5% more cold-start time at 100K tokens and changes response behavior. The earlier 56 MiB short-suite RAM saving belonged to that rejected configuration. It is not a final-deployment gain.

Reject logical batch 512 despite its favorable short test. It creates more retained recurrent checkpoints in the matched cache-pressure trace, consuming 340.75 MiB more accounted cache for the same seven entries.

The original setup already used MTP2. Final measured savings come from checkpoint sharing and draft Q4. The MTP-versus-disabled comparison is not a new speed improvement.

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
| Draft KV Q8_0 -> Q4_0 | 40.45 tok/s. GPU peak 11,255 -> 11,147 MiB, saving 108 MiB. All eight outputs match the MTP2 reference hashes. | Keep in the selected combination. Target KV and weights are unchanged. |
| Adaptive MTP, maximum 2 / 4 | 38.54 / 37.26 tok/s. No VRAM reduction at maximum 2. | Reject for the current workload. |
| Disable CUDA graphs | 39.20 versus 40.65 tok/s in the paired control. Saves 26 MiB GPU and about 26 MiB sampled PSS. | Keep graphs enabled. |
| Enable extra graph optimization | 40.61 tok/s, same GPU peak | No demonstrated benefit. Keep the default. |
| Graph chunk size 128 | 40.68 tok/s, saves 8 MiB GPU | No material benefit established. Keep the default. |
| CPU threads 2 | 40.70 tok/s, same GPU peak | No material speed or RAM benefit. Keep the existing setting. |
| Microbatch 128 | Saves another 62 MiB GPU. Its 100K cold-start time is 216.83 s versus 205.60 s original, with changed output behavior. | Reject. Keep microbatch 256. |
| Microbatch 512 | 40.09 tok/s, costs 184 MiB GPU. Prompt time falls to 18.63 s. | Reject for the memory goal. |
| Microbatch 64 | 40.16 tok/s. Saves 106 MiB GPU, but total time rises 56.84 -> 61.26 s. | Reject. |
| Draft Q4 plus microbatch 128, logical batch 2048 | 40.28 tok/s, 11,085 MiB GPU, 2,737.10 MiB peak PSS. Control: 40.65 tok/s, 11,255 MiB GPU, 2,793.19 MiB PSS. Total time 58.04 versus 56.84 s. | Rejected after the 100K comparison. These are not final-setting savings. |
| Disable lazy recurrent updates | 40.14 tok/s, costs 158 MiB GPU. Prompt processing is about 1.3 s faster on this suite. | Keep lazy updates for the memory goal. |
| Logical batch 512 / no warmup | Completed below | Reject both. Keep batch 2048 and normal warmup. |

Some changes produce different greedy output text even though repeated runs are stable. Microbatch changes differ from the MTP2 reference on the catalog prompt. Disabling lazy updates differs on explanation and catalog prompts. MTP depths 0, 1, and 4 also differ on some prompts. Draft Q4, adaptive MTP, graph changes, and two CPU threads match the reference hashes on the recorded cases. These checks are not a broad quality evaluation or proof of universal output identity.

### Short test that initially favored batch 512

All four runs contain the same eight short-suite requests. Source: `last-settings-summary.json` and `last-settings/*/{config,results}.json`.

| Variant | Generation tok/s | Total seconds | GPU peak MiB | PSS peak MiB | Outputs matching control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control: draft Q8, microbatch 256, logical batch 2048 | 40.79 | 56.40 | 11,255 | 2,786.66 | 8/8 |
| Logical batch 512 only | 40.75 | 56.38 | 11,255 | 2,674.00 | 8/8 |
| Disable startup warmup | 40.73 | 56.47 | 11,255 | 2,766.04 | 8/8 |
| Rejected after pressure test: draft Q4, microbatch 128, logical batch 512 | 40.37 | 57.66 | 11,085 | 2,616.62 | 6/8 |

The Q4/microbatch-128 combinations with either logical batch size differ on the two repeated catalog outputs. Both match the control on the other six responses and remain repeatable. This is not a claim that changing the microbatch preserves every output.

Logical batch 512 saves 112.66 MiB of sampled peak PSS and 112.64 MiB of kernel peak RSS on this suite. Only 24 MiB is the smaller fixed hidden-output reservation. The approximately 6K-token prompt also retains two checkpoints instead of three. Do not promise a fixed 112 MiB saving for every prompt length. Disabling warmup saves only about 20.6 MiB here by deferring allocations, so normal warmup remains enabled.

### Why cache pressure reverses the batch-512 decision

The matched comparison after branch 4 contains seven parked entries in both runs:

| Configuration | Accounted cache MiB | Parked checkpoint references |
| --- | ---: | ---: |
| Original: batch 2048, microbatch 256, draft Q8 | 1,800.00 | 8 |
| Rejected candidate: batch 512, microbatch 128, draft Q4 | 2,140.75 | 12 |

The four additional checkpoints cost about 351.82 MiB, partly offset by about 11.04 MiB less other state payload. Source inspection and logs identify the logical batch size as the cause. A 1,480-token suffix starts at position 6679. Batch 512 processes outer batches starting at 6679, 7191, and 7703, retaining an extra checkpoint at 7703. Batch 2048 handles that suffix in one outer batch. The microbatch reduction does not cause this count increase.

This is a checkpoint-placement tradeoff, not growing unbounded memory. The rejected candidate completed three rounds of 20 branches, each interleaved with 20 auxiliary requests. Maximum recorded per-request PSS was 6,763.28 / 6,763.73 / 6,763.96 MiB, a 0.68 MiB spread, with zero process swap. That validates a plateau for this trace but does not restore the lost cache capacity or validate the final configuration.

The earlier batch-512 candidate is now stored under `soak-b512/selected`. The later rejected microbatch-128 candidate is under `soak-ub128/selected`. The new `final-soak` run uses the final `-b 2048 -ub 256` settings. Historical microbatch-128 short-suite figures come from `followup/{control,q4_ub128}/results.json` and `followup-summary.json`.

### 100K comparison selects draft Q4 alone

All cold requests contain 100,052 prompt tokens. Times are single observations to first visible content.

| Configuration | Cold first content, seconds | Cold input tokens/s | Output result |
| --- | ---: | ---: | --- |
| Original, draft Q8, microbatch 256 | 205.602 | 486.759 | JSON, but incorrect middle and last values |
| Final, draft Q4, microbatch 256 | 205.669 | 486.596 | All five response hashes match original |
| Rejected, draft Q4, microbatch 128 | 216.830 | 461.560 | Prose instead of requested JSON, truncated at 96 tokens |

The original and final configurations both returned middle `7707` and last `54116`, instead of `77082` and `154116`. The 100K quality failure therefore exists in the original. Draft Q4 preserves the tested behavior. This arithmetic catalog is not a difficult retrieval benchmark. See [VALIDATION.md](VALIDATION.md) for reuse, restoration, and the remaining final gates.

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
- Including the one-layer MTP draft, combined target-plus-draft KV changes from 32,640 to 31,616 bytes/token. This is **32,896.5 -> 33,962.0 tokens/GiB**, a calculated **3.24%** increase. The exact calculation is `2**30 / bytes_per_token`.
- At 110,080 tokens, combined KV payload changes from 3.34625 to 3.24127 GiB, saving exactly 107.5 MiB before allocator rounding. The configured context limit remains unchanged.
- One full recurrent checkpoint is about 87.94 MiB. It captures the recurrent state across the model and does not grow one record per past token.
- The CPU image projector also holds 875.61 MiB of weights and reserves 248.10 MiB of compute scratch. Reserved scratch is not necessarily resident before an image request. The final vision test will measure its actual peak and idle RAM. No projector lifecycle change was made.
- Existing block deduplication remains useful. In the 22-request independent-prompt cache trace, peak logical cache reached 2,921.0 MiB while peak accounted storage was 1,891.4 MiB. All recorded entries stayed within the 4,096 MiB cache budget.
- The token-density figures exclude recurrent states, checkpoint copies, weights, scratch, and metadata. A 4 GiB parked-cache budget therefore does not store four times that many tokens automatically. Conversation lengths, shared prefixes, checkpoint counts, and branch ancestry determine actual capacity.

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

Pinned 4 MiB staging has a small copy-level benefit. It did not show a clear end-to-end cache benefit: sharing-off took 77.29 s and pinned-only 77.26 s. Sharing-on took 76.75 s and sharing-plus-pinned 76.67 s. Keep this experimental staging out of the release.

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
| Final no-warmup / logical-batch comparison | Complete. Reject batch 512 after cache-pressure analysis. Keep batch 2048 and normal warmup. |
| Context validation | Final draft-Q4 100K comparison complete: all five response hashes match original, including its wrong JSON answer. Final 8K/32K cold/repeat/followup checks remain pending. |
| Repeated-cache soak | Earlier batch-512 and microbatch-128 runs completed without swap/OOM. New final microbatch-256 run, 83 total requests including vision, remains pending. |
| Vision regression | Pending if included in final gate |
| Selected settings and release | MTP2, draft Q4, target Q8, `-b 2048 -ub 256`, `--ctx-size 110000`, `--cache-ram 4096`, `--ctx-checkpoints 3`, `--ctx-checkpoint-ram 384`. Clean release source `1af6247` builds successfully. Final runtime validation pending. |
| Source push | Root reports commit `3f8ed81` pushed and verified. Later changes may supersede it. |
| Running hub deployment | Pending: service, binary/commit, MTP depth, draft/target types |

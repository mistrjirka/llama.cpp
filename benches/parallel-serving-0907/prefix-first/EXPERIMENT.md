# Prefix-first restore experiment — 7 September 2026

**Status: experimental, not merged into `v100-optimized`.** Baseline is source `7571e2e17`; the baseline binary includes the vision-prefix guard correction. Production launch parameters have not been changed by this experiment.

## What was measured

Ornith AD-Q6_K-Q5_K on V100 32 GiB (200 W) + RTX 2080 Ti 22 GiB (250 W). Target and draft KV remain Q8_0; Shisa Q5_0 MTP3, CPU projector, layer split 14:35, 1.4M physical KV capacity, four logical 400k slots, batch/ubatch 2048/256. Actual histories: four 100,000-token histories sharing 98,000 tokens, appending 1,000 tokens and generating 128 each. The corpus is repetitive synthetic text, not a quality evaluation or four full 400k histories.

The first prototype prefers direct-prefix restoration rather than allocating full copies before deduplicating. `get_n_kv()` uses the highest occupied physical cell, so holes left by full-copy-then-dedup enlarge the attention span.

Four process arms ran A/B/B/A. Each arm had one excluded warm-up and two measured turns: four measured turns per condition. Exact same serialized histories and runtime parameters. Shared-library mappings were checked to prevent accidentally testing the same binary twice. Restore time is excluded from the turn measurements below.

| Metric | Baseline | Prototype v1 | Change |
|---|---:|---:|---:|
| Mean per-request reported PP | 146.96 tok/s | 210.36 tok/s | +43.14% |
| Mean per-request reported TG | 28.48 tok/s | 34.14 tok/s | +19.87% |
| Entire four-request turn | 12.603 s | 9.411 s | -25.33% |
| Total output / whole turn | 40.63 tok/s | 54.41 tok/s | +33.9% |

These PP numbers are **per-request server timings under concurrency**, not GPU kernel throughput; do not sum them as independent throughputs. All synthetic output hashes matched. Full-turn measured ranges were 12.472–12.724 s for A and 9.316–9.482 s for B.

## Correctness findings and why it is not a production claim

Four distinct code-review requests sharing ~12k source-code tokens exposed output differences. Inspection also found that the existing direct-prefix reader wrote serialized prefix tensors into already-live donor cells. The revised prototype (`v2`) skips these bytes instead, preserving the donor and avoiding unnecessary prefix transfers. Its token-only file metadata API passed ten checks, including bad magic/version, truncation, capacity limits and empty metadata.

The stricter A/B/A code test saved all four restored target states, draft states and speculative carries before generation. **Every v2 serialized state matched the baseline byte-for-byte**, and baseline state/outputs repeated identically. Nevertheless, three of the four B continuations differed from A, first at generated positions 8, 12 and 25; the fourth matched. This isolates the remaining issue beyond the serialized cache contents. Different attention reduction layouts or batch execution are plausible explanations, but no logit-level attribution or quality validation has been completed. Do not label the new inference path token-identical or deploy it on that assumption.

The ~34% throughput result above belongs to v1. v2 has not had a repeated long-context throughput comparison. On the separate short-code screen, A/B/A four-request wall times were 3.462/3.304/3.476 s, not a broad performance claim.

## Next acceptance gates

1. Compare logits at the first divergence using identical teacher-forced tokens; repeat with MTP disabled and controlled request ordering.
2. Trace physical KV extent, FP16 expansion and attention kernel/reduction choices. Match numerical tolerance against baseline rather than assuming that token differences prove quality loss or are harmless.
3. Re-run A/B/B/A on v2 after correctness attribution, with realistic 100k and near-350k histories, unequal lengths, and a three-decode/one-prefill workload.
4. Exercise empty, corrupt, legacy target-only and actual-image slot restores end to end. Token-metadata unit checks passed, but the earlier end-to-end guard script stopped at its code-output assertion and those later guard checks did not run.

Raw timings and hashes are retained beside this file. `direct-first-v1.patch` identifies the prototype used for the long-context timing table. Large model/snapshot fixtures remain in the persistent Development Sandbox, not Git.

## Related research

- Hydragen: https://arxiv.org/html/2402.05099v2 — separate shared-prefix and private-tail attention, combine with correctly normalized softmax.
- PAT: https://arxiv.org/html/2511.22333v3 — prefix-aware query packing and multi-tile kernels; its A100 Qwen3-30B-A3B tool-agent experiment reports 5.53–16.9% lower time per output token, not a forecast for our hardware.
- FlashInfer: https://arxiv.org/html/2501.01005v2 — composable cache layouts and workload-aware tiling. The described tensor-core templates cover SM75 and later, so V100 needs an adapted implementation.
- Sarathi-Serve: https://arxiv.org/html/2403.02310v1 — decode-first prefill budgeting; very small chunks can hurt PP through repeated KV reads.
- POD-Attention: https://arxiv.org/html/2410.18038v2 — co-execute complementary prefill/decode attention work.
- Nightjar, latest revision checked 15 June 2026: https://arxiv.org/html/2512.22420v5 — adapt speculation to concurrency and include switching/draft-cache reconstruction costs.
- Feather: https://arxiv.org/html/2605.06046v1 — prefix locality versus batch size; useful for queue grouping but not a reason to starve unrelated requests.

No lower target/draft KV precision, lossy eviction or approximate attention is proposed.

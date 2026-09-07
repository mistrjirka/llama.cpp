# MTP regression: cost analysis and next interventions — 7 September 2026

## Scope

This is analysis plus fresh diagnostic measurements, not a production fix. The source inspected is `80247767c`; measurements reuse its previously recorded immutable `parallel-refined-0907/final-bin`. All experimental attention, compact-restore, grouping, adaptive-depth, sampling-view and bulk-hidden flags are absent. Production `v100-optimized` and `local-llm-setup` are unchanged.

Hardware/model: V100-SXM2 32 GiB (200 W) and modified RTX 2080 Ti 22 GiB (250 W); Ornith AD-Q6_K-Q5_K, Shisa Q5_0 MTP, target/draft Q8_0 K/V, CPU vision projector. Four resident 100k actual C++ source histories sharing ~98k; suffixes 38/42/44/41 tokens, 128 output tokens per request. Same 1.4M physical pool, 400k logical caps, layer ratio 14:35, target ubatch256, draft ubatch128, one pipeline copy. No 350k or full-400k performance claim.

No profiling, compilation, or another GPU task runs during timed serving. Every successful arm has two process lifetimes, one excluded warm-up and one measured turn per lifetime. Two retained observations are a screen, not broad statistical confidence. State restoration is excluded from turn timing. Different MTP depths can change numerical execution and generated trajectories; these timings are not quality equivalence tests.

## 1. Shorter speculation helps, but does not reach the true off baseline

Mirrored order: off / requested-MTP0 / MTP1 / MTP2 / MTP3 / MTP3 / MTP2 / MTP1 / requested-MTP0 / off. The requested-MTP0 failures are excluded, not treated as measurements.

| Mode | Complete four-agent turn | Observed range | Mean TG per request |
|---|---:|---:|---:|
| True off, no speculative context | 5.732 s | 5.715–5.749 | 26.14 tok/s |
| MTP1 | 5.964 s | 5.935–5.992 | 25.36 tok/s |
| MTP2 | 6.345 s | 6.179–6.512 | 23.84 tok/s |
| MTP3 | 6.405 s | 6.394–6.416 | 24.08 tok/s |

MTP1 reduces turn time about 6.9% against MTP3, but is still 4.0% slower than true off. MTP3 is 11.7% slower than off in this screen. Do not infer that MTP1 wins at every context; prior short-context tests favored different depths.

The earlier repeated `gated-results.json` control data also show that acceptance did **not** collapse with concurrency: one agent accepted 320/564 (56.7%) and four agents 1285/2209 (58.2%). Under the ordinary one-initial-token/one-bonus-per-verification accounting, these counters imply ~2.70 and ~2.72 emitted tokens per verification, respectively. The four-agent case evaluates roughly 1.45 target token positions per committed generation token (excluding the initial token), plus the draft forwards. These are token-work counts, not a claim that GPU execution time or total FLOPs scale by precisely 1.45.

This motivates reducing cost rather than assuming the draft becomes substantially less accurate with four agents. MTP3 is three serial draft forwards (with agents batched at each step), followed by up to 16 target verification rows for four agents. Ordinary decoding batches four target rows and already amortizes weight loads.

## 2. Zero proposals is not a proper off mode

A separate off/warm/warm/off diagnostic interposes only `common_speculative_draft()` with a counted no-op. The warm arm retains the normal MTP3 target/draft contexts, hidden-state export, cache maintenance, and server pre-draft orchestration. It emits zero proposed tokens in every response; the shim reports 254 invocations in each process (see `diagnostic-checks.json`).

| Diagnostic | Complete four-agent turn | Observed range | Mean TG per request |
|---|---:|---:|---:|
| True off | 5.718 s | 5.701–5.736 | 26.15 tok/s |
| No proposals, MTP infrastructure retained | 6.633 s | 6.582–6.683 | 22.13 tok/s |

The retained machinery costs ~16.0% in this screen. This is **not an exact timing of the existing adaptive controller's zero-depth branch**: that branch also avoids some prompt-copy/checkpoint preparation skipped by neither the diagnostic shim nor ordinary pre-draft orchestration. The diagnostic establishes the narrower point that removing proposal forwards does not automatically recover the true-off runtime. The difference includes hidden export, recurrent configuration, bookkeeping and draft-cache refresh; it does not isolate refresh alone.

The source confirms the controller's zero-depth path still calls `common_speculative_process()` after target decoding. Its cost model is keyed only by active slot count, not actual context span, acceptance mix, or whether reactivation incurs catch-up. It must not compare its warm-zero arm to an assumed free baseline.

There is also a separate reproducible zero-limit bug: launching MTP with `--spec-draft-n-max 0` then restoring this fixture aborts at `GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)` during the continuation. Source inspection shows the draft loop can still append its first proposal before checking the zero limit, while server output capacity was sized for zero proposals. This is a supported-candidate explanation, not proof that every zero-depth failure has this cause. Both reproductions and the distinct safe no-proposal diagnostic are retained. Do not recommend n-max=0 as a production disable command.

## 3. Most targeted unimplemented code opportunity: K/V-only refresh

`common_speculative_impl_draft_mtp::process_impl()` refreshes the draft after target verification using token IDs paired with *verified target* hidden states. It currently calls ordinary MTP decode with output flags false. The Qwen35MoE single-layer MTP graph nevertheless builds attention, output projection and feed-forward layers before selecting output rows. Existing Nsight process-stage records contain attention, Q8 conversion and matmul kernels, so this is not merely dead source code.

For this **single-layer** head, the persistent K/V entries depend on the input token embedding, target-hidden projection/normalization, K/V projections, and K positional transform. They do not depend on this block's subsequent attention result or FFN. The caller does not consume a newly predicted draft hidden state during this refresh; its pending carry comes from the verified target.

Proposed implementation: a dedicated cache-update graph for no-output refresh, computing only those dependencies and writing K/V with existing index/rotation/quantization logic. Avoid constructing a full attention mask/read when no attention is evaluated. Keep the full MTP graph for proposing tokens. Scope to the validated single-layer Qwen/Ornith path; do not apply the shortcut to arbitrary multilayer draft models.

Second refinement: target processing currently refreshes all verified draft candidates *before* acceptance/rejection is known. Separate target hidden-row capture from draft refresh, then refresh only the retained target prefix, batched across agents. Preserve the correct next-token carry and all rollback/snapshot invariants.

**Do not simply keep all draft-generated KV or delete the refresh hook.** Later speculative draft states used predicted hidden states, which are not identical to target hidden states even when token IDs match. Only dependency-proven reuse is safe.

Acceptance gate: identical target/draft serialized states and pending carry on fixed inputs; first/partial/full rejection; uneven draft lengths; live-prefix forking; RAM parking; disk reload; no-output prompt continuation. Profile to demonstrate the removed attention/FFN kernels and measure full turns, not merely fewer launches. This optimization has not been implemented or timed here.

## 4. Next priorities and paper connections

1. **Genuine cost-aware suspend/resume.** Distinguish active drafting, cheap warm maintenance, and suspended drafting. Suppress maintenance only while preserving a bounded journal of required target token/hidden pairs; rebuild missing draft K/V through a cache-only path at reactivation. Avoid full 100k+ target replay, use minimum dwell times/hysteresis and charge actual resumption cost. Nightjar v5 motivates explicit switching-cost accounting. This is the first regression-prevention target; recovering the off baseline is not a new MTP acceleration.
2. **K/V-only accepted-prefix refresh**, above. More focused than another broad target-attention rewrite and useful for prompt catch-up as well as decode. Its end-to-end upside is not established; earlier synthetic MTP traces put process maintenance at only ~3.4% of GPU kernel time, and warm-zero invokes maintenance more frequently than MTP3.
3. **Draft-only sink+window attention.** Test 4k/8k/16k recent tokens plus 64 initial sinks while target attention and Q8 precision remain unchanged. Actually compact/index the draft read set and use a rollback-safe ring; masking the full cache still pays much of the cost. Windowed-MTP specifically studies built-in hybrid-model MTP. Its 1M-context B200/H100 results are not predictions for 100k/350k on our GPUs. Test acceptance and total latency jointly.
4. **Depth allocation using measured benefit/cost.** Use the already-fixed positive per-sequence early limit (research commit `9b579df8a`, not yet production), empirical per-position acceptance, context span and actual verification batch geometry. For four agents, 0..3 gives only 256 possible depth assignments; score them cheaply from measured timing tables, accounting for the hybrid splitter and kernel-shape discontinuities. TETRIS motivates unequal budgets, but its extra-drafting and selection costs must not be ignored. MagicDec warns that long context can make speculation beneficial again, so never disable purely by agent count.
5. **Draft vocabulary shortlist.** Ornith's MTP projection lacks dense Qwen's shortlist hook. FR-Spec motivates a draft-only compact vocabulary, retaining full target verification. Time the actual LM-head projection first; test code identifiers and missing-token acceptance rather than assuming all draft matmuls are LM-head work.

The target-side profiler remains relevant, but the existing 4-query Q8 candidate deliberately does not handle the common 16-row four-agent MTP3 verification shape; its 4–5% results do not constitute a fix for this regression. General pipelining and readback cleanup had weak measured end-to-end gains. Prefer the interventions above before another broad kernel sweep.

## Validation matrix

Use 1/2/4 active agents; actual 12k, 100k and near-350k histories; representative code, explanation and retrieval tasks; target sampling settings held fixed; Q8 unchanged. Baselines: true off, fixed 1/2/3, warm maintenance, suspended/resumed and each individual intervention. Separate cold prompt construction, incremental PP, pure TG, whole-turn latency and model-switch restore. Test fixed target inputs and cache-state equivalence; when arithmetic is deliberately changed, compare numerical error against a stronger reference rather than assuming either exact greedy agreement or disagreement proves quality parity/loss.

## Sources

- Source inspection: `common/speculative.cpp`, `tools/server/server-context.cpp`, `tools/server/server-serving-experiment.h`, `src/models/qwen35moe.cpp`, `src/models/qwen35.cpp`, `src/llama-graph.cpp` at `80247767c`.
- Previous measurement records: `../parallel-refined-0907/gated-results.json`, `../nsight-parallel-0907/mtp-attribution.json`; separately `research/mtp-costs-0907:benches/parallel-serving-0907/mtp-costs/RESEARCH.md`.
- Nightjar v5: https://arxiv.org/html/2512.22420v5
- Windowed-MTP: https://arxiv.org/html/2607.21535v1
- TETRIS: https://arxiv.org/html/2502.15197v2
- MagicDec: https://arxiv.org/abs/2408.11049
- FR-Spec: https://arxiv.org/abs/2502.14856

The script/fixture provenance and immutable binary reference are in `manifest.json`. Raw response token arrays and diagnostic logs remain in the persistent sandbox; summaries, stripped request timings, diagnostic source and reproducer scripts are committed here. No new production performance gain is claimed.

## Implemented follow-up

The [K/V-only refresh implementation and measurements](../mtp-kv-refresh-0907/REPORT.md) now provide exact-cache checks on Ornith and Qwen, raw snapshot/output parity, isolated zero-budget fixes and Nsight proof of eliminated refresh kernels. The full regression and later suspend/windowing proposals remain distinct.

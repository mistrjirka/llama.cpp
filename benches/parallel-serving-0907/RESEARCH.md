# Parallel PP/TG investigation — 7 September 2026

## Scope and evidence

This is a screening investigation, not a new general throughput claim. Test hardware was the V100-SXM2 32 GiB (200 W) plus modified RTX 2080 Ti 22 GiB (250 W), using the real Ornith AD-Q6_K-Q5_K target, Shisa Q5_0 MTP head, Q8_0 target and draft K/V, CPU vision projector, layer placement RTX:V100 14:35, batch/ubatch 2048/256, draft ubatch 128, pipeline copies 1. The server reserved a 1.4M physical KV pool with four logical 400,000-token slots and YaRN factor 400000/262144.

**Actual measured histories were 100,000 tokens each, sharing 98,000 tokens. These were not four full 400k histories.** The existing corpus is repeated “The quick brown fox…” text, not a representative coding evaluation. Repetition makes token verification deterministic but can bias speculative acceptance and throughput. Test real coding traces before choosing production settings.

Base source was `099ad5d13`; measurements include the text-only vision guard correction described below. Raw local scripts, logs, snapshots, metadata and token outputs are retained at `/workspace/oai-qwen38-pp-lab/results/parallel-research-0907/`. Tracked JSON files beside this document retain timings and output hashes. The latest binary used was the `llama-multiagent-cache/build-sm70-75` build, not the older main-worktree binary.

## Confirmed bug: a loaded projector prevented text-prefix reuse

`server_tokens.has_mtmd` records multimodal capability. Slot histories set this flag when a projector is loaded, even when the history contains no image/audio chunks. The prefix-sharing eligibility checks incorrectly interpreted it as actual media content.

With the original guards, a text-only child of the already cached 98k parent processed all 100,000 tokens again. With `has_media()` checks against actual media chunks, all three child forks reused 98,000 tokens and processed only their 2,000-token suffix. The correction also covers RAM donor eligibility and restored-prefix deduplication.

Validation with the projector still enabled: a 3,000-token parent + 200-token child reused exactly the parent and matched a fresh-prefill 32-token output hash. Two actual-image requests succeeded, and neither was automatically prefix-forked. Existing capability and multimodal-position bookkeeping are unchanged. See `vision-guard-result.json`.

## MTP depth depends on concurrency

Each request appended 1,000 tokens to a restored 100k history and generated 256. The four-request rows ran simultaneously. MTP3 was measured twice, bracketing MTP1, MTP2 and MTP-off (one run each). Reported TG is mean per-request generation speed; total turn time includes prompt work and serving overhead. Do not add per-request PP figures and call the result a kernel throughput.

| MTP depth | One active request TG | Four active requests: mean TG per request | Complete four-request turn |
|---|---:|---:|---:|
| Off | 35.77 tok/s | 26.23 tok/s | 17.34 s |
| 1 | 49.37 tok/s | 26.48 tok/s | 17.66 s |
| 2 | 54.81 tok/s | 25.12 tok/s | 18.58 s |
| 3 | 61.23 tok/s | 27.66 tok/s | 17.30 s |

Single-request total turn time improved from 9.234 s without MTP to a mean 6.209 s with MTP3. Four-request end-to-end output throughput was effectively unchanged: 59.06 versus 59.20 tok/s, including PP. MTP1/2 did not win this screen. All output hashes matched for this repetitive corpus. This does not establish that MTP-off should become the four-slot default.

## Physical restore layout affects both PP and TG

An additional same-server comparison used the same four 100k histories, +1k prompt each and **128 generated tokens each**:

| Layout before the requests | Complete four-request turn | Total output / whole-turn time |
|---|---:|---:|
| Full disk restore followed by prefix deduplication | 12.950 s | 39.54 tok/s |
| Restore one parent, then construct compact live forks | 9.457 s | 54.14 tok/s |

All four output hashes matched. This is 26.97% less latency / 36.93% more end-to-end throughput in one paired screen, not an ABBA-validated production gain. The raw per-request PP/TG timings are in `layout-results.json`.

The current file restore path ordinarily allocates a full sequence first and deduplicates afterward; it uses direct-prefix restoration only after an ordinary restore fails. KV attention extent is derived from the highest occupied physical cell, not simply the count of useful cells. Consequently a deduplicated layout can retain gaps and needlessly extend masks/attention spans. This is a code-supported explanation for the measured layout sensitivity; a kernel trace/physical-extent trace is still needed to attribute the entire difference.

**Next experiment:** restore directly onto a validated live prefix first, falling back to ordinary restore when unsupported. Preserve the destination recurrent state and MTP carry, and test deletion, eviction, reload and differing prompt lengths. Avoid duplicate allocations/gaps instead of relying on later compaction. No direct-prefix-first file restore change is included in this investigation.

## Paper-informed priorities

1. **Prefix-aware, direct-Q8 attention for the actual multi-agent shapes.** Hydragen separates shared-prefix and private-tail attention and merges with correct softmax normalization. PAT adds query packing, multiple tile sizes and scheduling. FlashInfer provides composable KV layouts and load-balanced graph-compatible execution. Our unified layout already puts several agents' queries into the same tensor, so current tiles can already share some K/V reads: it would be incorrect to assume every prefix is always read four times. The remaining opportunity is deliberate packing, avoiding masked/private-tail work, and widening Q8 tiles inside the kernel rather than staging the whole occupied KV span into FP16. The specialized Volta Q8 verification path currently requires a narrow Qwen D256/T4/head geometry, not arbitrary concurrent MTP shapes.

2. **Latency-aware prefill budgeting.** The server already batches decode and prefill; adding basic continuous batching is not the missing feature. Its fixed batch budget can nevertheless put a substantial prefill burst ahead of the next generation step. Sarathi-Serve motivates decode-first chunked prefill with a profiled token budget. POD-Attention goes further by co-executing complementary prefill/decode attention work. First test an adaptive token budget with three decoding agents and one long prefill. A smaller budget everywhere can hurt PP, so measure both TTFT and p95 inter-token latency.

3. **Adaptive speculation, based on measured cost rather than acceptance alone.** Nightjar adapts speculative depth to batch/load. Our screen directly illustrates the principle: speculation is valuable for one request, much less useful at four. Fit a lightweight controller using active sequences, context lengths, accepted tokens per target step and draft/verification times. Do not remove MTP re-evaluation just because a TODO exists: draft hidden states and verified target hidden states need not be interchangeable.

4. **Prefix-aware admission and in-flight prefill reuse.** Feather shows the tradeoff between larger heterogeneous batches and smaller prefix-homogeneous batches. The parked-session cache creates an opportunity to choose related queued agents while enforcing fairness. Our automatic fork currently requires an idle completed donor; independently arriving cold requests with the same long prefix can still compute it repeatedly. An in-flight prefix registry or explicit parent-prefill/fan-out operation addresses that without reducing KV precision.

The first two priorities also apply without shared prefixes: masked-span avoidance, direct-Q8 kernels and careful prefill scheduling are useful for unrelated concurrent agents. Prefix sharing itself only helps exact common prefixes.

## Papers consulted

- Hydragen (2024): https://arxiv.org/html/2402.05099v2
- FlashInfer (MLSys 2025): https://arxiv.org/html/2501.01005v2 — reports 13–17% parallel-generation improvement in its tested serving setups, not a forecast for this fork.
- PAT (ASPLOS 2026; revision 16 March 2026): https://arxiv.org/html/2511.22333v3 — Qwen3-30B-A3B tool-agent experiment reports 5.53–16.9% lower time per output token on A100.
- Sarathi-Serve (2024): https://arxiv.org/html/2403.02310v1
- POD-Attention (2025 version): https://arxiv.org/html/2410.18038v2
- Nightjar (February 2026 revision): https://arxiv.org/html/2512.22420v2
- Feather (May 2026): https://arxiv.org/html/2605.06046v1
- CoDec (March 2026 revision): https://arxiv.org/html/2505.17694v2
- MineDraft (March 2026): https://arxiv.org/html/2603.18016v1 — its extra draft GPU is not a free optimization on this already memory-constrained two-GPU setup.

These results use different GPUs, models, precisions and request counts. They justify experiments, not transplanting headline speedups. Target and draft KV remain Q8; no lossy eviction or reduced-precision cache is proposed here.

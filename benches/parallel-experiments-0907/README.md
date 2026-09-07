# Parallel-serving experiments — 7 September 2026

**Research branch: `perf/parallel-all-0907`. No experiment here is a new production default.** The stable `v100-optimized` branch and `local-llm-setup` settings were not changed by this experiment series.

Four directions were implemented and exercised: compact-first cache restore, direct-Q8 multi-query attention, latency-budgeted prefill, and cost-adaptive MTP. The outcome is not four unconditional speedups. Prefill budgeting offers a measurable responsiveness/throughput tradeoff; the first attention prototypes regress throughput; compact restoration still has an unresolved numerical-equivalence gate.

## Hardware and scope

The tests use the real Ornith `AD-Q6_K-Q5_K` weights, Shisa Q5_0 MTP head, Q8_0 **target and draft** K/V, and a CPU vision projector. GPUs are the Tesla V100-SXM2 32 GB (200 W) and modified RTX 2080 Ti 22 GB (250 W). Layer-placement ratio is RTX:V100 `14:35`, target batch/ubatch `2048/256`, draft ubatch `128`, and pipeline copies `1`.

The server reserves 1.4M physical KV tokens and exposes four logical 400,000-token slots. **Performance measurements use 100,000 actual tokens per agent, sharing 98,000 tokens—not four full 400k histories.** The performance corpus repeats “The quick brown fox…”. This is useful for shape/performance screening but biases acceptance and does not establish coding quality. Separate numerical checks use approximately 12k tokens of C++ source and four different code-review instructions.

Every comparison below is **within the fork**, not against vanilla llama.cpp. Per-request PP/TG rates are server-reported averages; whole-turn latency includes prompt processing and generation. Do not add per-request PP rates and label the sum GPU kernel throughput.

## 1. Compact-first restore: promising, still gated

The ordinary file restore allocates a whole sequence and subsequently deduplicates its prefix. The prototype instead reads token metadata first, finds a compatible live donor, and restores only the divergent tail into new physical cells. Its companion change skips serialized shared-prefix tensor bytes rather than overwriting the live donor. Target, draft, speculative carry, and recurrent state are retained.

The initial v1 A/B/B/A screen used four 100k histories, +1k prompt and 128 generated tokens per request. Four measured observations per arm, after warmups, gave:

| Initial restore policy | Whole four-request turn | Mean request PP | Mean request TG |
|---|---:|---:|---:|
| Full restore, then deduplicate | 12.603 s | 146.96 tok/s | 28.48 tok/s |
| Compact-first prototype v1 | 9.411 s | 210.36 tok/s | 34.14 tok/s |

This is **33.9% higher whole-turn throughput on that synthetic fixture**, not a released speedup. The first version was subsequently hardened to avoid writing into shared donor rows.

The hardened v2 produced **byte-identical serialized target, draft and speculative states** to the ordinary restore, including each sequence's recurrent state. However, the C++-source continuation did not meet an exact-output gate. A teacher-forced A/B/A test held the token histories identical and disabled MTP: **3 of 96 greedy selections differed for compact versus ordinary layout, while ordinary versus ordinary repeated exactly**. Probability differences were also observed; they have not been dismissed as harmless rounding. See `prefix-v2-state-and-output-check.json`, `prefix-v2-forced-token-check.json`, and `prefix-v2-probability-deltas.json`.

The leading investigation is layout-dependent attention/numerical execution, since serialized inputs match, but its root cause is not established. No claim of preserved coding quality is made. `LLAMA_EXPERIMENT_PREFIX_FIRST=1` is therefore required to prefer compact restore in the final branch. Without it, the established ordinary-first policy is retained, including the prefix-aware fallback when an ordinary restore cannot fit.

The final build with all experiments disabled passed the same teacher-forced check with **0/96 baseline differences**. The summary is retained in `final-default-forced-token-check.json`, and final GPU-reference output is in `final-q8-correctness.txt`. This is a focused regression check, not proof covering all models and workloads.

## 2. Direct-Q8 attention: correct primitive, slower serving

The isolated Volta prototype targets the actual Ornith attention geometry: D256, 16 query heads, 2 KV heads, GQA8, and 1–16 packed queries. Q8 K/V are dequantized into per-tile shared memory; split accumulators are FP32. The existing Qwen W4 implementation remains separate and unchanged.

The initial four-query kernel passed **16/16 CPU-reference tests** covering KV lengths 512/4096 and query counts 1, 2, 3, 4, 5, 8, 12, 16. These use the existing backend test's numerical tolerance, not byte equality. Serving comparisons use the same compact layout with the kernel disabled/enabled, and the synthetic generated-token hashes matched.

| Four-query kernel, same server fixture | Whole turn | Mean request PP | Mean request TG |
|---|---:|---:|---:|
| Existing attention | 9.327 s | 204.80 tok/s | 34.85 tok/s |
| Direct-Q8 prototype | 10.062 s | 206.38 tok/s | 29.09 tok/s |

The A/B/B/A screen retained two non-warmup observations per arm. **TG regressed about 16.5%.** Avoiding the full KV conversion is not sufficient to make this implementation faster.

A second implementation shares loaded K/V across 8- or 16-query tiles using larger dynamic shared memory. It also passed the 16 CPU-reference cases. A bracketed tuning screen tested 16-query packing with 40/80/160 split limits and 8-query packing with 80 splits. None won: the 8-query variant reached about 30.45 TG/s against 34.16 TG/s for its bracketed control; the 16-query variants reached only about 16.7–17.5 TG/s. These variants remain explicit experiments, not recommended settings. Larger packing increases per-block work and resource use; profiling is required to attribute the regression rather than assuming bandwidth was the only bottleneck.

This is a direct-Q8/query-packing prototype, **not a complete Hydragen/PAT implementation**. It does not implement a prefix tree, separate shared-prefix/private-tail attention passes, or all paper scheduling mechanisms.

## 3. Prefill budget: smoother generation at a throughput cost

The controller changes the number of prompt tokens admitted when decoding is already active. Pure-prefill batches retain their normal maximum. It estimates milliseconds per prompt token from completed mixed iterations and uses the requested budget to set the next chunk size.

Workload: three agents begin streaming 192 generated tokens from 100k caches; after all three emit content, a fourth agent appends 8k input tokens and generates 16. We measure complete-group latency and gaps between visible SSE content events. **Visible gaps are not exact per-token latency**, because speculative output may arrive in bursts.

The looser-budget screen bracketed the alternatives with two baseline runs:

| Budget | Whole group | Largest visible generation gap | Group latency cost |
|---|---:|---:|---:|
| Unrestricted control, mean | 14.414 s | 2.333 s | — |
| 250 ms | 19.996 s | 0.268 s | +38.7% |
| 500 ms | 16.769 s | 0.540 s | +16.3% |
| 1,000 ms | 15.856 s | 0.863 s | +10.0% |

The 1,000 ms budget reduced the worst visible gap by about **63%**, but took 10% longer to finish the entire group. This can be useful for interactive responsiveness, but it is **not a universal PP/TG throughput improvement**.

A prior, separate 100 ms screen reduced gaps from about 2.9 s to at most 0.23 s and finished the three decoding agents around 10.8 s instead of 16.5 s. The fourth agent's prefill slowed, so the full group took about 19.8 s. That earlier screen overlapped CPU compilation and is retained separately in `controller-results.json`; do not directly mix its absolute times with the later bracketed table.

An optional token-quantum setting rounds admitted prefill to stable tile sizes. A separate follow-up (`aligned-results.json`) used a 1,000 ms budget and tested 128/256-token alignment:

| Aligned follow-up | Whole group | Largest visible generation gap |
|---|---:|---:|
| Control A | 16.746 s | 2.902 s |
| 256-token alignment | 15.868 s | 0.731 s |
| 128-token alignment | 17.582 s | 0.835 s |
| Control B | 14.593 s | 2.360 s |

The 256-token alignment gives a promising **smoothness result with total runtime between the two controls**. Baseline runtime varies substantially, so this is not evidence of a statistically established throughput improvement. It is worth a longer randomized mixed-load experiment, not a default-setting change.

## 4. Cost-adaptive MTP: no demonstrated win yet

The controller keeps a small cost table per active-sequence count. It explores draft depths 0–3, measures elapsed work per generated token, uses an exponential moving average, and occasionally remeasures alternatives. A per-sequence MTP bound is now honored inside the draft loop; target verification is unchanged. Iterations containing prefill or reused partial drafts are excluded from the depth-cost observations.

The first uniform four-request screen gave 13.24 s with adaptation versus controls of 12.64 s and 13.59 s. Mixed-load adaptation and adaptation combined with a strict prefill budget did not establish a benefit either. These are small screens, not confidence intervals. The prototype is simpler than Nightjar and does not model all switch costs, context-length buckets, or draft-KV maintenance choices. **Fixed MTP3 remains the production setting.**

## Controls and validation

All controls below are off unless explicitly supplied. Unset presence-based flags to disable them; do not use `=0` as an opt-out.

| Environment variable | Experiment |
|---|---|
| `LLAMA_EXPERIMENT_PREFIX_FIRST=1` | Prefer compact, direct-prefix restore; numerical gate unresolved |
| `GGML_CUDA_VOLTA_Q8_MULTI=1` | Enable the isolated Ornith direct-Q8 Volta attention prototype |
| `GGML_CUDA_VOLTA_Q8_MULTI_PACK=4/8/16` | Select maximum query packing for that prototype |
| `GGML_CUDA_VOLTA_Q8_MULTI_SPLITS=N` | Bound split count, clamped to 1–512 |
| `LLAMA_EXPERIMENT_PREFILL_MS=1000` | Bound mixed prefill admission using estimated time |
| `LLAMA_EXPERIMENT_PREFILL_QUANTUM=128` | Optional token-alignment quantum; zero/unset means none |
| `LLAMA_EXPERIMENT_ADAPTIVE_MTP=1` | Test the per-load speculative-depth controller |

The prefix metadata reader has 10 checks for valid reads, empty metadata, truncated files, invalid versions/magic, null pointers, and insufficient capacity. The standalone controller test covers disabled behavior, tiny batch capacity, pure-prefill bypass, learning across independent load buckets, and token alignment. GPU attention tests cover incomplete query tiles as well as full groups. The full production-size concurrency and real-agent quality/stability matrix is not completed.

## Reproducing the retained screens

`summary.json` summarizes the raw JSON files in this directory. The harnesses use the persistent model/token fixtures and immutable binary snapshots under `/workspace/oai-qwen38-pp-lab/results/`. `PARALLEL_RESULTS_DIR` and `PARALLEL_FIXTURE_DIR` can relocate those fixture directories; model paths remain configurable in the helper. Snapshots and GGUF weights are deliberately not committed.

Historical screens used compact-first v2 before it was placed behind a flag. To repeat their memory layout with a fresh final-branch binary, explicitly set `LLAMA_EXPERIMENT_PREFIX_FIRST=1`. The historical binary snapshots are pinned in each harness; do not substitute a new binary and call it the same measurement. All GPU workloads were serialized with the Development Sandbox `gpu:all` resource lock.

Example correctness commands from the repository root:

```sh
g++ -std=c++17 -I. benches/parallel-experiments-0907/test-controller.cpp -o /tmp/test-serving-controller
/tmp/test-serving-controller
python3 benches/parallel-serving-0907/test-file-token-metadata.py --library build-sm70-75/bin/libllama.so
# Hide other GPUs first so CUDA0 is the V100.
GGML_CUDA_VOLTA_Q8_MULTI=1 GGML_CUDA_VOLTA_Q8_MULTI_PACK=16 \
  build-sm70-75/bin/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT \
  -p 'hsk=256,hsv=256,nh=2,nr23=\[8,1\]'
```

## Research behind the experiments

- [Hydragen](https://arxiv.org/html/2402.05099v2): shared-prefix/private-tail attention decomposition and query batching.
- [PAT](https://arxiv.org/html/2511.22333v3): prefix-aware query packing, tile choices and scheduling.
- [Sarathi-Serve](https://arxiv.org/html/2403.02310v1): decode-prioritized, latency-budgeted chunked prefill; tile granularity matters.
- [Nightjar](https://arxiv.org/html/2512.22420v2): speculative-depth adaptation to load and measured goodput.

These papers motivate the experiments. Their reported GPU/model speedups are not substituted for measurements on this V100/2080 Ti system. No experiment here lowers target or draft KV precision, evicts attention tokens, or bypasses target verification.

## Nsight follow-up

Actual Systems timelines and Compute counters are now recorded in [the Nsight report](../nsight-parallel-0907/REPORT.md). They identify target-side and orchestration costs, low two-GPU decode overlap, and resource/issue stalls in the direct-Q8 prototype. No production defaults changed.

## Refined implementation follow-up

See [the refined implementation report](../parallel-refined-0907/REPORT.md) for batched output reads, measured removal of local-memory traffic, cooperative QK, shape-selective attention benchmarks on real source prompts, and the unresolved numerical-parity checks. All new controls remain opt-in.

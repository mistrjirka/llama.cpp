# Refined parallel-serving experiments — 7 September 2026

**Branch: `perf/parallel-refined-0907`. Production defaults are unchanged.**

This series separates an algorithmic opportunity from flaws in its first implementation. It implements two synchronization/readback changes and three successive attention-kernel refinements, validates the proposed mechanisms with Nsight, and measures each separately. It does **not** implement a new two-GPU pipeline or resolve the existing compact-restore numerical discrepancy.

The strongest result is a shape-specific attention candidate: about 31% less operator time for four queries against 100k KV, and about 4–5% lower measured complete-turn latency in two serving configurations. The candidate still changes model probabilities and is not approved for production. The readback changes have stronger equivalence evidence, but only a small/noisy latency benefit despite removing many synchronization calls.

## Environment and controls

Base revision: `221ddc9f4f10dbff76cb17714d2dc15521bb11e0` in the experiment branch, not vanilla llama.cpp. Hardware is a V100-SXM2 32 GiB at 200 W plus a modified RTX 2080 Ti 22 GiB at 250 W. Ornith target weights are the real `AD-Q6_K-Q5_K`, with Shisa Q5_0 MTP weights, **Q8 target and draft K/V**, and CPU vision projector. Layer-placement ratio is RTX:V100 14:35; target batch/ubatch 2048/256; draft ubatch 128; pipeline copies 1.

The physical pool reserves 1.4M tokens and exposes four logical 400k slots. **Actual measured histories are 100,000 tokens per slot, sharing 98,000. No four-full-400k or near-350k benchmark is claimed here.** Compact-first restore, the adaptive MTP controller and prefill budget are disabled in these tests. Ordinary restoration preserves the established physical layout.

The new fixture contains actual C++ repository source, four different review topics and source excerpts. It is no longer the repeated fox sentence. The appended task/template lengths are 38/42/44/41 tokens; each active request generates 128 tokens. This is a controlled source-review workload, not an evaluation of complete coding-agent tasks. `manifest.json` records all source-file hashes and the complete external fixture hash. Large token fixtures, model weights and serialized states remain in the persistent sandbox.

GPU experiments were serialized with the `gpu:all` lock. Builds completed before timed runs. Timed serving tests do not run under Nsight or the validation interposer. Each relevant condition has two fresh process lifetimes, an excluded warmup in each, and two retained turns per lifetime: four retained turns per condition. The host experiment uses a mirrored order; the attention serving experiment uses A/B/B/A. These are **not randomized trials**. Process lifetimes and warmups are retained in the raw JSON. The small sample is not a broad confidence guarantee.

Immutable binary directories and library SHA256 values are recorded in `manifest.json`; benchmark launchers set matching `LD_LIBRARY_PATH`. The full-model timings include HTTP admission, prompt append and generation but exclude state restoration. TG is mean server-reported generation speed **per active request**, not aggregate throughput.

## 1. Read sampling outputs with one synchronization

`llama_get_sampling_output_ith()` returns a borrowed view of the sampled token, probabilities, logits, candidate IDs and counts after one context synchronization. Existing row accessors remain available; the hardened view resolves and validates its row once, with unchanged candidate/logit fallbacks. The view is invalidated by the next decode/encode, output reallocation or context destruction.

`common_sampler_sample()` can consume this view instead of repeatedly synchronizing for individual fields. This does not move sampling to a different backend, change the sampler chain, or omit a necessary GPU completion wait. The grammar fallback retains its established path.

Opt-in: `LLAMA_EXPERIMENT_SAMPLING_VIEW=1`.

## 2. Read MTP target hidden states once per completed batch

The MTP target exports dense, unmasked next-token hidden-state rows. The new path synchronizes that target buffer once and accesses its rows directly, instead of calling the individually synchronized accessor for every row. Deferred prompt capture continues using its already-owned saved buffer. The intervening draft-context decodes cannot invalidate target-context output storage.

Opt-in: `LLAMA_EXPERIMENT_MTP_BULK_HIDDEN=1`.

### Equivalence and lifetime checks

A validation-only interposer compares new and old APIs **on the same completed output buffers**, rather than comparing independently executed floating-point trajectories:

- Ornith: 613 sampling-output views and 396 dense hidden rows checked; all matched.
- Qwen3.8-27B with built-in MTP and four slots: 352 sampling views and 213 hidden rows checked; all matched. This Qwen check uses a 2k prefix and 32k total context allocation, not a Qwen 400k benchmark.
- A fixed-input Ornith A/B/A test covered 48 next-token probes (12 steps × four sequences). Tokens and the returned pre-sampling probability records were identical, with 0/48 A/B and 0/48 A/A disagreements.
- Four Ornith slots continued through target/draft/speculative snapshot save, erase and restore with view validation enabled.

The checks are in `views-validation-summary.json`; the retained scripts include `verify_views.cpp`, `validate.py` and `qwen-check.py`. This is focused equivalence evidence, not blanket validation of every architecture, sampler or multimodal path.

### Timing and Nsight attribution

| Four active MTP3 requests | Mean complete turn | Observed range | Mean per-request TG |
|---|---:|---:|---:|
| Existing reads | 6.539 s | 6.491–6.588 s | 23.75 tok/s |
| Sampling view only | 6.397 s | 6.340–6.526 s | 24.06 tok/s |
| Bulk hidden reads only | 6.395 s | 6.170–6.572 s | 24.27 tok/s |
| Both | 6.431 s | 6.332–6.543 s | 23.98 tok/s |

The approximately 1–2% mean differences overlap run-to-run variation. **They do not establish a robust production speedup.** The two changes are not additive in this screen.

A separate Nsight Systems capture measured 19,529 `cudaStreamSynchronize` calls with the old reads and 8,848 with both changes, about **54.7% fewer calls**. Summed CPU time inside those synchronization APIs fell only from 3.857 s to 3.727 s; profiled TG windows were 6.242/6.147 s. Synchronization time overlaps GPU work and must not be added to it. This supports the intended mechanism, but also shows that many removed calls were cheap while the necessary waits remained.

The same diagnostic tried the existing target backend-sampling toggle. Its profiled window was 9.117 s, so it is not enabled. This toggle test is not a new optimized GPU sampler, an unprofiled throughput comparison, or evidence that GPU sampling cannot be improved.

## 3. Direct-Q8 attention: fix implementation defects before judging the idea

The old experimental kernel remains intact for comparison. `fattn-q8-refined.cuh` is a separate route; ordinary attention and Qwen's existing kernel are unchanged.

### 3a. Remove thread-local memory traffic

The first refinement uses scalar lane shuffles instead of lane-dependent array indexing in softmax fragment conversion, and bounds QK-loop unrolling. In a same-shape Nsight Compute test (4k KV, one four-query tile):

| Counter | Old prototype | First refinement |
|---|---:|---:|
| Registers per thread | 255 | 146 |
| Local-memory load sectors | 136,016 | 0 |
| Local-memory store sectors | 142,972 | 0 |

Both changes were applied together; this table does not separately attribute the register reduction to one of them. The local-memory counters establish that the new compiled kernel removes the measured traffic. Shared memory still limits resident blocks, so theoretical occupancy remains 12.5%.

**Removing this traffic alone did not improve unprofiled operator time.** At 100k KV, four-query times were approximately 794 us for the old prototype and 819 us for the first refinement. A bottleneck counter improving is not sufficient evidence of a speedup.

### 3b. Stop recomputing the same QK product four times

Four warps previously repeated the whole QK score calculation before each consumed a different 64-dimensional slice of the PV result. The cooperative specialization divides the QK reduction dimension among them and combines their FP32 score partials with explicit barriers.

A second revision uses ordered pairwise shared-memory reduction. The scratch is 2 KiB per four-warp group; even a sixteen-query variant remains within Volta's per-block shared-memory limit. It avoids blindly increasing a register limit or overcommitting shared memory. The reduction order differs from the baseline, which is relevant to the numerical checks below.

At 100,096 KV tokens, the four-query cooperative operator takes about **667 us**, compared with approximately **965 us** for ordinary attention and approximately 819 us for the spill-free but redundant-QK prototype. The first bracketed cooperative comparison independently improved approximately 811 → 677 us. These are complete attention-operator timings, including reductions, not just one selected kernel.

### 3c. Dispatch by the measured crossover, not by feature availability

More query packing still does not universally win:

| 100,096 KV operator | Four queries | Eight queries | Sixteen queries |
|---|---:|---:|---:|
| Ordinary attention, representative bracketed result | 965 us | 1,100 us | 1,489 us |
| Pairwise cooperative pack4 | 667 us | 1,330 us | 2,670 us |
| Pairwise cooperative pack8 | 670 us | 1,180 us | 2,357 us |
| Pairwise cooperative pack16 | 671 us | 1,179 us | 2,360 us |

The latter packing rows are tuning screens, not independent statistical estimates. The important conclusion is stable: this implementation wins for four queries at long KV, not for wide query batches or short context.

`GGML_CUDA_VOLTA_Q8_REFINED_AUTO=1` therefore restricts replacement to exactly **four queries with KV length at least 65,536**, while all other shapes retain ordinary attention. This is an experimental dispatch rule based on the measured shapes, not a claim of optimality at every context length.

### Operator validation

The forced new routes passed all sixteen original CPU-reference cases (KV 512/4096; queries 1,2,3,4,5,8,12,16). The final test list additionally checks 65,536 and 100,096 KV with four queries: **18/18 pass**. The inherited normalized-error tolerance is 5e-4, not byte identity. Larger-pack checks also pass, despite their poor speed.

The exact Ornith shapes were added to the **performance case list** as well, fixing the earlier harness mistake where a successful perf invocation could contain zero matching cases. Scripts check nonzero expected test counts.

## 4. End-to-end impact on real source-review prompts

This A/B/B/A comparison changes only the shape-gated cooperative kernel. Bulk-read flags, adaptive speculation, compact restore and prefill budgeting are off in both conditions.

| Serving workload | Ordinary TG / agent | Candidate TG / agent | Ordinary whole turn | Candidate whole turn |
|---|---:|---:|---:|---:|
| One active agent, MTP3 | 52.28 tok/s | 54.83 tok/s | 2.707 s | 2.591 s |
| Four active agents, MTP off | 26.33 tok/s | 27.86 tok/s | 5.697 s | 5.440 s |
| Four active agents, MTP3 | 24.10 tok/s | 24.23 tok/s | 6.404 s | 6.358 s |

The first two measured complete turns are approximately **4.3% and 4.5% shorter**; per-request TG increases approximately 4.9% and 5.8%. Their four-observation ranges do not overlap in this experiment. Four-agent MTP3 is essentially unchanged, with overlapping ranges: its usual sixteen-query verification shape stays on ordinary attention. This is the purpose of the fallback, not a failure to activate an alleged universal optimization.

These are candidate performance results, **not production-quality claims**. Different kernels can change the generation trajectory. In the one-agent test each condition repeated deterministically, but the A/B output hashes differed. Four-agent unchanged controls also produced more than one hash across repetitions; concurrent grouping itself is a confound. Raw timings, acceptance counts and fingerprints are retained in `gated-results.json`. Equal input, output length, quantization and reservation are controlled; identical generated token trajectories are not claimed.

## 5. Numerical gate remains open; increased precision was tested, not assumed

A separate test uses the same four saved histories and the same forced C++ token appends, with four newly processed tokens per request to activate the exact gated shape. MTP is off. It measures 64 target distributions across A/B/A, returning the top 128 probabilities. Controls repeat exactly.

For the cooperative half-PV candidate:

- Greedy selections differ on 1/64 probes; control A/A differs on 0/64.
- Largest probability difference among mutually reported tokens is 0.276.
- Mean **coarse-grained** KL is 0.0117 nats. Unreported tokens are merged into an OTHER bucket; this is a lower bound, not full-vocabulary KL.

To test whether the original half-precision PV accumulation boundary explained this, an additional specialization accumulates the PV MMA result in FP32 and explicitly remaps its lane fragments. It also passes 18/18 operator tests, but does **not** restore baseline distributions: 2/64 greedy differences, maximum mutually reported probability difference 0.428, mean coarse KL 0.0193. Its four-query 100k operator is about 715 us. The end-to-end ABBA table above belongs to the earlier half-PV candidate, **not** this latest precision probe.

These findings neither establish a coding-quality loss nor justify dismissing the differences as harmless rounding. Increasing one accumulator precision is not sufficient to reproduce the original numerical trajectory. The kernel remains opt-in pending stronger model-quality evaluation and numerical attribution. The unchanged compact-restore discrepancy is a separate unresolved experiment.

## Controls

All new features are off when their flags are absent. Presence-based flags must be **unset** to disable them; setting `=0` still enables them.

- `LLAMA_EXPERIMENT_SAMPLING_VIEW=1`: synchronized sampling-output view.
- `LLAMA_EXPERIMENT_MTP_BULK_HIDDEN=1`: one dense target hidden read per MTP processing batch.
- `GGML_CUDA_VOLTA_Q8_MULTI=1` plus `GGML_CUDA_VOLTA_Q8_REFINED=1`: refined direct-Q8 route.
- `GGML_CUDA_VOLTA_Q8_COOP=1`: cooperative QK computation.
- `GGML_CUDA_VOLTA_Q8_MULTI_PACK=4/8/16`: query packing for forced operator experiments.
- `GGML_CUDA_VOLTA_Q8_REFINED_AUTO=1`: restrict to the measured four-query/long-KV crossover.
- `GGML_CUDA_VOLTA_Q8_PV_F32=1`: FP32-PV precision probe for four-query tiles; this explicitly selects the cooperative implementation.

The serving candidate table uses MULTI + REFINED + COOP + AUTO + PACK=4, with PV_F32 absent. `pre-pv-f32.cuh` retains the exact prior kernel source used for that timing table. The final source contains the additional precision probe behind its separate flag.

## Reproduction and evidence

`manifest.json` identifies source fixture, immutable libraries and raw traces. Default fixture/results root is `/workspace/oai-qwen38-pp-lab/results/parallel-refined-0907/`; models remain at the persistent paths in the helper. The Python harnesses accept relocation through `REFINED_RESULTS_DIR`. `prepare.py` rebuilds the source fixture; its hashes must be recorded again if the checkout changes.

- `bench.py --suite host --repeats 2`: mirrored single-change host-read ablations.
- `validate.py`: same-buffer API checks, fixed-input probability comparison and slot persistence.
- `qwen-check.py`: second-architecture MTP/view check.
- `check-kernels.sh`: reference tests, local-memory/register counters and unprofiled operator brackets.
- `gated.py`: three serving workloads with separate A/B/B/A comparisons.
- `probe-gated.py` and `probe-pv.py`: identical-input numerical probes.
- `profile-host.py`: separate Nsight Systems synchronization-count capture.

Nsight traces, operator timing outputs and serving timings are distinct measurements. GPU clocks were not locked, though power limits and serialized workload execution were held constant. Profiling replay/node tracing adds overhead; profiler time is not substituted for unprofiled application time. Large binary traces and state snapshots are not in Git.

## What this pass establishes and what remains

It establishes that the original direct-Q8 implementation had avoidable local-memory traffic and redundant QK computation, and that correcting these plus selective dispatch changes the result from a broad regression to a limited, measurable speed opportunity. It also establishes that halving synchronization call count is not equivalent to halving synchronization cost.

It does **not** establish a safe universal direct-Q8 replacement, production model-quality parity, near-400k performance, full Hydragen/PAT behavior, or an improved two-GPU pipeline. Pipeline work requires protected input buffers and intact recurrent/MTP rollback groups; simply removing the synchronization protecting reused graph inputs would be an invalid experiment. Those changes were not made in this pass.

Production `v100-optimized` and `local-llm-setup` remain unchanged.


## Hardened view and fixed-batch follow-up

The final sampling-view implementation resolves its output row once rather than internally calling each old accessor. It validates null arguments and out-of-range indices, clears the output on failure, preserves the LLAMA_TOKEN_NULL sentinel, and returns pointer/count pairs with documented borrowed lifetimes. A new validation interposer explicitly tests null context, null output, INT_MAX/INT_MIN indices, then compares every successful field against the established APIs on the same buffers. Its 598 sampling views and 375 dense hidden-row checks passed. The hardened A/B/A replay has **0/48 token differences and identical returned probability records**, including a four-slot target/draft/speculative save/erase/restore continuation. See `hardened-validation-summary.json` and `hardened-view-check.txt`. The hardened final build also passed a four-slot Qwen built-in-MTP run with 363 sampling views and 3,308 dense hidden rows compared against the original APIs; see `hardened-qwen-validation.txt`. This is a short-context second-architecture check, not a Qwen 400k benchmark.

A separate standalone test removes HTTP admission and sampled continuations as confounders. Each arm starts with four independently restored 100k histories, submits the exact same token IDs and positions, and compares full-vocabulary logits for shapes [4,0,0,0], [1,1,1,1], [4,4,4,4], and [1,2,3,4], including nonmonotonic sequence order. This covers 34 output rows and all 248,320 logits per row. The control repeats byte-identically, and the independent group-branch control also matches it exactly.

The gated Q8 kernel has 0/34 immediate greedy differences in this probe, **but is not probability-identical**. Maximum probability deltas are 0.0125, 0.0156, 0, and 0.0682 across those shapes. The 16-output case bypasses the candidate and matches all logits exactly, confirming the selective-dispatch guard. This strengthens the isolation of the affected shapes, not a general quality-parity claim; the longer teacher-forced probes above still show divergent selections. `fixed-comparison.json` retains full-distribution errors, Jensen–Shannon divergence, total variation, and maximum logit differences.

The grouped arms in this auxiliary 256-ubatch test fall back to serial scheduling and are **not pipeline-effect measurements**. Their results only diagnose changes in arithmetic grouping. The separate `perf/parallel-group-pipeline-0907` branch uses matched 128-token ubatches, confirms effective double buffering, compares it with identically grouped single-buffer controls, and records actual device overlap. Do not combine its absolute times with the 256-ubatch results here.

The hardened-build A/B/B/A comparison was repeated without profiler or compilation: four measured turns per arm after per-process warmups. Mean whole-turn time was **6.4481 s off versus 6.3959 s on**, about 0.8% lower; observed ranges 6.3106–6.5563 and 6.2747–6.5640 overlap. This final run does **not** establish a robust throughput win. It confirms the earlier conclusion that a large reduction in sync-call count mostly removes inexpensive calls, not the necessary device waits. Raw data are `final-host-results.json` and `final-host-summary.json`. Independently scheduled generations can differ as batch-admission timing changes, so same-buffer field tests and fixed-input probabilities, rather than those generated hashes, establish view equivalence.

The complete source harnesses and immutable binary/fixture hashes are in this directory. Raw full-vocabulary dumps and model snapshots remain in Development Sandbox, not Git. This follow-up keeps all experimental execution flags opt-in and does not alter either production branch.

# Vertical MoE expert-cache handoff

Date: 2026-07-26

## Executive conclusion

The specific performance claim requested by the user has **not yet been proven**:

> Load a subset of experts into VRAM, execute only experts that are resident, ignore all nonresident experts, and show that this unsafe upper-bound topology is faster than llama.cpp's default horizontal CPU/GPU placement.

Two relevant probes exist, but neither proves that claim:

| Probe | CUDA graphs | Decode throughput | What it proves |
|---|---:|---:|---|
| Default llama.cpp auto-fit horizontal placement | Off | 21.50 t/s | Trusted matched baseline for the short coding workload. |
| Resident-only fragmented dynamic graph | Off | 13.34 t/s | Valid unsafe resident-only execution; misses were omitted, but CPU route callbacks and fragmented submissions remained. It was slower than baseline. |
| Resident-only fragmented dynamic graph | On | 14.00 t/s | CUDA capture helped slightly, but the topology was still much slower than baseline. |
| Forced-rebuild GPU route-map | Off | 20.25 t/s | Initially interpreted as a near-baseline resident-only result, but later debugging invalidated that interpretation. |
| Forced-rebuild GPU route-map | On | 20.22 t/s | Same invalidation as above. |

The last trustworthy answer is therefore:

- We did prove that an unsafe mode can execute only currently resident experts and omit misses.
- That implementation achieved only 13.34–14.00 t/s versus the matched 21.50 t/s default baseline.
- We did **not** prove that a clean, frozen, vertically cached resident-only graph is faster than horizontal placement.
- The apparent 20.25 t/s near-baseline result was not a valid cache result because the cache path was disabled during the scheduler rebuild.

## Repository and worktree

Active implementation:

- repository: `/workspace/llama-mainline-cache`
- branch: `expert-cache-mainline`
- base commit: `555881ebc` (`upstream/master` when inspected)
- state: large uncommitted prototype, with both tracked modifications and many untracked benchmark artifacts

Earlier research archive:

- `/workspace/MOEs-Llama/.project/HANDOFF.md`
- `/workspace/MOEs-Llama/.project/HANDOFF-2026-07-26-dynamic-moe-cache.md`
- `/workspace/MOEs-Llama/.project/handoffs/2026-07-26-moe-expert-cache-handoff.md`

The active repository's `.project/` notebook contains later work than the earlier archive. This file should be treated as the current entry point.

## User's intended architecture

Use VRAM as a vertical expert cache across many MoE layers:

1. Keep the common non-expert model skeleton on the GPU.
2. Keep canonical expert weights in host memory as the exact fallback and promotion source.
3. Store selected expert bundles from many layers in compact GPU slots.
4. Route resident experts to the GPU.
5. Eventually handle misses exactly through CPU fallback, asynchronous prefetch, or a measured blocking-streaming crossover.
6. Optimize complete 8/8 selected-expert coverage per layer rather than average individual expert hit rate.

The requested unsafe experiment deliberately omits step 5. It is an architectural ceiling test, not a correctness-preserving inference mode.

## Implemented mechanisms

The current worktree includes:

- masked CPU and CUDA `MUL_MAT_ID` support;
- complementary hot/cold route construction;
- compact per-layer GPU expert slots;
- exact CPU execution for cold routes in the normal dynamic path;
- asynchronous expert promotion;
- completion-based READY publication;
- prefill route observation and warm-start experiments;
- segmented recency-aware metadata;
- same-layer transition prediction experiments;
- execution-level hot/cold branch suppression;
- GPU-side dynamic slot mapping in routed CUDA kernels;
- benchmark warmup with performance-counter reset;
- explicit phase markers around warmup and graph rebuild;
- detailed transfer, submission, route-residency, and complete-layer metrics.

Important modified source files include:

- `ggml/src/ggml-backend.cpp`
- `ggml/include/ggml-backend.h`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cpu/ggml-cpu.c`
- `src/llama-graph.cpp`
- `src/llama-context.cpp`
- `tools/completion/completion.cpp`
- `tests/test-backend-ops.cpp`

## Proven cache-policy findings

### Long coding trace

A 31,563-token coding prompt followed by 255 decode steps produced:

- 32 dynamic MoE layers;
- 1,775 total cache slots, approximately 7.1 GiB;
- 51.67% resident-route coverage;
- 5.15% complete 8/8 layer coverage;
- zero complete 32-layer GPU expert paths;
- a mean 30.35 CPU-dependent dynamic layers per token.

The cache entered decode cold and filled only around decode step 246 because the policy admitted at most eight experts per token. Promotion latency was approximately 1.39 ms median and 2.25 ms p95.

### Warm steady-state behavior

With 256 warmup tokens followed by 256 measured tokens:

- resident-route coverage improved to 76.49%;
- complete 8/8 layer coverage improved to 22.60%;
- no token achieved a complete GPU-resident expert path;
- measured throughput was approximately 7.65 t/s versus a fresh baseline near 19.29 t/s;
- steady state showed heavy churn: 1,316 admissions and 1,284 evictions in the warm half.

### Locality simulations

At the same 1,775-slot capacities, idealized immediate-residency simulations gave:

- LRU: 83.19% route coverage and 37.55% complete layers;
- online pairwise Markov: 82.01% and 35.43%;
- offline whole-trace frequency: 79.41% and 28.88%.

None produced a complete 32-layer GPU expert path in the 255-token trace.

### Capacity curve

Ideal static-selection estimates from the coding trace were approximately:

- 2,048 slots: 83.5% routes, 37.3% complete layers;
- 2,560 slots: 89.0%, 51.9%;
- 3,072 slots: 93.0%, 65.1%;
- 4,096 slots: 97.6%, 85.5%;
- roughly 6,144 slots for near-total trace coverage.

These are locality ceilings, not runtime results.

### Synchronous current-token loading

The measured cost does not support synchronously loading every current-token miss:

- full promotion latency: approximately 1.39 ms median;
- profiled all-eight-miss CPU cold branch: approximately 0.56 ms per layer.

Initial prediction/prefetch work should remain asynchronous unless an already-started transfer is close to completion and eliminates a final layer miss.

## Why the resident-only experiments were slow

The valid `evolved-gpu-only` mode did omit the CPU expert-compute branch, but it retained the CPU route-map callback and a fragmented scheduler graph. The matched 64/64 short coding runs were:

- graphs disabled: 13.34 t/s;
- graphs enabled: 14.00 t/s;
- matched baseline, graphs disabled: 21.50 t/s.

This demonstrates that removing expert arithmetic is not enough. Host intervention, graph fragmentation, backend splits, and many small submissions can dominate the saved compute.

## Critical correction: the forced-rebuild route-map result is invalid

The project notebook previously claimed that the 20.25–20.22 t/s forced-rebuild runs had removed the CPU route callback while preserving the GPU resident-expert graph. Later source changes and the final undocumented diagnostic run disproved that interpretation.

### Mechanism of failure

At the warmup boundary, `tools/completion/completion.cpp` sets:

- `GGML_MOE_DYNAMIC_FORCE_GPU_ONLY=1`;
- `GGML_MOE_DYNAMIC_GPU_ROUTE_MAP=1`;
- admissions and predictor prefetch to zero;

and then calls `llama_context_force_graph_rebuild()`.

`llama_context::force_sched_reserve()` in `src/llama-context.cpp` creates a new backend scheduler. The old expert-cache allocations still consume VRAM, but the new scheduler does not inherit the old cache ownership/state. Its automatic cache-budget calculation can consequently return zero available bytes. Auto slot planning then produces zero dynamic slots, so the graph builder disables the dynamic-cache feature.

### Direct diagnostic evidence

The final `graph-build-gates` probe enabled graph-build logging. In its measured phase:

- there were zero graph-build records with `gpu-route-map=1`;
- there were zero records with `force-gpu-only=1`;
- dynamic layers reported `feature=0`, `split-requested=0`, and `slots=0`;
- the reserved decode graph returned to 4,471 nodes and 68 backend splits;
- the single measured token ran at 16.96 t/s.

Related earlier diagnostics showed the new scheduler computing an expert-cache budget of zero after the old cache had consumed the free VRAM reserve.

Therefore the 20.25/20.22 t/s results cannot be used as evidence for resident-only GPU route mapping. Their near-baseline speed and matching output are consistent with rebuilding into the ordinary horizontally placed graph.

### Research-record implication

The conclusions in `.project/discoveries.md` sections 10–15 and `.project/experiments.md` about successful measured-phase GPU route mapping must be read with this correction. The topology investigation was still in progress when the previous assistant stopped, and the final 18:49–18:59 probes were not incorporated into those notes.

## Horizontal-versus-vertical comparison is still confounded

The short matrix did not compare pure horizontal placement against pure vertical placement.

The baseline auto-fit run used approximately 22.9 GiB of CUDA model buffers and horizontally overflowed expert tensors in later layers. The cache runs used a similar approximately 22.6 GiB horizontal model placement and then overlaid an additional expert cache from remaining VRAM.

Thus the experiments compared:

- horizontal auto-fit placement;
- horizontal auto-fit placement plus a dynamic vertical cache and its graph overhead.

They did not test a common GPU skeleton with all canonical expert tensors in host RAM and only compact cached experts in VRAM.

## Correct next experiment

### Name

**Startup-static frozen resident-only vertical ceiling**

### Question

With identical hardware, model, context, thread count, and VRAM limit, is a clean GPU common skeleton plus a frozen vertical expert set—executing resident routes only and ignoring misses—faster than llama.cpp's best default horizontal placement?

### Why this must be next

Policy, prediction, eviction, transfer overlap, and exact miss handling are irrelevant until the ideal resident-only topology demonstrates a speed advantage. If the unsafe ceiling cannot beat horizontal placement, improving hit rate cannot rescue the architecture without first reducing graph/kernel overhead.

### Required implementation change

Do not switch topology by destroying and recreating the scheduler after warmup.

Use one of these approaches, in preference order:

1. **Startup-static resident map.** Load a deterministic per-layer expert map synchronously before the measured graph is built, construct the GPU route-map graph from process startup, and never admit or evict during the measured run.
2. Move expert-cache ownership outside the scheduler and explicitly reattach the existing buffers, slot maps, READY state, and accounting to a replacement scheduler.
3. Rebuild only graph topology inside the existing scheduler without discarding cache state.

The startup-static map is the smallest and least ambiguous experiment.

### Pure vertical placement requirement

For the vertical arm:

- keep common non-expert tensors on CUDA where feasible;
- keep canonical gate/up/down expert tensors in host memory;
- allocate only compact cached expert bundles in CUDA VRAM;
- do not retain llama.cpp's horizontal partial expert-tensor overflow underneath the cache;
- use the same total VRAM ceiling as the baseline arm.

### Experimental arms

Run three matched arms:

#### A. Horizontal reference

- ordinary llama.cpp auto-fit placement;
- CUDA graphs disabled initially;
- no expert cache;
- expected-correct output.

#### B. Frozen resident-only vertical cache

- pure vertical placement as described above;
- startup-static compact expert map;
- GPU-side route-to-slot mapping;
- cold expert branch absent from graph construction;
- nonresident selected experts map to an invalid slot and contribute zero;
- no admission, eviction, promotion, or transfer during measurement;
- intentionally incorrect output; performance ceiling only.

#### C. Skeleton-only control

- same pure vertical placement and graph topology;
- skip all host-resident dynamic expert FFNs entirely;
- establishes the absolute ceiling for the common skeleton and non-expert work;
- intentionally incorrect.

### Resident map

Use a deterministic map so repeated runs are identical. A good first map is the top experts per layer from the saved coding trace, sized to the exact available VRAM. The map need not predict every route because misses are deliberately omitted, but it must produce substantial nonzero resident-route execution.

Record both:

- the exact map file/hash;
- per-layer slots and bytes.

### Matched workload

Start with the existing short coding workload:

- model: Qwen3.5-122B-A10B UD-Q3_K_XL;
- prompt: `profiling/coding-sanity-prompt.txt`;
- context: 4,096;
- seed: 1;
- temperature: 0;
- CPU threads: 40 decode / 48 batch, matching the existing script;
- CUDA graphs disabled;
- 64 warmup tokens;
- at least 256 measured tokens;
- five interleaved repetitions of A/B/C.

Then repeat at the target long coding context only if B or C establishes a clear ceiling advantage.

### Mandatory measured-phase assertions

Fail the run rather than record a throughput number unless all relevant assertions pass.

For B:

- every intended dynamic layer builds with `feature=1`;
- `split-requested=1`;
- `force-gpu-only=1`;
- `gpu-route-map=1`;
- configured slot count is nonzero;
- READY slot count equals the static-map population before measurement;
- resident GPU route count is nonzero;
- CPU cold expert-compute calls are zero;
- CPU route-map callback count is zero;
- admissions, evictions, promotions, and H2D expert bytes are zero during measurement;
- compact cache buffers remain at stable addresses;
- graph node/split count is recorded for the measured graph, not whole-run warmup.

For C:

- all expert compute for the target dynamic layers is absent or explicitly skipped;
- no hidden CPU expert work occurs.

These assertions would have rejected the invalid 20.25 t/s result.

### Metrics

Record per repetition:

- decode tokens/s and ms/token;
- graph nodes, leaf tensors, and backend splits;
- scheduler compute calls and backend submissions per token;
- cumulative CPU and GPU submit-return time;
- GPU utilization and idle fraction;
- CUDA model-buffer bytes, compact-cache bytes, compute/KV workspace, and minimum free VRAM;
- resident routes and complete 8/8 layers for B;
- output hash, labeled unsafe where appropriate.

Use trace-enabled runs only for one diagnostic verification. Keep tracing disabled in performance repetitions.

### Interpretation matrix

- **B > A with repeatable margin:** the vertical resident-only mechanism has a useful ceiling. Next work is a correctness-preserving, low-fragmentation miss path and realistic cache policy.
- **C > A but B <= A:** the common skeleton is fast enough, but compact expert kernels, slot mapping, or remaining graph splits erase the advantage. Optimize the resident GPU path before policy work.
- **C <= A:** even eliminating all target expert work does not beat horizontal placement. Scheduler placement/transfers or the supposedly common GPU skeleton are still wrong; profile topology before further caching work.
- **B is faster only at unrealistically low resident execution:** do not claim success. Sweep resident slots/routes to find the crossover between useful expert work and overhead.

### Optional second-stage sweep

After the three-arm proof, sweep static capacity while keeping topology frozen:

- 0% resident routes (skeleton control);
- approximately 25%;
- approximately 50%;
- approximately 75%;
- maximum feasible static map.

This measures whether resident expert work scales monotonically and estimates the performance ceiling at a hypothetical 100% hit rate without conflating policy or transfer costs.

## Next task in the user's words, normalized

Prove whether loading a subset of experts into VRAM and executing only the resident selected experts—while ignoring selected experts that are not resident—improves performance over llama.cpp's default horizontal CPU/GPU division. Treat this as an unsafe upper-bound experiment approximating the ideal 100% cache-hit steady state. Use a clean vertical placement, frozen cache state, no miss transfers or CPU expert fallback, and measured-phase assertions so a silently disabled cache cannot be mistaken for success.

## Existing benchmark artifacts

Most relevant:

- baseline and resident-only matrix:
  - `profiling/graph-mode-matrix-2026-07-26/`
- valid fragmented resident-only summaries:
  - `profiling/graph-mode-matrix-2026-07-26/gpu-only-off.summary.json`
  - `profiling/graph-mode-matrix-2026-07-26/gpu-only-on.summary.json`
- initially promising but invalidated route-map summaries:
  - `profiling/graph-mode-matrix-2026-07-26/route-map-off.summary.json`
  - `profiling/graph-mode-matrix-2026-07-26/route-map-on.summary.json`
- phase/scheduler tracing:
  - `profiling/scheduler-phase-trace-2026-07-26/`
- final graph-build diagnostic proving the cache path disabled:
  - `profiling/graph-build-gates-2026-07-26/`
- long-context baseline versus evolved cache:
  - `profiling/vertical-policy-benchmark-2026-07-26/`
- locality and capacity analysis:
  - `profiling/cache-locality-full-trace-2026-07-26/`
  - `profiling/analyze_cache_locality.py`
  - `profiling/analyze_cache_capacity.py`

## Current validation status

Before the last diagnostic probes, repeated commands completed successfully:

- `git diff --check` — exit 0;
- build of `llama-completion` and `test-backend-ops` — exit 0;
- focused CUDA `MOE_FFN_MASKED` tests — passed.

These validate compilation and focused masked kernels, not end-to-end model equivalence.

Correctness remains unresolved:

- free-running output equality is insufficient;
- some dynamic and baseline runs diverged;
- forced-token/full-logit replay is still required before any correctness-preserving claim.

## Worktree warning

The worktree is not a clean checkpoint. It contains thousands of lines of uncommitted code and many benchmark artifacts. Source files were modified after the last notebook update, especially:

- `src/llama-graph.cpp`;
- `src/llama-context.cpp`;
- `ggml/src/ggml-backend.cpp`;
- `tools/completion/completion.cpp`.

Do not make performance claims from the current tip without first checking graph-build assertions. Preserve or checkpoint the current diff before substantial restructuring.

## Bottom line

The project has established useful expert locality and identified graph fragmentation as a dominant cost. It has **not** established that the vertical-cache architecture is faster than horizontal placement, even under unsafe perfect-miss omission. The next experiment is not another cache-policy tweak. It is a clean startup-static, frozen resident-only vertical ceiling with hard assertions and a skeleton-only control.

---

## Experiment completed: frozen vertical ceiling (2026-07-26)

The recommended startup-static experiment has now been completed. The result is positive and supersedes the earlier statement that the performance ceiling was unproven.

Across three interleaved repetitions on the V100:

- horizontal auto-fit mean: **20.70 t/s**;
- frozen vertical resident-only mean: **42.10 t/s**;
- skeleton-only mean: **67.70 t/s**.

Frozen vertical was **2.03x faster** than horizontal (+103.36%). It used 4,678 READY expert slots in 144 persistent component buffers (18,298.22 MiB) and reduced batch-one scheduler splits from 66 to 7. All measured-phase no-transfer/no-cold-path assertions passed.

This is an unsafe upper bound: nonresident selected experts contribute zero, so output is intentionally incorrect. The architecture now has a proven mechanism-level speed advantage. The next research problem is to preserve that topology while restoring exact miss handling with minimal scheduler fragmentation.

Authoritative result: `profiling/static-vertical-ceiling-2026-07-26/RESULTS.md`.

---

## Experiment completed: exact CPU miss fallback (2026-07-26)

The frozen resident-only graph was extended with a complementary exact CPU miss branch. Route splitting is now implicit inside `MUL_MAT_ID`: the GPU path executes resident expert slots and the CPU path executes only canonical expert IDs absent from the immutable slot map. The old host route-map callback is not used.

Performance over three matched repetitions:

- horizontal auto-fit: **20.54 t/s**;
- ordinary `--cpu-moe`: **15.89 t/s**;
- frozen GPU hits + exact CPU misses: **8.55 t/s**;
- frozen GPU hits, misses omitted: **42.09 t/s**.

The exact graph is complete but incurs 49 CPU and 49 GPU submissions per token, 98 batch-one scheduler splits, and many small activation/partial-output transfers. Measured expert-weight copies remain zero. Thus the current miss path loses 58.35% versus horizontal and 46.16% versus ordinary CPU experts.

Teacher-forced full-logit replay on the 64-step coding workload showed exact split and ordinary `--cpu-moe` agreeing on all 64 top-1 choices. Both differed from horizontal on the same two steps with the same alternative tokens, consistent with ordinary backend-placement numerical drift.

The next task is **GPU spill-slot blocking miss service**: retain the seven-split resident graph, build a device-side miss list, upload missing expert bundles into reusable GPU spill slots, patch route IDs on device, and execute one GPU expert branch. A staged decode API or persistent executor is needed because miss IDs are data-dependent within each transformer layer.

Authoritative result: `profiling/static-exact-cpu-fallback-2026-07-26/RESULTS.md`.

---

## Adaptive-cache and predictor study completed (2026-07-26)

A real 1,024-token chat run confirmed that the dynamic cache warms and improves speed over time. Traced throughput rose from 4.72 t/s in the first 128-token window to 6.23 t/s around tokens 259-386 as resident-route coverage rose from 66.25% to 80.98%. It later settled near 6.0 t/s when routing changed phase. Overall coverage was 76.48%, with 10,209 admissions and 5,578 evictions.

The user's adjacent-layer predictor idea was implemented. The target layer learns normalized conditional transitions from both L-1 and L-2 routes. Offline K=8 recall is:

- old same-layer temporal predictor: 37.75%;
- L-1 conditional predictor: 50.45%;
- L-1 plus joint L-2/L-1 context: 55.64%.

A fair 512-token forced-history A/B showed cross-layer prediction improving complete layers from 32.86% to 33.55%, removing 169 CPU fallbacks, and saving about 451 MiB of H2D traffic, but aggregate traced throughput remained 5.8043 t/s in both arms.

The limiting fact is promotion latency: L-1 predictions reached READY in 7.64 ms median and L-2 predictions in 12.27 ms median. Zero predicted experts were READY before their target layer in the same token. These predictions improve future cache composition, not current-token exact misses. The next runtime mechanism must use direct GPU spill uploads outside the background promotion queue.

A full lossless compression scan found 12.32% overall zstd savings, but layers 0-1 save 72.77% while most later layers save only 1-4%. A compressed tier should be selective and GPU-resident/decompressed for early layers; compressing the already-quantized host model is unlikely to improve speed.

Authoritative report: `profiling/dynamic-real-chat-ab-2026-07-26/RESULTS.md`.

Next implementation target:

1. stable exact resident segment;
2. small fast-changing adaptive segment;
3. 2/4/8 exact GPU spill slots with direct staged upload;
4. L-2 early prediction and L-1 refinement feeding spill priority;
5. optional compressed VRAM L2 only for demonstrably compressible early-layer experts;
6. 2,048/4,096-token forced-chat evaluation with same-token READY rate, hit-rate windows, throughput, upload bytes, overflow, and full-logit replay.

---

## Latency floor and predictor algorithm research (2026-07-27)

Promotion latency is not fully hardware-limited. On the exact V100/CPU path, one normal expert bundle transfers in 0.3215 ms from pinned host memory or 0.4408 ms from pageable memory; one large bundle takes 0.4302/0.5594 ms. Eight normal bundles take 2.5507 ms pinned. The observed 7-12 ms prediction-to-READY delay is primarily caused by waiting to create transfer jobs until scheduler attachment reaches the target layer, plus component-level worker synchronization.

The current cross-layer predictor maintains conditional expert co-occurrence tables for L-1 -> L and L-2 -> L. It sums normalized conditional counts from the currently selected source experts, filters resident candidates, and admits the highest score under rotating budgets. It has no decay, joint feature interactions, router weights, hidden states, or deadline model.

On the same route-ID trace, K=8 recall improved from 50.45% to 55.49% by combining within-source expert-pair interactions, L-2/L-1 interactions, and the base conditional score. A sliding 64-token model alone reached 52.50%, confirming that conversation-phase recency is useful.

The immediate implementation target is a dedicated spill-service request ring that issues bundle copies at prediction time on a non-default CUDA stream, uses pinned candidate storage and bundle-level events, and bypasses graph-attachment timing. Keep the old worker for future-token background promotion.

Authoritative note: `profiling/latency-predictor-research-2026-07-27/RESULTS.md`.

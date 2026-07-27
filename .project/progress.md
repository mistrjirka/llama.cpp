# Project Progress

## Current objective

Move the MoE optimization toward a GPU-resident common layer skeleton plus a vertical expert cache that maximizes complete 8/8 selected-expert coverage per layer on realistic coding workloads.

The active investigation is cache locality and lifecycle correctness: determine why complete-route coverage is low, distinguish policy throttling from intrinsic expert unpredictability, and evaluate practical next-expert prefetching.

## Current state

The worktree contains an uncommitted dynamic expert-cache prototype on branch `expert-cache-mainline`.

Auto-fit keeps the common model skeleton on CUDA and selectively places later expert tensors in host RAM. The dynamic cache spans 32 MoE layers (16–47 in the latest full trace) and allocated 1,775 expert slots using about 7.1 GiB VRAM at 65k context.

Implemented mechanisms include masked CPU/CUDA `MUL_MAT_ID`, complementary hot/cold routes, compact slots, exact CPU miss execution, asynchronous promotion, completion-based publication, execution-level hot-branch suppression, VRAM-aware planning, detailed transfer/wait/submit tracing, and per-layer/per-token coverage metrics.

## Completed

- Added `.project/intent.md`, `.project/progress.md`, and `.project/todos.md`.
- Fixed auto slot planning to count actual host-resident gate/up/down expert tensors.
- Added complete-route coverage metrics and buffered full tracing.
- Added `profiling/analyze_cache_locality.py` with actual-cache, LRU, online-frequency, online-Markov, previous-token, and offline-frequency simulations.
- Found and fixed a concrete prefill placement bug: `force_split_host_moe_op` used `split_requested` even though dynamic splitting is decode-only, forcing ordinary expert FFN operations to CPU during multi-token prefill. It now requires `split_supported`.
- Focused masked MoE CUDA tests remain 2/2 passed after the fix.
- Completed a full trace on a 31,563-token coding prompt plus 255 decode steps.

### Full trace locality findings

- 32 dynamic layers, 8 routes/layer, 8,160 layer-decode steps.
- Cache capacity: 1,775 slots, approximately 50–57 experts/layer except layer 46 with 42.
- Actual resident-route coverage: 33,727/65,280 = 51.67%.
- Complete 8/8 layer coverage: 420/8,160 = 5.15%.
- Complete GPU expert paths: 0/255 tokens.
- Mean CPU-dependent dynamic layers: 30.35/32 per token.
- Coverage by decode quartile:
  - Q1: 30.18% resident routes, 0.20% full layers.
  - Q2: 63.28% resident routes, 7.57% full layers.
  - Q3: 50.24% resident routes, 7.13% full layers.
  - Q4: 62.63% resident routes, 5.62% full layers.
- Per-layer route coverage ranged from 35.10% (layer 17) to 66.13% (layer 30).
- Per-layer full8 coverage ranged from 0% (layer 16) to 13.73% (layer 30).

### Cache lifecycle findings

- 1,838 admissions, 1,838 READY publications, only 63 evictions.
- 5,392 candidates rejected solely by the global admission cap.
- Cache fill timeline:
  - 25% at decode step ~60.
  - 50% at step ~121.
  - 75% at step ~185.
  - 90% at step ~221.
  - 100% only at step ~246 of 255.
- Promotion latency: 1.39 ms median, 2.25 ms p95, 4.30 ms p99, 25.60 ms max.
- The cache does not learn from the 31,563-token prefill (`prefill-batches=0`); it enters decode cold.
- Current admission permits at most one candidate per layer invocation and eight globally per decode token. With 1,775 slots, the theoretical minimum fill time is 222 tokens.
- Eviction thrash is not the primary problem in this trace.

### Policy simulation findings

These simulations assume immediate residency updates and therefore are locality upper bounds, not measured runtime performance:

- Actual cache: 51.67% route coverage, 5.15% full layers.
- Online LRU at the same capacities: 83.19% route coverage, 37.55% full layers.
- Online frequency: 73.80% route coverage, 24.28% full layers.
- Online pairwise Markov predictor: 82.01% route coverage, 35.43% full layers.
- Offline whole-trace frequency upper bound: 79.41% route coverage, 28.88% full layers.
- None produced a complete 32-layer GPU expert path in this 255-token trace.

Previous-token overlap averages only 2.46–4.04 of 8 experts depending on layer. Reuse distance is nevertheless short: median 1–4 tokens and p90 roughly 14–35 tokens across layers.

Naive predictor churn is too large:

- Markov top-8: 37.35% route recall, approximately 234 MiB predicted additions/token.
- Markov top-16: 54.29%, approximately 393 MiB/token.
- Markov top-32: 70.96%, approximately 650 MiB/token.
- Full-capacity Markov: 82.01%, approximately 875 MiB/token.
- Idealized LRU: 83.19%, approximately 172 MiB of expert loads/token.

A practical predictor should therefore prefetch only a few high-value missing experts, especially experts predicted to convert a layer from 7/8 to 8/8, rather than rewrite the predicted cache set.

## In progress

- Design a prefill route-observation and bulk warm-start mechanism that does not perturb normal prefill placement.
- Replace the global rotating admission cap with a byte/time budget and value ranking.
- Add an explicit resident-route counter because the existing `ready-hits` field counts only routes actually executed on GPU after the full8 threshold; its printed 5.15% `hit-rate` is misleading relative to 51.67% residency.
- Evaluate constrained one-step predictors that prefetch 1–4 experts per layer/token and prioritize completion of 8/8 route sets.
- Run forced-token/full-logit correctness comparison; free-running baseline and dynamic outputs diverged early under deterministic generation.
- A benchmark-only same-process decode warmup is now implemented through `GGML_COMPLETION_BENCH_WARMUP_TOKENS`; it preserves the expert cache and resets llama.cpp performance counters after the requested number of generated tokens.
- A 256-token warmup followed by 256 measured coding tokens completed successfully:
  - cold half: 51.78% resident routes, 5.24% complete 8/8 layers;
  - warm half: 76.49% resident routes, 22.60% complete 8/8 layers;
  - measured warm decode: 7.65 t/s versus the fresh uncontended baseline's 19.29 t/s;
  - 0 complete 32-layer GPU expert paths in either half.
- The warm half had 1,316 admissions and 1,284 evictions, showing substantial steady-state churn after the cache filled. It also rejected 3,752 candidates because of the global admission cap.
- In the warm half, 2,171/8,160 layer steps (26.61%) were exactly 7/8 resident, and 69.74% had at least 6/8 resident. This makes completion-targeted prefetching a high-value policy direction.
- Added `profiling/analyze_cache_capacity.py` and generated capacity curves from the coding trace.
- At the current 1,775-slot budget, ideal static frequency gives about 79.4% route residency and 28.9% complete layers; ideal full-layer-oriented allocation gives 79.5%/29.9%. Per-layer reallocation is therefore useful but secondary to warm-start and replacement policy.
- Capacity curve under ideal static selection:
  - 2,048 slots: about 83.5% route / 37.3% full layers;
  - 2,560 slots: about 89.0% / 51.9%;
  - 3,072 slots: about 93.0% / 65.1%;
  - 4,096 slots: about 97.6% / 85.5%;
  - about 6,144 slots are needed for near-100% coverage on this trace.
- Cross-layer online pairwise prediction is promising offline: top-32 predicts 78.36% of next-layer routes and 26.12% complete layers; top-56 reaches 86.86% and 46.86%. A one-layer lead is likely too short for the current 1.39 ms median promotion, so practical prediction should target several layers ahead or the same layer on the next token.
- Current synchronous miss loading is not supported by measured costs: complete promotion latency is about 1.39 ms median while the profiled all-8-miss CPU cold branch averaged roughly 0.56 ms/layer. Predictor loading should initially remain asynchronous.

## Next useful steps

1. Add route observation during multi-token prefill without forcing expert computation onto CPU.
2. At the transition to decode, rank and bulk-populate useful experts from prefill observations within a measured H2D time/byte budget.
3. Raise/remove the fixed eight-admission cap in favor of queue-depth, bytes, and predicted payback limits.
4. Add resident-route versus executed-route metrics to avoid misleading hit-rate reporting.
5. Implement a constrained predictor using previous same-layer routes and online transition counts; prefetch only high-confidence missing experts with special value for 7/8 -> 8/8.
6. Compare LRU/decayed-frequency/predictor policies using identical forced token streams.
7. Profile transfer bandwidth and overlap after warm-start; do not optimize free-running output until numerical equivalence is characterized.
8. Resolve safe slot retirement and the standalone parallel zero-hot FFN test failure.

## Blockers and uncertainties

- No prefill-to-cache warm start exists.
- The fixed admission cap makes short and medium generations spend almost the entire run filling the cache.
- Full-token GPU paths remain unlikely even with stronger simulated policies because all 32 layers must simultaneously reach 8/8.
- Predictor simulations ignore real transfer scheduling and eviction safety.
- Free-running dynamic and baseline output SHA-256 values differ; forced-token logit analysis is required.
- Slot retirement and overwrite are not protected by last-use events.
- The existing `ready-hits` log field is an execution metric, not a residency metric.

## Verification

- `cmake --build build-v100 --target llama-completion test-backend-ops -j8` — exit 0.
- `build-v100/bin/test-backend-ops test -o MOE_FFN_MASKED -b CUDA0` — 2/2 passed.
- `git diff --check` — exit 0 before the latest build.
- Full trace run exit 0; trace size approximately 177 MiB.
- Trace event lifecycle: 1,838 admissions, 1,838 READY publications, 63 evictions.
- Promotion latency paired from `admit` to `ready_after_completion` for all 1,838 admissions.

## Important files

- `.project/intent.md`
- `.project/progress.md`
- `.project/todos.md`
- `ggml/src/ggml-backend.cpp`
- `ggml/include/ggml-backend.h`
- `src/llama-context.cpp`
- `src/llama-graph.cpp`
- `src/llama-graph.h`
- `profiling/analyze_cache_locality.py`
- `profiling/analyze_long_coding_trace.py`
- `profiling/run_long_coding_profile.sh`
- `profiling/cache-locality-full-trace-2026-07-26/`
- `profiling/long-coding-prefill-fix-2026-07-26/`

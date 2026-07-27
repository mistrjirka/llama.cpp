# Working Intent

## What is being achieved

Develop a MoE-oriented llama.cpp execution architecture that uses available VRAM as a vertical expert cache across the model. The target steady state is that all eight experts selected by each MoE layer are already executable from VRAM, allowing the token's active expert path to remain on the GPU without CPU expert work or a per-layer CPU/GPU merge wait.

The implementation should learn expert locality from realistic coding workloads, warm the cache before measured or production decode where possible, preserve useful residents, and asynchronously prefetch high-value missing experts before their layer deadline.

## Why it matters

The tested model activates only a small fraction of its experts per token. Keeping a GPU-resident common layer skeleton while caching useful experts from many layers can use VRAM more effectively than placing only complete layers on the GPU.

Current tracing shows useful temporal locality, but the existing policy enters decode cold, fills too slowly, churns heavily once full, and optimizes individual expert scores rather than elimination of CPU-dependent layers. The revised direction addresses those measured limitations directly.

The target is to outperform the best comparable default llama.cpp CPU/GPU placement on realistic, repeated coding-agent workloads. Cache activity, high individual-route residency, or isolated microbenchmark wins are not sufficient by themselves.

## User-stated requirements

- Use available VRAM aggressively but safely; reserve enough capacity for KV cache, graph workspace, compute buffers, promotion staging, and runtime variation.
- Prefer a GPU-resident common layer skeleton plus a vertical expert cache spanning MoE layers.
- Optimize for complete 8/8 selected-expert coverage per layer and ultimately a complete GPU-resident active path across layers.
- CPU/GPU mixed expert execution remains an exact cache-miss fallback, not the desired steady state.
- Add benchmark warmup so steady-state cache behavior can be measured separately from cold fill.
- Learn from prefill and prior route activations instead of entering decode with an empty policy state.
- Add asynchronous prediction and prefetching. Begin with interpretable online predictors, then evolve toward a small ANN per layer or shared model with per-layer heads when the baseline and logging are reliable.
- Analyze and support different cache capacities by layer when their marginal value, expert size, CPU cost, or predictability differs.
- Explore synchronous loading only where measured transfer plus GPU execution beats exact CPU fallback; do not block the current token by default.
- Performance claims require detailed profiling and careful comparison against default llama.cpp.
- Coding workloads are the primary optimization target.
- Random-token workloads are retained only for correctness, churn, queue, and race stress.
- Preserve exact routing and expert computation. Do not substitute approximate experts.

## Inferred intent

- Cache policy should value eliminating a layer's last CPU miss more than accumulating isolated expert hits. A predicted expert that changes 7/8 to 8/8 can remove the entire CPU branch and synchronization point.
- Cache capacity, cache replacement, admission bandwidth, and prefetch bandwidth are separate controls.
- Warm-start should use prompt observations without forcing normal prefill expert computation onto CPU.
- Initial replacement policy should be recency-aware and resistant to one-hit admissions. Segmented LRU or CLOCK-style probation/protection is preferred over pure decayed frequency.
- Admission should be limited by bytes, queue depth, measured transfer time, deadline, and expected payback rather than a fixed number of experts per token.
- Prediction should be deadline-aware. Same-layer next-token prediction has more lead time; cross-layer prediction can be more accurate but should target multiple layers ahead when one-layer lead is shorter than promotion latency.
- Preserve canonical CPU expert weights as exact fallback and promotion source unless a future design safely changes ownership.
- Common non-expert components should remain GPU-resident where feasible so 8/8 expert residency can produce a genuinely GPU-only layer.
- Cache state should eventually persist across turns and tool calls in coding-agent workloads when the same model context remains active.

## Active implementation direction

1. Benchmark-only same-process warmup that preserves cache state and resets timing counters.
2. Lightweight route observation during multi-token prefill.
3. Bounded warm-start population at the transition from prefill to decode.
4. Segmented recency-aware replacement with probationary and protected residents.
5. Byte/time/queue-depth admission budgets instead of the fixed eight-admissions-per-token cap.
6. Explicit resident-route, executed-route, 8/8-layer, CPU-dependent-layer, and full-path metrics.
7. Completion-targeted asynchronous prefetch using an interpretable online predictor.
8. Per-layer capacity allocation based on marginal 8/8 coverage, CPU time avoided, predictor confidence, and expert bytes.
9. Small ANN experimentation after the online baseline, forced-token replay, and cost-weighted evaluation are in place.
10. Event-safe slot retirement and overwrite before aggressive steady-state churn is enabled.

## Temporary implementation assumptions

- The current mixed CPU/GPU graph remains the exact miss-handling path while policy and prefetching improve.
- Dynamic expert execution remains decode-oriented initially; prefill observation must not require constructing the split graph.
- Predictor prefetch is asynchronous and speculative. CPU fallback proceeds if the predicted expert is not READY by its deadline.
- A synchronous wait may be considered only for an already-started transfer with small measured remaining latency and high value, especially 7/8 to 8/8 completion.
- Initial ANN work may be offline and trace-driven before becoming an in-process runtime dependency.
- Capacity analysis from one coding trace guides experiments but is not a universal allocation rule.

## Explicit non-goals

- Do not optimize the primary policy for uniformly random token streams.
- Do not claim improvement from one-off throughput measurements or trace-enabled timing.
- Do not synchronously transfer every current-token miss.
- Do not rewrite the entire predicted cache set every token; prefetch only a bounded number of high-value missing experts.
- Do not reduce cache capacity merely to reduce promotion traffic; control traffic and replacement separately.
- Do not treat raw individual expert hit rate as the sole success metric.
- Do not add a neural predictor before a simpler recency/transition baseline and reliable replay evaluation exist.

## Important uncertainties

- How much prefill observation predicts subsequent coding generation across different repositories, languages, tool outputs, and turns.
- The best warm-start byte/time budget and whether it can overlap the tail of prefill.
- The measured crossover among CPU fallback, asynchronous prefetch, and waiting for an already-near-complete transfer.
- How much per-layer capacity reallocation helps across diverse traces, rather than one captured generation.
- Whether cross-layer prediction retains its offline advantage when required to predict several layers ahead.
- The amount of predictor churn and wasted H2D traffic under realistic queue and eviction constraints.
- How much complete-token GPU coverage is achievable with realistic remaining VRAM. Good average layer coverage is insufficient across 32 dynamic layers.
- Whether free-running numerical divergence is benign split-order variation or indicates a correctness defect. Forced-token/full-logit replay is required.

## Useful completion indicators

- Multi-token prefill records route statistics without changing ordinary prefill placement or materially degrading prefill throughput.
- Warm-start fills high-value cache entries before measured decode under an explicit byte/time budget.
- Steady-state replacement churn and wasted promotions are substantially lower than the current policy.
- Admission and prefetch logs account for bytes, queue delay, READY time, deadline, actual use, 8/8 completion value, and eviction damage.
- Per-layer capacity is chosen by measured marginal value rather than near-even slot counts alone.
- Forced-token deterministic coding traces produce equivalent routing and acceptable logit error against baseline.
- Predictor evaluation beats recency/frequency baselines after accounting for transfer cost and unused predictions.
- Slot promotion, readiness, last use, retirement, and overwrite are event-safe.
- Repeated, interleaved coding benchmarks show a statistically credible end-to-end improvement over the best tuned default llama.cpp placement under the same VRAM, context, model, CPU settings, and output tokens.

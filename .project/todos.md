# Durable Tasks

## Active architecture track

- [ ] Fix auto VRAM slot planning so only dynamic-cache-eligible host-resident expert layers consume the slot-budget denominator.
- [ ] Add complete-route coverage metrics per layer/token: 8/8, 7/8, partial, zero, CPU-dependent layers, and full-token GPU paths.
- [ ] Measure common-skeleton VRAM cost at short and target coding contexts.
- [ ] Keep common non-expert layer components GPU-resident where feasible while expert weights use vertical caching.
- [ ] Change admission scoring to reward removal of the final CPU miss in a layer.
- [ ] Bundle gate/up/down promotion with per-expert completion events and safe slot retirement.
- [ ] Resolve the parallel zero-hot full-FFN test failure.
- [ ] Build and run deterministic multi-turn coding-agent replay benchmarks against tuned default llama.cpp.

## Deferred research directions

- [ ] Evaluate a compressed cold-expert tier in VRAM after the basic hot-cache and miss-path experiments are understood.
- [ ] Analyze the existing full per-token/per-layer expert route traces for next-layer and next-token prediction before implementing any predictor or prefetcher.

## Deferred stress work

- [ ] Retain random-token runs for churn, backpressure, race, and correctness stress only.

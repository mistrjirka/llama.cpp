# Vertical Cache Policy Evolution

## Purpose

Implement the active intent for warm-started, recency-aware, completion-targeted expert caching and predictor prefetching. This work is large enough to span multiple sessions and should remain grounded in forced-token coding traces and measured transfer costs.

## Workstreams

### Prefill learning and warm start

- Observe routed expert IDs during multi-token prefill without enabling the decode split graph or changing prefill device placement.
- Maintain per-layer frequency, recency, and transition summaries with bounded memory.
- Populate a bounded warm-start set at prefill-to-decode transition.
- Account for warm-start bytes, queue delay, completion time, and overlap with prefill tail.

### Replacement and admission

- Add probationary and protected resident states.
- Prefer eviction of unused or one-hit probationary experts.
- Promote repeatedly reused experts into the protected segment.
- Replace fixed admission-count throttling with byte, queue-depth, transfer-time, and expected-payback budgets.
- Reward candidates predicted to complete 8/8 layer coverage.

### Predictor prefetch

- Establish recency, frequency, same-layer transition, and cross-layer transition baselines.
- Predict several layers ahead and/or the same layer on the next token.
- Prefetch only bounded high-confidence missing experts.
- Never block current-token execution by default.
- Record prediction precision, transfer waste, deadline misses, and CPU branches eliminated.
- Evaluate a small ANN only after the online baselines and replay tooling are stable.

### Capacity allocation

- Estimate marginal route and complete-layer benefit per slot and per byte for each layer.
- Include measured CPU fallback cost and predictor confidence.
- Replan layer capacities at model load or warm-start boundaries, not continuously during decode.

### Correctness and lifecycle

- Add last-use events before aggressive slot overwrite.
- Resolve the standalone parallel zero-hot FFN failure.
- Build forced-token/full-logit replay and compare routing, logits, and outputs against baseline.

## Current smallest implementation slice

1. Preserve the benchmark decode warmup already implemented.
2. Add a prefill-observation path independent of decode split support.
3. Expose a bounded warm-start queue at the first decode step.
4. Add explicit residency and lifecycle metrics.
5. Verify unchanged prefill output and focused masked MoE tests.

## Useful completion

The first slice is useful when a long coding prompt produces non-zero prefill observations, starts decode with a partially populated cache under a bounded budget, preserves exact routing/output within the established tolerance, and improves first-quarter decode residency without forcing prefill expert computation onto CPU.

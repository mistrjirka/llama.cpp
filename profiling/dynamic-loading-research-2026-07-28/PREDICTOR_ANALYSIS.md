# Dynamic Predictor Headroom Analysis — 2026-07-28

## Question

Would combining previous-token expert choices, previous-layer expert choices, and a neural/token predictor materially improve dynamic expert loading?

## Current bottleneck position

Current measured throughput:

- mutable dynamic without prediction: 4.330021 tok/s;
- retained dynamic predictor: 4.551816 tok/s;
- frozen static: about 5.94 tok/s in current crossed tests.

Prediction already recovers roughly 5.1% over no prediction, but the mutable path remains roughly 30% slower than frozen static. The dominant demonstrated bottleneck is the 302-split mutable execution machinery versus the 152-split frozen decode topology. Partial READY conversion is also important: min-hot 1 was 6.03% faster than min-hot 4.

The true-configuration predictor trace recorded:

- 180 admissions over 32 measured tokens;
- 46.11% target precision;
- 85.56% READY before target use;
- 40.00% conversion into actual GPU routes;
- 5.625 admissions per token despite a nominal budget of 16.

Therefore the predictor has useful headroom, but improving prediction alone cannot remove the mutable topology or sparse branch/merge costs.

## Offline feature comparison

Global budget 16 precision on the retained chronological test:

| Method | In-distribution precision |
|---|---:|
| Previous-layer cross prediction | 49.90% |
| Previous-token temporal prediction alone | 27.24% |
| Naive cross + temporal | 47.52% |
| Naive cross + temporal + token | 55.01% |
| Learned route + temporal perceptron | 62.31% |
| Learned route + temporal + token perceptron | 64.36% |

Global budget 16 precision on the held-out prompt:

| Method | Cross-prompt precision |
|---|---:|
| Previous-layer cross prediction | 28.62% |
| Naive cross + temporal | 26.84% |
| Naive cross + temporal + token | 31.05% |
| Learned route + temporal perceptron | 38.22% |
| Learned route + temporal + token perceptron | 38.49% |

Conclusions:

1. Previous-token routes alone are weak.
2. Naively adding temporal scores can hurt because feature scales are not calibrated.
3. Learned fusion of previous-layer and previous-token routes is robust and materially better.
4. Token identity adds about 2.05 percentage points in-distribution but only 0.27 points cross-prompt. It should be a bounded correction, not the main predictor.
5. The retained experiments do not establish a benefit for a larger MLP over the sparse perceptron.

## Recommended fastest predictor

Use a sparse linear/low-rank fusion rather than a generic MLP.

For target expert `j` at target layer `L`:

```
score[j] = bias[L,j]
         + sum(W_cross[L, source_expert, j])
         + alpha * sum(W_temporal[L, previous_token_expert, j])
         + beta(confidence) * token_correction[token_bucket, L, j]
         + utility_adjustments[j]
```

Only eight source experts and eight previous-token experts are active, so the scorer sums a small number of rows instead of multiplying a dense feature vector. A rank-16 factorization can reduce each source row to a 16-dimensional embedding followed by 256 target scores.

Recommended feature priority:

1. current token's selected experts at layers L-1 and L-2;
2. previous token's selected experts at target layer L;
3. layer and prediction-distance embeddings;
4. current residency and transfer-deadline features;
5. token ID/hash embedding only when support and confidence are sufficient;
6. later, a small pre-attention hidden-state projection if route-only fusion saturates.

## Recommended runtime structure

Use a two-stage predictor:

1. Token-start temporal prior:
   - previous-token routes plus token correction predict all target layers;
   - provides long transfer lead;
   - issue only high-confidence or gate/up-only speculative transfers.
2. Same-token spatial refinement:
   - previous-layer routes update L+1/L+2 predictions;
   - cancel, reprioritize, or complete candidates;
   - router confirmation triggers remaining down-component transfer.

The score should optimize expected useful GPU execution, not route classification:

```
P(selected) * P(nonresident) * P(ready before use)
* P(route executes on GPU) * reuse_value
- transfer_bytes - queue/interference_cost - branch_cost
```

## Expected impact

The learned route-temporal fusion has enough offline headroom to justify implementation. A realistic initial target is approximately 1–3% throughput over the current predictor. More than about 5% from predictor quality alone is unlikely under the current architecture because:

- the current predictor already provides about 5.1% over no predictor;
- admissions average only 5.625 per token;
- 85.56% of admitted transfers are already ready before use;
- the 302-split mutable topology remains unchanged;
- partial route conversion and merge overhead remain.

The predictor becomes much more valuable when paired with a fixed or nearly fixed dynamic graph, event-driven publication, and component-selective transfer. In that architecture, prediction selects which fixed slots/nodes become active instead of forcing graph reconstruction.

## Implementation order

1. Implement sparse learned fusion of previous-layer and previous-token route sets.
2. Train/evaluate on multiple prompts with global-budget and useful-route objectives.
3. Add token correction behind confidence/support gating.
4. Run the scorer in the existing route callback first to establish benefit and latency.
5. If useful, move scoring and top-K to a small GPU kernel or pre-attention linear head.
6. Integrate with gate/up-first transfer and event-driven down completion.
7. Only then test a larger MLP.

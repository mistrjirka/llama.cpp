# GLM-5.2 per-layer slot-count plan — 2026-07-27

## Decision

Reject the per-layer slot-count override as not demonstrated.

The experiment preserved the existing static expert ordering and redistributed nearly the same cache bytes across layers. It passed output-hash, graph-topology, allocation, and VRAM gates, but five crossed repetitions showed no meaningful steady-state decode improvement.

## Accepted baseline

The accepted runtime remains:

- frozen static vertical expert map;
- automatic equal-byte layer planner;
- 3072 MiB cache reserve;
- 152 decode graph splits at batch size 1;
- deterministic output SHA-256 `e165b05c0568da0b099159b3045190e4f33ffbd97698d0a9881b982ead61f514`.

## Candidate plan

The candidate kept the current expert ranking and changed only each layer's prefix length. It was trained from the first portion of the 1025-step long route trace, with a minimum of 16 slots per layer and marginal route frequency per MiB as the allocation objective.

- baseline allocation: 14540.62 MiB, 1337 ready slots;
- override allocation: 14536.31 MiB, 1335 ready slots;
- candidate map: `slot-count-map.txt`.

Offline route coverage increased only slightly on the long trace and was essentially unchanged across the original and alternate short prompts. This was deliberately treated as a hypothesis, not a performance result.

## Runtime implementation tested

An opt-in `GGML_MOE_DYNAMIC_SLOT_COUNT_MAP` parser supplied per-layer slot counts after the normal auto-planner calculation. With no map, the default planner remained behaviorally unchanged. The GLM benchmark runner accepted a matching `SLOT_COUNT_MAP` parameter for crossed A/B tests.

## Correctness

Build:

```sh
cmake --build build-v100 --target llama-completion test-moe-split-backends -j 16
```

Result: exit code 0.

Focused test:

```sh
build-v100/bin/test-moe-split-backends
```

Result: exit code 0.

All ten GLM A/B runs produced:

- the exact accepted output SHA-256;
- 152 decode graph splits;
- no CUDA OOM;
- successful benchmark assertions.

## Five-run crossed A/B

| Mode | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | Median | Trimmed mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline tok/s | 5.9872 | 5.9393 | 6.0153 | 6.0338 | 5.8204 | 5.9592 | 5.9872 | 5.9806 |
| Override tok/s | 6.0219 | 5.6617 | 6.0730 | 6.0288 | 5.9182 | 5.9407 | 6.0219 | 5.9896 |

The raw override mean is 0.31% lower. The one-value-trimmed override mean is 0.15% higher. Both differences are far below run-to-run noise.

Per-token robust statistics reinforce the neutral result:

| Mode | Pooled median token latency | Mean excluding >=220 ms stalls | Stalls >=220 ms |
|---|---:|---:|---:|
| Baseline | 165.909 ms | 166.692 ms | 2 |
| Override | 166.052 ms | 166.670 ms | 3 |

The override's slow second run contained isolated 331 ms and 223 ms stalls; it was not uniformly slower. Conversely, its two best runs do not establish a repeatable improvement. The pooled robust difference is approximately 0.02 ms/token.

## Why coverage did not translate to speed

The static exact path always runs both a GPU hot branch and an exact CPU cold branch. Adding a slot is useful only when it removes CPU work from layers that lie on the token's critical path. Global route coverage treats all layer/expert hits equally and therefore cannot distinguish:

- CPU work already hidden behind the GPU branch;
- expensive versus cheap layer shapes;
- route hits that change the number of distinct cold experts;
- layers where synchronization, rather than arithmetic, dominates.

A useful planner needs measured marginal token latency, not marginal route count.

## Source disposition

The opt-in parser and runner plumbing were reverted after this result because they added runtime/repository complexity without a demonstrated performance benefit. The compact map, summaries, and analysis are retained as reproducible negative-result evidence.

## Next priority

Measure the actual CPU-cold critical path by layer. The next planner should value a slot by observed reduction in uncovered CPU expert work and token completion latency, ideally with coarse sampled instrumentation rather than the existing heavy profiler.

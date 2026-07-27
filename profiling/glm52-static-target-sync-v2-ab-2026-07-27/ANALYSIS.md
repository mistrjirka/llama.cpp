# GLM-5.2 frozen persistent target-sync experiment — 2026-07-27

## Decision

Reject both attempts to skip or defer the scheduler's target event wait for frozen persistent hot-weight inputs.

Both implementations preserved the exact output hash, memory use, and process correctness, but both reduced decode throughput. The original target wait is not a useful optimization target in the current 152-split mixed CPU/GPU graph.

## Motivation

An older exact-path profile reported approximately 78–85 ms of dynamic hot-input target synchronization over 12,288 hot splits, roughly 6–7 microseconds per hot split. The scheduler comment describes the wait as protection before overwriting a split input. Frozen static cache tensors are persistent and read-only after initialization, suggesting that steady-state reuse might not require that protection.

The expected benefit was small, so all throughput decisions used crossed five-run samples with the low-overhead decode marker.

## Common benchmark configuration

- frozen static GLM expert map;
- 3072 MiB cache reserve;
- 40 decode threads, 48 batch threads, poll 50;
- CUDA graphs disabled;
- verbose and graph-build diagnostics disabled;
- 16 warmup plus 32 measured tokens;
- identical forced-token stream.

All 20 A/B runs produced the accepted SHA-256:

`e165b05c0568da0b099159b3045190e4f33ffbd97698d0a9881b982ead61f514`

All runs completed with no CUDA OOM and the same 1364 MiB minimum free VRAM.

## Version 1: skip with an early cache lookup

Version 1 performed an additional `ggml_backend_sched_expert_cache_find()` before the existing attach path. When a frozen dynamic hot input already had a cache entry, it skipped the target wait.

| Mode | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | Median | Trimmed mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Skip enabled | 6.0189 | 5.9648 | 5.8177 | 5.7105 | 5.8349 | 5.8694 | 5.8349 | 5.8725 |
| Baseline | 6.0150 | 6.0190 | 5.9317 | 5.8818 | 5.8247 | 5.9344 | 5.9317 | 5.9428 |

Mean regression: **-1.10%**.

The implementation also doubled a linear cache-entry lookup on this path, scanning among as many as 225 entries. It was therefore replaced by a second implementation rather than being treated as a conclusive test of the wait itself.

## Version 2: defer until an actual overwrite

Version 2 added no cache lookup. It converted the wait into a one-shot lambda and called it only before an operation that could modify destination storage:

- VMM promotion;
- component promotion/upload;
- selective expert copy;
- ordinary backend copy.

A steady-state frozen persistent cache input with no pending component upload reached the existing persistent-cache `continue` path without invoking the wait.

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

Five crossed pairs:

| Mode | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | Median | Trimmed mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Deferred wait | 6.0310 | 6.0519 | 5.8966 | 5.6622 | 5.7237 | 5.8731 | 5.8966 | 5.8838 |
| Baseline | 6.0909 | 6.0658 | 5.9858 | 5.8844 | 5.8697 | 5.9793 | 5.9858 | 5.9787 |

Mean regression: **-1.78%**. Median regression: **-1.49%**.

The enabled arm lost every pair, from -0.23% to -3.78%. This is not a noise-only result.

## Interpretation

The target operation is an event/stream dependency, not merely host-side waiting. Its measured host cost is tiny, and it likely provides useful ordering or backpressure in the split-heavy mixed CPU/GPU schedule. Removing it can allow a less favorable queueing pattern even when the weight tensor itself is not overwritten.

The exact mechanism was not isolated, but the production decision is clear: retaining the wait is faster and simpler.

## Source disposition

The target-sync source changes and runner environment hook were reverted after the A/B. Compact summaries and this analysis are retained as negative-result evidence. The accepted 3072 MiB reserve optimization and low-overhead benchmark marker remain unchanged.

## Next priority

Do not pursue microsecond-scale split synchronization calls while CPU cold expert work remains the long branch. The next useful work should reduce CPU cold computation or remove CPU/GPU branch boundaries, for example through a bounded GPU spill-slot prototype or a layer-specific critical-path policy measured with coarse instrumentation.

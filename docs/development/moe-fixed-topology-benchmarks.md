# Fixed-topology MoE cache benchmark report

## Summary

This report covers an experimental fixed-topology dynamic MoE cache implemented in this branch of `llama.cpp`.

The key result is structural: with the same warmed expert set and exact output, replacing the legacy mutable 302-split decode graph with a persistent 152-split graph improved GLM-5.2 decode throughput substantially. Expert selection and upload policy were held constant in the strongest A/B, so the gain is attributable to reduced CPU/GPU coordination rather than faster weight copies.

The mode documented here remains experimental. The validated configuration uses one stable resident expert per MoE layer and disables decode-time replacement. Multi-slot mutation is not yet correct.

## Test system

| Item | Configuration |
|---|---|
| GPU | NVIDIA Tesla V100-SXM2, 32 GiB |
| CUDA architecture | 7.0 |
| Build | Release, CUDA enabled, CUDA Graphs enabled |
| Model | GLM-5.2 UD-IQ2_XXS, six GGUF shards |
| Model size | approximately 222 GiB across shards |
| Context | 2,048 tokens |
| CPU MoE | enabled for fixed/legacy expert-cache modes; disabled for horizontal baselines |
| Decode threads | 40 |
| Batch threads | 48 |
| Flash attention | enabled |
| Cache reserve | 3,072 MiB |
| Stable-map residency | one expert per each of 75 MoE layers |
| Decode-time admission | disabled |

These results are hardware-, model-, quantization-, and configuration-specific. They are not general `llama.cpp` performance claims. The compact machine-readable summary is stored in `profiling/dynamic-fixed-topology-prototype-2026-07-28/PUBLIC_BENCHMARKS.json`.

## Root cause being tested

The legacy mutable path produces four backend segments for each MoE layer:

```text
CPU route policy -> GPU resident branch -> CPU cold branch -> GPU continuation
```

Across 75 MoE layers this produces 302 scheduler splits, including approximately 151 CPU-to-GPU and 150 GPU-to-CPU transitions.

The fixed-topology path maintains a persistent device-side expert-slot map and uses two segments per layer:

```text
CPU cold branch -> GPU resident branch + merge + continuation
```

This reduces decode to 152 splits.

## Decode throughput

### Controlled topology A/B, CUDA Graphs disabled

The same synchronously warmed expert set was used in both paths. No post-warmup admissions were permitted.

| Path | Throughput | Output hash |
|---|---:|---|
| Fixed topology | **5.065196 tok/s** | `ac41c716...3ef7` |
| Legacy mutable topology | 3.594574 tok/s | `ac41c716...3ef7` |

Improvement: **40.91%**.

### Controlled topology A/B, CUDA Graphs enabled

This test used 16 warmup tokens followed by 32 measured tokens.

| Path | Throughput | Output hash |
|---|---:|---|
| Fixed topology | **6.339055 tok/s** | `e165b05c...f514` |
| Legacy mutable topology | 4.059881 tok/s | `e165b05c...f514` |

Improvement: **56.14%**.

A final rerun on the retained source reached **6.349746 tok/s** over the same 32-token measurement window with the exact expected output hash.

### Horizontal llama.cpp baselines

Two horizontal baselines are reported because this branch's modern `--fit` is more granular than classic whole-layer placement.

#### Modern `--fit` baseline

This baseline used `-fit on -fitc 2048`, no `--cpu-moe`, no manual `-ngl`, and the default 1,024 MiB margin. Verbose probing showed that it used tensor-level CPU overrides while keeping 79/80 layers logically offloaded; it is therefore **not** a pure whole-layer baseline.

A 256-token request produced **5.04 tok/s**, and the 1,793-token prefill produced **8.95 tok/s**.

#### Whole-layer-only baseline

For the classic horizontal comparison, autofit was disabled and the largest complete-layer count that fit was selected explicitly:

- `-fit off -ngl 11`: succeeds, 28.24 GiB CUDA model buffer;
- `-fit off -ngl 12`: fails while allocating the 1.75 GiB compute buffer.

The maximum valid pure whole-layer configuration is therefore `-ngl 11` on this 32 GiB V100.

The 256-token request completed 255 decode evaluations at **3.59 tok/s**. Fixed topology was **78.27% faster** than this maximum-capacity whole-layer baseline.

For reference, the four natural 256-token request results are:

| Path | Decode evaluations | Throughput |
|---|---:|---:|
| Fixed-topology expert cache | 255 | **6.40 tok/s** |
| Modern `--fit` with tensor overrides | 255 | 5.04 tok/s |
| Whole-layer-only `-ngl 11` | 255 | 3.59 tok/s |
| Legacy mutable expert cache | 255 | 4.17 tok/s |

These natural greedy runs did not produce identical complete output hashes, so this table is a performance comparison rather than an exact numerical-parity comparison. The controlled forced-token A/B above is the exact-output test.

### One-layer isolation

| Path | Decode splits | Throughput | Output parity |
|---|---:|---:|---|
| Fixed topology | 152 | 4.729537 tok/s | exact |
| Legacy topology | 154 | 4.579198 tok/s | exact |

This smaller A/B confirmed that eliminating only one extra dynamic-layer boundary improves throughput while retaining exact output.

## Token-length validation

The feature has been exercised at several lengths:

| Test | Tokens |
|---|---:|
| Short exact smoke | 3 generated tokens |
| Standard performance test | 16 warmup + 32 measured decode tokens |
| Long stability test | 256 generated-token request / 255 decode evaluations |
| Prompt-processing test | 1,793 tokens |

The 256-token request completed without a runtime failure:

| Path | Decode evaluations | Throughput | Complete output SHA-256 |
|---|---:|---:|---|
| Fixed topology | 255 | **6.40 tok/s** | `1f3fe9b9...ba4c` |
| Legacy mutable | 255 | 4.17 tok/s | `0c0a9103...4305` |

The unforced greedy token sequences diverged, so this is a **runtime-stability and performance result**, not an exact long-run parity result. Exact output parity is currently established on the 48-token forced decode sequence used by the controlled benchmark.

This does not establish unlimited-duration safety. It establishes the longest completed stable-map request currently included in this branch. Decode-time multi-slot mutation remains disabled.

## Prompt processing

The forced-token decode harness intentionally evaluates only a one-token prompt, so prompt speed was measured separately using normal tokenization of a 6,000-byte coding prompt.

Two crossed repetitions were run for the fixed and legacy dynamic paths:

| Path | Prompt throughput runs | Mean |
|---|---|---:|
| Fixed topology | 8.18, 8.16 tok/s | **8.17 tok/s** |
| Legacy mutable | 8.16, 8.14 tok/s | **8.15 tok/s** |
| Modern `--fit` with tensor overrides | 8.95 tok/s | **8.95 tok/s** |
| Whole-layer-only `-ngl 11` | 8.47 tok/s | **8.47 tok/s** |

Fixed topology was only **0.25%** faster than legacy mutable prompt processing, which is within ordinary run variance.

Interpretation:

The fixed topology has a large decode advantage but no demonstrated prefill advantage. Prompt evaluation uses batches up to 512 tokens and amortizes scheduler boundaries across many tokens, while autoregressive decode pays those boundaries once per generated token. Pure whole-layer placement was 3.67% faster than fixed topology for prefill, and modern tensor-level `--fit` was 9.55% faster.

Prompt processing is not the primary target of this optimization. The fixed graph mainly removes per-token decode coordination. Large-batch prefill uses different kernel shapes and amortizes scheduler overhead differently.

## Correctness and validation

Validated checks:

- exact SHA-256 parity on the 48-token forced decode sequence;
- 256-token request completion without a runtime failure; long-run natural output parity is not established;
- `build-v100/bin/test-moe-split-backends`: exit 0;
- release CUDA build: exit 0;
- `git diff --check`: pass;
- asynchronous one-slot-per-layer warm-start: exit 0;
- sampled promotion readback: byte-for-byte match.

Known failing check:

- multi-slot/decode-time mutation fails under NVIDIA Compute Sanitizer after upstream numerical/routing corruption reaches a later `SET_ROWS` kernel.

Therefore the experimental feature is not enabled by default.

## Reproduction

See [the run guide](moe-fixed-topology-cache.md) for the full environment and command.

The profiling runner used for the decode A/B is:

```sh
OUTDIR=profiling/my-run \
FORCE_TOKENS=/path/to/reference.tokens.txt \
WARMUP_TOKENS=16 \
MEASURE_TOKENS=32 \
MIN_HOT_ROUTES=1 \
CUDA_GRAPHS=1 \
DYNAMIC_FIXED_TOPOLOGY=1 \
DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=0 \
DYNAMIC_WARM_START_PER_LAYER=1 \
DYNAMIC_WARM_START_TOTAL=75 \
DYNAMIC_ASYNC_PROMOTION=1 \
URGENT_PREDICT_UPLOAD=0 \
CROSS_LAYER_PREFETCH_PER_LAYER=0 \
CROSS_LAYER_PREFETCH_TOTAL=0 \
bash profiling/run_glm52_scale_case.sh vertical 1
```

Use `DYNAMIC_FIXED_TOPOLOGY=0` for the legacy control.

## What the result proves

It proves that the large penalty in the tested legacy dynamic cache is primarily execution coordination and backend topology, not merely H2D expert-copy latency.

The fixed and legacy A/B used the same resident experts and exact output. A weight-transfer optimization cannot explain a 40–56% throughput difference when the transferred expert set is held constant.

## What the result does not prove

It does not prove that:

- arbitrary multi-slot dynamic replacement is correct;
- predictor-driven loading can already beat frozen static in production;
- this performance carries over to other GPUs or MoE architectures;
- the observed peak survives different context lengths, batch sizes, or prompts.

## Reddit-ready draft

**Title:** Experimental llama.cpp MoE cache: reducing 302 CPU/GPU scheduler splits to 152 improved GLM-5.2 decode by up to 56%

**Post:**

I have been experimenting with running a roughly 222 GiB quantized GLM-5.2 MoE model on a 32 GiB V100 using CPU fallback plus a GPU expert cache.

The surprising bottleneck was not mainly copying expert weights. The legacy mutable cache exposed four CPU/GPU graph segments per MoE layer, producing 302 scheduler splits and about 301 backend transitions per generated token. I added an experimental fixed-topology path with a persistent GPU `expert -> cache slot` map, reducing decode to 152 splits.

In a controlled A/B with the same warmed experts and exact output:

- CUDA Graphs off: 5.065 vs 3.595 tok/s, +40.9%
- CUDA Graphs on: 6.339 vs 4.060 tok/s, +56.1%
- final retained-source rerun: 6.350 tok/s

On a separate 256-token request, fixed topology reached 6.40 tok/s. Modern tensor-level `--fit` reached 5.04 tok/s, pure whole-layer `-ngl 11` reached 3.59 tok/s, and the legacy mutable cache reached 4.17 tok/s. The natural greedy outputs differed, so this is a speed comparison rather than an exact-parity test.

The longest current stability test requested 256 generated tokens. It completed 255 decode evaluations at 6.40 tok/s versus 4.17 tok/s for the legacy mutable graph. The unforced greedy outputs diverged, so I am treating that as a runtime-stability/performance result rather than exact long-run parity. Exact parity is currently validated on the 48-token forced sequence.

Prompt processing on a 1,793-token coding prompt was effectively tied between fixed topology and legacy mutable execution: 8.17 versus 8.15 tok/s. Pure whole-layer `-ngl 11` reached 8.47 tok/s, while modern tensor-level `--fit` reached 8.95 tok/s.

Important limitation: this is not production-ready dynamic replacement yet. The validated mode has one asynchronously warmed resident expert per MoE layer and disables decode-time admissions. Multi-slot mutation still has a CUDA correctness bug, so the feature is opt-in and documented as experimental.

The branch includes the source, run command, benchmark methodology, raw result summaries, and known failures. The next step is restoring numerical parity for multiple simultaneous resident routes, then attaching the learned expert predictor to the 152-split topology.

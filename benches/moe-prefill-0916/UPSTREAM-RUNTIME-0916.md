# Qwen3.8-Flash-Next fresh upstream / runtime-feature rebenchmark

Date: 2026-09-16

## Scope

Model: `Qwen3.8-Flash-Next-UD-IQ4_XS`, Q8_0 K/V, 131072 context capacity. Restore the same 100,000-token saved prefix, then time 1,000 new C++ fixture tokens. Eight prompt logits rows are requested. Model loading, prefix-state restore, continuation scoring, and first-setting warmup are excluded from the reported append medians.

Hardware: Tesla V100-SXM2-32GB followed by 22GB RTX 2080 Ti, 36/12 model-layer placement. Canonical expert tensors are pinned-host-backed in all matched arms. The request-wide executor may promote/cache selected experts in spare VRAM according to its stated 3072 MiB workspace and 16/10 base-residency allowances.

Upstream: `83078fec0db82d6b5a00d9599062c38c39145755`.
Fork runtime commit: `0ec1687d4e0c14db9ab9819b6b6368e8d9becb18`. The benchmark was run from the same runtime source immediately before that commit; only documentation/benchmark-record files were added afterward.
Fork loaded libraries: libllama `5205db4a716013b5140916f4d0ecc3321f9c13e1138ab971d455987af2f729c6`, libggml-cuda `e2b3941fd49fe99b6fc113ea545d5393427623c3031b8f0f918a6340a9aa506b`.

## Results

| Configuration | Median | Input tok/s | Throughput vs upstream | Throughput vs fork four-off |
|---|---:|---:|---:|---:|
| Current upstream | 8.325 s | 120.1 | baseline | -25.6% |
| Fork, four new runtime features off | 6.197 s | 161.4 | +34.3% | baseline |
| Request-wide executor; router fusion, exact-set top-k, selected attention off | 4.117 s | 242.9 | +102.2% | +50.5% |
| Request-wide + router fusion + exact-set top-k; selected attention off | 3.746 s | 267.0 | +122.3% | +65.5% |
| Full stack, including selected-entry attention | 2.860 s | 349.7 | +191.1% (2.91x) | +116.7% (2.17x) |

The full stack reduces measured new-input latency by 65.7% versus current upstream and 53.9% versus the same fork with all four new runtime features disabled.

Incremental effects in this tested preset:
- existing fork improvements with all four new switches off: +34.3% throughput over upstream;
- request-wide executor and its required internal memory/scheduling machinery: +50.5% over fork four-off;
- router fusion + exact-set top-k together: +9.9% over that request-wide baseline;
- selected-entry attention: +31.0% over the matched selected-attention-off configuration.

`request-wide` is not merely one CUDA kernel toggle: its harness includes the previously retained token-residency, expert-cache, prepared-route, plan-reuse, accumulation and transfer-overlap machinery required by that execution mode. The row should therefore not be read as a pure isolated `--moe-layer-first` microbenchmark.

## Repetition and output checks

Upstream and fork-four-off each produced a single repeated full-vocabulary hash within their six-round run. Their eight sampled top-token IDs agreed, although the complete score vectors were not byte-identical across the implementations.

In the ordered request-wide confirmation, the five measured selected-attention-off prompt files were byte-identical to one another and so were their fixed-continuation files. The five measured selected-attention-on prompt files were likewise byte-identical to one another, as were their continuation files. Dense versus selected-entry attention retained the same top token in all eight sampled prompt rows and all eight fixed-continuation rows, but their full score vectors differ.

This is a performance/repeatability check, not broad model-quality certification. The earlier FP32-request/FP16-value-accumulator issue and broader quality evaluation remain separate.

## Raw evidence

Results are under `/workspace/qwen-flashnext-upstream-bench-0916/results/mixed/`:
- `upstream-1000.jsonl` / `.log`
- `fork-off-1000.jsonl` / `.log`
- `request-base.jsonl` / `.log`
- `request-ordered.jsonl` / `.log` and per-round score vectors

The first attempted upstream run was rejected before timing because the test harness indexed requested logits incorrectly; the corrected harness asks for the actual eight flagged token positions. A first request-wide A/B was also rejected before timing because the per-round radix setting disagreed with the context parameter. Neither failed attempt contributes a row above.

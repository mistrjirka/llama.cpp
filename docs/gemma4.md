# Gemma 4

This fork has been benchmarked with both dense and routed-expert Gemma 4 variants. Use the normal hardware build first:

- [SM70](build-sm70.md)
- [SM75](build-sm75.md)
- [mixed SM70 + SM75](build-sm70-sm75.md)

No special global build flag is required just to use Gemma 4.

## What is automatic

The generic MoE router fusion is enabled by default with `--moe-router-fusion`. It only runs when the model graph is eligible.

On a standalone V100 with a 4,096-token prompt, Gemma 4 26B-A4B measured:

| Router fusion | Prompt processing |
|---|---:|
| off | 2595.1 tok/s |
| on | **2625.4 tok/s** |

That is a **+1.17%** throughput change in this workload. The E2B test produced no eligible fused-router calls, so its small timing variation is not attributed to this optimization. [Raw cross-model results](../benches/moe-prefill-0916/ROUTER-CROSSMODEL-0916.md).

The larger September 12 benchmark also includes Gemma 4 31B and 26B-A4B V100 rows in the [main comparison](benchmarks/upstream-table.md).

## Experimental features

The request-wide host-expert scheduler documented for Qwen3.8-Flash-Next has **not** been qualified for Gemma 4. Do not enable `--moe-layer-first` merely because a Gemma variant is MoE.

Selected-entry attention only activates when a model supplies the compatible sparse-history selection. It is not a generic way to make ordinary Gemma attention sparse.

[Back to the model chooser](../README.md#then-choose-your-model)

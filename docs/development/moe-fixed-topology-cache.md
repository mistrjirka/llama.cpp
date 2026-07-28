# Experimental fixed-topology MoE expert cache

This branch contains an experimental CUDA/CPU hybrid execution path for sparse MoE models that are larger than GPU memory.

The feature is **opt-in** and is currently intended for research and benchmarking. It has been validated with a stable, asynchronously loaded **one-resident-expert-per-MoE-layer** map. Decode-time multi-slot replacement is not yet safe and must remain disabled.

## Why this exists

The legacy mutable MoE cache exposes four backend segments per MoE layer:

1. CPU route-policy and residency masking;
2. GPU resident-expert computation;
3. CPU nonresident-expert computation and merge preparation;
4. GPU continuation.

On the tested 75-layer GLM-5.2 MoE model this produces 302 scheduler splits and approximately 151 CPU-to-GPU plus 150 GPU-to-CPU transitions per decode token.

The fixed-topology path keeps canonical expert IDs and a stable device-side `expert -> compact slot` map. Resident-route lookup and hot/cold merging stay in the fixed graph, reducing decode to 152 splits. The CPU still computes nonresident experts and manages cache policy; expert weight uploads remain a separate asynchronous worker operation.

## Status

Validated:

- fixed 152-split decode topology;
- one asynchronously warmed resident expert per MoE layer;
- CUDA Graph replay;
- exact output parity on the 48-token forced benchmark sequence;
- 256-token natural request completion; the unforced greedy output diverges from legacy;
- persistent device slot-map updates for the stable-map case.

Not validated for production:

- two or more simultaneously active mutable slots per layer;
- predictor-driven replacement during decode;
- eviction while a slot may still be in use.

Keep `GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=0`, disable predictive prefetch, and warm at most one expert per layer when using the documented safe configuration.

## Build

A CUDA build is required. The tested V100 build used compute capability 7.0:

```sh
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-completion -j
```

Choose the CUDA architecture appropriate for the target GPU.

## Run the validated experimental mode

The example below uses GLM-5.2 UD-IQ2_XXS on a 32 GiB V100. Cache reserve and fit targets are hardware/model specific.

```sh
MODEL=/path/to/GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf
PROMPT=/path/to/prompt.txt

GGML_EXPERT_CACHE_MIB=auto \
GGML_EXPERT_CACHE_RESERVE_MIB=3072 \
GGML_MOE_DYNAMIC_SPLIT_SLOTS=auto \
GGML_MOE_DYNAMIC_THRESHOLD=2 \
GGML_MOE_DYNAMIC_MIN_HOT_ROUTES=1 \
GGML_MOE_DYNAMIC_FIXED_TOPOLOGY=1 \
GGML_MOE_DYNAMIC_PREFILL_OBSERVE=1 \
GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=1 \
GGML_MOE_DYNAMIC_WARM_START_TOTAL=75 \
GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN=0 \
GGML_MOE_DYNAMIC_ASYNC_PROMOTION=1 \
GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD=0 \
GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER=0 \
GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL=0 \
GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_PER_LAYER=0 \
GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_TOTAL=0 \
build/bin/llama-completion \
  -m "$MODEL" \
  -f "$PROMPT" \
  -n 256 \
  -c 2048 \
  -b 512 \
  -ub 128 \
  -fa on \
  -dev CUDA0 \
  -t 40 \
  -tb 48 \
  --poll 50 \
  -fit on \
  -fitt 16384 \
  -fitc 2048 \
  --cpu-moe \
  -s 1 \
  --temp 0 \
  --ignore-eos \
  --single-turn \
  --no-conversation \
  --no-display-prompt \
  --simple-io
```

For models with a different number of MoE layers, set `GGML_MOE_DYNAMIC_WARM_START_TOTAL` to at most the number of MoE layers while retaining `GGML_MOE_DYNAMIC_WARM_START_PER_LAYER=1`.

## Run the horizontal whole-layer baseline

On the tested V100, the largest complete-layer placement that fits is 11 layers. Twelve layers fails while allocating the 1.75 GiB compute buffer.

```sh
build/bin/llama-completion \
  -m "$MODEL" \
  -f "$PROMPT" \
  -n 256 \
  -c 2048 \
  -b 512 \
  -ub 128 \
  -fa on \
  -dev CUDA0 \
  -t 40 \
  -tb 48 \
  --poll 50 \
  -fit off \
  -ngl 11 \
  -s 1 \
  --temp 0 \
  --ignore-eos \
  --single-turn \
  --no-conversation \
  --no-display-prompt \
  --simple-io
```

This is the pure whole-layer comparison. Current `--fit` in this branch can use tensor-level CPU overrides, so `-fit on` is a different, more granular horizontal baseline.

## Important environment variables

| Variable | Validated setting | Purpose |
|---|---:|---|
| `GGML_MOE_DYNAMIC_FIXED_TOPOLOGY` | `1` | Selects the persistent slot-map graph. |
| `GGML_MOE_DYNAMIC_WARM_START_PER_LAYER` | `1` | Limits the validated mode to one resident expert per layer. |
| `GGML_MOE_DYNAMIC_WARM_START_TOTAL` | `75` | Total warm-start admissions for the tested GLM model. |
| `GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN` | `0` | Prevents unsafe decode-time replacement. |
| `GGML_MOE_DYNAMIC_ASYNC_PROMOTION` | `1` | Uses the asynchronous promotion worker. |
| `GGML_MOE_DYNAMIC_MIN_HOT_ROUTES` | `1` | Allows partial resident coverage to execute on GPU. |
| `GGML_EXPERT_CACHE_RESERVE_MIB` | `3072` | Leaves measured VRAM headroom on the tested V100. |

## How the fixed graph works

Each layer has a persistent device-side map:

```text
slot_by_expert[layer][expert_id] -> compact GPU cache slot, or -1
```

The pointer and tensor shapes stay stable, so CUDA Graphs can be reused. When an asynchronously copied expert is READY, the map contents are updated without rebuilding graph topology.

The GPU branch maps canonical router IDs to compact cache slots. The CPU branch uses the published host map to skip resident experts. Their complementary outputs are added on GPU before router weighting and reduction.

## Limitations and safety

The multi-slot mutable path currently fails under CUDA Compute Sanitizer after upstream numerical/routing corruption reaches a later `SET_ROWS` kernel. Promotion readback confirmed that sampled expert weights were copied correctly, so the current blocker is not an out-of-bounds expert upload.

Do not enable predictor-driven admissions or increase warm-start residency beyond one slot per layer until multi-route numerical parity and slot-lifetime fencing are complete.

See [the benchmark and validation report](moe-fixed-topology-benchmarks.md) for measured performance, token counts, prompt processing results, exact hashes, and reproduction notes.

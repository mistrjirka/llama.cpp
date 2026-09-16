# Qwen3.8-Flash-Next: experimental MoE prefill

Qwen3.8-Flash-Next is a very large mixture-of-experts (MoE) model. The measured setup keeps canonical expert weights in system RAM and moves the experts needed by the new input onto the GPU. This is separate from the dense [Qwen3.8-27B](qwen38-27b.md) path. Build for your hardware first: [SM70](build-sm70.md), [SM75](build-sm75.md), or [mixed SM70 + SM75](build-sm70-sm75.md).

## What happens during a request

The executor takes the complete new-token suffix, works through one model layer at a time, and groups the expert calculations across all those tokens. Each needed offloaded expert projection can be uploaded once for that request and used for every token that selected it. Two GPU weight buffers rotate between expert groups; the main token state and running expert sum stay in GPU memory. Cached history tokens are not rerun through the experts.

This design matters most when a prompt contains multiple new-token chunks. The 100k-cached + 1k measurement is an important long-history case, but the 100k cached tokens are not 100k new tokens of expert work.

The optional selected-entry attention kernel skips masked history entries during calculation. It preserves the model's existing selection for identical inputs, but changes floating-point arithmetic. Different intermediate values can change later routing and attention selection. The [numerical audit](../benches/moe-prefill-0916/NUMERICS.md) found an inherited FP32-request/FP16-accumulator mismatch and did not certify broad quality equivalence.

The request-wide expert scheduler remains opt-in. **Selected-entry attention is on by default when this model, history length, and CUDA shape are eligible**; unsupported cases fall back to the previous attention path. Ordinary server launches do not automatically become host-expert-streaming runs.


## Runtime switches

The main optimization choices are context/runtime options, not environment variables:

| Option | Default | Purpose |
|---|---|---|
| `--moe-layer-first` | off | Enable request-wide MoE scheduling in supported single-sequence contexts. |
| `--moe-router-fusion` | on | Fuse compatible MoE routing graphs; use the `--no-*` form for controlled comparisons. |
| `--exact-set-top-k` | off | Enable exact-set radix top-k on compatible sparse-history selectors. |
| `--selected-attn` | **on** | Use direct selected-entry attention whenever the model and CUDA shape are eligible. |

Each has a matching `--no-*` form. `--no-selected-attn` is the comparison/escape hatch for the previous masked attention implementation. The C API also exposes `llama_context_params.selected_attn` and `llama_set_selected_attn()` for applications that need to change the attention implementation at runtime. Changing it invalidates the reusable graph reservation before the next decode.

## Start with a short benchmark

Build llama.cpp using the guide for your GPU: [SM70](build-sm70.md), [SM75](build-sm75.md), or [mixed SM70 + SM75](build-sm70-sm75.md). The benchmark wrapper needs Python 3.10 or later, a C++ compiler and CUDA headers. It compiles the included harness against your build; it does not download models, install packages or modify server defaults.

```bash
python3 benches/moe-prefill-0916/run.py \
  --preset v100 \
  --model /path/to/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --build build \
  --output results/flash-next-v100
```

The default test processes 4,096 new Python tokens with no saved prefix. Model loading and the first expert-cache population are reported separately from warmed prompt-processing times. At this short history the sparse attention gate does not activate; both settings are a control for ordinary short-history execution. Scores are sampled at eight prompt positions and eight fixed continuation inputs. These continuation inputs are not freely generated answers.

Each invocation runs sparse off/off/on/on/off/on/off. The report excludes the first observation of each setting and verifies that subsequent score files repeat exactly within that setting. It does not assert equality between the settings. Use a new output directory for each run; existing output is never overwritten.

| Preset | GPU order | Model layers | Base expert-cache allowances |
|---|---|---|---|
| `v100` | V100 32 GB | 48 | 14 |
| `rtx2080ti` | RTX 2080 Ti 22 GB | 48 | 6 |
| `v100-rtx2080ti` | V100, then RTX | 36 / 12 | 16 / 10 |
| `rtx2080ti-v100` | RTX, then V100 | 12 / 36 | 10 / 16 |

The wrapper detects matching GPU names and prints their UUIDs before loading. With several matching cards, pass `--devices GPU-uuid0,GPU-uuid1` in the preset order to select exact cards. It does not choose devices based on their current free memory or claim that each preset is optimal.

## Reproduce 100k cached + 1k new

A cached test needs both a compatible saved sequence state and the original token-ID file. The harness checks that the saved prefix tokens agree with that file. Keep all parts of the GGUF together and pass the first part as `--model`.

```bash
python3 benches/moe-prefill-0916/run.py \
  --preset v100-rtx2080ti \
  --model /path/to/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --build build \
  --prefix-state /path/to/next-100k-q8.state \
  --prefix-tokens /path/to/tokens.i32 \
  --prefix-length 100000 \
  --tokens 1000 \
  --output results/flash-next-100k
```

The original prefix state and model weights are not distributed with the repository. The supplied `tokens.i32` must contain the original prefix and enough additional token IDs for the harness; incompatible state is rejected, not silently reconstructed. Without a saved state, use the short test rather than treating its timing as a cached 100k test.

The cached test uses the included C++ suffix fixture; the default cold test uses the longer Python fixture. `--suffix /path/to/text.txt` tests another suffix; it must tokenize to more than the requested new tokens plus eight continuation tokens. The runner uses the selected suffix after the saved prefix, so changing the suffix does not change the identity check on the prefix.

## Memory settings

`--base` changes the persistent expert-cache allowance; `--workspace-mib` changes the per-GPU budget used by the planner for full-request token state and extra expert caching. For example:

```bash
python3 benches/moe-prefill-0916/run.py \
  --preset v100 \
  --model /path/to/first-model-part.gguf \
  --build build \
  --base 12 --workspace-mib 3072 \
  --output results/flash-next-smaller-expert-cache
```

Lowering the base leaves more space outside the expert cache, at the cost of additional host-to-GPU weight traffic. Increasing the workspace ceiling can support more new tokens, but must fit alongside model buffers, history, recurrent state and compute scratch. It can also increase optional expert caching. The value is not a total GPU-memory limit or an allocation of exactly that many bytes.

The base is expressed in layer-equivalent allowances, converted to actual weight bytes. Cached groups are distributed across layers; a base of 14 does not mean only the first 14 layers are computed or that every expert in exactly 14 layers stays resident.

The runtime currently reads `LLAMA_MOE_LAYER_FIRST_BASE_LAYERS_0` and `_1` only. Third and later devices receive no explicit base allowance. Two buffers per GPU are double buffering, not a two-GPU execution limit; nevertheless, useful three-card memory budgeting still needs this configuration extended.

## Support and memory limits

The measured model is Qwen3.8-Flash-Next UD-IQ4_XS, with Q8 K/V and one causal text sequence. The executor requires supported separate expert projections and CUDA layer backends. It does not support the full-request server path with multiple slots, speculative decoding or LoRA. The output head must remain on the final layer's GPU for the tested resident-state arrangement.

The model is not fully resident on either single card. Canonical expert sources remain in host RAM. The earlier loader recorded about 57,407 MiB of pinned host model buffers plus about 27,466 MiB of mapped per-layer embedding storage; mapped size is not the same as physical RAM use. Tests ran on a host with 256 GB RAM. This is not a low-RAM setup or a tested preset for an 11 GB RTX or 16 GB V100.

The new-token workspace is approximately 61,496 bytes per token plus padding per participating GPU in the tested layout. A larger suffix can require reducing cached expert weights. Failure to fit should be handled by changing explicit budgets, not quietly splitting the request into repeated expert-upload passes.

Two/three logical V100 devices passed on one physical V100 using the fork's virtual-device mode. Those tests exercise scheduling and state handoffs, not physical peer transfers, aggregate memory or real multi-card speed. [Hardware results](../benches/moe-prefill-0916/HARDWARE.md).

## Server integration

The C API exposes `llama_supports_prefill_request()` and `llama_prefill_request()`. The latter receives the whole new-token suffix; internal mixer chunks must not reset the request-wide expert lifetime. The server has a matching opt-in admission path.

The published performance measurements use the C API harness. The benchmark wrapper is not an HTTP server launcher: it sets additional scheduling flags, puts canonical expert weights in host buffers and checks layer placement. Copying four environment variables into an otherwise unchanged `llama-server` command does not reproduce its memory plan. Use the standard server for ordinary Qwen3.8-27B, and treat this feature as experimental integration work until the complete serving lifecycle has been qualified.

## Evidence

[Hardware measurements](../benches/moe-prefill-0916/HARDWARE.md), [numerical checks](../benches/moe-prefill-0916/NUMERICS.md), [merge tests](../benches/moe-prefill-0916/MERGE.md) and [recorded data](../benches/moe-prefill-0916/results/) describe separate test scopes. No new upstream ratio is inferred from the sparse off/on comparison.

# Terms used in this fork

This page defines the terms needed by the fork-specific guides. The upstream llama.cpp documentation remains the reference for the full option set.

## SM70 and SM75

`SM` is NVIDIA's name for a GPU compute architecture target. In this repository:

- **SM70 / compute capability 7.0**: Volta-class CUDA target; the main tested card is Tesla V100.
- **SM75 / compute capability 7.5**: Turing-class CUDA target; the main tested card is RTX 2080 Ti.

The architecture determines which CUDA kernels must be compiled. VRAM capacity is separate: two GPUs with the same SM version can still need different model/context settings.

## GGUF and model quantization

A **GGUF** file contains model metadata and weights in a format llama.cpp can load. A large model can be split across several GGUF files; keep all parts together and point llama.cpp at the first part.

Names such as `Q5_K_XL`, `Q4_K_M`, or `IQ4_XS` describe how the **model weights** are stored. Smaller quantizations usually use less memory, with model-dependent quality and speed trade-offs.

## Context and KV cache

The **context** is the token space available for the conversation, new input, and generated output. Attention-based layers retain history in a **KV cache**. `--cache-type-k` and `--cache-type-v` change that history representation; they do not change the GGUF weight quantization.

A larger context reserves more memory. It does not make a short prompt faster.

## Prompt processing, generation, and first-token time

A request has two performance phases:

1. **Prompt processing (PP)**: read the new input tokens.
2. **Token generation (TG)**: produce the answer tokens one by one.

**TTFT** (time to first token) includes the work before the first generated token appears. This fork has focused heavily on PP and TTFT for long histories.

A benchmark described as **100k cached + 1k input** restores an already-computed 100,000-token history and measures adding roughly 1,000 new tokens. It is not the time required to process a fresh 101,000-token prompt from zero.

## Batch and ubatch

`--batch-size` is the logical maximum number of input tokens submitted as a batch. `--ubatch-size` is the physical chunk llama.cpp processes at once. The latter affects temporary GPU memory and prompt-processing performance. Neither option is the number of users connected to the server.

## Dense and MoE models

A **dense** model uses the same feed-forward weights for every token. A **mixture-of-experts (MoE)** model has many expert weight sets and routes each token to a subset of them.

That distinction matters when the full expert pool is larger than VRAM. The experimental request-wide executor can keep canonical experts in system RAM and reuse each uploaded expert across all new tokens that select it.

## Router fusion and selected-entry attention

`--moe-router-fusion` is on by default. When a compatible MoE routing graph appears, the CUDA backend can combine several routing operations into one optimized path. If a graph is not eligible, nothing changes.

`--selected-attn` is also on by default. The current qualified integration is Qwen3.8-Flash-Next; it activates only for supported sparse-history shapes and history lengths. It does **not** turn an ordinary dense-attention model into an approximate sparse model.

## MMQ

**MMQ** is one of llama.cpp's quantized matrix-multiplication paths. Some MoE expert shapes on these older GPUs benefit from forcing or selectively using it, while other models do not. Treat MMQ settings as model-specific tuning, not as a general property of SM70 or SM75 hardware.

## MTP

**Multi-token prediction (MTP)** uses a draft/prediction head to propose future tokens that the main model verifies. It is optional and model-specific. A model guide will say when the tested configuration used it.

[Back to the README](../README.md)

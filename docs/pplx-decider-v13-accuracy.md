# Perplexity Decider v1.3 accuracy fix

## Root cause

The previous GGUF implementation read the classification logits after a
conventional causal chat prompt. The original Perplexity checkpoint instead
requires joint-chunked-readout protocol v4. It evaluates full attention
noncausally and reads the decision head at the final <|fim_suffix|> marker
following a <|fim_middle|> context separator. Recurrent linear-attention
layers retain their causal behavior. The sparse trained readout weights
were already mapped to the appropriate token IDs.

This difference, not quantization or speculative decoding (MTP), is the
identified source of the local/API inference incompatibility.

## Changes

- conversion/pplx_decider_v4.py: validate the official protocol and build
  the matching Jinja systemone prompt.
- conversion/pplx_decider.py: choose v4 only for matching checkpoints,
  embed qwen35.attention.causal=false and protocol_version=4 metadata.
- tools/server/server-decision.{cpp,h}: recognize protocol v4, verify marker
  placement, neutralize user-supplied special-token text, and disable prompt
  prefix sharing and cross-question batching. Each question uses one complete
  sequence. Existing decision-model versions retain their previous path.
- tools/upgrade_pplx_decider_v13_gguf.py: convert existing quantized GGUF files
  by copying the tensor data *unchanged* and rewriting header metadata. Source
  file is never overwritten.

## Usage

Build llama-server with the ordinary CUDA settings. Upgrade a legacy GGUF:

    python tools/upgrade_pplx_decider_v13_gguf.py \
      /path/to/pplx-decider-v1.3-Q6_K.gguf \
      /path/to/pplx-decider-v1.3-Q6_K-v4.gguf

Serve the corrected artifact:

    ./build-cuda/bin/llama-server \
      --model /path/to/pplx-decider-v1.3-Q6_K-v4.gguf \
      --mmproj /path/to/mmproj-pplx-decider-v1.3-F16.gguf \
      --no-mmproj-offload \
      --ctx-size 8192 --batch-size 8192 --ubatch-size 8192 \
      --parallel 1 --gpu-layers all --flash-attn on \
      --image-min-tokens 1024 --jinja

POST to /v1/systemone with state, questions and optional images. This
endpoint *does not generate* assistant tokens. Its logits represent class
scores. Entire noncausal prompt must fit in --ubatch-size. Use a larger
physical batch for images or long state.

## Tests run on grammetry (2026-11-23)

- Three pure protocol-format regressions passed.
- Q5_K_M and Q6_K GGUF-v4 models created without modifying the originals.
- Q5 tensors: 755 names, shapes and types match, and sampled head and tail
  tensor-byte signatures match exactly.
- Q5: choice, noul and numeric score API returned valid scores.
- Q5: three questions in one request versus three individual requests yielded
  identical probabilities and labels for all three.
- Q6 with the original vision projector: construction photo scored through
  /v1/systemone in 2.18 seconds (441 input tokens).
- No measured local-vs-hosted Perplexity logits parity yet. The comparison
  data use different sample counts and cannot establish a delta in accuracy.
- Perplexity's vision reference may use a different image patch budget than
  default llama.cpp; for vision, use --image-min-tokens 1024 and compare
  image/token representations before claiming numerical parity.
- Non-string JSON states may be re-keyed by the server JSON parser, changing
  rendered prompt token order. Strict parity requires identical state text.

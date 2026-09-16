# Qwen3.8-27B

Qwen3.8-27B is the simpler Qwen path in this fork: it is a dense model, so it does not need the experimental host-RAM expert scheduler used by Flash-Next.

Build for your hardware first:

- [SM70 only](build-sm70.md)
- [SM75 only](build-sm75.md)
- [SM70 + SM75](build-sm70-sm75.md)

The published examples use `Qwen3.8-27B-UD-Q5_K_XL.gguf` and Q8_0 history cache. Replace the model path with your own GGUF.

## V100 / SM70 starting point

```bash
./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --alias qwen \
  --device CUDA0 --split-mode none \
  --gpu-layers all --fit off \
  --ctx-size 131072 --parallel 1 \
  --batch-size 4096 --ubatch-size 4096 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --host 127.0.0.1 --port 8080
```

The 131,072-token context is a tested V100 32 GB starting profile, not a promise for every SM70 card or every quantization.

## RTX 2080 Ti 22 GB / SM75 starting point

```bash
./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --alias qwen \
  --device CUDA0 --split-mode none \
  --gpu-layers all --fit off \
  --ctx-size 32768 --parallel 1 \
  --batch-size 4096 --ubatch-size 2048 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --host 127.0.0.1 --port 8080
```

The benchmark also reached a 67,584-token context on the 22 GB card with little memory headroom. That is not a suitable default for an ordinary 11 GB RTX 2080 Ti.

## Send a request

From another terminal:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Explain what a GPU does in two sentences."}],"max_tokens":128}'
```

An OpenAI-compatible client can use `http://127.0.0.1:8080/v1` as the base URL and `qwen` as the model name.

## Important settings

| Setting | Meaning |
|---|---|
| `--ctx-size` | Maximum token space for history, new input, and output. Larger values use more memory. |
| `--batch-size` | Logical input batch. This is not the number of users. |
| `--ubatch-size` | Physical input chunk. Larger values can improve prompt speed but require more scratch memory. |
| `--cache-type-k/v q8_0` | Quantizes the attention history cache. This is separate from the GGUF weight quantization. |
| `--parallel 1` | One active server slot. Multi-slot serving needs separate memory tuning. |
| `--flash-attn on` | Uses the GPU attention path used by the published tests. |
| `--gpu-layers all --fit off` | Requests explicit GPU placement rather than letting automatic fitting change the tested profile. |

The server has architecture-specific automatic batch/ubatch defaults for recognized Qwen3.8-27B configurations when those values are omitted:

| Tested hardware profile | Context capacity | Batch / ubatch |
|---|---:|---:|
| V100 32 GB | 131,072 | 4096 / 4096 |
| RTX 2080 Ti 22 GB | 32,768 or 67,584 | 4096 / 2048 |
| V100 + RTX 2080 Ti | 409,600 | 4096 / 2048 |

These are model-and-memory presets, not defaults for every SM70/SM75 GPU. Explicit CLI/config values take priority. `LLAMA_V100_AUTO_BATCH=0` disables the automatic Qwen batch selection.

## Very long context on V100 + SM75

The tested 409,600-token configuration is advanced. It uses both GPUs and extends the model's original position range with YaRN:

```bash
GGML_CUDA_ALLREDUCE=internal \
GGML_CUDA_AR_COPY_THRESHOLD=131072 \
./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --alias qwen \
  --device CUDA1,CUDA0 --tensor-split 4,5 --split-mode tensor \
  --gpu-layers all --fit off --parallel 1 \
  --ctx-size 409600 \
  --override-kv qwen35.context_length=int:409600 \
  --rope-scaling yarn --rope-scale 1.5625 --yarn-orig-ctx 262144 \
  --flash-attn on \
  --batch-size 4096 --ubatch-size 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-checkpoints 32 --checkpoint-min-step 8192
```

The example assumes the RTX is `CUDA1` and V100 is `CUDA0`; verify your own device order. `--split-mode tensor` divides each model layer's matrix work across both GPUs, and the internal all-reduce combines those partial results without NCCL. `4,5` is the measured tensor-work ratio, not a universal SM70/SM75 rule. The `qwen35.context_length` name comes from the GGUF architecture metadata used by this model. The context checkpoints retain reusable recurrent state and also consume memory. YaRN makes a larger position range possible, but a successful speed test does not establish answer quality at every extended length.

## What not to enable globally

Do **not** build dense Qwen with global `GGML_CUDA_FORCE_MMQ=ON` just because you have a V100/Turing GPU; the measured dense Qwen path was slower with that policy. `GGML_CUDA_VOLTA_FORCE_MMQ=moe` only affects routed experts and therefore does not accelerate this dense model.

MTP is optional and is not part of the headline upstream comparison for this model.

See [terms used in these guides](fork-concepts.md) if GGUF, KV cache, batch/ubatch, YaRN, or MTP are unfamiliar.

[Back to the model chooser](../README.md#then-choose-your-model)

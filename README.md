# llama.cpp for NVIDIA SM70 and SM75 GPUs

This repository is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), a C/C++ project for running language models locally from **GGUF** model files. It includes the normal llama.cpp command-line tools and OpenAI-compatible HTTP server; this is a complete llama.cpp source tree, not an extension that must be installed on top of another copy.

The `v100-optimized` branch focuses on NVIDIA **compute capability 7.0 (SM70 / Volta)** and **7.5 (SM75 / Turing)**. The main tested cards are a Tesla V100 32 GB and an RTX 2080 Ti with 22 GB, separately and together. Other SM70/SM75 GPUs can use the same architecture builds, but model/context presets depend on VRAM and were not measured on every card.

The fork primarily targets faster long-prompt processing, lower time to the first generated token, better use of older CUDA hardware, and mixed SM70+SM75 systems. Most optimizations are automatic once the correct CUDA architecture is built; a few model-specific or experimental paths are documented separately.

## Choose your hardware

Start here. **The first build choice is the GPU architecture, not the model.**

| Hardware in the machine | Examples | Build guide |
|---|---|---|
| **SM70 only** | Tesla V100, Titan V | [Build for SM70](docs/build-sm70.md) |
| **SM75 only** | RTX 20-series, T4, Titan RTX | [Build for SM75](docs/build-sm75.md) |
| **SM70 + SM75 together** | V100 + RTX 2080 Ti | [Build for both architectures](docs/build-sm70-sm75.md) |

`SM70` and `SM75` are CUDA architecture targets. They do **not** describe VRAM capacity. On recent NVIDIA drivers you can check both the GPU name and compute capability directly:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
```

Choose the SM70 guide when it reports `7.0`, the SM75 guide when it reports `7.5`, and the mixed guide when both appear. For example, the RTX benchmarks in this repository use a 22 GB RTX 2080 Ti; their largest memory presets do not automatically fit an ordinary 11 GB card.

If terms such as GGUF, context/KV cache, batch/ubatch, MoE, MMQ, or MTP are unfamiliar, read [Terms used in this fork](docs/fork-concepts.md). It is intentionally short.

## Then choose your model

After building for the hardware, use the guide for the model family. This is where model-specific memory layouts and tuning belong.

| Model family | What matters here | Guide |
|---|---|---|
| **Qwen3.8-27B** | Dense model; simple single-GPU presets and an advanced mixed-GPU long-context preset | [Qwen3.8-27B](docs/qwen38-27b.md) |
| **Qwen3.8-Flash-Next** | Very large MoE; experimental expert streaming from system RAM (tested on a 256 GB host); model-defined sparse attention | [Qwen3.8-Flash-Next](docs/moe-prefill.md) |
| **Ornith 1.5 35B-A3B** | MoE; selective MMQ, mixed-GPU placement, optional MTP and multi-slot serving | [Ornith](docs/gpu-tuning.md) |
| **Gemma 4** | Dense and routed-expert variants; generic MoE router fusion where eligible | [Gemma 4](docs/gemma4.md) |
| **Other GGUF models** | Start with normal llama.cpp options; compatible fork kernels/fusions are selected automatically | [Upstream model documentation](docs/models.md) |

A flag used for one row in the benchmark graph is **not** necessarily a good global setting. In particular, globally forcing MMQ helps some measured MoE configurations but hurts dense Qwen3.8.

## Performance

A model first reads the new input and then writes the answer. **Prompt processing (PP)** measures the reading phase; **token generation (TG)** measures the answer-generation phase. Higher tokens per second is faster.

![Prompt-processing speed by model and GPU, comparing matched upstream llama.cpp with this fork, including Qwen3.8 Flash-Next](docs/benchmarks/long-context-prompt-processing.svg)

Most rows add about 1,000 new tokens to an already-computed long history. A label such as **100k cached + 1k input** does not mean the benchmark rereads 101,000 tokens from zero. See [the short terminology guide](docs/fork-concepts.md#prompt-processing-generation-and-first-token-time) if that distinction is unclear.

The September 12 rows compare against upstream `3057bb66`. The Qwen3.8-Flash-Next row is a later September 16 matched run against upstream `83078fec0`. Exact commands, revisions, model files, and caveats are kept with the benchmark evidence rather than on this landing page: [September 12 comparison](docs/benchmarks/upstream-table.md) · [Flash-Next September 16 comparison](benches/moe-prefill-0916/UPSTREAM-RUNTIME-0916.md).

## What is automatic, and what is experimental?

The normal build already contains the SM70/SM75 kernel work. You should not have to discover a long list of hidden environment variables before the fork is useful.

| Feature | Default | Scope |
|---|---|---|
| CUDA attention/quantization/kernel selection for supported shapes | automatic | General SM70/SM75 paths |
| `--moe-router-fusion` | **on** | Compatible MoE routing graphs; falls back when ineligible |
| `--selected-attn` | **on** | Currently qualified for Qwen3.8-Flash-Next sparse-history attention; ineligible graphs fall back |
| `--moe-layer-first` | off | Experimental request-wide host-expert scheduling; currently qualified for Flash-Next |
| `--exact-set-top-k` | off | Experimental Flash-Next exact-set selector optimization; intended to generalize to compatible selectors |

Selected-entry attention does **not** make an ordinary dense-attention model approximately sparse. It only changes how an already-selected sparse history is calculated. The Flash-Next numerical/quality caveats are documented in its model guide.

## What this fork changes

At a high level, the branch contains:

- SM70/SM75 FlashAttention and other CUDA kernel work for long-context prompt processing;
- faster quantized-weight conversion and matrix paths on the tested older GPUs;
- mixed V100/Turing execution and internal GPU-to-GPU reduction work;
- long-context cache/checkpoint and multi-slot serving improvements;
- generic MoE routing optimizations that can help models beyond Qwen;
- an experimental request-wide MoE path for models whose expert weights live partly in system RAM.

Not every item applies to every model. The model guides say which special settings were actually used.

## First run after building

Whichever hardware guide you use, first confirm the devices llama.cpp sees:

```bash
./build/bin/llama-server --list-devices
```

Then follow the relevant model guide rather than copying a command for a different architecture/model from a benchmark report.

The server exposes an OpenAI-compatible API. For general server endpoints, authentication, networking, and options, use the [llama.cpp server documentation](tools/server/README.md).

## Benchmarks and validation

The graph is a summary, not the source of truth for every setting. Detailed evidence is split by topic:

- [Published upstream comparison and exact settings](docs/benchmarks/upstream-table.md)
- [Qwen3.8-Flash-Next current-upstream comparison](benches/moe-prefill-0916/UPSTREAM-RUNTIME-0916.md)
- [Flash-Next hardware/placement checks](benches/moe-prefill-0916/HARDWARE.md)
- [Flash-Next numerical audit](benches/moe-prefill-0916/NUMERICS.md)
- [MoE router-fusion cross-model check](benches/moe-prefill-0916/ROUTER-CROSSMODEL-0916.md)
- [Quantization-kernel checks](benches/volta-kquant-0912/ADDITIONAL_FORMATS.md)
- [Saved-state correctness checks](benches/correctness-0912/nondeterminism/REPORT.md)

The branch is benchmarked on Linux with CUDA 12.9. Performance depends on model quantization, prompt shape, context length, GPU memory, and device placement.

## Upstream and help

The base project is maintained by [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). General llama.cpp build/platform documentation remains under [`docs/`](docs/). Use this fork's hardware/model guides for settings that differ from upstream.

For a fork-specific issue, include the commit, GPU model, VRAM, CUDA version, GGUF filename/quantization, and launch command. Remove credentials and private prompts from logs before sharing them.

See [LICENSE](LICENSE) and [CONTRIBUTING.md](CONTRIBUTING.md).

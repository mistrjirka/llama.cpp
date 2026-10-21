# llama.cpp for the AMD XDNA2 NPU

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that runs **prompt processing on the NPU of Ryzen AI 300 / Max chips** and keeps
token generation on the CPU. It is a ggml backend, not a separate program: `llama-server`, `llama-cli`, `llama-bench`, the OpenAI-compatible
API and ordinary GGUF models work exactly as upstream, and the NPU is used automatically.

Why: on battery, the NPU processes a long prompt at **half the power of the iGPU** and, on Qwen3.5 models, faster.

| Qwen3.5-4B, 4.3k-token prompt, power-saver profile | Prompt tok/s | Package power | Generation tok/s | Package power |
|---|---:|---:|---:|---:|
| **this fork (NPU + CPU)** | **560** | 16 W | 19.2 | 16 W |
| llama.cpp Vulkan (iGPU) | 528 | 33 W | **23.6** | 32 W |
| llama.cpp ROCm (iGPU) | 454 | 31 W | 22.3 | 31 W |
| FastFlowLM (NPU only) | 479 | **13 W** | 15.2 | **12 W** |

Token generation is not faster than the iGPU, and on Gemma 4 12B prompt processing is slower than the iGPU. [Full results](#benchmarks) and
[limitations](#limitations) below.

**Status:** works and passes llama.cpp's server tests (251 passed, 4 skipped, 1 failed; the failure is a model download timeout that also fails on a
stock build). Tested on one machine and one distribution only (see [Requirements](#requirements)).

## Contents

[Quick start](#quick-start) | [Requirements](#requirements) | [Benchmarks](#benchmarks) | [How it works](#how-it-works) |
[Troubleshooting](#troubleshooting) | [Limitations](#limitations) | [Staying current with upstream](#staying-current-with-upstream) |
[Credits and license](#credits-and-license)

## Quick start

```sh
sudo pacman -S xrt xrt-plugin-amdxdna        # Arch; other distributions: see docs/backend/XDNA2.md
git clone https://github.com/mistrjirka/llama.cpp-xdna2 && cd llama.cpp-xdna2
scripts/xdna2-setup.sh                       # checks the NPU stack, installs the kernel compiler (no root, ~1.4 GB)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA2=ON && cmake --build build -j
./build/bin/llama-server -m model.gguf -c 16384    # OpenAI API at http://127.0.0.1:8080/v1
```

The first prompt of a new model is 20-45 s slower while the NPU kernels compile; they are cached in `~/.cache/ggml-xdna2` and reused afterwards.
`llama-server --list-devices` should list `XDNA2: AMD XDNA2 NPU`. If the speed is far lower than the tables below, see [Troubleshooting](#troubleshooting).

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
./build/bin/llama-bench -m model.gguf -fa 1 -ub 2048 -p 512,2048 -n 64     # llama-bench needs -ub 2048 to match the server default
```

Full installation notes, other distributions, and the environment variables: [docs/backend/XDNA2.md](docs/backend/XDNA2.md).

## Requirements

- An AMD **XDNA2** NPU: Ryzen AI 300 (Strix Point, Krackan Point) or Ryzen AI Max (Strix Halo). XDNA1 (Ryzen 7040/8040) is not supported.
- Linux 6.14 or newer (mainline `amdxdna` driver), x86-64 with AVX-512 (all Ryzen AI 300 chips have it). Windows is not supported.
- XRT and the amdxdna plugin; Python 3.11 to 3.14 for the kernel compiler; GCC or Clang.
- 64 MiB of lockable memory for XRT (`ulimit -Hl`); the setup script checks this.

Tested only on an HP OmniBook Ultra 14 (Ryzen AI 9 365, 32 GB), Arch Linux, kernel 7.2, XRT 2.21.75, NPU firmware 1.1.2.64. The Ubuntu and
Fedora steps in the guide come from AMD's driver documentation and have not been run.

## Benchmarks

Same laptop, one engine at a time, each through its own OpenAI-compatible server and the same client. Prompt: one chat request of
4.3k-4.5k tokens, time to the first token. Generation: 256 tokens after a short prompt. Power is the whole-package power (CPU, iGPU, NPU, memory
controller) from the amdgpu driver, sampled every 20 ms; the screen and the rest of the laptop are not included. Each number is the second of
two runs; repeats differ by about 5 %.

Power-saver profile (the laptop default, what matters on battery). Prompt in tok/s at watts; generation likewise:

| Model | Engine | Prompt | Generation |
|---|---|---:|---:|
| Qwen3.5-4B Q4_K_M | **this fork** | **560** at 16.3 W | 19.2 at 16.4 W |
| | Vulkan (iGPU) | 528 at 33.2 W | **23.6** at 32.4 W |
| | ROCm (iGPU) | 454 at 31.0 W | 22.3 at 31.4 W |
| | FastFlowLM | 479 at 12.9 W | 15.2 at 12.0 W |
| Ornith-1.5-9B Q4_K_M | **this fork** | **408** at 18.3 W | 11.9 at 17.8 W |
| | Vulkan (iGPU) | 307 at 30.7 W | **15.1** at 29.7 W |
| | ROCm (iGPU) | 287 at 29.7 W | 13.6 at 29.4 W |
| | FastFlowLM (Qwen3.5-9B) | 338 at 11.8 W | 9.3 at 11.5 W |
| Gemma 4 12B QAT Q4_0 | this fork | 136 at 17.8 W | 9.2 at 18.4 W |
| | Vulkan (iGPU) | 261 at 29.2 W | **11.3** at 29.7 W |
| | ROCm (iGPU) | 179 at 28.8 W | 10.4 at 29.2 W |
| | FastFlowLM | **278** at 16.3 W | 6.8 at 14.3 W |

What this says:

- **Prompt processing, Qwen3.5:** fastest of the four engines, at about half the iGPU's power, i.e. about half the energy per prompt token
  (4B: 29 mJ/token against 63 for Vulkan). FastFlowLM is slightly more frugal on the 4B (27 mJ/token) but slower.
- **Prompt processing, Gemma 4 12B:** the slowest here. Its global-attention layers run on the CPU and the cost grows with the prompt.
- **Generation:** the iGPU is fastest everywhere. In power-saver this fork generates at about half the iGPU's power; in the performance
  profile the CPU cores clock up and the power is about the same. FastFlowLM generates slowest but at the lowest power.

The models are not byte-identical: llama.cpp runs the GGUFs named above, FastFlowLM its own conversions (Qwen3.5-9B stands in for Ornith, which
is a fine-tune of the same architecture). Output quality was not compared. [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) is the NPU
engine [Lemonade](https://github.com/lemonade-sdk/lemonade) uses on Linux; Lemonade itself was not run.

The performance profile (the iGPU engines draw 40-48 W there), energy per token, and methodology are in
[docs/backend/XDNA2.md](docs/backend/XDNA2.md#performance).

<details>
<summary>llama-bench prompt speed against prompt length (performance profile, tok/s)</summary>

`llama-bench -fa 1 -p 512,2048,4096 -n 64`. This fork at `-ub 2048`; the iGPU engines at their best `-ub` (Gemma 12B: `-ub 512` for all).

| Model | Engine | pp512 | pp2048 | pp4096 | tg64 |
|---|---|---:|---:|---:|---:|
| Qwen3.5-4B | this fork | 640 | 684 | 684 | 20.7 |
| | Vulkan | 614 | 598 | 593 | 27.1 |
| | ROCm | 556 | 548 | 536 | 26.4 |
| Ornith-1.5-9B | this fork | 440 | 487 | 472 | 12.7 |
| | Vulkan | 368 | 358 | 347 | 15.0 |
| | ROCm | 340 | 349 | 336 | 14.3 |
| Gemma 4 12B QAT | this fork | 196 | 172 | 152 | 10.4 |
| | Vulkan | 327 | 310 | 297 | 12.3 |
| | ROCm | 259 | 236 | 247 | 11.2 |

</details>

## How it works

| Work | Runs on |
|---|---|
| Weight matmuls during prompt processing (Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0) | NPU |
| Attention during prompt processing (head size 128/256, 4 or 8 KV heads, causal or sliding window; needs 32+ tokens and 1024+ context) | NPU |
| Everything else in prompt processing (norms, RoPE, Qwen3.5 linear attention, other formats, mixture-of-experts) | CPU |
| Token generation, and batches of up to 8 tokens | CPU (AVX-512), reading the NPU's weight copy in place |

Generation stays on the CPU because one token stream is limited by memory bandwidth, and the cores read DRAM faster (~70 GB/s here) than the
NPU can (~53 GB/s). The NPU computes in bfp16 (8-bit mantissas, one exponent per 8 values); perplexity stays within 0.2-0.6 % of the CPU build on
the models tested, but outputs are not bit-identical. The NPU kernels are generated for each model's shapes with
[mlir-aie](https://github.com/Xilinx/mlir-aie) (IRON) and compiled on first use; the sources are in `ggml/src/ggml-xdna2/kernels`.

Code: `ggml/src/ggml-xdna2/` (the backend), `scripts/xdna2-setup.sh` (setup), `docs/backend/XDNA2.md` (guide).

## Troubleshooting

- **Prompt speed is about 100 tok/s instead of 500+:** the NPU was not opened. The log says why. Usually the locked-memory limit is under 64 MiB: set
  `LimitMEMLOCK=infinity` for a service, or `memlock unlimited` in `/etc/security/limits.d`, then log in again.
- **`no kernel ... and no toolchain to build it`:** run `scripts/xdna2-setup.sh`, or point `GGML_XDNA2_MLIR_AIE_ENV` at a venv with `mlir_aie` and `llvm-aie`.
- **CMake says XRT was not found:** install XRT (on Ubuntu or Fedora build [amd/xdna-driver](https://github.com/amd/xdna-driver) and run
  `source /opt/xilinx/xrt/setup.sh`), then configure again.
- **First prompt is slow:** kernels are being compiled (20-45 s once per model shape).
- **Compare with the plain CPU path:** `GGML_XDNA2_DISABLE=1`.

## Limitations

- Generation is 0-5 % slower than a plain CPU build, and about 13 % slower with Q4_0 weights (measured on Llama 3.2 3B).
- Gemma 4: layers with head size 512, and all attention layers of E2B and E4B, run attention on the CPU, which limits long prompts.
- Only the formats listed under [How it works](#how-it-works) run on the NPU; others work but stay on the CPU.
- Single NPU, single machine type tested, Linux only, bfp16 arithmetic.

## Staying current with upstream

This branch is based on llama.cpp master of mid-October 2026 and tracks it by merging; it was about 150 commits behind on 21 October 2026, with
no merge conflicts. Anything not about the NPU (other backends, models, tools, the web UI) is upstream's: see the
[llama.cpp README](https://github.com/ggml-org/llama.cpp#readme) and [documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs).

## Credits and license

Built on [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp). NPU kernels are generated with [mlir-aie / IRON](https://github.com/Xilinx/mlir-aie)
and compiled with Peano ([llvm-aie](https://github.com/Xilinx/llvm-aie)); the device is driven through
[XRT](https://github.com/Xilinx/XRT) and the [amdxdna driver](https://github.com/amd/xdna-driver). [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM)
was used as the NPU comparison. MIT license, see [LICENSE](LICENSE).

# Build for an NVIDIA SM70 GPU

Use this guide when every NVIDIA GPU that llama.cpp will use has **compute capability 7.0 (SM70)**. The fork is tested primarily on a **Tesla V100 32 GB**. Other SM70 cards can use the same CUDA architecture build, but the memory and performance presets in this repository were not measured on every SM70 model.

If you have an SM75 card as well, use the [mixed SM70 + SM75 build](build-sm70-sm75.md) instead.

## 1. Prerequisites

You need Git, CMake, Ninja, a C++ compiler, an NVIDIA driver, and a CUDA toolkit. CUDA 12.9 is the version used for the published measurements.

Check that the CUDA compiler is installed:

```bash
nvcc --version
```

`nvidia-smi` can work even when the CUDA compiler is not installed, so it is not a substitute for this check.

## 2. Clone this branch

```bash
git clone --branch v100-optimized --single-branch \
  https://github.com/mistrjirka/llama.cpp.git
cd llama.cpp
```

## 3. Build only SM70 kernels

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DLLAMA_BUILD_UI=OFF

cmake --build build -j 8 --target llama-server llama-cli llama-bench
```

Building only architecture `70` avoids compiling Turing kernels you cannot use. Reduce `-j 8` if compilation runs out of system memory. `LLAMA_BUILD_UI=OFF` only omits the optional browser UI; the HTTP API server is still built.

## 4. Check the GPU llama.cpp sees

```bash
./build/bin/llama-server --list-devices
```

The output uses names such as `CUDA0`. Those names are llama.cpp device indexes, not permanent hardware identities.

## 5. Choose the model

The build is hardware-specific; the **run command is model-specific**. Do not add model-tuning flags just because you have a V100.

- [Qwen3.8-27B](qwen38-27b.md) — dense model; straightforward starting point.
- [Qwen3.8-Flash-Next](moe-prefill.md) — very large MoE model; experimental host-RAM expert streaming.
- [Ornith 1.5 35B-A3B](gpu-tuning.md) — MoE-specific MMQ and optional MTP tuning.
- [Gemma 4](gemma4.md) — dense and MoE variants; generic router fusion is automatic where eligible.
- Other GGUF models can use normal llama.cpp options; fork optimizations are selected automatically when their graph and CUDA shape match.

Do not add model-specific performance flags until you choose a model guide. A flag that helps one MoE model can slow a dense model on the same GPU.

[Back to the hardware chooser](../README.md#choose-your-hardware)

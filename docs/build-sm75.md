# Build for an NVIDIA SM75 GPU

Use this guide when every NVIDIA GPU that llama.cpp will use has **compute capability 7.5 (SM75)**. This includes Turing GPUs such as RTX 20-series cards and T4. The fork is benchmarked on an **RTX 2080 Ti with 22 GB of VRAM**; context and memory presets for that modified 22 GB card should not be assumed to fit an ordinary 11 GB 2080 Ti.

If the machine also has an SM70 card such as a V100, use the [mixed SM70 + SM75 build](build-sm70-sm75.md).

## 1. Prerequisites

You need Git, CMake, Ninja, a C++ compiler, an NVIDIA driver, and a CUDA toolkit. CUDA 12.9 is the tested toolkit.

```bash
nvcc --version
```

## 2. Clone this branch

```bash
git clone --branch v100-optimized --single-branch \
  https://github.com/mistrjirka/llama.cpp.git
cd llama.cpp
```

## 3. Build only SM75 kernels

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DLLAMA_BUILD_UI=OFF

cmake --build build -j 8 --target llama-server llama-cli llama-bench
```

`75` means the binary contains the CUDA kernels for SM75. Reduce the build parallelism if compilation uses too much RAM.

## 4. Check device names

```bash
./build/bin/llama-server --list-devices
```

## 5. Choose the model

The hardware build does not determine model tuning. Do not copy model-specific performance flags from another guide just because both models run on SM75.

- [Qwen3.8-27B](qwen38-27b.md)
- [Qwen3.8-Flash-Next](moe-prefill.md)
- [Ornith 1.5 35B-A3B](gpu-tuning.md)
- [Gemma 4](gemma4.md)

For another GGUF model, start from ordinary llama.cpp options and only add fork-specific flags when its model guide or benchmark calls for them.

[Back to the hardware chooser](../README.md#choose-your-hardware)

# Build for a mixed SM70 + SM75 system

Use this guide when llama.cpp should use **both an SM70 GPU and an SM75 GPU in the same machine**, for example a V100 together with an RTX 2080 Ti.

The published mixed-GPU measurements use a **Tesla V100 32 GB** and an **RTX 2080 Ti 22 GB**. Other SM70/SM75 combinations can use the same build, but the measured tensor splits, layer splits, context sizes, and memory budgets are specific to the tested cards and models.

## 1. Prerequisites

You need Git, CMake, Ninja, a C++ compiler, an NVIDIA driver, and a CUDA toolkit. CUDA 12.9 is the tested toolkit.

```bash
nvcc --version
nvidia-smi -L
```

## 2. Clone this branch

```bash
git clone --branch v100-optimized --single-branch \
  https://github.com/mistrjirka/llama.cpp.git
cd llama.cpp
```

## 3. Build both CUDA architectures

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES='70;75' \
  -DLLAMA_BUILD_UI=OFF

cmake --build build -j 8 --target llama-server llama-cli llama-bench
```

## 4. Check llama.cpp device order

```bash
./build/bin/llama-server --list-devices
```

Do this before copying any mixed-GPU command. `CUDA0` and `CUDA1` depend on the process/device ordering; a command from another machine may have them reversed.

## 5. Let the model determine the split

There is no single best multi-GPU split for the whole fork.

- **Qwen3.8-27B** uses the tested tensor-parallel arrangement for the very long-context preset. See [Qwen3.8-27B](qwen38-27b.md).
- **Qwen3.8-Flash-Next** uses a layer split in the experimental request-wide benchmark because canonical expert weights are host-backed. See [Flash-Next](moe-prefill.md).
- **Ornith 1.5 35B-A3B** has separate single-user and four-slot placements. See [Ornith](gpu-tuning.md).

Do not treat `--tensor-split`, `--split-mode`, internal all-reduce, or a layer ratio as hardware defaults. They describe how a **particular model workload** is divided between the cards.

[Back to the hardware chooser](../README.md#choose-your-hardware)

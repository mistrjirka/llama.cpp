# llama.cpp for V100 and RTX 2080 Ti

CUDA optimizations for NVIDIA **Volta (SM70)** and **Turing (SM75)**, tested on a Tesla V100-SXM2 32 GB and an RTX 2080 Ti 22 GB. The main target is Qwen3.8-27B `UD-Q5_K_XL` with llama.cpp's standard `q8_0` K/V cache.

The fork focuses on prompt-processing latency, especially when a long context is already cached. The current implementation adds tuned long-context Q8 attention paths for both GPUs, an INT8 Tensor-Core QK path for the RTX 2080 Ti, and measured batch defaults for the tested Qwen3.8 V100/RTX 2080 Ti topologies.

**Branch:** `v100-optimized` · **Synced through upstream:** `43f3dda62` · **Benchmark baseline:** `5bda51bfb` (2026-09-11) · **Optimization commit:** `0fc400871`

## Benchmarks

The main workload is **100,000 cached tokens followed by a 1,000-token prompt append**. The headline comparisons were rerun after syncing the fork through upstream `5bda51bfb`. Qwen uses `UD-Q5_K_XL`, `q8_0` K/V, FlashAttention, and MTP disabled on both engines.

### 100k cached + 1k append

| Hardware | Upstream PP | `v100-optimized` PP | PP gain | Upstream TTFT | `v100-optimized` TTFT | TTFT reduction |
|---|---:|---:|---:|---:|---:|---:|
| V100 32 GB | 298.24 tok/s | **430.87 tok/s** | **+44.47%** | 3.404 s | **2.370 s** | **-30.37%** |
| V100 + RTX 2080 Ti | 405.41 tok/s | **690.44 tok/s** | **+70.31%** | 2.524 s | **1.503 s** | **-40.44%** |

The V100 row uses native 131072 context and matched `batch=4096`, `ubatch=4096`. The dual-GPU row uses the production-style 409600-token YaRN context, a 4:5 RTX 2080 Ti:V100 tensor split, and `batch=4096`, `ubatch=2048`.

With 64 generated tokens after the dual-GPU append, generation improves from **23.96 to 26.47 tok/s (+10.45%)** and total request time falls from **5.160 to 3.891 seconds (-24.59%)**.

### RTX 2080 Ti: 65k cached + 1k append

The 22 GB RTX 2080 Ti can run the long-Q8 path at a 67,584-token context. Using 65,536 cached tokens leaves enough room for a 1,000-token append and directly exercises the Turing long-context kernel.

| Metric | Upstream `5bda51bfb` | `v100-optimized` | Change |
|---|---:|---:|---:|
| Prompt processing | 384.91 tok/s | **496.44 tok/s** | **+28.98%** |
| TTFT | 2.627 s | **2.048 s** | **-22.03%** |
| 64-token decode | 17.35 tok/s | 17.37 tok/s | +0.10% |

The allocation left about 463 MiB free on the 22 GB card. Cold-prompt and earlier comparison data are retained in the benchmark reports.

### Cold prompt processing

For reference, the same-day pre-sync comparison against upstream `b0dcb8192` also measured cold 1k and 16k prompts. These cells were not rerun after the final upstream merge, so they are kept separate from the current-sync headline above.

| Hardware | 1k upstream | 1k fork | Gain | 16k upstream | 16k fork | Gain |
|---|---:|---:|---:|---:|---:|---:|
| V100 32 GB | 857.45 | **901.59** | **+5.15%** | 849.80 | **921.07** | **+8.39%** |
| RTX 2080 Ti 22 GB | 670.78 | **923.28** | **+37.64%** | 641.91 | **923.07** | **+43.80%** |
| V100 + RTX 2080 Ti | 982.50 | **1085.45** | **+10.48%** | 1031.86 | **1239.70** | **+20.14%** |

Full current-sync methodology and measurements are in [`benches/upstream-sync-0911/REPORT.md`](benches/upstream-sync-0911/REPORT.md). The previous `b0dcb8192` benchmark set is retained in [`benches/upstream-vs-optimized-0911/REPORT.md`](benches/upstream-vs-optimized-0911/REPORT.md).

## Build and run

### Build

Clone and build one binary containing both SM70 and SM75 kernels:

```bash
git clone --branch v100-optimized --single-branch https://github.com/mistrjirka/llama.cpp.git && cd llama.cpp && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_ARCHITECTURES='70;75' -DLLAMA_BUILD_UI=OFF && cmake --build build -j --target llama-server
```

`70;75` builds kernels for both V100 and RTX 2080 Ti, so the same build works on either GPU or on a mixed system. `-DLLAMA_BUILD_UI=OFF` skips the web UI and its build/download step; remove it if you use the built-in UI.

The same CUDA build is recommended for Qwen and Ornith. Qwen needs no MMQ environment setting; the Ornith section below shows the selective Volta MoE setting.

### V100

For Qwen3.8-27B on a single V100, the server automatically selects the measured `batch=4096`, `ubatch=4096` defaults when `-b/-ub` are omitted.
Replace the `CUDA_VISIBLE_DEVICES=0` index below with the `nvidia-smi` index of the card you want to expose. Once only one GPU is visible, llama.cpp sees it as `CUDA0`.

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --gpu-layers all \
  --ctx-size 131072 --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0
```

At 16k prompt processing, the measured V100 ubatch sweep was **876.49 tok/s at 1024**, **923.31 at 2048**, and **952.18 at 4096**.

### RTX 2080 Ti

For a single RTX 2080 Ti, the server automatically selects `batch=4096`, `ubatch=2048` for the tested Qwen3.8-27B model.

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --gpu-layers all \
  --ctx-size 32768 --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0
```

At 16k prompt processing, `ubatch=1024/2048/4096` measured **890.95 / 923.63 / 930.62 tok/s**. `ubatch=2048` keeps almost all of the performance while leaving more VRAM for context and server state.

### V100 + RTX 2080 Ti

The Qwen batch/ubatch choice is automatic on the tested mixed pair, but tensor placement and context remain explicit because free VRAM and context size materially change the best split. The following is the tested 400k-context configuration used for the long-context benchmark. Check `./build/bin/llama-server --list-devices` first; the device order below assumes the RTX 2080 Ti is `CUDA1` and the V100 is `CUDA0`.

```bash
export GGML_CUDA_ALLREDUCE=internal
export GGML_CUDA_AR_COPY_THRESHOLD=131072
./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --device CUDA1,CUDA0 --tensor-split 4,5 --split-mode tensor \
  --gpu-layers all --fit off --parallel 1 \
  --ctx-size 409600 \
  --override-kv qwen35.context_length=int:409600 \
  --rope-scaling yarn --rope-scale 1.5625 --yarn-orig-ctx 262144 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-checkpoints 32 --checkpoint-min-step 8192
```

When `-b/-ub` are omitted, this pair automatically selects `4096/2048`; that is also the serving default for this 400k setup. `ubatch=4096` was a little faster in the sweep but left only about 1 GiB free on the RTX 2080 Ti before adding other server state.

### Batch defaults

| Setup | Qwen3.8 default | Selection |
|---|---:|---|
| Single V100, 131072 ctx | `4096/4096` | automatic |
| Single RTX 2080 Ti, 32768 or 67584 ctx | `4096/2048` | automatic |
| V100 + RTX 2080 Ti, 409600 ctx | `4096/2048` | automatic |
| Other model/topology/context | upstream behavior | unchanged |

Explicit `-b/-ub`, `LLAMA_ARG_BATCH`/`LLAMA_ARG_UBATCH`, and non-default configuration values take precedence. `LLAMA_V100_AUTO_BATCH=0` disables the hardware-aware Qwen batch defaults. The automatic table is deliberately limited to benchmarked context/topology combinations.

## RTX 2080 Ti optimizations

The RTX 2080 Ti path targets the parts of long-context attention where Turing differs from Volta.

### Long-Q8 attention dispatch

For the tested D256/GQA6 tensor-parallel geometry, Q8 K/V uses a dedicated `16x2` long-prompt attention specialization. This was the first large SM75 kernel improvement during the optimization work.

### INT8 Tensor-Core QK

The optimized path uses the existing `q8_0` K cache as the source data. K is packed into Tensor-Core-friendly INT8 codes and scales, Q is quantized once for the attention call, and the QK score calculation uses Turing's INT8 Tensor Cores. The mask, softmax, and value accumulation remain on the established attention path.


Together these paths produce the standalone RTX 2080 Ti long-context gain shown above while keeping decode performance unchanged.

## V100 optimizations

The Volta path lets long-context Q8 K/V use the compact V100 Tensor-Core attention specialization. Q8-backed KV is now eligible for the tuned compact kernel after conversion to the FP16 attention input.

The compact path remains specific to the validated long-context geometry. The regular FlashAttention dispatch handles other shapes.

## Ornith and MoE

Use the same build shown above. On V100 or a V100-containing system, set:

```bash
export GGML_CUDA_VOLTA_FORCE_MMQ=moe
```

This selects MMQ for routed expert matmuls on Volta. The setting is safe to keep in a shared launcher used for both dense Qwen3.8 and MoE models.

### Ornith 100k cached + 1k append

`Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf`, one V100, Q8 K/V, MTP off:

| Metric | Upstream `5bda51bfb` | `v100-optimized` + `MMQ=moe` | Change |
|---|---:|---:|---:|
| Prompt processing | 534.97 tok/s | **803.26 tok/s** | **+50.15%** |
| TTFT | 1.933 s | **1.306 s** | **-32.44%** |
| 64-token decode | 56.97 tok/s | 56.92 tok/s | -0.08% |

### Why use the selective MMQ setting

A mirrored Qwen test measured **429.86 PP/s** with the selector unset and **429.17 PP/s** with `GGML_CUDA_VOLTA_FORCE_MMQ=moe` (-0.16%). Dense Qwen has no routed experts, so the selector leaves its matmul policy unchanged.

Global `GGML_CUDA_FORCE_MMQ=ON` is much less suitable as a common build: the same Qwen 100k+1k test fell to **295.70 PP/s**, while Ornith measured **776.86 PP/s** globally forced versus **804.23 PP/s** with selective `MMQ=moe`. The normal build plus the runtime selector therefore gives the better shared Qwen/Ornith configuration.

The tested Ornith Q6/Q5 model is about 25 GiB and fits on the V100 or the combined V100 + RTX 2080 Ti setup. MTP serving has separate benchmarks under [`benches/mtp-final-integration-0907/`](benches/mtp-final-integration-0907/).

## Other fork features

The branch also contains earlier work on MTP serving, exact-prefix KV sharing, parked agent sessions, V100 FlashAttention/GatedDeltaNet tuning, PXQ, and mixed-GPU scheduling. These features have separate benchmarks and controls:

- [`benches/mtp-final-integration-0907/REPORT.md`](benches/mtp-final-integration-0907/REPORT.md)
- [`benches/parallel-serving-0907/RESEARCH.md`](benches/parallel-serving-0907/RESEARCH.md)
- [`benches/readme-current-0908/REPORT.md`](benches/readme-current-0908/REPORT.md)
- [`benches/upstream-vs-optimized-0911/REPORT.md`](benches/upstream-vs-optimized-0911/REPORT.md)
- [`benches/upstream-sync-0911/REPORT.md`](benches/upstream-sync-0911/REPORT.md)

## Benchmark notes

Benchmark percentages are measurements for the configurations above. Context length, quantization, tensor placement, batch size, and ubatch can move the bottleneck substantially. The current sync report contains the exact controls, retained measurements, MMQ comparison, and correctness checks used for the headline tables.

For general llama.cpp APIs, platform support, and build documentation, see upstream [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), [`docs/`](docs/), and [`tools/server/README.md`](tools/server/README.md).

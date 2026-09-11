# llama.cpp for V100 and RTX 2080 Ti

CUDA optimizations for NVIDIA **Volta (SM70)** and **Turing (SM75)**, tested on a Tesla V100-SXM2 32 GB and an RTX 2080 Ti 22 GB. The main target is Qwen3.8-27B `UD-Q5_K_XL` with llama.cpp's standard `q8_0` K/V cache.

The fork focuses on prompt-processing latency, especially when a long context is already cached. The current implementation adds tuned long-context Q8 attention paths for both GPUs, an INT8 Tensor-Core QK path for the RTX 2080 Ti, and measured batch defaults for common single-GPU Qwen3.8 setups.

**Branch:** `v100-optimized` · **Current upstream benchmark:** `b0dcb8192` (2026-09-11) · **Optimization commit:** `0fc400871`

## Benchmarks

The main benchmark is the workload this fork is optimized for: **100,000 cached tokens followed by a 1,000-token prompt append**. All comparisons use Qwen3.8-27B `UD-Q5_K_XL`, llama.cpp `q8_0` K/V cache, FlashAttention, MTP disabled, and current upstream `b0dcb8192` as the baseline.

### 100k cached + 1k append

| Hardware | Upstream PP | `v100-optimized` PP | PP gain | Upstream TTFT | `v100-optimized` TTFT | TTFT reduction |
|---|---:|---:|---:|---:|---:|---:|
| V100 32 GB | 298.34 tok/s | **431.90 tok/s** | **+44.77%** | 3.409 s | **2.365 s** | **-30.62%** |
| V100 + RTX 2080 Ti | 408.79 tok/s | **691.80 tok/s** | **+69.23%** | 2.497 s | **1.499 s** | **-39.97%** |

The single-V100 row uses native 131072 context and matched `batch=4096`, `ubatch=4096` on both engines. Six measurements per side were collected in A/B/B/A process order after a warm request; the generated control token matched across every measured run.

The dual-GPU row uses the production-style 409600-token YaRN context, a 4:5 RTX 2080 Ti:V100 tensor split, and `batch=4096`, `ubatch=2048`. With 64 generated tokens after the append, upstream measured **24.02 tok/s** generation and **5.131 s** end-to-end, while `v100-optimized` measured **26.68 tok/s** and **3.867 s**.

These long-context results are where the Q8 attention work matters most: the request spends substantial time attending over the existing 100k-token KV cache, so the optimized Volta and Turing attention paths have much more impact than they do on a cold short prompt.

### Cold prompt processing

Cold PP is included as a secondary comparison. These runs start without a 100k cached prefix and use matched `batch=4096`, `ubatch=2048` settings on upstream and the fork.

| Hardware | 1k upstream | 1k `v100-optimized` | Gain | 16k upstream | 16k `v100-optimized` | Gain |
|---|---:|---:|---:|---:|---:|---:|
| V100 32 GB | 857.45 | **901.59 tok/s** | **+5.15%** | 849.80 | **921.07 tok/s** | **+8.39%** |
| RTX 2080 Ti 22 GB | 670.78 | **923.28 tok/s** | **+37.64%** | 641.91 | **923.07 tok/s** | **+43.80%** |
| V100 + RTX 2080 Ti | 982.50 | **1085.45 tok/s** | **+10.48%** | 1031.86 | **1239.70 tok/s** | **+20.14%** |

The RTX 2080 Ti also benefits strongly without a long cached prefix because the fork adds Turing-specific prompt-processing and attention paths.

Full methodology, correctness checks, and retained measurements are in [`benches/upstream-vs-optimized-0911/REPORT.md`](benches/upstream-vs-optimized-0911/REPORT.md).

## Build and run

### Build

Clone and build one binary containing both SM70 and SM75 kernels:

```bash
git clone --branch v100-optimized --single-branch https://github.com/mistrjirka/llama.cpp.git && cd llama.cpp && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='70;75' -DLLAMA_BUILD_UI=OFF && cmake --build build -j --target llama-server
```

`70;75` builds kernels for both V100 and RTX 2080 Ti, so the same build works on either GPU or on a mixed system. `-DLLAMA_BUILD_UI=OFF` skips the web UI and its build/download step; remove it if you use the built-in UI.

The same CUDA build is recommended for Qwen and Ornith. On V100, `GGML_CUDA_VOLTA_FORCE_MMQ=moe` selects MMQ for routed MoE experts at runtime, keeping dense and MoE models on one build.

### V100

For Qwen3.8-27B on a single V100, the server automatically selects the measured `batch=4096`, `ubatch=4096` defaults when `-b/-ub` are omitted.

```bash
export GGML_CUDA_VOLTA_FORCE_MMQ=moe
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --device CUDA0 --gpu-layers all --split-mode none \
  --ctx-size 131072 --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0
```

The `moe` selector only changes routed MoE expert matmuls on Volta. It can stay in a shared launcher used for dense Qwen3.8 and MoE models.

At 16k prompt processing, the measured V100 ubatch sweep was **876.49 tok/s at 1024**, **923.31 at 2048**, and **952.18 at 4096**.

### RTX 2080 Ti

For a single RTX 2080 Ti, the server automatically selects `batch=4096`, `ubatch=2048` for the tested Qwen3.8-27B model.

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --device CUDA0 --gpu-layers all --split-mode none \
  --ctx-size 32768 --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0
```

At 16k prompt processing, `ubatch=1024/2048/4096` measured **890.95 / 923.63 / 930.62 tok/s**. `ubatch=2048` keeps almost all of the performance while leaving more VRAM for context and server state.

### V100 + RTX 2080 Ti

Mixed-GPU placement remains explicit because context size and free VRAM materially change the best split. The following is the tested 400k-context configuration used for the long-context benchmark. Check `./build/bin/llama-server --list-devices` first; the device order below assumes the RTX 2080 Ti is `CUDA1` and the V100 is `CUDA0`.

```bash
export GGML_CUDA_VOLTA_FORCE_MMQ=moe
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
  --batch-size 4096 --ubatch-size 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-checkpoints 32 --checkpoint-min-step 8192
```

`4096/2048` is the serving default for this 400k setup. `ubatch=4096` was a little faster in the sweep but left only about 1 GiB free on the RTX 2080 Ti before adding other server state.

### Batch defaults

| Setup | Qwen3.8 default | Selection |
|---|---:|---|
| Single V100 | `4096/4096` | automatic |
| Single RTX 2080 Ti | `4096/2048` | automatic |
| V100 + RTX 2080 Ti | `4096/2048` | explicit in the tested 400k profile |
| Unknown model/GPU | upstream behavior | unchanged |

Explicit `-b/-ub`, configuration-file values, or environment values take precedence. `LLAMA_V100_AUTO_BATCH=0` disables the hardware-aware Qwen batch defaults.

## RTX 2080 Ti optimizations

The RTX 2080 Ti path targets the parts of long-context attention where Turing differs from Volta.

### Long-Q8 attention dispatch

For the tested D256/GQA6 tensor-parallel geometry, Q8 K/V uses a dedicated `16x2` long-prompt attention specialization. This was the first large SM75 kernel improvement during the optimization work.

### INT8 Tensor-Core QK

The optimized path uses the existing `q8_0` K cache as the source data. K is packed into Tensor-Core-friendly INT8 codes and scales, Q is quantized once for the attention call, and the QK score calculation uses Turing's INT8 Tensor Cores. The mask, softmax, and value accumulation remain on the established attention path.


These two changes explain why the standalone RTX 2080 Ti improves much more than the V100 in the cold-prompt upstream comparison.

## V100 optimizations

The Volta path lets long-context Q8 K/V use the compact V100 Tensor-Core attention specialization. Q8-backed KV is now eligible for the tuned compact kernel after conversion to the FP16 attention input.

The compact path remains specific to the validated long-context geometry. The regular FlashAttention dispatch handles other shapes.

## Ornith and MoE

Use the same build shown above. On V100 or a V100-containing system, set:

```bash
export GGML_CUDA_VOLTA_FORCE_MMQ=moe
```

This selects MMQ for routed expert matmuls on Volta. Dense Qwen3.8 has no routed expert matmuls, so the same environment setting can remain enabled in a common launcher.

The fair Ornith control used global `GGML_CUDA_FORCE_MMQ=ON` on both upstream and the fork. Those numbers are kept in [`benches/upstream-vs-optimized-0911/REPORT.md`](benches/upstream-vs-optimized-0911/REPORT.md). For serving, use the runtime `GGML_CUDA_VOLTA_FORCE_MMQ=moe` setting above.

The tested `Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf` is about 25 GiB and fits on the V100 or the combined V100 + RTX 2080 Ti setup.

## Other fork features

The branch also contains earlier work on MTP serving, exact-prefix KV sharing, parked agent sessions, V100 FlashAttention/GatedDeltaNet tuning, PXQ, and mixed-GPU scheduling. These features have separate benchmarks and controls:

- [`benches/mtp-final-integration-0907/REPORT.md`](benches/mtp-final-integration-0907/REPORT.md)
- [`benches/parallel-serving-0907/RESEARCH.md`](benches/parallel-serving-0907/RESEARCH.md)
- [`benches/readme-current-0908/REPORT.md`](benches/readme-current-0908/REPORT.md)
- [`benches/upstream-vs-optimized-0911/REPORT.md`](benches/upstream-vs-optimized-0911/REPORT.md)

## Benchmark notes

Benchmark percentages are measurements for the configurations above. Context length, quantization, tensor placement, batch size, and ubatch can move the bottleneck substantially. The upstream comparison report contains the exact controls, retained outputs, and correctness checks used for the headline tables.

For general llama.cpp APIs, platform support, and build documentation, see upstream [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), [`docs/`](docs/), and [`tools/server/README.md`](tools/server/README.md).

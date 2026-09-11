# llama.cpp for V100 and RTX 2080 Ti

[![GitHub stars](https://img.shields.io/github/stars/mistrjirka/llama.cpp?style=flat-square&logo=github)](https://github.com/mistrjirka/llama.cpp/stargazers)

A CUDA-focused `llama.cpp` fork for NVIDIA **Volta (SM70)** and **Turing (SM75)**. It is tested primarily on a **Tesla V100-SXM2 32 GB** and **RTX 2080 Ti 22 GB**, with **Qwen3.8-27B `UD-Q5_K_XL`** and Q8 K/V cache.

The main goal is lower prompt-processing latency, especially for long cached contexts. The normal GGUF format is unchanged; the headline Qwen results do **not** require a custom model quantization.

Current benchmark comparison: upstream `b0dcb8192`, measured 2026-09-11. Raw results and methodology are in [`benches/upstream-vs-optimized-0911/REPORT.md`](benches/upstream-vs-optimized-0911/REPORT.md).

## What this fork changes

For the tested Qwen3.8 long-context path:

- **V100:** Q8 K/V can use the compact Volta Tensor-Core attention path instead of falling back because the source cache is quantized.
- **RTX 2080 Ti:** the guarded D256 long-context Q8 path packs K once, quantizes Q once, and evaluates QK with **SM75 INT8 Tensor Cores**. V remains on the existing FP16 value path.
- **Turing dense quantized matmuls:** large prompt batches cross over from MMQ to dequantize + cuBLAS at the measured SM75 threshold.
- **Volta MoE:** routed experts can opt into MMQ at runtime with `GGML_CUDA_VOLTA_FORCE_MMQ=moe`; a second globally forced-MMQ build is not required.

The long-Q8 paths are geometry- and architecture-gated. They do not make the SM75 INT8 path run on V100 or change every FlashAttention call.

## Build

Use the normal build. Do **not** compile with global `GGML_CUDA_FORCE_MMQ=ON` for ordinary serving.

A single binary can contain both SM70 and SM75 kernels:

```bash
git clone --branch v100-optimized --single-branch \
  https://github.com/mistrjirka/llama.cpp.git
cd llama.cpp

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES='70;75' \
  -DLLAMA_BUILD_UI=OFF

cmake --build build -j --target llama-server
```

For a machine that will only ever use one architecture, replace `70;75` with `70` for V100 or `75` for RTX 2080 Ti. Omit `-DLLAMA_BUILD_UI=OFF` if you want the built-in web UI.

## Qwen3.8-27B quick start

The examples below use:

```text
Qwen3.8-27B-UD-Q5_K_XL.gguf
```

and Q8 K/V cache.

### One V100

Use **batch 4096 / ubatch 2048** as the conservative starting point. At 32k context, ubatch 4096 measured about 3% faster, so it is worth trying when you have enough VRAM headroom.

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --device CUDA0 \
  --gpu-layers all \
  --split-mode none \
  --ctx-size 131072 \
  --parallel 1 \
  --flash-attn on \
  --batch-size 4096 \
  --ubatch-size 2048 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

For the 32k single-V100 benchmark, `--ubatch-size 4096` reached 952.18 tok/s versus 923.31 tok/s at 2048. The command above keeps 2048 as the safer starting point for larger contexts.

### One RTX 2080 Ti

For the tested 22 GB card, use **batch 4096 / ubatch 2048**. The Q5_K_XL model fits the tested 32k Q8-KV configuration.

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --device CUDA0 \
  --gpu-layers all \
  --split-mode none \
  --ctx-size 32768 \
  --parallel 1 \
  --flash-attn on \
  --batch-size 4096 \
  --ubatch-size 2048 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

`--ubatch-size 4096` was only about 0.8% faster in the 16k sweep and uses more VRAM, so 2048 is the default recommendation for this card.

### V100 + RTX 2080 Ti

Mixed-GPU placement is explicit because available VRAM, context size and device enumeration matter. This is the tested 400k-context Qwen setup used for the long-context measurements. Check `./build/bin/llama-server --list-devices` before copying the device order literally.

```bash
export GGML_CUDA_ALLREDUCE=internal
export GGML_CUDA_AR_COPY_THRESHOLD=131072

./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --device CUDA1,CUDA0 \
  --tensor-split 4,5 \
  --split-mode tensor \
  --gpu-layers all \
  --fit off \
  --parallel 1 \
  --ctx-size 409600 \
  --override-kv qwen35.context_length=int:409600 \
  --rope-scaling yarn \
  --rope-scale 1.5625 \
  --yarn-orig-ctx 262144 \
  --flash-attn on \
  --batch-size 4096 \
  --ubatch-size 2048 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ctx-checkpoints 32 \
  --checkpoint-min-step 8192
```

In the 400k-context sweep, `ubatch=2048` was the safer choice; `ubatch=4096` left only about 1 GiB free on the 2080 Ti.

## Performance vs current upstream

All headline Qwen comparisons use the same `UD-Q5_K_XL` model, Q8 K/V, FlashAttention, MTP disabled, and one server slot.

### Cold prompt processing — matched settings

The table below keeps `batch/ubatch = 4096/2048` matched between upstream and the fork so the comparison does not credit the fork for a different ubatch.

| Hardware | 1k upstream | 1k fork | Change | 16k upstream | 16k fork | Change |
|---|---:|---:|---:|---:|---:|---:|
| V100 32 GB | 857.45 | **901.59** | **+5.15%** | 849.80 | **921.07** | **+8.39%** |
| RTX 2080 Ti 22 GB | 670.78 | **923.28** | **+37.64%** | 641.91 | **923.07** | **+43.80%** |
| V100 + RTX 2080 Ti | 982.50 | **1085.45** | **+10.48%** | 1031.86 | **1239.70** | **+20.14%** |

### Recommended single-GPU ubatch

A follow-up sweep on the fork found:

| GPU | ubatch 1024 | ubatch 2048 | ubatch 4096 | Recommendation |
|---|---:|---:|---:|---|
| V100, 16k PP | 876.49 | 923.31 | **952.18 tok/s** | **4096** when VRAM permits |
| RTX 2080 Ti, 16k PP | 890.95 | 923.63 | **930.62 tok/s** | **2048** for the small performance/VRAM trade-off |

This is why the single-V100 auto default differs from the matched benchmark table above.

### 100k cached + 1k append

Main long-context workload: 409600-token YaRN context, V100 + RTX 2080 Ti tensor parallel, restored 100,000-token Q8 KV state, then a 1,000-token append.

| Metric | Upstream `b0dcb8192` | `v100-optimized` | Change |
|---|---:|---:|---:|
| Prompt processing | 408.79 tok/s | **691.80 tok/s** | **+69.23%** |
| TTFT | 2.497 s | **1.499 s** | **-39.97%** |
| 64-token decode | 24.02 tok/s | **26.68 tok/s** | **+11.1%** |
| 64-token end-to-end | 5.131 s | **3.867 s** | **-24.6%** |

The final merged tree was also rebuilt with the old performance environment switches unset. Its warm 100k+1k samples settled at about **693.9 tok/s**, and the exact D256/GQA6/KV101120/Q1000 attention test passed against the CPU reference on both the V100 and RTX 2080 Ti.

See [`benches/upstream-vs-optimized-0911/REPORT.md`](benches/upstream-vs-optimized-0911/REPORT.md) for the retained raw data and benchmark details.

## Why long-context Q8 is faster

The persistent KV cache remains ordinary llama.cpp `q8_0`; there is no new cache format.

On the tested Turing path:

1. the D256/GQA6 Q8 cache is allowed into the tuned two-local-KV-head long-prompt dispatch;
2. K is repacked from `q8_0` into contiguous INT8 codes plus scales;
3. Q is quantized once for the attention call;
4. QK uses Turing INT8 Tensor Cores;
5. the value path remains FP16.

On V100, the corresponding Q8 source is accepted by the compact Volta FP16 Tensor-Core attention path. Volta does not use the SM75 INT8-QK kernel.

The fast paths are guarded by architecture, D256 geometry, KV type, head layout and long-context conditions rather than being global replacements for FlashAttention.

## Ornith / MoE

Use the **same normal build**. Do not maintain a second globally forced-MMQ binary just for MoE.

On V100, enable MMQ only for routed expert matmuls with:

```bash
export GGML_CUDA_VOLTA_FORCE_MMQ=moe
```

This leaves dense matmuls on their normal dispatch path. The environment variable is an explicit runtime selector; leaving it unset keeps the standard policy.

A fairness control was also run with global `GGML_CUDA_FORCE_MMQ=ON` on **both** upstream and the fork. That control did not show a cold-prompt Ornith win for the fork, so the README does not present global FORCE_MMQ as a recommended build mode. Details are in the upstream-comparison report.

The tested `Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf` is about 25 GiB, so it does not fit fully on a 22 GB RTX 2080 Ti by itself.

## Recommended settings and opt-outs

| Behavior | Recommendation | Override / opt-out |
|---|---|---|
| Qwen batch/ubatch | start with `4096/2048` | pass different `-b/-ub` values |
| V100 with spare VRAM | try ubatch `4096` | keep `2048` for more headroom |
| Mixed-GPU placement | explicit | set devices and tensor split manually |
| SM75 long-Q8 dispatch | guarded, on | `GGML_CUDA_TURING_TP_Q8_ATTN=0` |
| SM75 INT8 QK | guarded, on | `GGML_CUDA_TURING_INT8_QK=0` |
| Volta compact long-Q8 dispatch | guarded, on | `GGML_CUDA_VOLTA_TP_Q8_ATTN=0` |
| Volta MoE-only MMQ | opt-in | `GGML_CUDA_VOLTA_FORCE_MMQ=moe` |

## Other fork features

This branch also contains earlier work that is separate from the headline Qwen benchmark above, including:

- MTP/draft-head serving improvements and per-request draft limits;
- exact-prefix KV sharing and parked agent sessions;
- V100 FlashAttention and GatedDeltaNet tuning;
- PXQ support and SM75 PXQ prompt-processing optimizations;
- mixed-GPU scheduling and long-context experiments.

Useful detailed reports include:

- [`benches/mtp-final-integration-0907/REPORT.md`](benches/mtp-final-integration-0907/REPORT.md)
- [`benches/parallel-serving-0907/RESEARCH.md`](benches/parallel-serving-0907/RESEARCH.md)
- [`benches/readme-current-0908/REPORT.md`](benches/readme-current-0908/REPORT.md)
- [`benches/upstream-vs-optimized-0911/REPORT.md`](benches/upstream-vs-optimized-0911/REPORT.md)

Those experiments use different controls and should not be added arithmetically to the current-upstream numbers above.

## Benchmark notes

- Benchmark percentages are measured examples, not universal speedups.
- Context length, quantization, GPU placement and ubatch can move the bottleneck substantially.
- The first request after restoring/loading state can be warmup-sensitive; retained benchmark means follow the methodology in the corresponding report.
- FORCE_MMQ materially changes the compute balance of MoE models, so compare it only against a matched FORCE_MMQ control.
- PXQ results are separate from the Qwen `UD-Q5_K_XL` results on this page.

For general llama.cpp documentation, APIs and platform support, use upstream [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), [`docs/`](docs/), and [`tools/server/README.md`](tools/server/README.md).

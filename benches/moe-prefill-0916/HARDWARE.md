# Hardware portability: standalone and multi-device checks

16 September 2026

## Conclusion

Both physical single-GPU configurations passed the full-model tests: V100 alone and the available 22 GiB RTX 2080 Ti alone. No sparse-kernel or executor source change was needed; a new benchmark launcher removed the old mixed-GPU placement preset and used smaller single-card expert-residency budgets.

Two and three logical V100 backends also passed using the fork's existing virtual-device mode on one physical V100. This is a functional scheduling/state-transfer test, not a real two/three-V100 hardware or performance test. Only one physical V100 and one physical RTX were available in this Sandbox.

The important remaining portability limitation is in the resident-expert budget configuration: the frozen executor reads only BASE_LAYERS_0 and BASE_LAYERS_1. A third device participates in execution, but its explicit base allowance cannot be configured. It receives zero base plus the common spare-workspace allowance. This limits capacity use; it is not a hard two-GPU execution limit.

## Test scope

Model: Qwen3.8-Flash-Next UD-IQ4_XS; Q8_0 K/V; one causal text sequence; layer placement, not tensor/expert parallel execution. All canonical expert weights remain in host RAM and the existing executor caches or streams them to the selected GPU. Native PLE storage remains CPU-backed. Standalone means one physical GPU, not a model entirely resident in that GPU's VRAM and not a low-system-RAM qualification.

The primary test restores the same saved 100,000-token prefix and appends the same 1,000 C++ input tokens. Context capacity is 131,072; mixer chunk and expert tile are 1,024. It samples eight full-vocabulary prompt rows and then scores eight fixed continuation inputs. Those are not eight freely generated output tokens.

Latencies below measure new-input processing. Model loading, saved-prefix restore, first expert-cache population, and fixed continuations are excluded. Each configuration runs sparse off/off/on/on/off/on/off. The first off and first on observations are excluded, leaving three off and two on observations. Tables report medians, not a broad statistical confidence claim.

The two sides use the same new layer-first executor. Sparse off is the previous attention kernel, not the original V100-optimized chronological executor. The sparse variant still uses the existing model-selected attention mask; this audit does not resolve the earlier precision-policy or general-quality questions.

## Fresh cached-prefill measurements

| Physical configuration | Previous attention | Native sparse attention | Sparse input tokens/s | Latency reduction |
|---|---:|---:|---:|---:|
| V100 alone (32 GiB) | 4.494 s | 3.672 s | 272.3 | 18.3% |
| RTX 2080 Ti alone (22 GiB) | 4.911 s | 4.059 s | 246.4 | 17.4% |
| V100 first / RTX second (36 + 12 layers) | 4.007 s | 3.103 s | 322.2 | 22.6% |
| RTX first / V100 second (12 + 36 layers) | 3.811 s | 2.930 s | 341.3 | 23.1% |

The reversed mixed placement changes which model layers and the output head execute on each GPU. It is not merely renaming identical work. Its result demonstrates that CUDA0 need not be the V100; the faster recorded time is not a universally validated replacement preset.

| Configuration | Off observations (ms) | On observations (ms) | First off observation (ms) |
|---|---|---|---:|
| V100 alone (32 GiB) | 4494.26, 4514.32, 4471.64 | 3717.95, 3626.79 | 8069.28 |
| RTX 2080 Ti alone (22 GiB) | 5106.60, 4900.67, 4911.50 | 4074.56, 4043.11 | 5889.19 |
| V100 first / RTX second (36 + 12 layers) | 4134.33, 4007.12, 3975.17 | 3175.87, 3030.77 | 7222.65 |
| RTX first / V100 second (12 + 36 layers) | 4026.18, 3810.77, 3804.43 | 2940.08, 2919.73 | 7137.11 |

## Uncached multi-chunk checks

Each single GPU also processed 4,096 Python input tokens from an empty prompt cache, as one request spanning four chronological mixer chunks. Model loading and first population are not included in the warm-model median. The selected-entry attention gate does not activate at this short history; off/on output equality is a negative control, not a sparse-attention speed comparison.

| GPU | Median of three warmed 4k requests | Input tokens/s | Sparse toggle changes checked outputs |
|---|---:|---:|---|
| V100 alone | 7.047 s | 581.2 | No; byte-identical |
| RTX 2080 Ti alone | 6.481 s | 632.0 | No; byte-identical |

## Device-count and device-order checks

| Test | Model layers per logical device | Requests | Resident stores | State-copy payload per 1k request | Result |
|---|---|---:|---:|---:|---|
| V100 alone (32 GiB) | 48 | 7 | 1 | 0.00 MB | Passed |
| RTX 2080 Ti alone (22 GiB) | 48 | 7 | 1 | 0.00 MB | Passed |
| V100 first / RTX second (36 + 12 layers) | 36 / 12 | 7 | 2 | 40.96 MB | Passed |
| RTX first / V100 second (12 + 36 layers) | 12 / 36 | 7 | 2 | 40.96 MB | Passed |
| Two logical V100 devices on ONE physical V100 | 24 / 24 | 7 | 2 | 40.96 MB | Passed |
| Three logical V100 devices on ONE physical V100 | 16 / 16 / 16 | 7 | 3 | 81.92 MB | Passed |

All these runs completed at the expected sequence positions. Every repeated sampled prompt-score file and fixed-continuation score file was byte-identical within its own variant. The executor reported fully GPU-resident token state, with zero host backing for the main state, expert inputs, expert sum and shared-expert full-window arrays. These are software state/counter checks, not a new independent physical-copy audit.

The virtual tests use GGML_CUDA_DEVICES=2 or 3 with only the real V100 exposed. The devices share that one GPU's memory and compute. The three-device test exercised three backend stores and two successive state handoffs. It cannot test physical peer access, NVLink, extra aggregate VRAM, bandwidth contention, or speed scaling on real multi-V100 hardware. Its timings are retained in raw evidence but intentionally omitted from the physical performance table.

## Actual kernel dispatch

Two separate instrumented standalone runs verify the main sparse kernels rather than accepting a successful fallback as proof. Their sampled prompt and continuation outputs match the corresponding uninstrumented sparse runs byte-for-byte. Profiling times are not mixed into the primary benchmark medians.

| Physical GPU | Main sparse kernel shape | Main sparse launches | Mask compaction launches |
|---|---|---:|---:|
| V100 | D256, 1 x 32 grouped columns | 12 | 12 |
| RTX 2080 Ti | D256, 1 x 8 grouped columns | 12 | 12 |

## Hardcoded assumptions and remaining limits

**Launcher preset, removed locally.** The previous benchmark assumes 48 layers divided 36/12, GPU-specific offload counts and split weights 36/13 including the output layer. The new bench-portable.cpp takes BENCH_GPU_LAYERS explicitly, checks the number of visible CUDA backends, and verifies actual layer ownership and canonical host expert buffers after loading. It keeps the output head on the final layer device.

**Two-entry residency configuration, still in the frozen runtime.** src/llama-layer-first.cpp:197-199 constructs a two-entry base allowance array. Three guarded reads later substitute zero for higher ordinals. The ordinals follow first appearance in model layer order. A BASE_LAYERS_2 environment value is never read. The new runner rejects nonzero third-device allowances instead of silently pretending they work. Generalizing this configuration is the next small portability change.

**Memory is manually budgeted.** The adaptive planner uses actual expert weight bytes under a configured ceiling, but it is not a universal fit-to-free-VRAM algorithm. The common workspace allowance is 3,072 MiB in these tests. The base allowances are shown below; they are working test presets, not an exhaustive optimum.

| Configuration | Base resident-layer allowances |
|---|---|
| V100 alone (32 GiB) | 14 |
| RTX 2080 Ti alone (22 GiB) | 6 |
| V100 first / RTX second (36 + 12 layers) | 16 / 10 |
| RTX first / V100 second (12 + 36 layers) | 10 / 16 |

**Two banks are not two GPUs.** Each backend owns two rotating weight banks. Main token stores and weight pools are dynamic maps keyed by backend, and layer ownership is read from the model. The three-device execution confirms this beyond a static inspection.

**Architecture specialization is intentional.** The kernel chooses the Volta or Turing specialization by compute capability, not by CUDA ordinal or GPU marketing-name matching. The new model path is specialized for D256 and turns on only for multiple queries with at least 32,768 history entries. Its selected-count bound comes from model selection geometry; 2,051 in replay tools is a fixture constant.

**Model and serving restrictions remain.** This is the QWEN4EXP layer-first adapter with supported separate expert projections, one causal text sequence and CUDA layer backends. It is not a universal MoE executor or a qualification of tensor parallelism, arbitrary output-only GPU placement, multi-node execution, concurrent slots, speculative decoding, LoRA, or other architectures.

## What to conclude about two or three physical V100 cards

Two V100 cards in one host are a credible supported target for the tested layer-placement design: both ordinals have configurable residency allowances, every layer can execute on a V100, and two logical Volta backends passed. Physical V100-to-V100 communication and its measured performance remain untested here.

Three V100 cards are not blocked by a fixed two-device scheduler: three logical Volta backends completed the real model request. The third-card residency configuration should be generalized before expecting useful use of all three cards' memory. A real three-card test is still required for peer transfers, capacity, and performance.

Layer placement is sequential through model layers. Extra GPUs can keep more expert weights resident and reduce host uploads, but this executor does not split every layer's calculation over all GPUs simultaneously. Therefore neither the virtual tests nor the mixed-pair results justify a 2x/3x compute-speed claim. Actual peer access depends on the CUDA/PCIe/NVLink topology.

## Preservation, limitations and saved evidence

Completed 54 requests in 10 explicit test invocations. Source status, audited source hashes, and both frozen library hashes were verified unchanged. No production launcher change, commit, push or deployment was made. The earlier FP32-request/FP16-value-accumulator mismatch remains unresolved, and sparse attention remains opt-in.

These tests do not qualify an 11 GiB RTX 2080 Ti, a 16 GiB V100, a low-RAM host, cold 100k prefill, arbitrary request sizes, or broad task quality. The 100k prefix was restored from the existing fixture rather than recomputed on each hardware layout. GPU memory samples are periodic NVML observations, not guaranteed instantaneous maxima; the first V100 cached run predates the added sampler.

All new work is under /workspace/moe-layer-first-0914/hardware-portability-0916/. prepare.py and bench-portable.cpp implement only the local test launcher changes, including safe padding for the quality harness's fixed-continuation fallback. run.py records explicit commands, environment, hardware layout and runtime hashes. analyze.py verifies repetition hashes and resident-state counters; analyze-profiles.py verifies Nsight kernel names and counts. source-audit.json records exact code excerpts and hashes. The profiler-name parser was adjusted for Nsight's numeric boolean template arguments; that was a parser correction, not a kernel failure.

Raw score vectors, model weights, prefix state, frozen libraries and Nsight databases remain in the persistent Sandbox. The small evidence archive contains scripts, manifests, summaries and logs, not those large binary artifacts.

Reproduction examples (use the shared gpu:all lock and a NEW output name):

```sh
cd /workspace/moe-layer-first-0914/hardware-portability-0916
python3 prepare.py
python3 run.py new-v100-run --devices GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79 --layers 48 --base 14
python3 run.py new-rtx-run --devices GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29 --layers 48 --base 6
python3 run.py new-virtual-three --devices GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79 --virtual 3 --layers 16,16,16 --base 2,2,0
```

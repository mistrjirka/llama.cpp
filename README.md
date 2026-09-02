# llama.cpp — V100 / Volta optimized fork

This is the `v100-optimized` runtime-integration branch of [`mistrjirka/llama.cpp`](https://github.com/mistrjirka/llama.cpp), focused on long-context inference on NVIDIA Volta, especially the Tesla V100.

The main target is **Qwen3.8-27B** in coding/agent workloads with a large reusable context. The branch also carries Volta tuning that benefits **Ornith-1.5-35B-A3B**, plus MTP batching and long-session prompt-cache improvements.

This branch is newer than `qwen38-lossless-agent-cache` and is based on much newer upstream llama.cpp. The original experimental work was later decomposed into smaller, cleaner candidates, benchmarked again, and then integrated here.

## Performance overview

The latest clean reconstruction before this integration branch was cut compared the final CUDA candidate stack against the same current-upstream revision.

Primary Qwen workload:

```text
Qwen3.8-27B UD-Q5_K_XL
Tesla V100 32 GB + RTX 3060 Ti 8 GB
100,000-token restored state + 1,000 new prompt tokens + 64 generated tokens
262,144 context, parallel 1
q8_0 target KV
MTP n-max 2
all model layers on GPU, layer split 64,2
```

| build | prompt processing | change | result |
|---|---:|---:|---|
| upstream | 329.306 tok/s | — | reference |
| clean Volta CUDA stack | **478.146 tok/s** | **+45.20%** | exact output, MTP 37/52 every run |

Two independent runs with 256 generated tokens made the decode comparison less noisy:

| run | PP change | TG change | result |
|---|---:|---:|---|
| A | **+44.70%** | +1.38% | exact output, MTP 153/202 |
| B | **+40.01%** | +0.86% | exact output, MTP 153/202 |

The large gain is therefore in **long-context prompt processing**. Token generation is essentially unchanged.

The clean stack above contains the CUDA work integrated in this branch: the FlashAttention barrier fix, Volta 256x256 configuration, strict adaptive 2-CTA specialization, and Volta scalar GatedDeltaNet specialization. This branch additionally carries the runtime/cache features described below.

## Qwen3.8: clean isolated 256x256 FlashAttention result

The strongest small upstream candidate is one explicit sm70 FlashAttention configuration. It is submitted separately as [llama.cpp PR #27997](https://github.com/ggml-org/llama.cpp/pull/27997).

### Simulated agent turn: 100k cached + 1024 PP + 512 TG

Single V100, 200 W, q8_0 K/V, no forced MMQ. A 100,000-token state was serialized and restored for each measured arm; PP and TG were timed separately.

| model | upstream PP | edited PP | PP change | upstream TG | edited TG | TG change |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 276.063 | **352.926** | **+27.84%** | 16.269 | 16.366 | +0.60% |
| Ornith-1.5-35B-A3B | 485.493 | **580.357** | **+19.54%** | 59.262 | 59.386 | +0.21% |
| Gemma 4 12B | 483.430 | 486.664 | +0.67% | 39.445 | 39.457 | +0.03% |

The change is geometry-specific, not model-name-specific. Qwen3.8 and Ornith both use the relevant 256-wide attention geometry.

### Qwen3.8 scaling with KV depth

Single V100, 200 W, q8_0 K/V, 1024 new prompt tokens:

| prefilled KV depth | upstream | edited | change |
|---:|---:|---:|---:|
| 4,096 | 736.718 tok/s | 754.633 tok/s | +2.43% |
| 16,384 | 609.015 tok/s | 657.612 tok/s | **+7.98%** |
| 65,536 | 356.232 tok/s | 434.180 tok/s | **+21.88%** |
| 100,000 | 275.180 tok/s | 349.180 tok/s | **+26.89%** |

This is why a normal short `pp512` benchmark understates the benefit for agent workloads.

### Independent 2x-V100 validation

A separate tester later benchmarked PR #27997 on another DGX-1 using **2x Tesla V100-SXM2 32 GB**, Qwen3.8-27B `Q8_K_XL`, `llama-server`, tensor split, and a different benchmark path:

| prompt depth | upstream | + PR #27997 | change |
|---:|---:|---:|---:|
| 71,713 | 955.9 tok/s | 1067.1 tok/s | **+11.6%** |
| 122,869 | 761.7 tok/s | 892.7 tok/s | **+17.2%** |

Decode remained effectively unchanged. See the [independent result](https://github.com/ggml-org/llama.cpp/pull/27997#issuecomment-5478564345).

## Ornith 1.5 benchmarks

Ornith-1.5-35B-A3B is an important second target because it uses the same **256x256 FlashAttention family** and `S_v=128` scalar GatedDeltaNet, but has a different GQA layout and routed-MoE feed-forward path.

The benchmarked quantization in the clean PR-sized tests was:

```text
Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf
```

### Isolated Volta 256x256 config

Standard short-context `llama-bench`, single V100:

| test | upstream | edited | change |
|---|---:|---:|---:|
| pp512 | 865.224 | 890.623 | +2.94% |
| pp2048 | 840.202 | 871.888 | **+3.77%** |
| pp4096 | 821.086 | 837.496 | +2.00% |
| tg128 | 104.578 | 104.646 | +0.06% |

At a realistic restored **100k KV + 1024 PP + 512 TG**, the same isolated FA config gives the much larger result shown above:

```text
PP  485.493 -> 580.357 tok/s   +19.54%
TG   59.262 ->  59.386 tok/s    +0.21%
```

### Ornith with `GGML_CUDA_FORCE_MMQ=ON`

For routed MoE on V100, forcing MMQ is a large independent optimization. It keeps large quantized `MUL_MAT_ID` work on the GPU instead of falling into a much slower host-assisted path.

Matched PR #27997 builds at short context:

| test | PR02 normal build | PR02 + FORCE_MMQ | FORCE_MMQ gain |
|---|---:|---:|---:|
| pp512 | 862.600 | **1464.154** | **+69.74%** |
| pp2048 | 857.413 | **1452.536** | **+69.41%** |
| pp4096 | 847.210 | **1413.810** | **+66.88%** |
| tg128 | 105.246 | 106.063 | +0.78% |

The FA optimization still helps after MMQ has made the MoE matmuls much faster. In a controlled 10-sample confirmation, the short-context PR02 delta with FORCE_MMQ was +0.19% at pp2048 and +0.60% at pp4096. As attention becomes a larger fraction of runtime at long KV depth, its gain grows again:

| KV depth | upstream + FORCE_MMQ | edited FA + FORCE_MMQ | FA change |
|---:|---:|---:|---:|
| 16,384 | 1171.403 ± 5.112 | 1238.417 ± 3.333 | **+5.72%** |
| 65,536 | 760.651 ± 2.346 | 895.968 ± 2.160 | **+17.79%** |

So for Ornith on V100 the best practical combination is generally **this branch + a FORCE_MMQ build**.

Build an MMQ-forced variant with:

```bash
cmake -S . -B build-mmq \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DLLAMA_BUILD_UI=OFF

cmake --build build-mmq -j --target llama-server llama-cli
```

Do **not** use FORCE_MMQ globally for Qwen3.8; its dense Volta matmuls benefit from the normal FP16/cuBLAS path.

### Volta GatedDeltaNet x4

The branch also contains a separate sm70 specialization for scalar-gate `S_v=128` GatedDeltaNet prefill. It processes four independent state/output columns per warp while sharing Q/K/gate loads.

Representative isolated Ornith kernel time:

```text
v_repeat=2: ~1642 us -> ~1067 us
```

Controlled same-process, CUDA-graphs-disabled whole-model measurements gave:

```text
Ornith 8k context   ~+2.32% PP
Ornith 16k context  ~+2.03% PP
```

The outputs were exact. Focused GDN correctness tests passed **40/40 on V100** and **40/40 on RTX 3060 Ti**.

### Adaptive 2-CTA does not broaden onto Ornith

The later Qwen-oriented adaptive 2-CTA optimization is deliberately restricted to the exact Volta `256x256, ncols1=32, ncols2=2` layout. Ornith's `8x8` legacy path remains outside that specialization. During the clean PR03 work the extracted Ornith-style kernel SASS was byte-identical between the PR02 baseline and strict PR03 build.

This restriction replaced an earlier broader K96 implementation specifically to avoid perturbing layouts such as Ornith's.

## Quick start: Qwen3.8 on one V100

### 1. Build

You need a C++ compiler, CMake, the CUDA toolkit, and a Volta GPU.

```bash
git clone --branch v100-optimized --single-branch \
  https://github.com/mistrjirka/llama.cpp.git
cd llama.cpp

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DLLAMA_BUILD_UI=OFF

cmake --build build -j --target llama-server llama-cli
```

Omit `-DLLAMA_BUILD_UI=OFF` if you want the built-in web UI. For general build problems, see [`docs/build.md`](docs/build.md).

### 2. Get Qwen3.8-27B

The primary development quantization is:

```text
unsloth/Qwen3.8-27B-GGUF
Qwen3.8-27B-UD-Q5_K_XL.gguf
```

Using the Hugging Face CLI:

```bash
mkdir -p models/Qwen3.8-27B
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --local-dir models/Qwen3.8-27B

export MODEL="$PWD/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf"
```

### 3. Start the optimized server

If the V100 is the only visible NVIDIA GPU:

```bash
export CUDA_VISIBLE_DEVICES=0

./build/bin/llama-server \
  --model "$MODEL" \
  --alias qwen3.8-27b \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 262144 \
  --parallel 1 \
  --fit off \
  --gpu-layers 63 \
  --split-mode layer \
  --flash-attn on \
  --batch-size 4096 \
  --ubatch-size 1024 \
  --prefill-reuse 1024 \
  --pipeline-copies 2 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-type-k-draft f16 \
  --cache-type-v-draft f16 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-ubatch 1024 \
  --cache-ram 65536 \
  --cache-idle-slots \
  --ctx-checkpoints 32 \
  --checkpoint-min-step 8192 \
  --jinja \
  --reasoning on \
  --reasoning-preserve \
  --perf \
  --metrics
```

`--gpu-layers 63` was the tested tight 262k MTP-safe placement on the development 32 GB V100. Exact headroom can change with CUDA, quantization, or later allocator changes, so re-check VRAM on another build.

`--cache-ram 65536` allows up to 64 GiB of host prompt-cache data. Lower it or omit it on systems with less RAM. It improves long-session reuse but is unrelated to the CUDA kernel speedup.

Check the server:

```bash
curl http://127.0.0.1:8080/health
```

The OpenAI-compatible endpoint is `http://127.0.0.1:8080/v1`.

## V100 + RTX 3060 Ti

For a 32 GB V100 plus an 8 GB RTX 3060 Ti, build both architectures:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES='70;86' \
  -DLLAMA_BUILD_UI=OFF

cmake --build build -j --target llama-server llama-cli
```

Put the V100 first in CUDA's visible-device order, then use the tested Qwen placement:

```bash
export CUDA_VISIBLE_DEVICES=1,0

./build/bin/llama-server \
  --model "$MODEL" \
  --alias qwen3.8-27b \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 262144 \
  --parallel 1 \
  --fit off \
  --gpu-layers all \
  --split-mode layer \
  --tensor-split 64,2 \
  --flash-attn on \
  --batch-size 4096 \
  --ubatch-size 4096 \
  --prefill-reuse 1024 \
  --pipeline-copies 2 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-type-k-draft f16 \
  --cache-type-v-draft f16 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-ubatch 1024 \
  --cache-ram 65536 \
  --cache-idle-slots \
  --ctx-checkpoints 32 \
  --checkpoint-min-step 8192 \
  --jinja \
  --reasoning on \
  --reasoning-preserve \
  --perf \
  --metrics
```

The `64,2` split is specific to the tested V100 32 GB + 3060 Ti 8 GB setup. Retune it for a different VRAM layout.

## What this fork changes

The CUDA optimizations are selected by hardware/tensor geometry, not model names.

### 1. FlashAttention barrier correctness

The upstream FA combine path had complementary warp branches reaching different static `__syncthreads()` sites. On V100, Compute Sanitizer `synccheck` reported **1024 divergent-barrier errors** for the reproducer.

The branch moves the synchronization to a single uniform block-wide barrier. The clean fix is submitted as [llama.cpp PR #27955](https://github.com/ggml-org/llama.cpp/pull/27955).

Validation of the clean fix:

```text
V100 synccheck   0 errors
V100 racecheck  0 hazards / errors / warnings
V100 memcheck   0 errors
```

Temperature-normalized whole-model testing measured it as performance-neutral.

### 2. Volta 256x256 FlashAttention config

Qwen3.8's full-attention layers use 256-wide Q/K/V heads. The old sm70 path fell through to an Ampere-oriented config.

The clean Volta config changes the target kernel from roughly:

```text
upstream fallback   255 registers/thread, 552 B stack
Volta config        255 registers/thread, ~48-56 B stack
```

It stages Q in shared memory and reduces V scratch. This small config change is responsible for the large long-KV improvement shown above and is [PR #27997](https://github.com/ggml-org/llama.cpp/pull/27997).

Full existing 256x256 FA correctness matrix:

```text
V100        132/132 passed
RTX 3060 Ti 132/132 passed
```

### 3. Strict adaptive Volta 2-CTA specialization

The next Qwen-specific step creates a separate compact kernel only for exact sm70 `256x256, 32x2` and uses:

```text
K scratch        96 half2
V scratch        64 half2
combine scratch  64 half2
shared memory    49,152 B/block
```

That allows two 128-thread blocks to fit in the V100's 96 KiB shared-memory budget. The launch is enabled only when its whole-tile wave-efficiency guard says two CTAs are useful; otherwise the normal 256x256 kernel remains the fallback.

A clean 100k-KV + 1k-PP sanity run measured **+13.11%** on top of the PR02 baseline. The strict version intentionally leaves Ornith's 8x8 layout unchanged.

### 4. Volta scalar GatedDeltaNet x4

For exact sm70, scalar gate, `S_v=128`, prefill-only GDN, four independent state columns are processed per warp while reusing Q/K/gate inputs. Decode and non-target architectures remain on the normal path.

### 5. Additional Volta attention geometry tuning

The runtime branch also contains later guarded work for other Volta FA shapes:

- `128x128`: safer Q-shared/K96/V32 config, primarily useful for high-GQA layouts;
- `192x128`: layout-aware FA batch tuning for the `4x16` GQA16 layout while keeping `8x8` unchanged.

These are independent of the core Qwen3.8 `256x256` optimization.

### 6. Lossless quantized-weight reuse during prefill

`--prefill-reuse 1024` allows a larger physical prompt batch to reuse a converted quantized weight while keeping the smaller cuBLAS GEMM tile. It targets the Volta Q5_K/Q6_K -> F16 cuBLAS path.

This remains an experimental fork-level tuning knob. After the newer FA work its incremental Qwen gain is comparatively small, and it should not be assumed safe/performance-positive for every architecture.

### 7. Smaller pipeline scheduler allocation

`--pipeline-copies 2` reduces cross-backend scheduler input copies. Its main purpose is reducing compute-buffer VRAM so larger long-context graphs fit.

### 8. Separate MTP draft ubatch

`--spec-draft-ubatch 1024` lets the MTP/draft context use a smaller physical ubatch than the target context. This is useful when the target uses a large prompt ubatch but the transient draft graph has tighter VRAM constraints.

### 9. Agent prompt-cache and recurrent checkpoint behavior

The branch carries server changes intended for coding-agent traffic:

- prompt-cache states are chosen by the deepest usable absolute prefix rather than ratio alone;
- recurrent checkpoints preserve likely replay boundaries;
- duplicate exact checkpoint prefixes are refreshed rather than stored twice;
- observed replay hits influence checkpoint retention;
- plain-attention draft KV is not duplicated into recurrent checkpoints when it can be suffix-trimmed normally.

A regression test covers the case where a short live prompt would otherwise beat a much deeper cached prefix.

## Validation summary

The clean candidate work was checked beyond a single generated response:

```text
256x256 FA matrix        V100 132/132, RTX 3060 Ti 132/132
focused GDN cases        V100 40/40,   RTX 3060 Ti 40/40
V100 FA synccheck        0 errors
V100 FA racecheck        0 hazards / errors / warnings
V100 FA memcheck         0 errors
```

The Qwen candidate stack also reproduced exact token/output trajectories in the long-context ABBA comparisons quoted above.

## Upstream PR work

The integration branch intentionally contains more than should be proposed upstream in one change. Clean work has been split into independent pieces.

Submitted upstream:

- [#27955 — CUDA: fix divergent FlashAttention barrier](https://github.com/ggml-org/llama.cpp/pull/27955)
- [#27997 — add sm70 FlashAttention config for DKQ=256, DV=256, ncols=64](https://github.com/ggml-org/llama.cpp/pull/27997)

Other work remains intentionally separate: adaptive 2-CTA, Volta GDN x4, 128x128/192x128 tuning, and server-cache policy.

## Notes on benchmark interpretation

Do not compare absolute numbers from different tables unless their hardware/power/build settings match. In particular:

- some early short-context model sweeps were captured at the V100's 300 W limit;
- the later controlled KV-depth and agent-turn results use an enforced **200 W** V100 limit;
- FORCE_MMQ materially changes Ornith's compute balance, so its absolute PP numbers should be compared only against the matched FORCE_MMQ baseline;
- the older `qwen38-lossless-agent-cache` README's ~41% headline came from the complete earlier fork, while the newer tables above isolate individual clean changes and the reconstructed final CUDA stack.

For general llama.cpp usage, APIs and platform documentation, use upstream [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), [`docs/`](docs/), and [`tools/server/README.md`](tools/server/README.md).

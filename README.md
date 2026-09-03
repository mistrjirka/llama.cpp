# llama.cpp V100 / Volta optimized fork

`v100-optimized` is a [`llama.cpp`](https://github.com/mistrjirka/llama.cpp) branch tuned for long-context inference on NVIDIA Volta, especially the Tesla V100 (sm70). The main workload is Qwen3.8-27B with a large reusable coding/agent context. The branch also includes Volta tuning for Ornith-1.5-35B-A3B, MTP batching, and long-session prompt-cache changes.

> [!WARNING]
> `GGML_CUDA_FORCE_MMQ=ON` is for routed quantized MoE workloads such as Ornith. Keep it off for dense Qwen3.8. In the matched 100k test below, forcing MMQ on Qwen reduced prompt-processing speed from 452.8 to 323.2 tok/s (-28.6%).

## Recommended configurations

| model | recommended setup | why |
|---|---|---|
| **Qwen3.8-27B** | normal CUDA build, **MMQ off**, **MTP on with `n-max=3`**; use the Qwen quick-start settings below | Recommended long-context Qwen setup used by this fork. |
| **Ornith-1.5-35B-A3B** | build with `GGML_CUDA_FORCE_MMQ=ON` | FORCE_MMQ provides most of the V100 speedup for this routed MoE model. |

If you run both model classes, keep separate normal and FORCE_MMQ binaries.

## Key long-context results

`PP` is prompt-processing/prefill throughput. `TG` is token-generation/decode throughput. All values are tokens/s.

| workload | speculative decoding | baseline | `v100-optimized` | speedup |
|---|---|---:|---:|---:|
| **Qwen3.8-27B**, 100k + 1k prompt + 256 generated, V100 + 3060 Ti | **MTP on, `n-max=3` on both sides** | upstream: 294.11 PP / 26.64 TG | **424.07 PP / 33.55 TG** | **+44.19% PP / +25.93% TG** |
| **Ornith-1.5 AD-Q6_K**, 100k + 1k + 64 generated, V100 + 3060 Ti | **MTP3 and FORCE_MMQ on both sides** | upstream: 696.94 PP / 69.01 TG | **887.80 PP / 69.08 TG** | **+27.39% PP / +0.10% TG** |

The Qwen row is a direct upstream-to-fork comparison. MTP is enabled on both sides, so the TG gain is **not** an MTP-on versus MTP-off comparison. Both Qwen arms use the same model, q8_0 target KV, FP16 draft KV, 262k context, `64,2` layer split, and `n-max=3`. Generated tokens were identical in the matched A/B/B/A run.

See [Benchmarks](#benchmarks) for the exact baselines and component-level measurements.

## Quick start: Qwen3.8 on one V100

### Build

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

### Download Qwen3.8-27B

Tested Qwen quantization:

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

### Start the server

If the V100 is the only visible NVIDIA GPU:

```bash
export CUDA_VISIBLE_DEVICES=0

# Qwen3.8-27B V100 token-generation optimizations.
export GGML_CUDA_VOLTA_Q8_FATTN_TC=1
export GGML_CUDA_VOLTA_Q5_X4=1
export GGML_CUDA_VOLTA_Q6_W4R4=1
export GGML_CUDA_QWEN35_MTP_SHORTLIST="$PWD/data/mtp-shortlists/qwen38-27b-exact-131072.i32"

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
  --spec-draft-n-max 3 \
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

The four environment variables are opt-in and narrowly gated:

- `GGML_CUDA_VOLTA_Q8_FATTN_TC=1` only selects the validated sm70 Qwen target-verification geometry (`D=256`, 24 Q heads, 4 KV heads, `T=4`, q8_0 K/V, masked attention, no logit softcap).
- `GGML_CUDA_VOLTA_Q5_X4=1` enables exact-arithmetic Q5_K weight-decode reuse for non-`MUL_MAT_ID` `T=4` MMVQ.
- `GGML_CUDA_VOLTA_Q6_W4R4=1` enables the validated Q6_K `T=4` Volta launch geometry.
- `GGML_CUDA_QWEN35_MTP_SHORTLIST=...` affects only the MTP **proposal** head with the exact Qwen3.5/3.8 27B `5120 x 248320` Q6_K geometry. Invalid maps or other geometries fall back to the full head. The included map SHA-256 is `d2f2c068e3a637224b44741d40c914c987111807450d4d6f3c7ffd04b7f0f8cd`.

Use `--spec-draft-n-max 3` with this stack: three draft tokens produce the target verification width `T=4` that the optimized q8 attention and T4 weight kernels are designed for.

`--gpu-layers 63` is the tested 262k MTP-safe placement on the 32 GB V100. VRAM headroom can change with CUDA, quantization, or allocator changes, so check it on other builds.

`--cache-ram 65536` allows up to 64 GiB of host prompt-cache data. Lower it or omit it on systems with less RAM. It improves long-session reuse but is unrelated to the CUDA kernel speedup.

Check the server:

```bash
curl http://127.0.0.1:8080/health
```

The OpenAI-compatible endpoint is `http://127.0.0.1:8080/v1`.

### V100 + RTX 3060 Ti

For a 32 GB V100 plus an 8 GB RTX 3060 Ti, build both architectures:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES='70;86' \
  -DLLAMA_BUILD_UI=OFF

cmake --build build -j --target llama-server llama-cli
```

The `64,2` split expects the V100 to be CUDA0 and the RTX 3060 Ti to be CUDA1. Check CUDA's device order first:

```bash
./build/bin/llama-server --list-devices
```

If it already reports the V100 as `CUDA0`, leave `CUDA_VISIBLE_DEVICES` unset. Otherwise reorder the CUDA devices so the V100 becomes CUDA0 (for example, use `CUDA_VISIBLE_DEVICES=1,0` only when the V100 is currently CUDA1). Do not copy `nvidia-smi` indices blindly; CUDA ordinals can differ.

Then enable the tested Qwen paths:

```bash
export GGML_CUDA_VOLTA_Q8_FATTN_TC=1
export GGML_CUDA_VOLTA_Q5_X4=1
export GGML_CUDA_VOLTA_Q6_W4R4=1
export GGML_CUDA_QWEN35_MTP_SHORTLIST="$PWD/data/mtp-shortlists/qwen38-27b-exact-131072.i32"

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
  --spec-draft-n-max 3 \
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

## Benchmarks

The benchmark groups use different baselines because they isolate different changes:

| benchmark | baseline | optimized side |
|---|---|---|
| Qwen upstream vs fork | upstream parent `6d1479c14`, MTP `n-max=3` | `v100-optimized`, same MTP3 setup + recommended Qwen generation paths |
| isolated Qwen generation paths | same fork, paths off, MTP `n-max=3` | q8 W4 attention + 131k MTP shortlist + Q5x4 + Q6 w4r4 |
| Qwen / Ornith isolated FA | matching upstream build | isolated sm70 256x256 FA config (PR #27997) |
| Ornith upstream vs fork | upstream parent `6d1479c14`, `GGML_CUDA_FORCE_MMQ=ON`, MTP `n-max=3` | `v100-optimized`, same FORCE_MMQ + MTP3 setup |

Do not compare absolute values across benchmark groups unless the hardware, power limit, model quantization, GPU placement, and build settings match.

### Qwen upstream vs fork (MTP3)

This is the Qwen comparison used in the headline table. Both sides use MTP speculative decoding with `n-max=3`; the comparison is upstream llama.cpp versus the recommended fork paths, not MTP off versus on.

```text
Qwen3.8-27B UD-Q5_K_XL
100,000-token restored state + 1,000 new prompt tokens + 256 generated tokens
262,144 context, q8_0 target KV, FP16 draft KV
V100 + RTX 3060 Ti, layer split 64,2
MTP n-max=3 on both sides
```

| build | PP tok/s | TG tok/s | MTP accepted / drafted |
|---|---:|---:|---:|
| upstream `6d1479c14` | 294.11 | 26.64 | 167 / 261 |
| `v100-optimized` | **424.07** | **33.55** | 170 / 252 |
| speedup | **+44.19%** | **+25.93%** | - |

The figures are means from a warmed A/B/B/A run. Each measured arm followed a 256-token warm-up so first-use CUDA JIT/capture did not enter the result. All four measured arms produced identical generated tokens and content.

### Isolated Qwen token-generation paths

This section isolates the four newer generation paths **within the fork**. It is useful for attributing the TG gain, but it is not the upstream-versus-fork number shown above.

Four opt-in sm70 paths accelerate Qwen3.8 token generation when MTP uses `n-max=3` (target-verification width `T=4`):

- q8_0 KV tiles are widened to FP16 in shared memory and consumed by Volta tensor cores for the validated target-verification attention geometry;
- the MTP proposal head can evaluate a validated 131,072-row shortlist instead of all 248,320 vocabulary rows;
- Q5_K `T=4` MMVQ reuses each decoded weight fragment across four activation columns;
- Q6_K `T=4` uses the validated `4 warps x 4 rows/CTA` launch geometry.

The q8 attention topology is adapted from the small-T INT8 Volta work in [NInfer-V100](https://github.com/geoffwatts/ninfer-v100) while retaining llama.cpp's q8_0 KV format.

Single-V100 ABBA at `100k restored + 1k new + 256 generated`, q8_0 target KV, MTP `n-max=3`, MMQ off:

```text
PP  410.750 -> 413.418 tok/s    +0.65%
TG   27.591 ->  36.471 tok/s   +32.19%
```

All four full generated-token SHAs were identical. MTP acceptance was 167/261 with the paths off and 170/252 with them on. The q8 attention sub-kernel dropped from about 2.674 ms to 1.419 ms at ~101k KV and from 6.879 ms to 3.363 ms at ~260k KV.

The T=4 weight paths were also checked on Ornith with `GGML_CUDA_FORCE_MMQ=ON`, using the V100 for the target and RTX 3060 Ti for the MTP draft:

| Ornith quant / workload | paths off | paths on | change | correctness |
|---|---:|---:|---:|---|
| AD-Q6_K, 100k + 1k + 256 TG | 70.295 TG | 70.561 TG | +0.38% TG | identical 161/280 acceptance and token SHA |
| AD-Q5_K-Q4_K, 10k + 1k + 256 TG | 1444.256 PP / 116.868 TG | 1437.878 PP / 117.778 TG | -0.44% PP, +0.78% TG | identical 178/229 acceptance and token SHA |

On the tested Ornith configurations these paths are neutral to slightly positive. FORCE_MMQ provides the larger gain for routed MoE.

### MMQ at 100k

For Ornith, the upstream-to-fork comparison must use FORCE_MMQ on both sides. Otherwise most of the apparent gain comes from changing the build mode rather than from this fork.

Matched warmed A/B/B/A, `100k cached + 1k new + 64 generated`, AD-Q6_K target on V100, fixed MTP3 draft on RTX 3060 Ti, q8_0 target/draft KV, 131,072-token allocation:

| build | PP tok/s | TG tok/s | MTP accepted / drafted |
|---|---:|---:|---:|
| upstream `6d1479c14` + FORCE_MMQ | 696.94 | 69.01 | 39 / 70 |
| `v100-optimized` + FORCE_MMQ | **887.80** | **69.08** | 39 / 70 |
| speedup | **+27.39%** | +0.10% | - |

Both measured arms were compiled with `GGML_CUDA_FORCE_MMQ=ON`. All four generated-token SHAs and response content were identical. This is the fair Ornith branch comparison used in the headline table.

A second warmed A/B/B/A with 512 generated tokens confirmed the same PP result: 689.86 -> 881.44 PP tok/s (+27.77%), while TG stayed flat at 72.49 -> 72.58 tok/s (+0.13%). MTP acceptance was 327 / 551 in every arm and all generated-token SHAs matched.

Qwen is different. In an earlier matched within-fork 100k test, forcing MMQ reduced PP from 452.8 to 323.2 tok/s (-28.6%) while TG stayed effectively flat (26.66 vs 26.54 tok/s).

For a machine that runs both models, build two binaries: a normal build for dense Qwen3.8 and a `GGML_CUDA_FORCE_MMQ=ON` build for routed MoE such as Ornith.

### Isolated 256x256 FlashAttention benchmark

[PR #27997](https://github.com/ggml-org/llama.cpp/pull/27997) isolates the sm70 FlashAttention configuration for `DKQ=256`, `DV=256`, `ncols=64`.

#### 100k cached + 1024 PP + 512 TG

Single V100, 200 W, q8_0 K/V, no forced MMQ. A 100,000-token state was serialized and restored for each measured arm; PP and TG were timed separately.

| model | upstream PP | edited PP | PP change | upstream TG | edited TG | TG change |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 276.063 | **352.926** | **+27.84%** | 16.269 | 16.366 | +0.60% |
| Ornith-1.5-35B-A3B | 485.493 | **580.357** | **+19.54%** | 59.262 | 59.386 | +0.21% |
| Gemma 4 12B | 483.430 | 486.664 | +0.67% | 39.445 | 39.457 | +0.03% |

The change is geometry-specific, not model-name-specific. Qwen3.8 and Ornith both use the relevant 256-wide attention geometry.

#### Qwen3.8 scaling with KV depth

Single V100, 200 W, q8_0 K/V, 1024 new prompt tokens:

| prefilled KV depth | upstream | edited | change |
|---:|---:|---:|---:|
| 4,096 | 736.718 tok/s | 754.633 tok/s | +2.43% |
| 16,384 | 609.015 tok/s | 657.612 tok/s | **+7.98%** |
| 65,536 | 356.232 tok/s | 434.180 tok/s | **+21.88%** |
| 100,000 | 275.180 tok/s | 349.180 tok/s | **+26.89%** |

Short `pp512` benchmarks understate the gain at agent-scale KV depth.

#### Independent 2x-V100 validation

An independent DGX-1 test of PR #27997 used **2x Tesla V100-SXM2 32 GB**, Qwen3.8-27B `Q8_K_XL`, `llama-server`, tensor split, and a different benchmark path:

| prompt depth | upstream | + PR #27997 | change |
|---:|---:|---:|---:|
| 71,713 | 955.9 tok/s | 1067.1 tok/s | **+11.6%** |
| 122,869 | 761.7 tok/s | 892.7 tok/s | **+17.2%** |

Decode remained effectively unchanged. See the [independent result](https://github.com/ggml-org/llama.cpp/pull/27997#issuecomment-5478564345).

### Ornith 1.5 benchmarks

Ornith-1.5-35B-A3B uses the same 256x256 FlashAttention family and `S_v=128` scalar GatedDeltaNet, with a different GQA layout and routed-MoE feed-forward path.

PR-sized Ornith tests use:

```text
Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf
```

#### Isolated Volta 256x256 config

Standard short-context `llama-bench`, single V100:

| test | upstream | edited | change |
|---|---:|---:|---:|
| pp512 | 865.224 | 890.623 | +2.94% |
| pp2048 | 840.202 | 871.888 | **+3.77%** |
| pp4096 | 821.086 | 837.496 | +2.00% |
| tg128 | 104.578 | 104.646 | +0.06% |

At **100k KV + 1024 PP + 512 TG**, the same isolated FA config gives:

```text
PP  485.493 -> 580.357 tok/s   +19.54%
TG   59.262 ->  59.386 tok/s    +0.21%
```

#### Ornith with `GGML_CUDA_FORCE_MMQ=ON`

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

For Ornith on V100, use this branch with a FORCE_MMQ build.

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

#### Volta GatedDeltaNet x4

The sm70 scalar-gate `S_v=128` GatedDeltaNet prefill path processes four independent state/output columns per warp while sharing Q/K/gate loads.

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

#### Ornith exclusion from adaptive 2-CTA

The Qwen-oriented adaptive 2-CTA path is restricted to the exact Volta `256x256, ncols1=32, ncols2=2` layout. Ornith's `8x8` legacy path stays on the existing path. The extracted Ornith-style kernel SASS was byte-identical between the PR02 baseline and strict PR03 build.

The strict gate leaves non-target layouts such as Ornith's unchanged.

## Implementation

CUDA paths are selected by hardware and tensor geometry, not by model name.

### FlashAttention barrier correctness

The upstream FA combine path had complementary warp branches reaching different static `__syncthreads()` sites. On V100, Compute Sanitizer `synccheck` reported **1024 divergent-barrier errors** for the reproducer.

The branch moves synchronization to one uniform block-wide barrier. The fix is [llama.cpp PR #27955](https://github.com/ggml-org/llama.cpp/pull/27955).

Validation:

```text
V100 synccheck   0 errors
V100 racecheck  0 hazards / errors / warnings
V100 memcheck   0 errors
```

Temperature-normalized whole-model testing measured it as performance-neutral.

### Volta 256x256 FlashAttention config

Qwen3.8's full-attention layers use 256-wide Q/K/V heads. The old sm70 path fell through to an Ampere-oriented config.

The Volta config changes the target kernel from roughly:

```text
upstream fallback   255 registers/thread, 552 B stack
Volta config        255 registers/thread, ~48-56 B stack
```

It stages Q in shared memory and reduces V scratch. [PR #27997](https://github.com/ggml-org/llama.cpp/pull/27997) isolates this change.

Full existing 256x256 FA correctness matrix:

```text
V100        132/132 passed
RTX 3060 Ti 132/132 passed
```

### Adaptive Volta 2-CTA specialization

The next Qwen-specific step creates a separate compact kernel only for exact sm70 `256x256, 32x2` and uses:

```text
K scratch        96 half2
V scratch        64 half2
combine scratch  64 half2
shared memory    49,152 B/block
```

That allows two 128-thread blocks to fit in the V100's 96 KiB shared-memory budget. The launch is enabled only when its whole-tile wave-efficiency guard says two CTAs are useful; otherwise the normal 256x256 kernel remains the fallback.

A 100k-KV + 1k-PP run measured **+13.11%** over the PR02 baseline. The gate excludes Ornith's 8x8 layout.

### Volta scalar GatedDeltaNet x4

For exact sm70, scalar gate, `S_v=128`, prefill-only GDN, four independent state columns are processed per warp while reusing Q/K/gate inputs. Decode and non-target architectures remain on the normal path.

### Other Volta attention geometries

The branch also contains guarded paths for other Volta FA shapes:

- `128x128`: Q-shared/K96/V32 config for high-GQA layouts;
- `192x128`: layout-aware FA batch tuning for the `4x16` GQA16 layout while keeping `8x8` unchanged.

These are independent of the core Qwen3.8 `256x256` optimization.

### Quantized-weight reuse during prefill

`--prefill-reuse 1024` allows a larger physical prompt batch to reuse a converted quantized weight while keeping the smaller cuBLAS GEMM tile. It targets the Volta Q5_K/Q6_K -> F16 cuBLAS path.

This is an experimental fork-level tuning knob. Its incremental Qwen gain is small after the FA changes; do not assume it helps other architectures.

### Smaller pipeline scheduler allocation

`--pipeline-copies 2` reduces cross-backend scheduler input copies. Its main purpose is reducing compute-buffer VRAM so larger long-context graphs fit.

### MTP draft ubatch

`--spec-draft-ubatch 1024` gives the MTP/draft context a smaller physical ubatch than the target context, reducing transient draft-graph VRAM use.

### Prompt cache and recurrent checkpoints

Server changes for coding-agent traffic:

- prompt-cache states are chosen by the deepest usable absolute prefix rather than ratio alone;
- recurrent checkpoints preserve likely replay boundaries;
- duplicate exact checkpoint prefixes are refreshed rather than stored twice;
- observed replay hits influence checkpoint retention;
- plain-attention draft KV is not duplicated into recurrent checkpoints when it can be suffix-trimmed normally.

A regression test covers the case where a short live prompt would otherwise beat a much deeper cached prefix.

## Validation

Validation includes backend correctness tests and CUDA sanitizers:

```text
256x256 FA matrix        V100 132/132, RTX 3060 Ti 132/132
focused GDN cases        V100 40/40,   RTX 3060 Ti 40/40
V100 FA synccheck        0 errors
V100 FA racecheck        0 hazards / errors / warnings
V100 FA memcheck         0 errors
```

The Qwen candidate stack reproduced exact token/output trajectories in the long-context ABBA comparisons.

## Upstream PRs

Upstream-facing changes are split into independent pieces.

Submitted upstream:

- [#27955: CUDA: fix divergent FlashAttention barrier](https://github.com/ggml-org/llama.cpp/pull/27955)
- [#27997: add sm70 FlashAttention config for DKQ=256, DV=256, ncols=64](https://github.com/ggml-org/llama.cpp/pull/27997)

Separate upstream candidates include adaptive 2-CTA, Volta GDN x4, 128x128/192x128 tuning, and server-cache policy.

## Benchmark notes

Do not compare absolute numbers from different tables unless their hardware/power/build settings match. In particular:

- some early short-context model sweeps were captured at the V100's 300 W limit;
- controlled KV-depth and agent-turn results use an enforced **200 W** V100 limit;
- FORCE_MMQ materially changes Ornith's compute balance, so its absolute PP numbers should be compared only against the matched FORCE_MMQ baseline;
- the `qwen38-lossless-agent-cache` README's ~41% headline came from a different branch snapshot; use the tables in this README for this branch.

For general llama.cpp usage, APIs and platform documentation, use upstream [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), [`docs/`](docs/), and [`tools/server/README.md`](tools/server/README.md).

# llama.cpp Qwen3.8 Volta fork

A performance fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) for Qwen3.8-27B on NVIDIA Volta, especially the Tesla V100.

The main win is prompt processing at long context. On the 100k-cached + 1k-new-token workload used during development, the fork is about 41% faster than the matching upstream build on a single V100. Token generation is essentially unchanged.

| setup | fork PP | upstream PP | PP gain | fork TG | upstream TG |
|---|---:|---:|---:|---:|---:|
| V100 32 GB | **433.1 tok/s** | 306.7 tok/s | **+41.2%** | 23.19 tok/s | 23.29 tok/s |
| V100 32 GB + RTX 3060 Ti 8 GB | **450.0 tok/s** | 317.7 tok/s | **+41.6%** | 26.65 tok/s | 26.48 tok/s |

These numbers use Qwen3.8-27B `UD-Q5_K_XL`, q8_0 target KV, MTP, a real 100,000-token cached state, 1,000 new prompt tokens, and 64 generated tokens. A later check against vanilla upstream `bb4caa754` measured 313.98 -> 447.11 tok/s on the dual-GPU setup (+42.40%).

If you only have one V100, start there. The RTX 3060 Ti is optional.

## Quick start: single V100

### 1. Build the fork

You need a C++ compiler, CMake, the CUDA toolkit, and a V100 with enough free VRAM.

```bash
git clone --branch qwen38-lossless-agent-cache --single-branch \
  https://github.com/mistrjirka/llama.cpp.git
cd llama.cpp

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DLLAMA_BUILD_UI=OFF

cmake --build build -j --target llama-server llama-cli
```

`LLAMA_BUILD_UI=OFF` keeps the build smaller. Omit it if you want the built-in web UI.

For general build problems, use the upstream [`docs/build.md`](docs/build.md).

### 2. Get the Qwen3.8 model

The benchmarked model is:

```text
unsloth/Qwen3.8-27B-GGUF
Qwen3.8-27B-UD-Q5_K_XL.gguf
```

With the Hugging Face `hf` CLI:

```bash
mkdir -p models/Qwen3.8-27B
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --local-dir models/Qwen3.8-27B

export MODEL="$PWD/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf"
```

You can use another model path, but the settings and benchmark numbers below are for this quantization.

### 3. Start the optimized server

If the V100 is the only visible NVIDIA GPU, use:

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

On the development V100, `--gpu-layers 63` was the tightest tested MTP-safe placement at 262k context. Layer 64 did not leave enough room for the transient MTP graph.

`--cache-ram 65536` allows up to 64 GiB of host-side prompt cache. Lower it or omit it on machines with less RAM. It is useful for long agent sessions but is not responsible for the CUDA kernel speedup.

Check the server:

```bash
curl http://127.0.0.1:8080/health
```

The OpenAI-compatible endpoint is:

```text
http://127.0.0.1:8080/v1
```

### 4. Benchmark the optimized single-V100 setup

For a quick benchmark, put a representative long prompt in `prompt.txt` and run the same performance settings through `llama-cli`:

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
  --model "$MODEL" \
  --ctx-size 262144 \
  --fit off \
  --gpu-layers 63 \
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
  --file prompt.txt \
  --predict 64 \
  --seed 1234 \
  --no-display-prompt \
  --perf
```

The final timing lines report prompt processing and token generation speed.

This is the easiest way to test your own workload. It does not reproduce the exact 100k cached-continuation test because that benchmark restores a saved recurrent/KV state. The exact benchmark scripts and cache preparation commands are in [`benches/v100-qwen38/v100-qwen38.md`](benches/v100-qwen38/v100-qwen38.md).

## Optional: V100 + RTX 3060 Ti

The second GPU helps token generation more than prompt processing. On the tested setup, adding the 3060 Ti changed Qwen3.8 PP from 433.1 to 450.0 tok/s and TG from 23.19 to 26.65 tok/s.

Build for both architectures:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES='70;86' \
  -DLLAMA_BUILD_UI=OFF

cmake --build build -j --target llama-server llama-cli
```

Put the V100 first in CUDA's visible-device order. For example, if `nvidia-smi -L` shows the V100 as GPU 1 and the 3060 Ti as GPU 0:

```bash
export CUDA_VISIBLE_DEVICES=1,0
```

Then run:

```bash
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

The `64,2` split is for a 32 GB V100 plus an 8 GB 3060 Ti with the V100 presented as the first CUDA device. Retune it for different GPUs or VRAM sizes.

## What this fork changes

The fork combines several independent changes. The CUDA optimizations are guarded by hardware and tensor shape; they are not selected by model name.

### Volta 256x256 FlashAttention

Qwen3.8 uses 256-wide full-attention heads. The upstream Volta path falls back to an Ampere-oriented configuration that puts heavy pressure on V100 registers.

This fork adds an sm70-specific configuration that:

- stages Q in shared memory instead of keeping it in registers;
- uses smaller K/V scratch tiles;
- preserves the original MMA accumulation order;
- uses a smaller shared-memory combine window;
- enables the two-CTA launch only for the query sizes where the logical schedule stays unchanged.

On the isolated Qwen3.8 attention geometry, the tuned kernel was about 1.46-1.48x faster across 4k-24k KV lengths.

### Volta GatedDeltaNet prefill

Qwen3.8's recurrent layers use scalar-gate GatedDeltaNet with a 128-wide state. The sm70 kernel processes four independent state columns per warp so they can share Q/K/gate loads.

The isolated 32-head, 128-wide, 1024-token test dropped from about 1.797 ms to 1.175 ms.

### Quantized-weight reuse during prefill

`--prefill-reuse 1024` lets a larger physical prompt batch reuse a converted quantized weight while keeping the smaller cuBLAS GEMM width. This avoids repeated Q5_K/Q6_K -> F16 conversion on the Volta cuBLAS path.

This optimization is opt-in because changing the physical prompt batching policy is model-sensitive. Qwen3.8 passed the strict lossless checks used here; other models should be checked separately.

### Smaller pipeline scheduler allocation

`--pipeline-copies 2` reduces cross-backend scheduler copies. That frees enough VRAM for the larger Qwen3.8 prompt graph on the tested setup.

### Separate MTP ubatch

`--spec-draft-ubatch 1024` lets the MTP context use a smaller physical batch than the target context. This matters on the dual-GPU configuration, where the target runs at `--ubatch-size 4096`.

### Agent prompt-cache changes

The server also changes long-session cache behavior:

- cached states are ranked by reusable prefix length instead of requiring two similarity ratios to improve at once;
- recurrent checkpoints retain likely replay boundaries;
- exact checkpoint prefixes are refreshed instead of duplicated;
- observed replay hits influence checkpoint retention;
- plain attention draft KV is not duplicated when it can be suffix-trimmed normally.

These changes target coding-agent traffic where a long conversation can be interrupted by a small unrelated request and then resumed.

## Benchmark results

### Qwen3.8 against upstream

Workload: real 100,000-token cached state, 1,000 new prompt tokens, then 64 generated tokens.

| hardware | fork PP | upstream PP | PP change | fork TG | upstream TG |
|---|---:|---:|---:|---:|---:|
| V100 only | **433.1** | 306.7 | **+41.2%** | 23.19 | 23.29 |
| V100 + 3060 Ti | **450.0** | 317.7 | **+41.6%** | 26.65 | 26.48 |

The fresh comparison against vanilla upstream `bb4caa754` on the dual-GPU setup measured:

```text
upstream: 313.98 tok/s PP, 26.17 tok/s TG
fork:     447.11 tok/s PP, 26.20 tok/s TG
```

That run was +42.40% in prompt processing. Both builds generated the same 64 tokens and had the same MTP acceptance, 37/52.

### Other models

The speedup is not generic.

| model | hardware | fork PP | upstream PP | PP change |
|---|---|---:|---:|---:|
| Qwen3.5-122B-A10B | V100 + 3060 Ti | 283.5 | 257.4 | +10.2% |
| Qwen3.5-122B-A10B | V100 only | 280.8 | 241.4 | +16.3% |
| Laguna-S-2.1 | V100 + 3060 Ti | 317.4 | 319.5 | -0.6% |
| Laguna-S-2.1 | V100 only | 312.3 | 320.5 | -2.5% |

Qwen3.5 shares the relevant 128-wide GatedDeltaNet and 256-wide attention shapes, so some of the Volta kernel work carries over. Laguna does not and acts as a useful negative control.

GLM-5.2 was tested separately at 10k cached + 1k new tokens because a real 100k semantic prime was too slow. Fork PP stayed within about 0.5% of upstream. MTP itself was useful on GLM, improving TG by about 31-42% for roughly a 3% PP cost.

See [`benches/v100-qwen38/v100-qwen38.md`](benches/v100-qwen38/v100-qwen38.md) for the full matrix, placements, cache preparation, and raw benchmark workflow.

## Lossless checks

The promoted Qwen3.8 CUDA path was checked beyond sampled output equality.

Tests included:

- exact top-100 probability-object comparison;
- exact 64-step token and probability replay;
- exact 100k -> 101k saved-state continuation;
- raw q8_0 FlashAttention output comparison on the active Q=1000 geometry;
- CUDA synccheck, racecheck, and memcheck on the adaptive two-CTA path;
- V100 and RTX 3060 Ti backend correctness tests.

Broader two-CTA and larger softmax variants were rejected because they changed the numerical trajectory. They are not enabled in this branch.

The larger-batch weight-reuse policy should not be assumed lossless on unrelated model architectures. Qwen3.5 and Laguna showed measurable probability drift in some large-ubatch experiments even when throughput improved.

## PFlash proxy

[`tools/pflash/`](tools/pflash/) contains a separate optional PFlash proxy for cold long prompts. It removes some old prompt tokens before target inference and is therefore approximate, not lossless.

It is disabled unless you run the proxy yourself. It is not part of the performance numbers above.

See [`tools/pflash/README.md`](tools/pflash/README.md) if you want to experiment with it.

## Development and upstream PR work

This branch is the complete working fork. It intentionally contains changes that should be split before proposing anything to upstream llama.cpp.

Useful development notes:

- [`benches/v100-qwen38/HANDOFF.md`](benches/v100-qwen38/HANDOFF.md) - full project handoff and workspace history
- [`benches/v100-qwen38/pr-readiness.md`](benches/v100-qwen38/pr-readiness.md) - code-quality review and suggested PR split
- [`benches/v100-qwen38/v100-qwen38.md`](benches/v100-qwen38/v100-qwen38.md) - benchmark commands and results
- [`README.old`](README.old) - README from the original upstream base

The likely upstream candidates are separate changes rather than one large PR: server cache-state selection, basic Volta 256x256 FlashAttention tuning, Volta GatedDeltaNet prefill, adaptive FA occupancy, and MTP draft batching.

The complete fork currently sits on upstream base `0e1d9185c`. It was also tested against a separately built vanilla `bb4caa754`; the intervening upstream commits did not touch files modified by this fork at the time of that check.

For general llama.cpp usage and APIs, use the upstream documentation in [`docs/`](docs/) and [`tools/server/README.md`](tools/server/README.md).

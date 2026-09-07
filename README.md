# llama.cpp optimized for V100 & RTX 2080 Ti

[![GitHub stars](https://img.shields.io/github/stars/mistrjirka/llama.cpp?style=flat-square&logo=github)](https://github.com/mistrjirka/llama.cpp/stargazers)

A CUDA performance fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) for NVIDIA Volta (SM70) and Turing (SM75), tested on a Tesla V100-SXM2 32 GB and an RTX 2080 Ti 22 GB. The main target is long-context Qwen3.8-27B serving; Ornith-1.5-35B-A3B has additional routed-MoE tuning.

**Branch:** `v100-optimized` · **Validated multi-agent code:** `439f2245e3` · **Upstream merged through:** `465e49b9c`

## Performance vs vanilla llama.cpp

The headline vanilla-vs-fork tables below are the retained direct comparison from **5 September 2026**: fork snapshot `547593d21` versus upstream `6a1a922d2`. The branch has since been merged forward to upstream `465e49b9c`; the new code was regression-gated separately and is documented in [Post-upstream integration validation](#post-upstream-integration-validation). The same model and common runtime settings are used on both sides of each comparison, and mixed V100 + RTX results are not reported as single-GPU numbers.

`PP` is prompt-processing throughput and `TG` is token-generation throughput. Higher is better.

### Tesla V100 32 GB (SM70)

#### 100k cached Qwen3.8-27B — main workload

One V100, Qwen3.8-27B `UD-Q5_K_XL`, 100,000 restored tokens, q8_0 K/V, MTP off. Every request reused exactly 100,000 tokens and processed only the listed suffix. Six measurements per side were collected in A/B/B/A order; the generated control token matched exactly across all runs.

| appended prompt | vanilla llama.cpp PP | `v100-optimized` PP | speedup |
|---:|---:|---:|---:|
| +128 | 155.59 tok/s | **190.58 tok/s** | **+22.49%** |
| +256 | 211.10 tok/s | **290.56 tok/s** | **+37.64%** |
| +1,000 | 300.94 tok/s | **433.19 tok/s** | **+43.94%** |

At +1,000 tokens, prompt processing falls from **3.323 s to 2.308 s**, saving about **1.014 s**. The fork is primarily optimized for this long-KV workload.

#### Standard `llama-bench`

| test | vanilla llama.cpp | `v100-optimized` | change |
|---|---:|---:|---:|
| pp512 | 784.14 tok/s | **796.50 tok/s** | **+1.58%** |
| pp2048 | 780.25 tok/s | **800.43 tok/s** | **+2.59%** |
| tg128 | 27.89 tok/s | 27.72 tok/s | -0.62% (effectively unchanged) |

Short `pp512`/`pp2048` runs do not represent the long-context gain above.

### RTX 2080 Ti 22 GB (SM75)

#### Standard `llama-bench`

Single-GPU Qwen3.8-27B, q8_0 K/V, FlashAttention on, three repetitions per process in A/B/B/A order:

| test | vanilla llama.cpp | `v100-optimized` | change |
|---|---:|---:|---:|
| pp512 | 657.36 tok/s | **675.88 tok/s** | **+2.82%** |
| pp2048 | 658.78 tok/s | **678.37 tok/s** | **+2.97%** |
| tg128 | 24.70 tok/s | 24.67 tok/s | -0.12% (neutral) |

The long-context RTX investigation also contains isolated 101k-KV attention tests and V100 + RTX tensor-parallel tests. They are kept in [Detailed benchmarks and methodology](#detailed-benchmarks-and-methodology) because they are not RTX-only end-to-end measurements.

## Recommended configurations

| GPU / model | recommended setup |
|---|---|
| **V100 — Qwen3.8-27B** | normal CUDA build, **MMQ off**, MTP `n-max=3`; add `--spec-mtp-defer-prompt` for lower agent-turn TTFT |
| **V100 — Ornith-1.5-35B-A3B** | separate build with `GGML_CUDA_FORCE_MMQ=ON` |
| **RTX 2080 Ti — Qwen3.8-27B** | SM75 build; leave `GGML_CUDA_VOLTA_*` unset; Turing paths are selected automatically |
| **V100 + RTX 2080 Ti — Qwen3.8 Flash-Next** | q8_0 K/V, `n-cpu-moe=18`, layer split **35:14**; enable the QSA PP raw-q8 stack documented below with tile 16 |

Build for `70`, `75`, or `70;75` for a mixed V100 + RTX 2080 Ti system. If you serve both Qwen and Ornith on V100, keep separate normal and FORCE_MMQ binaries.

> [!WARNING]
> `GGML_CUDA_FORCE_MMQ=ON` is intended for routed quantized MoE workloads such as Ornith. Do not enable it globally for dense Qwen3.8; it is a known prompt-processing regression there.

## Multi-agent KV sharing and parked sessions

For the measured parallel PP/TG investigation, the projector/text-prefix guard correction, and remaining optimization work, see [parallel serving research](benches/parallel-serving-0907/RESEARCH.md). Its layout and MTP results are screening measurements, not new vanilla-vs-fork claims.

This fork also reduces the memory cost of parallel long-context agents. The server's exact-prefix fork is **enabled by default for unified-KV serving**: when a new text-only slot is an exact extension of an idle slot, the shared attention prefix is referenced by both sequence IDs instead of allocating a second K/V copy. The path is implemented through llama.cpp's generic sequence-memory interface rather than an Ornith-specific model hook.

For server mode, **multiple slots now default to unified KV even when `--parallel N` is supplied explicitly**, so this optimization is not tied to Ornith or to the auto-slot path. `--no-kv-unified` remains an explicit compatibility opt-out. Single-slot behavior is unchanged.

The automatic path remains conservative. It requires unified KV, an empty destination, an idle exact-prefix donor and compatible adapter state; multimodal prompts and LoRA requests fall back to normal slot scheduling. Use `--no-slot-fork-prefix` to disable it. It has been runtime-validated on both **Qwen3.8-27B** (dense FFN with hybrid attention, built-in MTP) and **Ornith-1.5-35B-A3B** (hybrid recurrent/attention, external Shisa MTP). Qwen's 8k-parent + 2k-child test reused all 8,000 parent tokens and produced the same deterministic output SHA and 42/61 MTP acceptance as a clean full-prefill control.

For `N` histories of lengths `L_i` with one common prefix of length `P`, the attention-KV occupancy is approximately:

```text
physical KV tokens = sum(L_i) - (N - 1) * P
```

### Measured four-slot Ornith VRAM

Real `AD-Q6_K-Q5_K`, V100 32 GB + RTX 2080 Ti 22 GB, four logical 400k slots, q8_0 target **and** draft KV, Shisa Q5_0 MTP3, CPU vision projector, identical model placement. Only the physical unified-KV pool changes:

| physical KV pool | RTX 2080 Ti used | V100 used | total GPU memory used |
|---:|---:|---:|---:|
| 1.2M tokens | 19,230 MiB | 30,045 MiB | **49,275 MiB** |
| 1.3M tokens | 20,110 MiB | 31,015 MiB | **51,125 MiB** |
| **1.4M tokens** | **20,990 MiB** | **31,985 MiB** | **52,975 MiB** |

The measured slope is almost exactly **1,850 MiB per 100k physical q8 KV tokens**. Four independent 400k caches would require a 1.6M-token pool; relative to that reservation, the validated 1.4M shared pool avoids about **3,700 MiB (~3.6 GiB) of GPU memory**. Extrapolating the measured slope puts a 1.6M configuration at about 56,675 MiB, beyond this 54 GB GPU pair. The 1.4M pool therefore turns an otherwise non-fitting 4x400k layout into a working one whenever the four full histories share at least **66,667 prefix tokens**. A 100k common parent would need only about 1.3M physical tokens; 1.4M is retained as the safer default headroom.

### More agent sessions than GPU slots

A GPU slot is now execution capacity rather than a permanent agent identity. The server's host prompt cache can park **multiple complete agent states** behind a smaller number of live slots. Parked entries include target state, draft/MTP state and speculative carry. On reuse, the deepest matching history is loaded into a free/replaced slot; if a compatible prefix is still live in another unified-KV slot, the parked target and draft states restore **directly onto that prefix** and allocate only the divergent tail.

This was forced in a Qwen test with an ~8.5k physical pool: a 5.5k child could not coexist as a full copy beside its 4k parent, but the parked child restored successfully by reusing the live 4k parent in both target and draft KV. The resumed output SHA and 47/47 MTP acceptance matched the always-live child control exactly. A separate one-slot A -> B -> C -> A test restored 5,000 cached tokens from RAM and matched the live-cache continuation exactly, including 42/63 MTP acceptance. Ornith's hybrid/recurrent A -> B -> C -> A test also matched exactly, including 43/57 MTP acceptance.

The parked RAM bank can be persisted with:

```text
POST /prompt-cache?action=save     {"filename":"prompt-cache.bin"}
POST /prompt-cache?action=restore  {"filename":"prompt-cache.bin"}
```

`--slot-save-path` supplies the directory, as with ordinary slot snapshots. In an end-to-end wrapper test, two parked Qwen agents occupied a 625.5 MB bank; after the server process was killed and recreated, the bank restored in **141 ms** before serving traffic, the live slot restored separately, and the parked-agent continuation matched the live-cache control SHA exactly. RAM-cache eviction remains size-bounded/oldest-first through `--cache-ram`.

For multi-agent unified-KV serving, prefer a host cache without aggressive idle clearing: keep useful live slots resident and let LRU replacement or KV pressure park states to RAM. This is the mode used by the companion `local-llm-setup` configuration.

## Quick start: Qwen3.8 on one V100 (SM70)

### Build

You need a C++ compiler, CMake, the CUDA toolkit, and a Volta GPU (SM70).

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
  --spec-mtp-defer-prompt \
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

## Quick start: Qwen3.8 on Turing (SM75)

For an RTX 2080 Ti or another SM75 GPU, build the Turing path:

```bash
cmake -S . -B build-sm75 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DLLAMA_BUILD_UI=OFF

cmake --build build-sm75 -j --target llama-server llama-cli
```

Do not set the `GGML_CUDA_VOLTA_*` variables in an SM75 run. The Turing paths are selected automatically from the GPU architecture and tensor geometry. Model placement depends on available VRAM; check `--list-devices` and tune `--gpu-layers`, `--split-mode`, and `--tensor-split` for the model and context length.

For a mixed V100 + RTX 2080 Ti system, compile with `-DCMAKE_CUDA_ARCHITECTURES='70;75'` and retune layer/tensor placement. Mixed-GPU benchmark notes are kept below; do not infer single-RTX performance from them.

## Feature-specific measurements

### Lower MTP refresh overhead

Single-layer Qwen/Ornith MTP heads now refresh K/V without running unused draft attention, feed-forward layers or vocabulary projection. This is **enabled by default for supported, non-shared heads**, together with synchronized sampling/hidden-state reads and zero/per-request draft-budget fixes. Target attention, Q8 KV precision, configured draft depth, shared prefixes and parked-session persistence are unchanged.

The integrated build versus the previous production revision measured **2.5% shorter four-agent MTP1 turns** (5.936 → 5.786 s) at 100k context on the V100 + 2080 Ti pair, with Q6/Q5 weights and Q8 target/draft KV unchanged. MTP-off remained effectively unchanged; four-agent MTP3's 1.2% mean reduction overlapped timing variation. These within-fork results are not additive, not a claim that MTP always beats ordinary decoding, and not the vanilla baseline used in the headline tables.

See [the production integration report and regression evidence](benches/mtp-production-merge-0907/REPORT.md) for the integrated build's tests and measurements. To diagnose regressions, set `LLAMA_MTP_KV_ONLY=0`, `LLAMA_MTP_BULK_HIDDEN=0`, or `LLAMA_SAMPLING_VIEW=0` before starting the engine. Existing launch configurations pick up the defaults after rebuilding; no lower-precision KV or changed MTP depth is required. Unresolved attention, compact-layout and grouping experiments remain excluded.


The tables below isolate individual fork options. They are useful for choosing settings, but they are **not** the vanilla-vs-fork headline comparison above.

### Post-upstream integration validation

The current patch stack is synced through upstream `465e49b9c` and regression-gated as code commit **`f3f0eef26`**. Hardware was the mixed **Tesla V100-SXM2 32 GB + RTX 2080 Ti 22 GB** system; logical `CUDA0` was the V100 and `CUDA1` the 2080 Ti.

For Qwen3.8 Flash-Next, the new raw-q8 tiled QSA prefill path uses a **35:14** layer split, 100,000 restored tokens, +1,000 prompt tokens, q8_0 K/V, `n-cpu-moe=18`, batch/ubatch `2048/1000`, and tile width 16. The 36:13 raw-q8 placement is intentionally **not** recommended: it can run out of V100 workspace memory during the prompt.

The benchmarked opt-in prefill stack is:

```bash
export GGML_CUDA_QSA_PP_REDUCE=1
export QWEN4EXP_QSA_PP_BLOCK_TOPK=1
export QWEN4EXP_QSA_PP_GATHER=1
export QWEN4EXP_QSA_PP_GATHER_Q8=1
export QWEN4EXP_QSA_PP_GATHER_TILE=16
```

Use it with `--tensor-split 35,14` for the tested V100 + 2080 Ti placement. Decode QSA gather remains enabled by default; the variables above are for the new prefill path.

| validation | control / old | post-sync `f3f0eef26` | result |
|---|---:|---:|---:|
| Flash-Next 100k + 1k PP, same 35:14 split | 244.71 PP/s | **266.70 PP/s** | **+8.98% PP** |
| Flash-Next final 128-token run | — | **265.43 PP/s / 23.17 TG/s** | exact 128-token SHA |
| Flash-Next final 512-token run | — | **266.98 PP/s / 22.88 TG/s** | exact 512-token SHA |
| Ornith Q5/Q4, pp1024 | 1830.74 PP/s | **1851.75 PP/s** | **+1.15%** |
| Ornith Q5/Q4, tg128 | 97.55 TG/s | **98.09 TG/s** | **+0.55%** |

Flash-Next generated output stayed exact. The retained 128-token hash is `fa4eabafcd8e2e384c8918d644386b7b9a66a1bb330c3420903e4244411f2046`; the 512-token hash is `0c52eae49d84c1068a2a69f4bbf9ca2f28c897c441a44afeea8d65cb46e0555b`. The raw-q8 PP comparison is an alternating same-split test, so its gain is not a layer-placement artifact.

Dense Qwen3.8 was treated as a hard regression gate. Its 8k PP result was order-sensitive: old-first A/B/A/B gave **-0.51%** for the merged build, while reversing the order to new/old/new/old gave **+0.33%** (`997.79` vs `994.46` PP/s). A separate 4096-token Nsight run had the post-sync build at **1019.67 vs 1014.92 PP/s (+0.47%)**, and the short tg128 comparison was effectively flat (~-0.03%). The sign flip with ordering plus the profile result is why this is classified as **no stable dense-Qwen regression**, rather than claiming a speedup.

A merge-resolution guard initially disabled Flash-Next's sparse text-decode path for IMRoPE batches and caused a large TG regression. That change was rejected and replaced by a token-input guard; after the fix, matched 128-token runs were **22.94/23.19 TG/s** on the merged build versus **18.44/20.54 TG/s** on neighboring old-build runs, with identical generated hashes. The final 512-token raw-q8 run above sustained **22.88 TG/s**.

### Faster first token for MTP agent turns on one V100

For a long coding/agent session, throughput alone can hide the delay the user actually feels. `--spec-mtp-defer-prompt` reduces **time to first streamed token (TTFT)** when a large prefix is already cached and a relatively small tool/user suffix is appended. It keeps the target prompt work unchanged, but moves MTP prompt catch-up and publication of the replay checkpoint off the first-token critical path.

Single Tesla V100-SXM2-32GB, Qwen3.8-27B `UD-Q5_K_XL`, 100,000 restored tokens, q8_0 target KV, FP16 draft KV, MTP `n-max=3`, one generated control token per request:

| appended prompt | normal MTP TTFT | `--spec-mtp-defer-prompt` | answer starts sooner |
|---:|---:|---:|---:|
| 64 | 455 ms | **445 ms** | 9 ms / 2.0% |
| 128 | 746 ms | **693 ms** | **53 ms / 7.1%** |
| **177** | 868 ms | **814 ms** | **54 ms / 6.3%** |
| 256 | 954 ms | **898 ms** | **56 ms / 5.9%** |
| 512 | 1,572 ms | **1,547 ms** | 26 ms / 1.6% |
| 1,000 | 2,400 ms | **2,356 ms** | 44 ms / 1.8% |

The 177-token point is representative of the median appended-prompt size in the agent trace used for this tuning. The final cleaned/public-option A/B/B/A run with **257 generated tokens** measured **864.9 -> 819.3 ms TTFT (-5.28%, 45.6 ms sooner)**. Generation throughput was 36.92 -> 36.78 tok/s (-0.38%), MTP acceptance stayed exactly **171 / 255**, and all four generated-token SHAs were identical.

The improvement is not limited to the first SSE event. In a separate matched 64-token streaming run, token 1 arrived 48 ms earlier, token 2 was still 32 ms earlier, and token 64 was 36 ms earlier; the generated token stream was identical.

Replay behavior is preserved. After a 100k + 177 request, an exact repeat still restored **100,173 cached tokens and recomputed only the final 4**. Extending the prompt by another 32 literal tokens restored **100,177** and processed only those 32 new tokens.

Enable the latency path together with the normal MTP3 setup:

```bash
--spec-type draft-mtp \
--spec-draft-n-max 3 \
--spec-mtp-defer-prompt \
--ctx-checkpoints 32 \
--checkpoint-min-step 8192
```

This path is opt-in and currently intended for **draft-MTP agent serving on one V100**. It has not been validated as a mixed-GPU TTFT configuration.

### Agent prompt checkpointing on one V100

For **non-speculative** Qwen3.8 agent serving, `--checkpoint-recurrent-prev` removes a fixed checkpoint-tail cost that is disproportionately expensive when a large cached context receives a small new tool/user suffix. It keeps one recurrent rollback plane during multi-token prompt processing and stores the immediately preceding recurrent state as a device-backed checkpoint. Suffixes of 64 tokens or fewer stay on the existing checkpoint path because that split is faster at the 64-token boundary.

Single Tesla V100-SXM2-32GB, 100,000-token restored state, q8_0 K/V, MTP off:

| appended prompt | normal checkpoints | `--checkpoint-recurrent-prev` | change |
|---:|---:|---:|---:|
| 64 | 157.29 PP/s | 156.70 PP/s | -0.38% (neutral) |
| 128 | 185.76 PP/s | **211.92 PP/s** | **+14.08%** |
| **177** | 226.57 PP/s | **252.62 PP/s** | **+11.50%** |
| 256 | 292.83 PP/s | **323.97 PP/s** | **+10.63%** |
| 512 | 341.38 PP/s | **360.37 PP/s** | **+5.56%** |
| 1,000 | 435.04 PP/s | **449.53 PP/s** | **+3.33%** |

The 177-token row is included because it is representative of the median appended-prompt size in the agent trace used during this tuning work. In a five-branch replay test, the optimized path restored 100,127 tokens, recomputed one prompt token, and produced the same 16 generated tokens as full recomputation for every branch. Separate 64-TG and 512-TG A/B/B/A runs produced identical generated-token SHAs and no TG regression.

Enable it with:

```bash
--ctx-checkpoints 32 \
--checkpoint-min-step 8192 \
--checkpoint-recurrent-prev
```

`--checkpoint-recurrent-prev` remains the **non-speculative** optimization. When draft-MTP is enabled, use the separate `--spec-mtp-defer-prompt` path above instead.

## Detailed benchmarks and methodology

The headline tables above are the retained **5 September vanilla-vs-fork snapshot**: upstream `6a1a922d2`; fork CUDA/runtime code `547593d21`; CUDA 12.9. The current branch is newer (`f3f0eef26`, upstream merged through `465e49b9c`), so use the post-upstream validation section for the current integration gate. The relevant methodology for the historical headline is summarized below.

### Current-upstream V100 100k methodology

The V100 headline restores the same saved 100,000-token state before every request, sends the full original token sequence through `100000 + suffix`, and asserts `cache_n=100000` plus `prompt_n=suffix`. Both binaries use the common options only: one V100, 131,072 context, q8_0 K/V, FlashAttention on, batch/ubatch 2048/1024, MTP off, and 32 context checkpoints. Fork-only `--prefill-reuse` and `--checkpoint-recurrent-prev` are disabled for this comparison.

### Current-upstream `llama-bench` methodology

The V100 and RTX short-context tables use the same command on each binary, with only the selected GPU and executable/library path changed:

```bash
llama-bench \
  -m Qwen3.8-27B-UD-Q5_K_XL.gguf \
  -p 512,2048 -n 128 -r 3 \
  -fa on -ctk q8_0 -ctv q8_0 \
  -ngl 99 -sm none -o json
```

Each GPU was run upstream/fork/fork/upstream. The table values are the mean of the two process-level `avg_ts` values per side; each process-level value already contains three benchmark repetitions.

### RTX 2080 Ti long-context operator screen

This lower-level screen predates the fresh current-upstream table and therefore uses the fork parent `6d1479c14` as its vanilla control. It is retained because it shows where the SM75 work helps and where it does not. One RTX 2080 Ti, Qwen D256/GQA6 attention, q8_0 K/V, 101,120 KV tokens:

| query width | vanilla latency | fork latency | change |
|---:|---:|---:|---:|
| 128 | 17.087 ms | **13.125 ms** | **-23.19%** |
| 177 | **23.658 ms** | 33.598 ms | **+42.02% regression** |
| 256 | 31.993 ms | **23.889 ms** | **-25.33%** |
| 512 | 63.230 ms | **47.591 ms** | **-24.73%** |
| 1,000 | 119.678 ms | **91.047 ms** | **-23.92%** |

The Q=177 regression is real in this isolated full-head q8_0 shape, so this table is not presented as a universal SM75 speedup. The current-upstream full-model `llama-bench` table above is the user-facing RTX comparison.

### Mixed V100 + RTX 2080 Ti — incremental integration results

These rows use **both GPUs** and are intentionally excluded from the single-GPU headline tables. They compare successive accepted fork builds, not vanilla llama.cpp. Workload: restored 100k prefix, FP16 target KV, MTP3, 257 generated tokens.

| model / append | previous fork build | optimized fork build | change | TTFT |
|---|---:|---:|---:|---:|
| Qwen3.8 +1,000 | 631.13 PP/s | **645.29 PP/s** | **+2.24%** | 1636.93 → **1608.94 ms** |
| Ornith-1.5 +1,000 | 1512.76 PP/s | **1548.71 PP/s** | **+2.38%** | 718.49 → **699.54 ms** |
| Ornith-1.5 +1,000, later V100 Q6 tile | 1557.30 PP/s | **1594.47 PP/s** | **+2.39%** | 698.30 → **679.73 ms** |

Generation was effectively unchanged in these comparisons. Output token hashes matched; the later V100 Q6 integration also matched speculative acceptance-count sets.

### Other benchmark groups

The older/component-isolation groups below use different baselines because they answer different questions. Do not use them as the current upstream headline comparison:

| benchmark | baseline | optimized side |
|---|---|---|
| isolated Qwen generation paths | same fork, paths off, MTP `n-max=3` | q8 W4 attention + 131k MTP shortlist + Q5x4 + Q6 w4r4 |
| **MTP agent TTFT** | same fork, normal MTP3 prompt path | `--spec-mtp-defer-prompt`, same target segmentation and MTP3 setup |
| Qwen / Ornith isolated FA | matching upstream build | isolated sm70 256x256 FA config (PR #27997) |

Do not compare absolute values across benchmark groups unless the hardware, power limit, model quantization, GPU placement, and build settings match.

### NInfer vs this fork: exact-token 100k cached append

This is the closest cross-runtime comparison used during the NInfer investigation. Both runtimes use **one Tesla V100-SXM2-32GB only**; any secondary GPU is hidden, speculative decoding is disabled, and each measured request reuses the literal same first 100,000 token IDs before appending the same continuation tokens. The cold 100k construction/prime is excluded on both sides so the table measures the steady agent-style cached append.

The llama.cpp side uses the `dfdcd8add` branch snapshot retained for this historical comparison, Qwen3.8-27B `UD-Q5_K_XL`, q8_0 K/V, MMQ off. `checkpoint` means the opt-in `--checkpoint-recurrent-prev` path described above. NInfer uses the Qwen3.8-27B `groupwise-int` artifact, INT8-G64 KV, 1024-token prefill chunks, and CUDA Graphs. PP is prompt-processing throughput; higher is better.

| 100k reused + suffix | fork, normal checkpoints | fork, `--checkpoint-recurrent-prev` | NInfer groupwise | fastest |
|---|---:|---:|---:|---:|
| +128 | 191.15 PP/s | 219.19 PP/s | **224.74 PP/s** | NInfer +2.53% |
| +256 | 291.78 PP/s | **323.60 PP/s** | 311.76 PP/s | fork +3.80% |
| +1000 | 434.28 PP/s | **450.47 PP/s** | 326.95 PP/s | fork +37.78% |

The fork figures are warmed A/B/B/A means with six measured samples per condition. Every request asserted `cache_n=100000` and `prompt_n=suffix`; the one generated control token was identical across checkpoint-on/off arms. The NInfer figures are three steady repeated measurements from its retained private endpoint, likewise asserting exactly 100,000 reused tokens.

The two artifacts use different quantization layouts and KV formats, so this is a workload comparison rather than a quantization-equivalent kernel comparison. NInfer reports about 15.92 GiB of weights with a lower-precision groupwise layout in several projections (for example Q4 gate/up and Q5 down); the tested GGUF is about 18.83 GiB and uses Q5_K gate/up and Q6_K down. The KV formats also differ: INT8-G64 versus q8_0.

No +177 or +512 NInfer row is reported: the preserved +177 raw-token harness failed its 100k-reuse assertion (`reused_prompt_tokens=0`), and the +512 sweep hit a CUDA-graph preparation failure. Those points are omitted rather than interpolated. NInfer's published short fresh-prompt `pp2048` result is also not mixed into this long-cache table.

### Isolated Qwen token-generation paths

This section isolates the four newer generation paths **within the fork**. It is useful for attributing the TG gain and is separate from the single-V100 whole-model rows above.

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

For routed MoE models such as Ornith, build a separate binary with `GGML_CUDA_FORCE_MMQ=ON`. Do not use FORCE_MMQ for dense Qwen3.8; it can reduce prompt-processing throughput.

### Isolated 256x256 FlashAttention benchmark

[PR #27997](https://github.com/ggml-org/llama.cpp/pull/27997) isolates the sm70 FlashAttention configuration for `DKQ=256`, `DV=256`, `ncols=64`.

#### 100k cached + 1024 PP + 512 TG

Single V100, 200 W, q8_0 K/V, no forced MMQ. A 100,000-token state was serialized and restored for each measured arm; PP and TG were timed separately.

| model | upstream PP | edited PP | PP change | upstream TG | edited TG | TG change |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 276.063 | **352.926** | **+27.84%** | 16.269 | 16.366 | +0.60% |
| Ornith-1.5-35B-A3B | 485.493 | **580.357** | **+19.54%** | 59.262 | 59.386 | +0.21% |
| Gemma 4 12B | 483.430 | 486.664 | +0.67% | 39.445 | 39.457 | +0.03% |

The gain follows the 256-wide attention geometry; both Qwen3.8 and Ornith use it.

#### Qwen3.8 scaling with KV depth

Single V100, 200 W, q8_0 K/V, 1024 new prompt tokens:

| prefilled KV depth | upstream | edited | change |
|---:|---:|---:|---:|
| 4,096 | 736.718 tok/s | 754.633 tok/s | +2.43% |
| 16,384 | 609.015 tok/s | 657.612 tok/s | **+7.98%** |
| 65,536 | 356.232 tok/s | 434.180 tok/s | **+21.88%** |
| 100,000 | 275.180 tok/s | 349.180 tok/s | **+26.89%** |

Short `pp512` benchmarks understate the gain at agent-scale KV depth.

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

The outputs were exact. Focused GDN correctness tests passed **40/40 on V100**.

#### Ornith exclusion from adaptive 2-CTA

The Qwen-oriented adaptive 2-CTA path is restricted to the exact Volta `256x256, ncols1=32, ncols2=2` layout. Ornith's `8x8` legacy path stays on the existing path. The extracted Ornith-style kernel SASS was byte-identical between the PR02 baseline and strict PR03 build.

The strict gate leaves non-target layouts such as Ornith's unchanged.

## Patch map

The fork is best understood as **upstream llama.cpp plus the patches below**. Documentation-only commits are omitted. Percentages are measured examples rather than additive estimates: gains depend on model, context length, quantization and GPU placement, and later patches often move the bottleneck somewhere else.

### Core V100 / Turing patches

| patch | commit | main scope | what it changes | measured impact / expectation |
|---|---|---|---|---|
| Validated MTP refresh and readbacks | `3ec3c0ff5`, `98babb266`, `105d66183` | Single-layer Qwen/Ornith MTP | K/V-only maintenance graph and consolidated sampling/hidden reads; automatic for validated heads with rollback switches. | Integrated four-agent MTP1 turn **-2.5%** at 100k; MTP3 effect within noise. See the [integration report](benches/mtp-production-merge-0907/REPORT.md). |
| Correct draft budgets | `f238050da`, `acda209a4`, `f9d1c3a2b` | MTP and server | Skip zero-budget preparation; stop at the per-request cap; preserve negative=no-override semantics. | Prevents extra drafts/zero-budget failures; tested separately and in the integration bundle. |
| Volta runtime foundation | `0471a9885` | V100, Qwen/Ornith | Integrated the first V100-specific FA/GDN/speculative/cache/scheduler stack used by the later patches. | Bundled foundation; **do not assign one percentage** to this commit. Use the isolated rows below. |
| Qwen MTP3 decode stack | `4cb009882` | Qwen3.8-27B, V100 | q8_0 tiled target-verification attention, exact 131072-row MTP proposal shortlist, Q5_K T=4 weight reuse and Q6_K T=4 `w4r4` scheduling. | **+32.19% TG** in the within-fork 100k + 1k + 256 MTP3 A/B, PP +0.65%; generated-token SHA matched. |
| Volta Qwen prompt-attention dispatch | `d3c14522d` | Qwen3.8, V100 | Adds the sm70 D256 32-column prompt configuration and a measured small/medium-suffix dispatch gate. | Part of the D256 FA stack. The isolated D256 stack measured **+27.84% Qwen PP at 100k KV**; that whole number should not be attributed to this commit alone. |
| Recurrent previous-state checkpoint | `be6567cea` | hybrid/recurrent models, mainly non-MTP Qwen | Avoids replaying a separate checkpoint tail for cached agent turns; preserves the previous recurrent state on device. | Qwen 100k cached: **+14.08% PP at +128**, +11.50% at +177, +10.63% at +256, +5.56% at +512. |
| Deferred MTP prompt catch-up | `214200a2c` | Qwen + draft-MTP | Defers MTP hidden-state catch-up until after the first target token is queued. | Representative +177 agent turn: **TTFT -5.28%** (864.9 → 819.3 ms), TG -0.38%, same MTP acceptance. |
| Volta MoE MMQ + mirrored-copy/runtime cleanup | `cee72e8c8` | Ornith/MoE on V100; mixed GPU | Adds expert-only `GGML_CUDA_VOLTA_FORCE_MMQ=moe`, Volta MMQ sizing/register fixes and large mirrored-input async fan-out. | **Ornith Q5/Q4: +54.7% PP** at 100k in the current test (917.6 → 1419.3), TG flat and exact SHA. The mirrored-input subpath separately measured about +1.16% PP on a large 1k fan-out workload. |
| Tensor-parallel head balancing | `130965904` | mixed V100 + second GPU | Rebalances Qwen tensor-parallel attention heads instead of relying only on byte/layer split. | Useful only in mixed-GPU placement; **no clean standalone percentage** retained. Evaluate as part of the mixed-GPU integration benchmark. |
| Long-context TP attention dispatch | `86949610a` | mixed GPU, long KV | Adds geometry/long-K dispatch tuning for tensor-parallel attention. | Context- and split-dependent; no single isolated headline number. |
| Turing prompt kernels | `2bb7aca44`, `2e468c655` | RTX 2080 Ti / SM75 | SM75-specific Qwen prompt-kernel tuning plus an optional large-prompt cuBLAS crossover. | See the RTX 2080 Ti benchmark section; these paths are intentionally separate from the V100 claims. |
| Correct GDN RMSNorm-scale fusion | `5eac15130` | CUDA, hybrid/GDN models | Preserves upstream's corrected `rsqrt(sum(x^2)+eps)` GDN normalization while fusing the RMSNorm + scalar-scale pair into one CUDA launch. | Removes 480 extra launches in the profiled dense-Qwen case; current Qwen/Ornith regression results are reported in the post-upstream validation table. |

For Ornith specifically, the dominant fork feature is the **expert-only Volta MMQ route** in `cee72e8c8`. A matched upstream-vs-fork 100k + 1k + 512 MTP test with FORCE_MMQ enabled on both sides measured **689.86 → 881.44 PP/s (+27.77%)** with essentially unchanged TG. The newer downloaded Q5/Q4 model isolates the MMQ routing decision itself more strongly: **917.65 → 1419.32 PP/s (+54.67%)**, identical generated-token SHA.

### Qwen3.8 Flash-Next patches

| patch | commit | main scope | what it changes | measured impact |
|---|---|---|---|---|
| One-weight-ahead MoE prefetch | `9f2bf9bc2` | partially CPU-offloaded MoE prefill | Uses an independent CUDA copy context plus events to overlap the next host weight transfer with current GPU compute while keeping only one extra weight tensor live. | On the final `v100-optimized` base, Flash-Next `n-cpu-moe=18`, `ubatch=1000`: **192.59 → 202.48 PP/s (+5.13%)**, TTFT **5.232 → 5.014 s (-4.16%)**, exact output. Decode is not attributed to this prefetch path. Forced-CPU-expert Ornith test was only +0.3% PP (noise), so no Ornith win is claimed. |
| Append-only restored hybrid cache | `4129c41b6` | Flash-Next / hybrid state restore | Preserves restored recurrent/KV/indexer state when continuing append-only instead of invalidating the reusable long-context state. | **Correctness/enabler, not a kernel speed claim.** It makes same-state 261k+ continuation benchmarking possible and bit-for-bit reproducible. |
| Gather-based QSA attention | `fe255b46a` | Qwen3.8 Flash-Next | Physically gathers the QSA-selected original K/V cells and runs attention over the compact selection instead of rebuilding a sparse mask over the full KV window. | On the earlier final `v100-optimized` base at ~262k context: **7.974 → 9.399 TG/s (+17.88%)**, same generated-token SHA and cache endpoint. |
| Remove unused indexer V cache | `d1ccec3ea` | Qwen3.8 Flash-Next | Stops allocating a V cache for the QSA indexer because the indexer consumes K only. | Small VRAM saving; **no meaningful speedup claimed**. |
| Block-first QSA selection | `4bf80db6d`, `383db326c`, `70b604998`, `8f5d6a7e3` | Qwen3.8 Flash-Next direct decode | Selects 512 compressed QSA blocks before expanding only their selected physical cells; avoids materializing a context-sized cell-score vector. | Exact-output structural win and prerequisite for the fused pooling row below; no standalone percentage is assigned because the final stack was validated as a whole. |
| Fused q8 QSA block-key pooling | `09d9e1bf1`, `cce048d1d`, `5191933b8`, `c9e919e73` | Qwen3.8 Flash-Next direct decode | Dequantizes and mean-pools each 4-row q8_0 indexer block directly into one F32 summary; uses a private QSA GET_ROWS marker; keeps 65,536 block norms off CUDA `grid.y`; enabled by default with an opt-out env flag. | 100k+1k production: **TG +17.58% at 64** and **+11.97% at 512**, PP +0.17--0.18%, exact SHA. Near 262k, warm 4x256: **12.005 → 15.675 TG/s (+30.6%)**, exact. Ornith generic-backend ABBA: PP -0.325%, TG +0.43%, exact. |
| Fused 4-head / block-first QSA prefill | `52bac9bda`, `8010e9d13` | Qwen3.8 Flash-Next prefill | Reduces the four QSA heads together and selects complete compressed blocks before expanding them back to physical cells. | Accepted pre-sync stack moved the 100k+1k workload from roughly 208 PP/s to ~245 PP/s with exact generation. |
| Raw-q8 tiled QSA prefill gather | `de2c9a338`, `273a76f6a`, `f3f0eef26` | Qwen3.8 Flash-Next prefill + text decode integration | Gathers only selected q8_0 K/V rows in query tiles, isolates gather/non-gather graph inputs, and keeps the sparse decode fast path enabled for token-input IMRoPE text batches. | Same-split 35:14 PP: **244.71 → 266.70 tok/s (+8.98%)**; final 512-token run **266.98 PP / 22.88 TG**, exact SHA. |

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
V100 132/132 passed
```

### Adaptive Volta 2-CTA specialization

The adaptive Volta path uses a separate compact kernel for exact sm70 `256x256, 32x2` and uses:

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

### Turing (SM75) paths

SM75 uses separate geometry-gated paths:

- the Qwen `256x256` attention configuration reduces the large prompt tile for Turing's register/shared-memory balance;
- the prefill GatedDeltaNet x4 path is enabled for both Volta and Turing;
- long-K Qwen GQA6 prompt tiles can use the smaller `ncols2=2` dispatch;
- `GGML_CUDA_TURING_CUBLAS_MIN_BATCH` is an optional threshold for sending large dense Q5_K/Q6_K prompt matmuls through cuBLAS; leave it unset unless it is measured on the target SM75 GPU.

These controls do not replace the SM70 Volta paths. Single-GPU V100 results, single-GPU RTX operator results, and mixed-GPU end-to-end results are labeled separately in this README.

### Recurrent previous-state checkpoints

`--checkpoint-recurrent-prev` is an opt-in server optimization for hybrid/recurrent prompt replay. For prompt suffixes above 64 tokens, it avoids forcing a separate final checkpoint tail. Instead, the server saves the immediately preceding recurrent snapshot on-device and can later restore that state and replay one token at a branch point. Rollback-plane work is disabled during single-token decode, so the optimization does not add TG work. Device-backed checkpoints are excluded from the persistent RAM prompt cache and from child-slot clones; only portable host-backed checkpoints cross those boundaries.

### Deferred MTP prompt catch-up

`--spec-mtp-defer-prompt` is the speculative/MTP counterpart for **user-visible latency**. The target still evaluates the same prompt segmentation, but the MTP hidden-state catch-up is captured asynchronously and consumed after the first response token is queued. The N-4 replay checkpoint is also queued device-side before the final target batch and completed after token 1, so exact prompt replay keeps the same deep cached boundary instead of falling back to the older 100k prefix.

The option is deliberately restricted to draft-MTP. If the split-phase capture is unavailable, the normal speculative processing path remains the fallback.

### Quantized-weight reuse during prefill

`--prefill-reuse 1024` allows a larger physical prompt batch to reuse a converted quantized weight while keeping the smaller cuBLAS GEMM tile. It targets the Volta Q5_K/Q6_K -> F16 cuBLAS path.

This is an experimental fork-level tuning knob. Its incremental Qwen gain is small after the FA changes; do not assume it helps other architectures.

### Smaller pipeline scheduler allocation

`--pipeline-copies 2` reduces cross-backend scheduler input copies. Its main purpose is reducing compute-buffer VRAM so larger long-context graphs fit.

### Tensor-parallel copies and Volta register pressure

Large contiguous host inputs that are marked as mirrored are submitted through the backend's asynchronous setter before one explicit synchronization. This reduces copy fan-out overhead while preserving the caller's input lifetime.

On Volta, large-row Q6_K MMQ uses a DP4A launch configuration instead of the higher-register Ampere configuration. This is the register-pressure mitigation used by the current branch. `GGML_CUDA_VOLTA_FORCE_MMQ=moe` is an opt-in override for routed MoE workloads; leave it unset for dense Qwen3.8.

The optional `GGML_CUDA_VOLTA_GQA8_NCOLS2=2` path is restricted to long-K Qwen 256-wide GQA8 attention and remains geometry- and environment-gated. Validate it on the target GPU before enabling it.

### MTP draft ubatch

`--spec-draft-ubatch 1024` gives the MTP/draft context a smaller physical ubatch than the target context, reducing transient draft-graph VRAM use.

### Prompt cache and recurrent checkpoints

Server changes for coding-agent traffic:

- multi-slot server mode defaults to unified KV unless `--no-kv-unified` is explicitly requested;
- exact-prefix sequence sharing is enabled by default for unified-KV serving and can be disabled with `--no-slot-fork-prefix`;
- prompt-cache entries preserve target, draft and speculative/MTP state, and the whole parked-agent bank can be saved/restored through `/prompt-cache`;
- parked states can restore directly onto an already-live compatible prefix instead of temporarily duplicating the prefix in KV;
- prompt-cache states are chosen by the deepest usable absolute prefix rather than ratio alone;
- recurrent checkpoints preserve likely replay boundaries;
- duplicate exact checkpoint prefixes are refreshed rather than stored twice;
- observed replay hits influence checkpoint retention;
- plain-attention draft KV is not duplicated into recurrent checkpoints when it can be suffix-trimmed normally.

A regression test covers the case where a short live prompt would otherwise beat a much deeper cached prefix.

## Validation

Validation includes backend correctness tests and CUDA sanitizers:

```text
256x256 FA matrix        V100 132/132
focused GDN cases        V100 40/40
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

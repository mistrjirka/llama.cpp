# llama.cpp — Qwen3.8 long-context / agent-cache fork

This repository is a small performance fork of upstream [`llama.cpp`](README.old), focused on **lossless** long-context Qwen3.8 serving on the V100 + RTX 3060 Ti system used for development here.

The original upstream README is preserved as [`README.old`](README.old). Upstream build, model, API, and general usage documentation remains there.

## What differs from upstream

The fork intentionally keeps the default behavior compatible with upstream unless a new optimization is explicitly enabled.

### 1. Deeper absolute-prefix prompt-cache selection

Upstream prompt-cache state selection can let a short currently-live prompt beat a much deeper cached branch because it compares ratio-style similarity/retention measures. This fork keeps the existing viability guard but ranks usable cached states by **absolute common-prefix token count**.

This is aimed at agent workloads such as:

1. long coding conversation A,
2. small unrelated request,
3. return to conversation A.

The cached long branch should win because it avoids far more prefill work.

Commit: `server: prefer prompt-cache states with deeper prefixes`.

### 2. Recurrent prompt checkpoint retention for agent replay

For recurrent/hybrid models such as Qwen3.8, ordinary KV-prefix reuse is not sufficient because recurrent state must also be restored at the correct boundary.

This fork improves prompt checkpoints by:

- marking high-value user/exact-replay boundaries,
- refreshing an existing checkpoint instead of duplicating the same prefix,
- retaining replay boundaries during ordinary spacing cleanup,
- recording successful replay hits,
- value-based checkpoint eviction using checkpoint spacing, semantic replay value, and observed hits,
- avoiding redundant draft KV storage when the draft cache can be suffix-trimmed normally.

Commit: `server: improve recurrent prompt checkpoint retention`.

### 3. Lossless Volta quantized-weight reuse during prefill

New option:

```text
--prefill-reuse N
```

On the Volta CUDA path, large Q5_K/Q6_K prompt matrix multiplications convert the static quantized weight to F16 before `cublasGemmEx`. Repeating the ordinary 1024-token physical ubatch therefore converts the same static weight repeatedly.

This fork can use a larger physical ubatch while preserving the original GEMM width:

```text
large physical ubatch
    -> convert quantized src0 weight once
    -> GEMM columns 0..1023      (baseline 1024 shape)
    -> GEMM columns 1024..2047   (same baseline shape)
    -> GEMM columns 2048..3071   (same baseline shape)
    -> ...
```

The converted weight buffer is reused, while each cuBLAS GEMM keeps the baseline tile dimension and stream order. This is specifically guarded to the quantized F16 **Volta** path; Ampere/MMQ and other CUDA paths are unchanged.

For this hardware the validated tile is:

```text
--prefill-reuse 1024
```

`0` disables the feature and preserves upstream behavior.

### 4. Configurable pipeline scheduler copies

New option:

```text
--pipeline-copies N
```

Upstream pipeline scheduling reserves up to four copies of cross-backend graph inputs. On this two-GPU Qwen3.8 configuration that can consume enough graph workspace to prevent otherwise valid larger prefill graphs and can leave too little headroom for runtime cuBLAS workspace.

`0` keeps the upstream/default scheduler behavior. On this system `2` is the tested sweet spot.

### 5. Independent draft / MTP ubatch

New option:

```text
--spec-draft-ubatch N
```

The MTP context no longer has to inherit the target's physical ubatch. If the target processes a larger prompt batch, MTP catch-up is streamed through the draft context in its own smaller chunks while preserving the exact token positions and shifted target hidden rows.

For Qwen3.8 here:

```text
--spec-draft-ubatch 1024
```

This is needed to combine large target prefill reuse with a memory-efficient MTP graph.

### 6. Volta fused GatedDeltaNet prefill column reuse

Qwen3.8's scalar-gate recurrent layers use a 128-wide GatedDeltaNet state. In the upstream CUDA kernel, one warp owns one state/output column, so independent warps repeatedly load the same Q/K vectors and scalar gate/beta values.

On NVIDIA Volta only, this fork adds a specialized `S_v = 128`, scalar-gate, prefill-only kernel that keeps **four independent state columns in one warp**. The four columns share the same Q/K registers while each column preserves its original token recurrence and warp reduction order. Decode (`n_tokens == 1`), KDA, other head sizes, Ampere, HIP, and MUSA continue through the upstream kernel.

The specialized sm70 kernel uses 72 registers/thread with no local-memory spill. On the Qwen-like isolated GDN benchmark (`head_size=128`, `head_count=32`, `n_seq_tokens=1024`) it reduced kernel time from about **1.797 ms to 1.175 ms** (~34.6% lower GDN time).

A matched three-run 23,289-token production-reuse A/B measured:

```text
reuse baseline: 29.537 / 29.738 / 29.900 s, median 29.738 s = 783.13 tok/s
reuse + GDN:    28.934 / 29.191 / 29.500 s, median 29.191 s = 797.80 tok/s
```

This is **+1.87% prompt-processing throughput** from the GDN kernel on top of the already-validated prefill-reuse configuration. The captured native Pi tool-response canonical SHA and all paired retrieval token SHAs were identical.

The optimization follows the scalar-gate Gated DeltaNet recurrence described by **Gated Delta Networks: Improving Mamba2 with Delta Rule** and the related delta-rule work **Parallelizing Linear Transformers with the Delta Rule over Sequence Length**. Flash Linear Attention is a useful reference implementation of the same model family:

- https://arxiv.org/abs/2412.06464
- https://arxiv.org/abs/2406.06484
- https://github.com/fla-org/flash-linear-attention

This fork's sm70 path is deliberately the existing **token-by-token recurrent** formulation. It does not replace the recurrence with the papers' WY/chunkwise formulation, because reassociating the recurrence changes floating-point execution and has not passed this fork's bitwise-lossless gate.

Commit: `cuda: optimize Volta GDN prefill column reuse`.

### 7. Volta 256x256 FlashAttention register-pressure tuning

Qwen3.8-27B full-attention layers use 256-wide Q/K/V heads with 24 query heads and 4 KV heads (GQA ratio 6). For large prefill this selects the 64-column `flash_attn_ext_f16<256,256,32,2>` specialization.

Upstream's Volta configuration table does not have a dedicated 256x256 entry and falls back to the Ampere tuning. On sm70 that configuration keeps Q permanently in registers. The resulting kernel compiled at **255 registers/thread with a 552-byte stack frame**, indicating heavy spill pressure.

This fork adds a Volta-only 256x256 / 64-column configuration that keeps the same thread count, occupancy target, KV batch, K/V load geometry, combine width, and pipeline staging, but sets `Q_in_reg = false`. Q is therefore staged through shared memory instead of being held permanently in registers. The compiled stack frame falls from **552 bytes to 48 bytes** while the attention/rescaling work order remains unchanged.

On the Qwen3.8 full-attention geometry (`D=256`, 4 KV heads, GQA=6, query batch 4096, q8_0 K/V), isolated V100 timings were:

| KV length | upstream-derived Volta config | Q-in-shared Volta config | kernel speedup |
|---:|---:|---:|---:|
| 4,096 | 19.58 ms | 16.62 ms | 1.18x |
| 8,192 | 37.51 ms | 31.96 ms | 1.17x |
| 16,384 | 75.44 ms | 64.47 ms | 1.17x |
| 24,576 | 113.48 ms | 96.37 ms | 1.18x |

A matched three-run 23,289-token production-stack A/B measured:

```text
reuse + GDN baseline: 28.507 / 28.548 / 28.760 s, median 28.548 s = 815.78 tok/s
+ Volta FA Q staging:  27.887 / 27.931 / 27.972 s, median 27.931 s = 833.80 tok/s
```

This is **+2.21% whole prompt-processing throughput** on top of the existing lossless reuse + GDN stack. All six generated-token hashes were identical. A separate production-ub4096 top-100 probability A/B was byte-identical (probability SHA256 `21101d2100c0d29a0683ff83d507505d3ff6ace70fb96964c75316ef8ed7c4e7`).

A more aggressive variant also doubled the FlashAttention rescaling chunk and reached roughly 32 TFLOP/s, but it changed the target probability object and is therefore **rejected from the lossless path**. Only the register-placement change above is promoted.

Commit: `cuda: reduce Volta 256x256 FlashAttention register pressure`.

## Recommended Qwen3.8-27B configuration for this machine

Validated hardware:

- Tesla V100 SXM2 32 GB — main Qwen trunk compute
- GeForce RTX 3060 Ti 8 GB — LM-head / MTP side
- Qwen3.8-27B UD-Q5_K_XL
- 262,144-token context

Current validated lossless candidate:

```bash
./build/bin/llama-server \
  --model /models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --alias qwen3.8-27b \
  --ctx-size 262144 \
  --parallel 1 \
  --split-mode layer \
  --fit off \
  --gpu-layers all \
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
  --perf --metrics
```

The manual `64,2` placement is intentional for this exact hardware/model. Do not copy it blindly to a different GPU pair.

## Measured lossless prefill result

The original prefill-reuse implementation was validated with a matched three-run 23,289-token production-context A/B:

| configuration | median prompt time | median PP | result |
|---|---:|---:|---:|
| baseline: batch4096 / ub1024 | **31.639 s** | **736.1 tok/s** | reference |
| reuse: batch4096 / ub4096 / tile1024 | **30.007 s** | **776.1 tok/s** | **+5.44% PP** |

After the GDN optimization, a fresh three-run series on the current code measured:

| configuration | prompt runs | median prompt time | median PP |
|---|---|---:|---:|
| original ub1024 path | 31.597 / 31.842 / 32.031 s | **31.842 s** | **731.38 tok/s** |
| reuse baseline | 29.537 / 29.738 / 29.900 s | **29.738 s** | **783.13 tok/s** |
| reuse + GDN | 28.934 / 29.191 / 29.500 s | **29.191 s** | **797.80 tok/s** |

In that fresh series the reuse + GDN stack was **9.08% faster** than the original ub1024 path. The GDN kernel itself contributed **+1.87% PP** over the already-optimized reuse path. The older independently matched reuse result (**+5.44%**) remains the conservative standalone number for weight reuse; benchmark noise means these percentages should not simply be added.

A later same-session matched test of the Volta 256x256 FlashAttention register-pressure fix measured **28.548 s -> 27.931 s**, or **+2.21% PP** on top of reuse + GDN. This percentage has its own fresh control and should likewise not be algebraically added to results from earlier sessions.

The paired 23,289-token generated token SHAs were identical. A captured native Pi request produced the same canonical tool-response SHA, and a 64-step replay produced exactly equal target tokens, content, and top-20 probability structures. On three deterministic replays of the historical 39-call `pydicom-1256` Pi trajectory, the paired summed-prompt-processing gains were **+1.47%**, **+0.79%**, and **+1.00%**. Averaged across the three runs, baseline summed PP was **48.428 s** versus **47.907 s** with GDN (**+1.09%**); mean replay wall time improved by **0.61%**. Every replay had identical prompt/cache geometry, restoring about **530k cached tokens** and processing **27,109 new prompt tokens**, so its incremental gain is naturally smaller than a cold long prompt.

A full **isolated** five-task Pi/SWE smoke run with the normal permissive Pi instruction scored **3/5**: `1413`, `1694`, and `1256` passed; `901` and `1139` failed. `901` is also the historical baseline failure. The additional `1139` miss came from a stochastic live trajectory that stopped at incomplete iteration semantics. The older baseline run was also stochastic and scored 4/5, so this cross-run 4/5-vs-3/5 difference is recorded but is not treated as a numerical-regression oracle. The deterministic probability/token/state gates above are stronger evidence for the lossless CUDA claim, and live tool-using traces can diverge after nondeterministic generation/tool output even when inference math is identical.

A physical ubatch of 8192 was also tested and rejected because the target compute graph did not fit; using logical batch 8192 was slower in absolute terms on this hardware.

Validation used fixed sampling seeds only for A/B reproducibility; the normal serving configuration does **not** set temperature and uses server defaults.

## Approximate opt-in PFlash proxy

The fork also includes [`tools/pflash/`](tools/pflash/) as a separate **approximate** optimization. It is not part of `llama-server` and is disabled unless you explicitly run the proxy.

PFlash removes query-irrelevant aged assistant/tool tokens before a cold target prefill, using a Qwen3-0.6B BF16 Lucebox scorer. The integration is deliberately **cold-long only** in `auto` mode: if a session's first request is short, the proxy leaves that session byte-for-byte pass-through forever instead of rewriting a valuable warm llama.cpp prefix when the conversation later grows.

For genuinely cold long histories the proxy freezes the compressed old prefix. Full omitted text remains in proxy memory so later queries can recover it near the current tail without rewriting earlier prompt bytes. Exact/rare identifiers use bounded lexical recovery; semantic recovery uses a small 1% PFlash pass on new user turns. Long scorer inputs are split into bounded 22k-token windows so the scorer can coexist with the target on the 8 GB 3060 Ti.

Validated cold 48.5k-token retrieval case:

| path | target prompt | end-to-end wall | result |
|---|---:|---:|---|
| direct target | 48,512 | 78.75 s | `A731|B284|C915` |
| PFlash + target | 36,020 | about 67.0 s | `A731|B284|C915` |

That is roughly **1.17× / 15% faster end-to-end** in this case. PFlash remains approximate: removing tokens can change model behavior, so it is not included in the fork's lossless performance claims.

The proxy, tested defaults, Lucebox scorer build helper, future-query recovery design, lifecycle caveats, and per-request controls are documented in [`tools/pflash/README.md`](tools/pflash/README.md).

## Building for V100 + RTX 3060 Ti

A CUDA build that contains native code for both GPUs can be configured with:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES='70;86'
cmake --build build -j
```

See [`README.old`](README.old) for the complete upstream build instructions and supported platforms.

## Upstream relationship

This fork should be kept rebased on upstream `master`. The code documented here was validated against local upstream base `9731ad3f29da96f588711a0d1eb08cf210721e16`. The sandbox blocked a fresh network fetch during final validation, so this README does not claim a newer unverified upstream base. The fork-specific commits are intentionally small and separate from general llama.cpp development so they can be reviewed, rebased, or upstreamed independently.

To inspect the exact delta from upstream in a checkout with the `upstream` remote configured:

```bash
git log --oneline upstream/master..HEAD
git diff --stat upstream/master...HEAD
```

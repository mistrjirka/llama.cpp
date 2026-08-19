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

On the 23,289-token controlled prompt with 262k context, MTP enabled and identical generated token sequence:

| configuration | median prompt time | median PP |
|---|---:|---:|
| baseline: batch4096 / ub1024 | **31.639 s** | **736.1 tok/s** |
| reuse: batch4096 / ub4096 / tile1024 | **30.007 s** | **776.1 tok/s** |

The validated ub4096 configuration is approximately **5.44% faster in cold prompt processing** than the matched baseline. The full generated token sequence was identical. A physical ubatch of 8192 was also tested and rejected because the target compute graph did not fit; using logical batch 8192 was slower in absolute terms on this hardware.

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

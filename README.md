# llama.cpp - Qwen3.8 long-context / agent-cache fork

This repository is a small performance fork of upstream [`llama.cpp`](README.old), focused on **lossless** long-context Qwen3.8 serving on the V100 + RTX 3060 Ti system used for development here. The fork-specific commits currently sit on upstream `0e1d9185c` (2026-08-20). A fresh vanilla `master` at `bb4caa754` (2026-08-21) was also built and benchmarked; the fork has not yet been rebased onto that commit.

The README from the original upstream base is preserved as [`README.old`](README.old). Upstream build, model, API, and general usage documentation remains there. Reproducible benchmark commands and scripts are under [`benches/v100-qwen38/`](benches/v100-qwen38/).

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

### 7. Volta 256x256 FlashAttention tuning

Qwen3.8-27B full-attention layers use 256-wide Q/K/V heads with 24 query heads and 4 KV heads (GQA ratio 6). Large prefill selects the 64-column `flash_attn_ext_f16<256,256,32,2>` specialization.

Upstream's Volta configuration table has no dedicated 256x256 entry and falls back to the Ampere tuning. On sm70 that configuration pins Q in registers and compiles at **255 registers/thread with a 552-byte stack frame**. The first lossless fork optimization moved Q to shared memory (`Q_in_reg=false`), reducing the stack frame to **48 bytes** and improving the isolated FA kernel by roughly 17-18%.

The final implementation goes further without changing the 32-row softmax/rescaling order:

1. Q remains staged in shared memory, avoiding the original register spill.
2. K scratch is reduced from 128 to **96 half2 values** and V scratch from 128 to **64 half2 values**.
3. Because K is now split into 96+32 subtiles, the non-MLA Volta path explicitly walks those subtiles **forwards**. The resulting MMA update sequence is the same 0->128 K-dimension order as the original one-piece K tile.
4. `nbatch_fa=32` and the arithmetic `nbatch_combine=128` are deliberately unchanged. For the validated continuation regime, the two independent 64-half2 output halves are materialized through the same 64-wide shared-memory window. This reduces the launch footprint from 67,584 to **49,152 bytes** and permits two CTAs per V100 SM without changing the combine arithmetic.
5. The 2-CTA launch is **adaptive and conservative**: it is enabled only for the exact sm70 256x256 specialization, one sequence, 768-1023 query tokens, and only when both the legacy and compact occupancy regimes remain on the same >=75% efficient whole-tile/non-Stream-K schedule. Decode/small batches and >=1024-token prefills automatically reserve the legacy shared-memory footprint.
6. The compact specialization also uses a uniform block barrier for its parallel-warp metadata combine; non-target FA specializations retain their original code path.
7. The specialization remains Volta-only; Ampere and all other architectures retain upstream dispatch/configuration.

On the Qwen3.8 full-attention geometry (`D=256`, 4 KV heads, GQA=6, query batch 4096, q8_0 K/V), isolated V100 timings are:

| KV length | upstream-derived config | first Q-shared version | final Q-shared + K96/V64 | final vs upstream |
|---:|---:|---:|---:|---:|
| 4,096 | 19.58 ms | 16.62 ms | **13.19 ms** | **1.48x** |
| 8,192 | 37.51 ms | 31.96 ms | **25.60 ms** | **1.47x** |
| 16,384 | 75.44 ms | 64.47 ms | **51.65 ms** | **1.46x** |
| 24,576 | 113.48 ms | 96.37 ms | **77.66 ms** | **1.46x** |

A fresh interleaved three-pair 23,289-token production-stack A/B measured:

```text
reuse + GDN baseline:      28.511 / 29.015 / 29.024 s, median 29.015 s = 802.67 tok/s
+ final Volta FA tuning:   26.649 / 26.789 / 26.986 s, median 26.789 s = 869.36 tok/s
```

That is **+8.31% whole prompt-processing throughput** over the same-session reuse + GDN control. Every one of the six runs produced the same canonical token/probability SHA256 `1405080cf8dbad123b94b29a908fc6baccc2f88aa3a41a73bb97661c784403af`. A subsequent build from the production tree measured 26.517 s / 878.3 tok/s in a one-shot 23k confirmation.

Lossless validation is stronger than sampled-token equality:

- captured 4,118-token top-100 probability object: exactly equal, SHA256 `fdbcc8cdc33c1c78d0f077c9f210b38bd646186247237615f403fde8e14995ac`;
- 64-step replay: every token, content byte and top-20 probability structure exactly equal, probability-object SHA256 `3d936d1bcd4a37de70b488c46726b11ce4fc98950c2b12e33d91a05b9c12bdb1`;
- CUDA backend correctness includes the Qwen GQA geometry and q8_0 K/V;
- CUDA racecheck on the candidate reported **0 hazards / 0 errors / 0 warnings**;
- CUDA1/Ampere fallback passed 114/114 filtered FlashAttention tests.

The adaptive 2-CTA follow-up was validated separately on the active Q=1000/q8_0 geometry with all three CUDA sanitizer modes: **synccheck 0 errors, racecheck 0 hazards/errors/warnings, and memcheck 0 errors**. The compact Volta metadata path replaces the previously divergent block-barrier structure with a uniform barrier.

Two broader variants were deliberately rejected rather than enabled globally:

- doubling the softmax/rescaling work chunk changed the 64-step trajectory: **numerical/quality failure**;
- enabling the compact 49,152-byte / 2-CTA launch indiscriminately changed probability structures on scheduling regimes such as smaller queries/Stream-K and large prefills. Investigation showed that the same compact kernel is bit-exact when the logical whole-tile schedule is unchanged, which led to the conservative adaptive gate above rather than an approximate global mode.

Commits in this area start with `cuda: reduce Volta 256x256 FlashAttention register pressure`; the K/V-subtile and adaptive 2-CTA follow-ups retain the same lossless policy.

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

A later fresh interleaved three-pair test of the final Volta 256x256 FlashAttention implementation measured **29.015 s -> 26.789 s**, or **+8.31% PP** on top of the same-session reuse + GDN control. This supersedes the earlier Q-shared-only +4.02% result. As with the other measurements, do not algebraically add percentages taken from different benchmark sessions.

For the more realistic long-agent continuation target, a saved **100,000-token production KV state** was restored into each build and the identical next **1,000 prompt tokens** were evaluated. Using the same state removes cache-construction and cache-selection differences from the A/B:

| implementation | +1k prompt time | +1k PP | vs upstream-derived CUDA baseline |
|---|---:|---:|---:|
| historical upstream-derived CUDA baseline (`474446df1`; only server cache-selection commits differ from its upstream base) | **3.010 s** | **332.18 tok/s** | reference |
| previous pushed lossless stack (`7718be7bd`) | **2.566 s** | **389.65 tok/s** | **+17.30%** |
| adaptive Volta 2-CTA continuation, rebased on `0e1d9185c` | **2.236 s** | **447.21 tok/s** | **+34.63%** |

The rebased adaptive path is **+14.77% PP throughput** over the previous pushed fork for this shared-state continuation, with **12.87% lower prompt latency**. It is only **0.19% slower** than the pre-rebase 448.08 tok/s confirmation, i.e. effectively unchanged within run-to-run noise despite taking upstream's new CUDA/cuBLAS workspace changes. The generated token, content, and complete top-100 probability object are exactly equal to the pre-rebase validated build. Raw Q8 FlashAttention output at Q=1000 was also bit-identical across all **6,144,000 float outputs** between the one-CTA and two-CTA launch. Boundary gates at Q=767/768/1023/1024 were exactly equal, and a matched 64-token direct replay had identical tokens, content, and every probability object.

Post-rebase validation used a clean `build-rebased/` tree at commit `45c26d803`: the top-100 and 64-step probability objects remained byte-for-byte equal to the pre-rebase validated build; the shared 100k -> 101k continuation remained probability-identical; and the active Q=1000/q8_0 CUDA path passed **synccheck (0 errors), racecheck (0 hazards/errors/warnings), and memcheck (0 errors)**.

The paired 23,289-token generated token SHAs were identical. A captured native Pi request produced the same canonical tool-response SHA, and a 64-step replay produced exactly equal target tokens, content, and top-20 probability structures. On three deterministic replays of the historical 39-call `pydicom-1256` Pi trajectory, the paired summed-prompt-processing gains were **+1.47%**, **+0.79%**, and **+1.00%**. Averaged across the three runs, baseline summed PP was **48.428 s** versus **47.907 s** with GDN (**+1.09%**); mean replay wall time improved by **0.61%**. Every replay had identical prompt/cache geometry, restoring about **530k cached tokens** and processing **27,109 new prompt tokens**, so its incremental gain is naturally smaller than a cold long prompt.

A full **isolated** five-task Pi/SWE smoke run with the normal permissive Pi instruction scored **3/5**: `1413`, `1694`, and `1256` passed; `901` and `1139` failed. `901` is also the historical baseline failure. The additional `1139` miss came from a stochastic live trajectory that stopped at incomplete iteration semantics. The older baseline run was also stochastic and scored 4/5, so this cross-run 4/5-vs-3/5 difference is recorded but is not treated as a numerical-regression oracle. The deterministic probability/token/state gates above are stronger evidence for the lossless CUDA claim, and live tool-using traces can diverge after nondeterministic generation/tool output even when inference math is identical.

A physical ubatch of 8192 was also tested and rejected because the target compute graph did not fit; using logical batch 8192 was slower in absolute terms on this hardware.

Validation used fixed sampling seeds only for A/B reproducibility; the normal serving configuration does **not** set temperature and uses server defaults.

## Current controlled benchmark matrix

The main portability check uses an exact **100,000-token real cached state + 1,000 new prompt tokens + 64 generated tokens**. Model-specific GPU placement was tuned independently, but each upstream/fork row uses the same saved semantic state and prompt sequence.

| Model | Hardware | Fork PP | Upstream PP | PP delta | Fork TG | Upstream TG | TG delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | V100 + 3060 Ti | **450.0** | 317.7 | **+41.6%** | 26.65 | 26.48 | +0.6% |
| Qwen3.8-27B | V100 only | **433.1** | 306.7 | **+41.2%** | 23.19 | 23.29 | -0.4% |
| Qwen3.5-122B-A10B | V100 + 3060 Ti | **283.5** | 257.4 | **+10.2%** | 19.80 | 19.83 | -0.1% |
| Qwen3.5-122B-A10B | V100 only | **280.8** | 241.4 | **+16.3%** | 21.30 | 21.29 | +0.0% |
| Laguna-S-2.1 | V100 + 3060 Ti | 317.4 | **319.5** | -0.6% | 20.81 | **21.24** | -2.0% |
| Laguna-S-2.1 | V100 only | 312.3 | **320.5** | -2.5% | 22.26 | **23.28** | -4.4% |

The large PP gain is therefore **Qwen/model-shape specific**, not a generic speedup. Laguna is a useful negative control: it does not use the promoted 256-wide FA/GDN paths and does not benefit from the fork.

A separate Qwen3.8 freshness check was run against vanilla `master` `bb4caa754` rather than the older controlled base. On the same 100k + 1k + 64 dual-GPU workload, vanilla master measured **313.98 tok/s PP / 26.17 tok/s TG** and this fork measured **447.11 tok/s PP / 26.20 tok/s TG**. That is **+42.40% PP** with identical generated tokens/content and the same **37/52 MTP acceptance**.

Qwen3.5 has the same scalar-gate GatedDeltaNet `S_v=128` shape and 256-wide full-attention layers, so the promoted Volta kernels also apply. Isolated Qwen3.5-shaped tests measured about **34.9% lower GDN kernel latency** and about **1.51x FlashAttention throughput**. However, the larger-ubatch weight-reuse policy is model-sensitive: Qwen3.5 and Laguna showed small-to-material probability drift when that policy was enabled, so Qwen3.8's strict lossless recipe must not be generalized without a model-level quality check.

### GLM-5.2 orientation

GLM-5.2 is a roughly 223 GiB `glm-dsa` model with 576/512 MLA attention and no GatedDeltaNet. A real 100k semantic prime was disproportionately slow, so it was measured separately at **10,000 cached + 1,000 new + 64 generated**. These numbers are orientation-only and are not directly comparable with the 100k table. RTX 3060 Ti-only was intentionally excluded.

| Hardware | MTP | Fork PP | Upstream PP | Fork TG | Upstream TG |
|---|---|---:|---:|---:|---:|
| V100 + 3060 Ti | off | 46.00 | 46.03 | 4.47 | 4.40 |
| V100 + 3060 Ti | on | 44.44 | 44.68 | 6.29 | 6.26 |
| V100 only | off | 45.94 | 45.91 | 4.78 | 4.57 |
| V100 only | on | 44.34 | 44.51 | 6.25 | 6.23 |

MTP accepted **41/42 drafted tokens (97.6%)** in every GLM MTP cell. It improved TG by roughly **31-42%** for about a **3% PP cost**. Fork PP is effectively unchanged on GLM, which is consistent with the CUDA optimization guards.

Full commands, fit settings, and benchmark runners are documented in [`benches/v100-qwen38/v100-qwen38.md`](benches/v100-qwen38/v100-qwen38.md).

## Upstreaming / PR decomposition

The fork is intentionally useful as one deployment branch, but it should **not** be proposed upstream as one PR. A review against current llama.cpp contribution rules suggests this split:

1. **Server prompt-cache prefix selection** - small bug-fix candidate with an existing regression test. It is related to the long-agent recurrent-cache failures tracked upstream, including issue `#22746`.
2. **Volta 256x256 FlashAttention Q/shared + K/V scratch tuning** - backend-only performance PR with no public API. This should be the first FA PR because the diff is much smaller than the adaptive occupancy follow-up.
3. **Volta scalar-gate GatedDeltaNet prefill kernel** - independent backend-only performance PR, guarded to sm70, `S_v=128`, non-KDA prefill.
4. **Adaptive Volta 2-CTA FA occupancy** - separate follow-up because it changes shared-memory scheduling and needs the existing synccheck/racecheck/memcheck evidence.
5. **Independent MTP draft ubatch** - separate speculative-decoding change with dedicated tests.
6. **Prefill-reuse / pipeline-copy controls** - not yet a clean upstream candidate. They add CLI and public API surface and need a stronger generic design than a machine-specific tuning knob.
7. **Recurrent checkpoint retention policy** - related to a real upstream problem, but the current value-based eviction policy is too large to bundle with the simpler cache-selection fix and needs focused regression tests.
8. **PFlash proxy** - keep fork-only. It is approximate, adds an external workflow/dependency, and is outside the lossless CUDA claims.

The source comments in the fork have also been shortened to follow upstream's current rule: comments should usually explain a non-obvious invariant in one or two lines rather than preserve experiment history. No inference logic was changed by that cleanup.

Detailed code-quality findings, test status, and per-candidate upstreaming requirements are in [`benches/v100-qwen38/pr-readiness.md`](benches/v100-qwen38/pr-readiness.md).

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

The fork-specific commits currently sit on upstream `0e1d9185c`. On 2026-08-21, `upstream/master` was fetched at `bb4caa754` (`llama.cpp 0.2.0-dev`). It is 20 upstream commits ahead of the fork base, and those 20 commits do **not** overlap any source file modified by this fork.

A fresh binary from `bb4caa754` was benchmarked directly against the fork on the main Qwen3.8 100k + 1k workload: **313.98 -> 447.11 tok/s PP (+42.40%)**, with effectively unchanged TG and identical generated tokens. The branch should still be rebased onto current master before any upstream candidate is submitted, followed by the same Qwen3.8 A/B and relevant backend/server tests.

To inspect the current delta:

```bash
git fetch upstream master
git log --oneline upstream/master..HEAD
git diff --stat upstream/master...HEAD
```

The fork benchmark scripts are intentionally isolated under `benches/v100-qwen38/` so they can remain deployment/research documentation without being included in small upstream PRs.

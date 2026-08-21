# llama.cpp Qwen3.8 / Volta optimization handoff

Date: 2026-08-21
Workspace: `/workspace/oai-qwen38-pp-lab`
Primary fork checkout: `/workspace/oai-qwen38-pp-lab/llama.cpp`
Current fork branch: `qwen38-lossless-agent-cache`
Last committed fork HEAD: `b403a781e`
Current vanilla upstream master checked/built: `bb4caa754` (`llama.cpp 0.2.0-dev`)

This document is intended to survive total chat/context loss. It records what was done, why it was done, where the useful artifacts are, benchmark methodology/results, important failures, and how the work should be split before attempting upstream PRs.

---

## 1. Goal and scope

The project goal is to improve **code-level inference performance** in llama.cpp for the local Qwen3.8-27B workload, especially long-context prompt processing on NVIDIA Volta/V100, while keeping inference lossless whenever a change is promoted as lossless.

Primary hardware:

- Tesla V100-SXM2 32 GB: main compute GPU
- GeForce RTX 3060 Ti 8 GB: secondary/display GPU and, where useful, model/MTP side placement
- host RAM: 256 GB

Primary model:

```text
/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf
```

Important scope constraints from the user:

- No instruction/prompt tuning as a performance technique. Improvements should come from code/runtime behavior.
- Normal serving should not force a temperature; use server defaults.
- Exact/lossless behavior is strongly preferred. Small numerical differences were explored during research, but promoted lossless paths were validated much more strongly than sampled-token equality.
- Qwen3.8-27B is the most important model and the benchmark that should be rerun after any potentially performance-affecting cleanup.
- GLM-5.2 must **not** be benchmarked on RTX 3060 Ti alone. V100-only and V100+3060Ti are allowed.
- For GLM, a 10k real-cache orientation benchmark was accepted instead of 100k because real 100k semantic priming is extremely slow.

---

## 2. Workspace layout

Main paths:

```text
/workspace/oai-qwen38-pp-lab/
├── llama.cpp/                         # optimized fork, branch qwen38-lossless-agent-cache
├── llama.cpp-upstream-current/        # clean detached upstream worktree, currently bb4caa754
├── build-rebased/                     # fork CUDA build, currently rebuilt from b403a + local working-tree cleanup
├── build-upstream-current/            # current vanilla master CUDA build, bb4caa754
├── results/                           # benchmark and validation output
└── HANDOFF-llama-cpp-qwen38-volta-2026-08-21.md
```

Useful benchmark/reproduction files now live in the fork at:

```text
llama.cpp/benches/v100-qwen38/
├── qwen38-prepare-cache.sh
├── qwen38-100k-ab.sh
├── qwen38-v100-only-ab.sh
├── glm52-10k-prepare-cache.sh
├── glm52-10k-run-one.sh
├── v100-qwen38.md
└── pr-readiness.md
```

These are currently **uncommitted working-tree files**.

The main benchmark summaries are outside the git checkout under:

```text
results/resumed-100k-benchmark-20260821/FINAL-BENCHMARK-SUMMARY.md
results/resumed-100k-benchmark-20260821/FINAL-BENCHMARK-SUMMARY.json
results/resumed-100k-benchmark-20260821/GLM52-10K-ORIENTATION.md
results/resumed-100k-benchmark-20260821/GLM52-10K-ORIENTATION.json
results/qwen38-master-bb4caa754-ab/summary.json
results/pr-readiness-20260821/
```

---

## 3. Git state: committed vs local-only work

### 3.1 Last committed/pushed fork state

Current committed HEAD:

```text
b403a781e docs: record cross-model optimization portability
```

Recent fork-specific commits after upstream base `0e1d9185c`:

```text
3607825ec server: improve recurrent prompt checkpoint retention
eb119fdbf server: prefer prompt-cache states with deeper prefixes
850c85277 cuda: add lossless Volta prefill weight reuse
f00a8b4bd tools: add optional cache-stable PFlash proxy
bad7e3952 cuda: optimize Volta GDN prefill column reuse
d422385c6 docs: document validated Qwen3.8 lossless optimizations
e793205ee cuda: reduce Volta 256x256 FlashAttention register pressure
8c96f72e6 docs: update validated Volta FlashAttention results
e75d47ad9 cuda: optimize Volta FlashAttention K/V scratch
45c26d803 cuda: add lossless adaptive Volta FA occupancy
11ac587b5 docs: record rebased Volta continuation results
b403a781e docs: record cross-model optimization portability
```

The previous known-working fork state is preserved on GitHub at commit `7718be7bd` with:

```text
branch: backup/qwen38-lossless-pre-adaptive
tag:    qwen38-lossless-pre-adaptive-2026-08-21
```

The optimized fork was previously rebased onto upstream `0e1d9185c` and pushed. That rebase was validated for exactness and performance.

### 3.2 Current local uncommitted work

As of this handoff, `git -C llama.cpp status --short` shows:

```text
 M README.md
 M common/arg.cpp
 M common/common.h
 M common/speculative.cpp
 M ggml/src/ggml-cuda/common.cuh
 M ggml/src/ggml-cuda/fattn-mma-f16.cuh
 M ggml/src/ggml-cuda/gated_delta_net.cu
 M ggml/src/ggml-cuda/ggml-cuda.cu
 M include/llama.h
 M tools/pflash/mmvq-bf16-stub.cu
 M tools/server/server-context.cpp
 M tools/server/server-task.cpp
 M tools/server/tests/unit/test_kv_keep_only_active.py
?? benches/v100-qwen38/
```

These local changes are mostly PR-readiness/documentation cleanup:

- root README updated with the final benchmark matrix, current-master A/B, GLM orientation, and recommended upstream PR split;
- long experiment-history comments shortened to llama.cpp's current comment style;
- new `if (...) throw` statements changed to braced form;
- one standards/portability fix changed arithmetic on `void *` to `static_cast<char *>(dst_ptr) + byte_offset`;
- `--prefill-reuse` CLI text no longer claims universally lossless behavior;
- the new server regression test explicitly sets `server.n_ctx = 1024` because its ~725-token prompt does not fit TinyLlama's default 512 context;
- benchmark/cache-preparation scripts and PR-readiness notes added under `benches/v100-qwen38/`.

The local cleanup does **not** intentionally change inference math or scheduling. The only C++ executable changes are semantically equivalent braces and the standards-correct byte-pointer cast. Do not assume these are committed or pushed.

### 3.3 Current upstream state

Fresh upstream fetch on 2026-08-21:

```text
upstream/master = bb4caa754
version         = llama.cpp 0.2.0-dev
fork base       = 0e1d9185c
```

There are 20 upstream commits between `0e1d9185c` and `bb4caa754`.

Important: those 20 commits **do not modify any source file modified by this fork**. The upcoming rebase should therefore be structurally clean, although all performance/correctness tests must still be repeated after rebasing.

Do not create or submit a PR automatically. Current llama.cpp `AGENTS.md` explicitly forbids autonomous PR submission and AI-written PR/reviewer responses. The human contributor must review/understand every line and write the final PR text. AI usage must be disclosed in the PR template.

---

## 4. Current hardware/power state

At handoff:

```text
GPU 0: NVIDIA GeForce RTX 3060 Ti, 7839 MiB free, power limit 200 W
GPU 1: Tesla V100-SXM2-32GB,     32495 MiB free, power limit 300 W
```

The sandbox did not have permission to restore the historical user-selected lower caps of approximately 130 W / 200 W. Therefore the fresh controlled A/B runs in this round use **200 W 3060 Ti / 300 W V100**.

This matters when comparing absolute throughput to earlier historical low-power benchmarks. A/B comparisons within the same fresh session are still fair.

At the end of benchmarking/validation the GPUs were cleaned: no model server left occupying VRAM.

---

## 5. Models used

### Qwen3.8-27B - primary target

```text
/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf
```

### Qwen3.5-122B-A10B - portability check

```text
/models/Qwen3.5-122B-A10B-GGUF/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf
```

### Laguna-S-2.1 - negative/control model

```text
/models/laguna-s-2.1/Laguna-S-2.1-UD-IQ2_M.gguf
```

### GLM-5.2 - very large orientation model

```text
/models/GLM-5.2-UD-IQ2_XXS/UD-IQ2_XXS/GLM-5.2-UD-IQ2_XXS-00001-of-00006.gguf
```

GLM is roughly 222-223 GiB in this quantization, ~753.9B parameters, 256 experts / 8 active, `glm-dsa`, ~1M context, nextn/MTP1. `--load-mode none` is not operationally safe on this 256 GiB host because copying the model creates severe host-memory pressure; use mmap for GLM.

---

## 6. What was optimized

### 6.1 Server prompt-cache state selection

Commit:

```text
eb119fdbf server: prefer prompt-cache states with deeper prefixes
```

Problem: a tiny currently-live prompt could beat a much deeper cached branch because upstream ranked candidates using retention/similarity ratios. For agent workloads this can throw away tens of thousands of reusable prefix tokens.

Fork behavior:

- retain the existing viability guard;
- among viable states, choose the state with the largest absolute common-prefix token count.

This is architecture-independent and is one of the cleanest upstream bug-fix candidates.

Regression test:

```text
llama.cpp/tools/server/tests/unit/test_kv_keep_only_active.py::test_prompt_cache_prefers_deeper_absolute_prefix
```

After fixing the test's context-size mistake, the full neighboring test file passes 3/3.

### 6.2 Recurrent/hybrid prompt-checkpoint retention

Commit:

```text
3607825ec server: improve recurrent prompt checkpoint retention
```

Main ideas:

- mark semantic/exact replay boundaries;
- refresh an existing checkpoint at the same prefix instead of duplicating it;
- protect replay boundaries during periodic spacing cleanup;
- count successful exact restores;
- value-based eviction using checkpoint spacing, semantic replay value, and hit history;
- avoid storing redundant draft KV in recurrent checkpoints when a plain-attention draft can suffix-trim normally.

This helps agent replay on hybrid/recurrent models such as Qwen3.8, where ordinary KV-prefix reuse is insufficient because recurrent state has to be restored at the correct boundary.

This is useful but **not yet a good single PR in its current form**. It needs focused tests for invalidation, branch changes, refresh, capacity eviction, hit weighting, and draft-state restoration.

### 6.3 Volta quantized-weight conversion reuse during prefill

Commit:

```text
850c85277 cuda: add lossless Volta prefill weight reuse
```

New fork option:

```text
--prefill-reuse N
```

For Q5_K/Q6_K matmul on Volta, the large quantized static weight is converted to F16 before cuBLAS. With ordinary 1024-token physical ubatches, the same weight conversion repeats for each ubatch.

The fork can use a larger physical ubatch while preserving the original cuBLAS GEMM width:

```text
large physical ubatch
  -> convert quantized src0 once
  -> GEMM columns 0..1023
  -> GEMM columns 1024..2047
  -> ...
```

Validated Qwen3.8 tile:

```text
--prefill-reuse 1024
```

Important portability finding: the mechanism is useful, but the strict lossless large-ubatch policy is model-sensitive. Qwen3.5 and Laguna showed probability drift in their large-ubatch/reuse tests, so do not upstream a blanket "lossless" claim.

This implementation currently adds CLI/public-context plumbing and a CUDA backend proc-address setter. The functionality is useful in the fork, but the API design should be simplified/generalized before an upstream PR.

### 6.4 Configurable scheduler pipeline copies

New option:

```text
--pipeline-copies N
```

On this dual-GPU setup, upstream's default pipeline scheduler may reserve four copies of cross-backend graph inputs. Reducing it to 2 can free enough VRAM for the larger target prefill graph and runtime cuBLAS workspace.

Validated local value:

```text
--pipeline-copies 2
```

Again: useful local tuning, but currently exposed through additional scheduler/public API surface. Not a clean first upstream candidate.

### 6.5 Independent MTP/draft ubatch

New option:

```text
--spec-draft-ubatch N
```

The draft/MTP context does not have to inherit the target's large physical ubatch. Target prompt data is streamed through the draft in draft-sized chunks with the same token positions and shifted target hidden rows.

Validated local value:

```text
--spec-draft-ubatch 1024
```

This is a plausible generic upstream improvement but needs explicit tests for:

- multi-sequence batches;
- chunk boundaries;
- MTP chain heads;
- pending hidden-row handling;
- target/draft ubatch ratios.

### 6.6 Volta GatedDeltaNet scalar-gate prefill kernel

Commit:

```text
bad7e3952 cuda: optimize Volta GDN prefill column reuse
```

Target geometry:

- sm70/Volta only;
- scalar-gate GatedDeltaNet, non-KDA;
- `S_v = 128`;
- prefill (`n_tokens > 1`), not decode.

Upstream's scalar path effectively gives one state/output column to a warp. The fork keeps four independent state columns in the same warp, sharing Q/K/g/beta loads while preserving the per-column recurrence and warp-reduction order.

Isolated Qwen-like geometry measured roughly:

```text
~1.797 ms -> ~1.175 ms
~34.6% lower GDN kernel time
```

Qwen3.5-shaped isolated test:

```text
1.633 ms -> 1.064 ms
~34.9% lower kernel latency
```

Whole-prompt contribution on one matched 23,289-token test was about +1.87% PP over an already optimized prefill-reuse baseline.

This is a good independent CUDA PR candidate because it is backend-only, shape/hardware guarded, and adds no public API.

### 6.7 Volta 256x256 FlashAttention tuning

Commits:

```text
e793205ee cuda: reduce Volta 256x256 FlashAttention register pressure
e75d47ad9 cuda: optimize Volta FlashAttention K/V scratch
45c26d803 cuda: add lossless adaptive Volta FA occupancy
```

Target Qwen3.8 full-attention geometry:

- DKQ=256, DV=256;
- 64-column prefill specialization;
- V100/sm70;
- Qwen3.8 uses 24 Q heads, 4 KV heads, GQA=6.

Upstream had no dedicated Volta 256x256 tuning and fell back to an Ampere configuration. On sm70 that pinned Q in registers and compiled near 255 registers/thread with a stack frame.

Optimizations:

1. stage Q in shared memory rather than registers;
2. K scratch reduced to 96 half2 and V scratch to 64 half2;
3. split K subtiles evaluated forward to preserve the original MMA update order;
4. preserve `nbatch_fa=32` and 128-value combine arithmetic while using a narrower 64-half2 shared-memory window for the two independent output halves;
5. adaptively use the smaller shared-memory launch only in a schedule-stable Q range, allowing 2 CTAs/SM on V100;
6. use a uniform block barrier in the compact metadata-combine path.

Isolated V100 Qwen3.8-like FA measurements:

| KV length | upstream-derived | final fork | speedup |
|---:|---:|---:|---:|
| 4,096 | 19.58 ms | 13.19 ms | 1.48x |
| 8,192 | 37.51 ms | 25.60 ms | 1.47x |
| 16,384 | 75.44 ms | 51.65 ms | 1.46x |
| 24,576 | 113.48 ms | 77.66 ms | 1.46x |

Matched 23,289-token whole-prompt A/B for the final FA tuning:

```text
reuse + GDN control: 29.015 s median = 802.67 tok/s
+ final Volta FA:     26.789 s median = 869.36 tok/s
whole-prompt gain:    +8.31% PP
```

Important PR-quality finding: the adaptive 2-CTA change duplicates a substantial metadata-combine block just to guarantee uniform synchronization. For upstream, split this work:

1. first PR: Q-shared + K96/V64 tuning, smaller/simple diff;
2. later PR: adaptive compact 2-CTA occupancy with sanitizer and schedule-stability evidence.

### 6.8 PFlash proxy

Commit:

```text
f00a8b4bd tools: add optional cache-stable PFlash proxy
```

Path:

```text
llama.cpp/tools/pflash/
```

This is **approximate and opt-in**, separate from the lossless llama.cpp server changes. It uses a Qwen3-0.6B BF16 Lucebox scorer to remove query-irrelevant old assistant/tool text on cold long sessions while trying to preserve prefix-cache stability.

Validated cold 48.5k retrieval example:

```text
direct target: 48,512 target tokens, 78.75 s, result A731|B284|C915
PFlash:        36,020 target tokens, ~67.0 s, same result in this test
~15% end-to-end gain
```

Do not include PFlash in a lossless upstream CUDA/server PR. It changes model input and has an external scorer workflow. Keep it fork-only unless separately discussed upstream.

---

## 7. Lossless validation already performed

The promoted Qwen3.8 CUDA stack was validated much more strongly than merely checking one sampled token.

Important prior gates include:

- captured top-100 probability object exactly equal;
- 64-step deterministic replay with identical token sequence, content, and probability structures;
- same 100k production semantic KV/recurrent state + identical +1k continuation;
- raw q8_0 FA output at Q=1000 bit-identical across all 6,144,000 float outputs between the one-CTA and two-CTA variants;
- boundary checks at Q=767/768/1023/1024 exactly equal;
- CUDA sanitizer validation for adaptive FA:
  - synccheck: 0 errors;
  - racecheck: 0 hazards/errors/warnings;
  - memcheck: 0 errors;
- RTX 3060 Ti/Ampere fallback was previously checked on the filtered FA suite.

Two broader FA variants were explicitly rejected:

- larger softmax/rescaling work chunk changed the 64-step trajectory: numerical/quality failure;
- enabling the compact 49,152-byte / 2-CTA launch globally changed probability structures on other scheduling regimes. This led to the conservative adaptive schedule gate instead.

Do not resurrect these rejected variants without redoing the numerical gates.

---

## 8. Main controlled benchmark methodology

For Qwen3.8, Qwen3.5 and Laguna the main comparison is:

```text
100,000 real cached tokens
+ 1,000 identical new prompt tokens
+ 64 generated tokens
```

The same saved semantic state and token sequence are restored into fork/upstream. This removes cache construction and cache selection from the throughput A/B and directly measures the continuation path.

Main summary:

```text
results/resumed-100k-benchmark-20260821/FINAL-BENCHMARK-SUMMARY.md
```

JSON version:

```text
results/resumed-100k-benchmark-20260821/FINAL-BENCHMARK-SUMMARY.json
```

The fresh benchmark session used current GPU power limits 200 W / 300 W.

---

## 9. Main 100k benchmark results

| Model | Hardware | Fork PP | Upstream PP | Fork PP delta | Fork TG | Upstream TG | Fork TG delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | V100 + 3060 Ti | **450.0** | 317.7 | **+41.6%** | 26.65 | 26.48 | +0.6% |
| Qwen3.8-27B | V100 only | **433.1** | 306.7 | **+41.2%** | 23.19 | 23.29 | -0.4% |
| Qwen3.5-122B-A10B | V100 + 3060 Ti | **283.5** | 257.4 | **+10.2%** | 19.80 | 19.83 | -0.1% |
| Qwen3.5-122B-A10B | V100 only | **280.8** | 241.4 | **+16.3%** | 21.30 | 21.29 | ~0.0% |
| Laguna-S-2.1 | V100 + 3060 Ti | 317.4 | **319.5** | -0.6% | 20.81 | **21.24** | -2.0% |
| Laguna-S-2.1 | V100 only | 312.3 | **320.5** | -2.5% | 22.26 | **23.28** | -4.4% |

Interpretation:

- The fork's large PP gain is **not generic**.
- Qwen3.8 gains ~41% PP in both dual and V100-only configurations.
- Qwen3.5 also benefits, but less: ~10-16% PP.
- Laguna is a useful negative control and does not benefit. Its slight fork regression supports the conclusion that the big gain is tied to Qwen-like model/tensor shapes rather than an accidental global speedup.
- TG is mostly unchanged by the prompt-processing optimizations.

---

## 10. Fresh Qwen3.8 benchmark against current vanilla master

After fetching/building current upstream master `bb4caa754`, the most important benchmark was rerun against it.

Path:

```text
results/qwen38-master-bb4caa754-ab/summary.json
```

Workload:

```text
100,000 cached + 1,000 new + 64 generated
V100 + RTX 3060 Ti
same state and sequence
```

Results:

| build | PP | TG | MTP accepted |
|---|---:|---:|---:|
| vanilla master `bb4caa754` | 313.9845 tok/s | 26.1698 tok/s | 37/52 |
| fork `b403a781e` | **447.1078 tok/s** | **26.2016 tok/s** | 37/52 |
| fork delta | **+42.398% PP** | +0.122% TG | identical |

Generated token IDs and content were identical.

This is currently the cleanest top-line number for the project: **~42.4% faster Qwen3.8 continuation PP than current vanilla master, with unchanged TG/output for this deterministic test.**

Note: the measured fork binary here was built before the later local comment/style cleanup. That cleanup does not change the inference logic; only the byte-pointer cast/braces changed executable source semantics in a standards-equivalent way.

---

## 11. Qwen3.8 placement details

### 11.1 Dual GPU: V100 + RTX 3060 Ti

Main model placement:

```text
--gpu-layers all
--split-mode layer
--tensor-split 64,2
--batch-size 4096
--ubatch-size 4096
```

Fork additions:

```text
--prefill-reuse 1024
--pipeline-copies 2
--spec-draft-ubatch 1024
```

Target KV:

```text
--cache-type-k q8_0
--cache-type-v q8_0
```

Draft/MTP:

```text
--cache-type-k-draft f16
--cache-type-v-draft f16
--spec-type draft-mtp
--spec-draft-n-max 2
```

The `64,2` split was the maximum tested safe discrete layer placement with the desired large target graph; moving more weight onto the 3060 Ti OOMed.

100k sequence/state:

```text
results/cache100k-sharedstate-ab/seq.json
results/cache100k-sharedstate-ab/slots/cache100k.bin
```

### 11.2 V100 only

V100 UUID used in scripts:

```text
GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79
```

Tight MTP-safe placement found:

```text
--gpu-layers 63
--ubatch-size 1024
```

`ngl64` did not fit. `ngl63` with larger ubatch also ran out of memory when the MTP graph was allocated, so `ngl63 + ubatch1024` was the tight tested MTP-safe configuration.

Final V100-only Qwen3.8 results:

```text
upstream: PP 306.6845, TG 23.2887
fork:     PP 433.12,   TG 23.19
```

Idle V100 free memory was ~1958 MiB because transient MTP graph allocation still needs substantial headroom.

---

## 12. Qwen3.5 portability details

Qwen3.5-122B-A10B shares important Qwen-like structures:

- 36 scalar-gate GatedDeltaNet layers with `S_v=128`;
- 12 full-attention layers with 256-wide K/V.

Therefore the promoted Volta GDN and FA kernels also apply.

### Dual GPU final

Exact saved state/sequence:

```text
results/model-portability-100k/qwen35-shared/slots/cache100k.bin
results/resumed-100k-benchmark-20260821/qwen35/seq.json
```

Final manual fit arguments:

```text
results/resumed-100k-benchmark-20260821/qwen35/manual-fit/manual-packed-plus-2gate.args
```

Important operational result: `--load-mode none` was materially better than mmap + CPU overrides for this model.

Final:

```text
upstream: PP 257.368923, TG 19.825560
fork:     PP 283.512126, TG 19.801663
```

Fork used:

```text
--prefill-reuse 1024
--pipeline-copies 2
```

### V100 only final

Fit data:

```text
results/resumed-100k-benchmark-20260821/v100-only-fit/qwen35-ub1024/
```

Final:

```text
upstream: PP 241.36, TG 21.29
fork:     PP 280.77, TG 21.30
```

Interesting hardware result: dual-GPU Qwen3.5 TG was slower than V100-only TG (~19.8 vs ~21.3), likely because the cross-GPU overhead is not worth it for decode.

---

## 13. Laguna negative-control details

Laguna has 128-wide attention and no GatedDeltaNet, so the promoted Qwen-like Volta FA/GDN paths do not apply.

Sequence:

```text
results/model-portability-100k/laguna/seq.json
```

Laguna has hybrid SWA=512. Saved slot token IDs exist, but restoring the desired prefix across a server restart was not sufficiently reliable for this model, so the final benchmark primed 100k then appended +1k in the same process.

### Dual GPU

Manual `m512` placement, `--load-mode none`.

```text
upstream: PP 319.478203, TG 21.236282
fork:     PP 317.433411, TG 20.805331
```

### V100 only

`m384`, ubatch 1024.

```text
upstream: PP 320.503267, TG 23.277235
fork:     PP 312.339243, TG 22.262050
```

Forcing the Qwen-specific prefill/pipeline knobs made Laguna worse, so the final portability comparison intentionally did **not** force them.

This model is important because it demonstrates that the fork is not simply making every CUDA workload faster.

---

## 14. GLM-5.2 orientation benchmark

GLM real 100k semantic priming was attempted and was impractically slow. A previous real prime reached roughly 69k before being stopped. The user explicitly allowed switching to **10,000 real cached tokens + 1,000 new + 64 generated** for orientation.

Do not compare these GLM values directly to the 100k Qwen/Laguna table.

Do not benchmark GLM on RTX 3060 Ti alone unless the user explicitly changes that instruction.

Detailed summary:

```text
results/resumed-100k-benchmark-20260821/GLM52-10K-ORIENTATION.md
results/resumed-100k-benchmark-20260821/GLM52-10K-ORIENTATION.json
```

Shared real semantic state:

```text
results/resumed-100k-benchmark-20260821/glm52-10k/seq11k.json
results/resumed-100k-benchmark-20260821/glm52-10k/shared-slot/cache10k.bin
```

### Final GLM 10k results

| Hardware | MTP | Fork PP | Upstream PP | Fork PP delta | Fork TG | Upstream TG | Fork TG delta |
|---|---|---:|---:|---:|---:|---:|---:|
| V100 + 3060 Ti | off | 45.9990 | 46.0281 | -0.06% | 4.4721 | 4.4041 | +1.54% |
| V100 + 3060 Ti | on | 44.4423 | 44.6807 | -0.53% | 6.2861 | 6.2607 | +0.41% |
| V100 only | off | 45.9438 | 45.9113 | +0.07% | 4.7768 | 4.5703 | +4.52% |
| V100 only | on | 44.3439 | 44.5076 | -0.37% | 6.2506 | 6.2284 | +0.36% |

All eight final measured cells generated identical 64-token output/content.

MTP acceptance in every MTP cell:

```text
41 / 42 = 97.619%
```

### MTP effect on GLM

Dual GPU:

```text
upstream: PP -2.93%, TG +42.16%
fork:     PP -3.38%, TG +40.56%
```

V100 only:

```text
upstream: PP -3.06%, TG +36.28%
fork:     PP -3.48%, TG +30.85%
```

Conclusion: **MTP is strongly worthwhile on GLM**, giving roughly +31% to +42% TG for only ~3% PP cost.

Fork PP is effectively neutral on GLM, reinforcing that the main CUDA gains are Qwen/model-shape specific.

The 3060 Ti contributes almost nothing to GLM at this 10k workload. V100-only can even be slightly better for no-MTP TG.

### GLM final placements

Dual no-MTP:

```text
ubatch 4096
fit m768
57/22 layer split
post-warm free ~1085 MiB 3060 Ti / ~3066 MiB V100
```

A tighter m512 55/24 configuration could load and had a short ~49.9 tok/s warmup, but it failed/crashed during the real longer workflow. m768 is the stable selected placement.

Dual MTP:

```text
ubatch 4096
fit m3072
69/10 layer split
post-warm free ~1345 MiB 3060 Ti / ~294 MiB V100
```

V100-only no-MTP:

```text
ubatch 4096
fit m1024
post-warm free ~868 MiB
```

V100-only MTP:

```text
ubatch 4096
m3072 loaded but OOMed when the actual MTP graph was allocated
m3584 was the tightest tested working placement
post-warm free ~288 MiB
```

Final canonical GLM result files:

```text
results/resumed-100k-benchmark-20260821/glm52-10k/dual-nomtp-upstream/result.json
results/resumed-100k-benchmark-20260821/glm52-10k/dual-nomtp-fork/result.json
results/resumed-100k-benchmark-20260821/glm52-10k/dual-mtp-upstream/result.json
results/resumed-100k-benchmark-20260821/glm52-10k/dual-mtp-fork/result.json
results/resumed-100k-benchmark-20260821/glm52-10k/v100-nomtp-upstream/result-final-m1024.json
results/resumed-100k-benchmark-20260821/glm52-10k/v100-nomtp-fork/result-final-m1024.json
results/resumed-100k-benchmark-20260821/glm52-10k/v100-mtp-upstream-clean-m3584/result-final.json
results/resumed-100k-benchmark-20260821/glm52-10k/v100-mtp-fork/result-final.json
```

### Synthetic GLM 100k state - do not use for TG/quality

A helper was created to synthesize an occupied 100k DSA KV state:

```text
fake-kv-state.cpp
./fake-kv-state
results/resumed-100k-benchmark-20260821/glm52-fakekv/cache100k-fake.bin
```

It restores successfully and is useful only for valid 100k memory/PP geometry experiments. Its fake KV values make MTP acceptance and token generation meaningless. Do **not** report TG/quality from this state.

---

## 15. Benchmark scripts / commands

Primary permanent documentation:

```text
llama.cpp/benches/v100-qwen38/v100-qwen38.md
```

### Build/rebuild the exact Qwen3.8 100k real state

From the fork checkout:

```bash
./benches/v100-qwen38/qwen38-prepare-cache.sh
```

The sequence JSON must already contain at least the required token IDs. The script processes exactly the first 100,000 real tokens and saves `cache100k.bin`.

### Dual-GPU Qwen3.8 fork-vs-upstream A/B

```bash
./benches/v100-qwen38/qwen38-100k-ab.sh
```

Environment-variable overrides are supported for model path, sequence, slot directory, fork/upstream binary, result directory, and port.

### V100-only Qwen3.8 A/B

```bash
./benches/v100-qwen38/qwen38-v100-only-ab.sh
```

### Prepare GLM real 10k state

```bash
./benches/v100-qwen38/glm52-10k-prepare-cache.sh
```

### Generic GLM 10k runner

```bash
./benches/v100-qwen38/glm52-10k-run-one.sh MODE BIN FIT_ARGS HW MTP UBATCH PORT
```

Examples and the actual fit paths are documented in `v100-qwen38.md`.

### Important historical /tmp scripts

The permanent scripts above were created from the actual development runners. If needed, historical raw versions may still exist in `/tmp`, including:

```text
/tmp/resume_bench_qwen38.sh
/tmp/qwen38_v100_only_ab_final.sh
/tmp/qwen38_dual_upstream_tune.sh
/tmp/qwen38_split4096.sh
/tmp/glm10k_run_one.sh
/tmp/glm10k_prime_dual_nomtp_m768.sh
```

Prefer the permanent `benches/v100-qwen38/` versions going forward.

---

## 16. Build commands

Validated mixed V100 + RTX 3060 Ti build:

```bash
cmake -S llama.cpp -B build-rebased \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES='70;86'
cmake --build build-rebased -j "$(nproc)"
```

Current vanilla master build:

```bash
cmake -S llama.cpp-upstream-current -B build-upstream-current \
  -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES='70;86'
cmake --build build-upstream-current --target llama-server -j "$(nproc)"
```

Current binaries:

```text
fork:    build-rebased/bin/llama-server
vanilla: build-upstream-current/bin/llama-server
fit:     build-upstream-current/bin/llama-fit-params
```

Current version strings at last check:

```text
fork:    0.1.2-dev, commit b403a781e
vanilla: 0.2.0-dev, commit bb4caa754
```

The fork binary version does not include the dirty working-tree state in the commit string.

---

## 17. Current code-quality / PR-readiness findings

Detailed persistent notes:

```text
llama.cpp/benches/v100-qwen38/pr-readiness.md
```

### llama.cpp contribution rules learned

Current upstream `AGENTS.md` / `CONTRIBUTING.md` emphasize:

- contributor must understand every submitted line and maintain it;
- simpler/less invasive changes are strongly preferred;
- open/discuss significant features before large PRs;
- avoid extra dependencies/files/API surface unless justified;
- comments should usually be 1-2 concise lines and explain a non-obvious invariant rather than experiment history;
- use simple English/ASD-STE100 style;
- use ASCII in source/comments (`->`, `x`, `...` rather than Unicode arrows/multiplication/ellipsis);
- follow surrounding 4-space/brace/snake_case style;
- public API uses sized integer types;
- do not bundle unrelated changes;
- server changes should fit `tools/server/README-dev.md` scope;
- AI-generated PR descriptions/reviewer replies are prohibited;
- autonomous agents must not push/create a PR;
- PR template requires AI usage disclosure.

Relevant code ownership:

```text
ggml CUDA:   @ggml-org/ggml-cuda
llama server: @ggml-org/llama-server
llama common: @ggml-org/llama-common
```

### Current code-quality assessment

Strongest parts:

- cache-prefix selection: tiny, understandable bug fix with regression test;
- GDN kernel: isolated backend-only specialization, tight hardware/shape guard;
- basic Volta 256x256 FA Q/shared + K/V scratch tuning: backend-only and performance-motivated.

Main concerns:

- branch as a whole mixes too many unrelated features and cannot be a single PR;
- adaptive 2-CTA FA currently has duplicated metadata-combine code, which increases maintenance burden;
- prefill-reuse/pipeline-copy tuning expands CLI/public/backend scheduler API and is too machine-specific in current form;
- recurrent checkpoint value eviction is heuristic/large and needs much stronger focused tests;
- draft-ubatch needs dedicated multi-sequence/chunk tests;
- PFlash is outside a clean lossless upstream PR.

---

## 18. Recommended PR decomposition

Do **not** upstream this branch as one PR.

Suggested order:

### PR candidate 1 - server absolute-prefix cache selection

Scope:

```text
tools/server/server-task.cpp
tools/server/tests/unit/test_kv_keep_only_active.py
```

Why first:

- small bug fix;
- architecture-independent;
- simple behavioral rationale;
- new regression test;
- entire neighboring test file now passes 3/3.

Related upstream long-agent cache/recurrent problem area includes issue #22746.

### PR candidate 2 - basic Volta 256x256 FA tuning

Start with only the simpler changes:

- dedicated Volta 256x256 config;
- Q in shared memory;
- K96/V64 scratch;
- forward split-K order preserving the original MMA accumulation order.

Do **not** include adaptive 2-CTA occupancy in the first FA PR if avoidable.

Required validation:

- isolated performance;
- `test-backend-ops` 256x256 q8_0 cases on V100;
- Ampere fallback;
- Qwen3.8 deterministic probability/token equality;
- Qwen3.8 fastest 100k+1k benchmark.

### PR candidate 3 - Volta scalar-gate GDN 128x4 kernel

Independent backend-only PR.

Required validation:

- GDN correctness on V100 and fallback GPU;
- prefill-specific `n_tokens=64/256/512/1024` cases need to be explicitly included in test mode, not only performance-mode tests;
- Qwen3.8 and Qwen3.5 whole-model performance;
- Qwen3.8 exactness gate.

### PR candidate 4 - adaptive FA 2-CTA occupancy

Only after the basic FA tuning is accepted/discussed.

Need:

- simplify duplicated combine code if possible;
- synccheck/racecheck/memcheck;
- scheduling boundary tests;
- Q=767/768/1023/1024 exactness;
- Qwen3.8 fastest benchmark.

### PR candidate 5 - independent MTP draft ubatch

Generic speculative-decoding change.

Need tests for multi-sequence/chunk behavior before considering it ready.

### Later / redesign before upstream

- prefill-reuse + pipeline-copy controls: useful locally but current public/API design is not upstream-clean;
- recurrent checkpoint retention/value eviction: real problem, but split from the simple cache-selection fix and add tests;
- PFlash: keep fork-only.

---

## 19. PR-readiness tests already run

Raw logs:

```text
results/pr-readiness-20260821/
```

### Build/style

- full fork source rebuilt successfully after local style cleanup;
- `git diff --check` passes;
- new benchmark shell scripts pass `bash -n`;
- added source lines checked for accidental Unicode in the cleanup.

### FlashAttention backend correctness

Filtered 256x256 q8_0 cases:

```text
V100:        10/10 passed
RTX 3060 Ti: 10/10 passed
```

Logs:

```text
results/pr-readiness-20260821/v100-fa256.log
results/pr-readiness-20260821/rtx3060ti-fa256.log
```

Note on backend test naming: `test-backend-ops` enumerates backend devices as:

```text
CUDA0 = Tesla V100-SXM2-32GB
CUDA1 = NVIDIA GeForce RTX 3060 Ti
```

This is opposite the `nvidia-smi` index labels used elsewhere on the host. Do not confuse them.

### GatedDeltaNet backend correctness

Current filtered `head_size=128` test-mode cases:

```text
V100:        2/2 passed
RTX 3060 Ti: 2/2 passed
```

Logs:

```text
results/pr-readiness-20260821/v100-gdn128.log
results/pr-readiness-20260821/rtx3060ti-gdn128.log
```

Caveat: current test-mode cases only hit a small subset. The realistic 64/256/512/1024 prefill cases are currently in the performance-case generator, so a PR should add/ensure real test-mode coverage of the specialized prefill path.

### Server cache regression

Official llama.cpp Python test harness was set up in:

```text
/workspace/oai-qwen38-pp-lab/.venv-server-tests
```

The new test initially failed because it constructed ~725 tokens while TinyLlama preset uses context 512. The test was fixed with:

```python
server.n_ctx = 1024
```

Final result:

```text
unit/test_kv_keep_only_active.py: 3/3 passed
```

This includes the two pre-existing neighboring tests plus the new absolute-prefix regression.

---

## 20. Important recurrent-state test finding

`test-recurrent-state-rollback` was run against Qwen3.8 with `--ctx-size 1024` to avoid meaningless ~16 GiB KV allocation per test context.

The dirty-context restore check fails on the fork at:

```text
position 6, token 0
3.36596 != 6.54106
```

The **identical test against current vanilla master `bb4caa754` fails at exactly the same position with exactly the same values**.

Therefore:

- this is not a regression introduced by the fork;
- it is useful evidence of an upstream Qwen3.8 recurrent-state restore limitation;
- it is relevant background when working on the recurrent/checkpoint problem and issue #22746;
- do not hide or "fix" the test in the fork just to make the suite green.

Logs:

```text
results/pr-readiness-20260821/recurrent-qwen38-c1024.log
results/pr-readiness-20260821/recurrent-qwen38-current-master.log
```

---

## 21. Useful performance observations learned

### Qwen-like shape specificity

The largest gains correlate with models using the promoted tensor structures:

- scalar-gate GDN with state width 128;
- full attention with 256-wide K/V on Volta;
- long context where FA dominates continuation processing.

Qwen3.8 gets ~41-42% PP improvement; Qwen3.5 gets ~10-16%; Laguna/GLM do not benefit.

### 3060 Ti contribution is workload/model dependent

Qwen3.8:

- dual vs V100-only fork PP: roughly +3.9%;
- dual vs V100-only fork TG: roughly +14.9%.

Qwen3.5 and Laguna:

- dual GPU TG is slower than V100-only due to cross-GPU overhead.

GLM:

- 3060 Ti contributes almost nothing to the 10k workload.

### MTP is especially valuable on GLM

GLM TG improves roughly +31-42% for only ~3% PP cost. Keep MTP enabled for GLM unless there is a model-quality reason not to.

### Large physical ubatch is not automatically lossless

The Qwen3.8 prefill-reuse strategy was designed to preserve baseline cuBLAS GEMM width precisely because simply increasing ubatch can change cuBLAS kernels/math and output probabilities.

On Qwen3.5:

```text
4096-token PP:
~368.45 -> 464.82 tok/s with reuse (+26.2%)
but top-100 TV ~1.8e-5
plain ubatch4096 ~618.26 tok/s but TV ~6.8e-5
```

Laguna showed even larger drift (~0.0128 TV) in the generic large-ubatch experiment.

Do not generalize Qwen3.8's exactness claim to other models without checking probability/state equality.

---

## 22. Current recommended production Qwen3.8 command

The root README contains the fully documented command. The validated dual-GPU recipe is approximately:

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

The manual `64,2` split is specific to this V100 32 GB + RTX 3060 Ti 8 GB setup. Do not copy it blindly to another GPU pair.

---

## 23. Things that should not be repeated unless needed

To save future context/time:

- Do not rerun GLM 100k real semantic priming unless explicitly requested. The user accepted 10k orientation because 100k is disproportionately slow.
- Do not benchmark GLM on RTX 3060 Ti alone.
- Do not use the synthetic GLM 100k KV state for TG/quality/MTP acceptance.
- Do not force Qwen-specific prefill/pipeline knobs onto Laguna for portability claims; they made it worse.
- Do not assume an old stale benchmark job is active just because a sandbox job record says "running". Check real processes and GPU allocations. Previous duplicate/stale jobs often left zombie metadata while no server was actually running.
- Do not submit/push an upstream PR automatically. llama.cpp's current AI policy forbids automated PR submission and AI-written reviewer responses.
- Do not combine PFlash with the lossless CUDA PR story.

---

## 24. What to do next when resuming after context loss

Recommended immediate sequence:

1. Read this file.
2. Read:

```text
llama.cpp/benches/v100-qwen38/pr-readiness.md
llama.cpp/benches/v100-qwen38/v100-qwen38.md
results/resumed-100k-benchmark-20260821/FINAL-BENCHMARK-SUMMARY.md
```

3. Check current git state:

```bash
git -C llama.cpp status --short --branch
git -C llama.cpp log -5 --oneline --decorate
git -C llama.cpp log -1 --oneline upstream/master
```

4. Decide which **single PR candidate** the user wants to work on first. The safest first candidate is the absolute-prefix server cache-selection fix.
5. For a CUDA candidate, create a clean branch/worktree containing only that isolated change rather than trying to prune the deployment fork in place.
6. Rebase/cherry-pick the candidate onto current `upstream/master`.
7. Run the candidate-specific correctness tests.
8. If the source change can affect performance, rerun the fastest/most important Qwen3.8 100k+1k A/B using `benches/v100-qwen38/qwen38-100k-ab.sh`.
9. For CUDA math/scheduling changes, rerun deterministic probability/token exactness gates and sanitizer tests as appropriate.
10. Let the human contributor inspect/write the final commit/PR text according to llama.cpp's AI policy.

---

## 25. Suggested first PR workflow: cache-prefix bug fix

If the user chooses the server cache-selection PR, isolate only:

```text
tools/server/server-task.cpp
tools/server/tests/unit/test_kv_keep_only_active.py
```

Use the small `eb119fdbf` change as the logical starting point, plus the current local test fix `server.n_ctx = 1024`.

Validation already known:

```text
unit/test_kv_keep_only_active.py: 3/3 passed
```

Then check current upstream for any issue/PR duplication before submission. The final human-authored PR should explain the concrete failure mode: a short live prompt with excellent retention ratio can block a much deeper cached branch even though restoring the deeper branch saves far more prefill tokens.

Do not copy an AI-generated PR description verbatim; llama.cpp explicitly disallows AI-written PR text.

---

## 26. Suggested first CUDA PR workflow: basic Volta 256x256 FA

If the user chooses CUDA instead, start from a clean current-master branch and extract only the smaller FA tuning before adaptive occupancy.

Candidate conceptual pieces:

- dedicated sm70 256x256 config with `Q_in_reg=false`;
- K scratch 96, V scratch 64;
- forward split-K order so MMA accumulation remains equivalent to the original one-piece K traversal.

Try to exclude:

- adaptive two-CTA scheduling;
- large duplicated metadata-combine restructuring;
- unrelated GDN, server, scheduler and prefill-reuse changes.

Then validate:

1. build sm70 + sm86;
2. filtered V100 256x256/q8 tests;
3. RTX 3060 Ti fallback;
4. isolated kernel benchmark;
5. Qwen3.8 deterministic exactness;
6. Qwen3.8 100k+1k performance A/B.

This should produce a far more reviewable PR than trying to upstream the final deployment kernel in one step.

---

## 27. Final concise state summary

At handoff:

- Optimized Qwen3.8 fork is committed/pushed through `b403a781e` on branch `qwen38-lossless-agent-cache`.
- Local working tree has uncommitted README, comment/style, test and benchmark-reproducibility cleanup.
- Fresh upstream master is `bb4caa754`; none of its 20 new commits overlap fork-modified source files.
- Fresh current-master Qwen3.8 A/B: **313.98 -> 447.11 tok/s PP (+42.40%)**, TG unchanged, identical generated tokens/content and MTP 37/52.
- Main 100k portability matrix: Qwen3.8 +41%, Qwen3.5 +10-16%, Laguna slightly negative.
- GLM 10k orientation: fork neutral; MTP strongly beneficial (+31-42% TG for ~3% PP cost).
- Server cache absolute-prefix test file passes 3/3.
- Filtered 256x256 q8 FA tests pass 10/10 on V100 and 10/10 on RTX 3060 Ti.
- Basic GDN head-size-128 correctness tests pass 2/2 on each GPU, but specialized prefill-path test coverage should be improved before PR.
- Qwen3.8 dirty-context recurrent rollback fails identically on fork and vanilla master, so it is an upstream limitation, not a fork regression.
- Best first upstream candidate: **server absolute-prefix cache selection**. Best CUDA candidates after that: **basic Volta 256x256 FA tuning**, then **Volta GDN**.

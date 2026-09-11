# Gemma 4 V100 optimization notes — 2026-09-11

## Goal

Evaluate Gemma 4 on the `v100-optimized` fork, with the Tesla V100 32 GB as the primary target and V100 + RTX 2080 Ti as a secondary target. The main workload is agent-style long-context inference: **100,000 cached tokens + 1,000-token append**, with Q8 K/V and 64-token decode also recorded.

The user will download Gemma 4 26B-A4B. Do **not** download the 26B model from the sandbox. Until it appears under `/models`, use the already-downloaded Gemma 4 31B as the dense Gemma control.

## Models

### Gemma 4 31B

File:

`/models/Gemma4-31B/gemma-4-31B-it-UD-Q4_K_XL.gguf`

Size: 18,822,970,304 bytes on disk; llama reports 18,807,134,448 model bytes and 30,697,345,596 parameters.

Important GGUF/model geometry:

- architecture: `gemma4`
- context length: 262,144
- layers: 60
- embedding width: 5,376
- attention heads: 32
- K/V head dimension: 512
- K/V heads: mostly 16, with every sixth layer using 4
- sliding window: 1,024
- shared-KV layers: 0
- FFN width: 21,504
- dense model, so `GGML_CUDA_VOLTA_FORCE_MMQ=moe` should not affect it

This is substantially different from the Qwen3.8 D256/GQA6 geometry that received the strongest existing fork attention tuning. D512 Gemma attention is therefore a likely optimization target.

## Fair build setup

Both comparison builds were freshly rebuilt before Gemma benchmarking.

Fork:

- repo: `/workspace/llama-v100-optimized`
- branch: `v100-optimized`
- commit at start of Gemma work: `d3c996f22` (`docs: split build commands into separate lines`)
- build: `build-compare-sm70-75`

Upstream runtime control:

- repo: `/workspace/llama-upstream-head-0911`
- commit: `43f3dda62`
- build: `build-sm70-75`

Matched CMake configuration on both:

- `CMAKE_BUILD_TYPE=Release`
- `GGML_CUDA=ON`
- `GGML_CUDA_GRAPHS=ON`
- `CMAKE_CUDA_ARCHITECTURES=70;75`
- `LLAMA_BUILD_UI=OFF`
- `GGML_CUDA_FORCE_MMQ=OFF`

The README uses the same SM70+SM75 normal build and recommends selective `GGML_CUDA_VOLTA_FORCE_MMQ=moe` only for routed MoE expert matmuls. Do not globally force MMQ for the dense 31B comparison.

## Early measurements — NOT the final fair comparison

These are diagnostic only and must not be presented as upstream-vs-fork deltas.

### Fork smoke test

Gemma 4 31B Q4_K_XL, single V100, Q8 K/V, FA on, full GPU offload, batch 4096 / ubatch 1024:

- 1,024-token PP: **777.08 tok/s**
- roughly 20.4 GiB VRAM during the smoke test

This was a one-repetition fork-only smoke run.

### Incomplete upstream 16k ubatch sweep

Single V100, Q8 K/V, batch 4096, 16,384-token PP, 4 repetitions:

- ubatch 512: **515.06 tok/s**, sd 2.28
- ubatch 1024: **597.62 tok/s**, sd 0.65

The sweep was intentionally stopped before ubatch 2048/4096 and before the fork arm, because the user prioritized the 100k+1k workload. Therefore these numbers are **not** a fair fork-vs-upstream comparison.

## 100k + 1k benchmark methodology

Harness:

`/models/.bench-gemma31-harness.py`

Large cache files/logs live under `/models/.bench-gemma31/` because `/workspace` is nearly full. The small notes/results directory remains in the repo.

Matched server settings planned for both implementations:

- single Tesla V100 32 GB
- `CUDA_VISIBLE_DEVICES=GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79`
- `--ctx-size 106496`
- `--parallel 1`
- `--fit off`
- `--gpu-layers all`
- `--device CUDA0`
- `--split-mode none`
- `--flash-attn on`
- `--batch-size 4096`
- `--ubatch-size 1024`
- `--cache-type-k q8_0`
- `--cache-type-v q8_0`
- `--cache-ram 0`
- `--no-warmup`
- `--perf`
- MTP disabled
- no fork-only CLI tuning
- experiment environment variables cleared for both arms

The deterministic 101k token sequence is generated once through Gemma's own `/tokenize` endpoint from a repeated fixed sentence and saved as `/models/.bench-gemma31/seq101k.json`. Both implementations consume the exact same token IDs.

Each implementation creates its **own** 100k saved KV state from that identical token sequence, avoiding cross-version cache serialization assumptions. The benchmark then restores that state and submits the same first 101k tokens, so the server should report `cache_n=100000`, `prompt_n=1000`.

The requested benchmark measures:

- 1,000-token append prompt-processing throughput
- wall TTFT
- 64-token decode throughput
- end-to-end wall time
- generated-token hash for control

## Upstream 100k prime result

The first full upstream prime completed successfully and remained fully resident on the V100 (~25.0 GiB VRAM observed during the run).

Upstream `43f3dda62`, Gemma 4 31B Q4_K_XL, Q8 K/V, batch 4096 / ubatch 1024:

- full 100,000-token cold prime: **323.7617 tok/s**
- prompt time: **308.869 s**
- saved KV state: **4,799,258,588 bytes**
- slot save time: **2.381 s**

This is **not** the 100k+1k result; it is the cost of producing the 100k baseline cache.

Observed cumulative prime throughput illustrates strong context-length scaling:

- 4,096: 774.31 tok/s
- 16,384: 622.41 tok/s
- 32,768: 525.10 tok/s
- 65,536: 404.32 tok/s
- 98,304: 329.41 tok/s
- final 100,000: 323.76 tok/s

This makes the restored-cache 100k+1k benchmark especially important.

## Existing fork changes relevant to Gemma

The current fork differs from upstream in several CUDA areas, but many guards target Qwen-specific shapes. Relevant observations before making any new Gemma changes:

- `fattn-mma-f16.cuh` already has explicit Volta D512 configurations for ncols 8/16/32/64.
- Existing long-Q8 and INT8-QK specializations are primarily D256/Qwen guarded and therefore should not directly accelerate Gemma D512.
- Volta D256 Qwen-specific dispatch changes in `fattn.cu` do not match Gemma D512.
- The fork contains generic norm fusion work that may benefit Gemma if its graph patterns match.
- Volta Q6_K MMQ register-pressure tuning exists, mainly relevant once 26B-A4B quant sweeps include Q6.
- `GGML_CUDA_VOLTA_FORCE_MMQ=moe` is relevant to 26B-A4B routed experts, not dense 31B.

## Optimization experiment log

No new Gemma-specific optimization has been applied yet. Add every attempted change below, including negative results, with exact commit/patch, benchmark geometry, result, and whether it was kept or reverted.

| Experiment | Target | Result | Decision |
|---|---|---|---|
| Baseline only | Gemma 31B D512 | in progress | establish fair upstream/fork controls first |

## Qwen tricks mapped to Gemma

### 1. Volta FlashAttention tile/config tuning — highest priority for Gemma 31B

Qwen gain: `d3c14522d` added a measured SM70 D256 32-column configuration and shape-specific dispatch; later long-context work relaxed/tuned compact dispatch for the real cached-append geometry.

Gemma mapping:

- Current fork has **no corresponding Gemma D512-specific tuning relative to upstream**.
- Gemma 31B has two materially different attention geometries. Its global long-context layers are D512 with GQA8; its sliding-window layers use the smaller SWA geometry and only see a 1,024-token window.
- Current Volta dispatch chooses `ncols2=8` for GQA8 globally. The D512 Volta config table exists, but is still essentially generic and the source literally retains `TODO tune specifically for Volta` after the explicit rows.
- This makes D512/GQA8 global layers the leading explanation for the strong context-length scaling seen during the 100k prime.

Experiments to try after the fair baseline:

1. D512/GQA8 long-context `ncols2` sweep: 8 (control), 4, 2, possibly 1 where legal.
2. For each layout, sweep the D512 Volta config parameters (`nthreads`, `nbatch_fa`, `nbatch_K2`, `nbatch_V2`, `nbatch_combine`, occupancy/Q-in-reg where legal), guided by register/shared-memory occupancy.
3. Test compact/2-CTA scheduling at D512 for 128..1000-token appends and >=64k KV rather than relying on the generic occupancy heuristic.
4. Validate each candidate with backend CPU-reference attention tests before keeping it.

### 2. Direct/tighter Q8 KV attention path — very high priority for 100k+1k

Qwen gain: `0fc400871` added long-Q8 attention work, including keeping Q8 K data compact longer and specialized Tensor-Core paths instead of paying generic expansion/staging costs. The strongest existing special cases are D256/Qwen-shaped and do not match Gemma's D512 global layers.

Gemma mapping:

- The requested workload uses Q8 K/V and 100k cached tokens.
- Only the global layers pay the full 100k attention span, but each global head is very wide (D512), so K/V traffic and conversion cost are large.
- A D512 Q8 path that consumes q8 tiles directly or performs fused q8->FP16 tile conversion in shared memory may be more valuable than optimizing the 1K-window layers.
- Extending the Turing INT8-QK approach directly to V100 is not possible in the same form because Volta lacks Turing's INT8 Tensor Core MMA; on V100 the likely win is compact Q8 storage/load plus FP16 Tensor-Core MMA, not INT8 Tensor-Core QK.

Candidate: implement/benchmark a D512/GQA8 Volta Q8 specialization only after profiling shows Q8 conversion/load is significant.

### 3. GQA layout specialization — high priority

Qwen gain: for long-context TP shards we stopped blindly using the generic GQA tile shape when it wasted work; a smaller `ncols2` was much faster despite looking worse under the generic occupancy heuristic.

Gemma mapping:

- Gemma's global layers are naturally GQA8, exactly a geometry where current Volta dispatch hard-selects an 8-wide group.
- With D512, grouping eight Q heads per tile is much heavier than with D256. A smaller group may reduce register/shared-memory pressure and increase useful residency.
- This is cheap to test first because it can be exposed as an experimental dispatch switch without rewriting the kernel.

This is the **first optimization experiment to run** once baseline numbers are locked.

### 4. Batch/ubatch tuning — useful but not the core optimization

Qwen gain: measured hardware/model-specific batch defaults avoided generic settings that left performance on the table.

Gemma observation already measured upstream at 16k:

- ub512: 515.06 tok/s
- ub1024: 597.62 tok/s (+16.0%)

The 2048/4096 sweep was stopped before completion to prioritize 100k+1k, so 1024 is the current fair control, not claimed as optimal. External Gemma 4 measurements also report strong hardware-specific batch sensitivity. The final Gemma defaults should therefore be derived from a V100 sweep, not copied from Qwen. 

### 5. Quantized matmul dispatch — important for the 26B quant sweep

Qwen/Turing gain: large-N dense quantized matmuls were sent to dequantize->FP16 cuBLAS once that overtook MMQ; Q5/Q6 also received targeted kernel work. On V100, globally forcing MMQ hurt dense Qwen badly.

Gemma 31B Q4_K_XL:

- dense Q4 matmul should be profiled, but current 100k scaling strongly points to attention as the first target.
- We should still sweep MMQ vs cuBLAS crossover by operation shape after attention is understood.

Gemma 26B-A4B:

- routed experts change the answer: selective `GGML_CUDA_VOLTA_FORCE_MMQ=moe` is directly relevant.
- The wide quant ladder (IQ2/IQ3/IQ4/Q2/Q3/Q4/Q5/Q6/Q8/MXFP4_MOE) lets us identify quant-specific kernel cliffs instead of assuming one dispatch policy.
- MXFP4_MOE deserves its own path analysis; our earlier fork already contains V100 recurrent/MXFP4 decode work, but Gemma's expert shapes may differ from Ornith.

### 6. Norm / elementwise / graph fusion — medium priority, potentially unusually useful on Gemma

The fork contains generic RMS/scale/add fusion work from Qwen/recurrent optimization. Gemma performs many more normalization-like operations around attention and FFN than a simple decoder block: Q norm, K norm, V RMS norm, attention post norm, FFN pre/post norms, and for 26B the shared-MLP + MoE branches add more.

Potential Gemma-specific fusions:

- K/V normalization + RoPE staging where safe;
- residual add -> RMS norm -> weight multiply patterns;
- Gemma MoE router's `RMS norm -> scale -> elementwise router scale -> matmul` chain;
- post-attention/post-FFN residual/norm sequences.

These are likely smaller than the D512 long-attention opportunity for 31B, but may matter more on 26B-A4B because its active expert compute is lower and fixed graph overhead becomes a larger fraction.

### 7. Gemma-only graph TODOs — investigate after profiling

`src/models/gemma4.cpp` contains two explicit optimization TODOs relevant to prompt processing:

- strip unused token rows after the last KV layer when possible;
- improve per-layer embedding handling.

31B may not exercise the per-layer-embedding path, whereas 26B-A4B may. These should be profiled rather than assumed important.

### Working priority

For **31B / V100 / 100k+1k**:

1. lock fair upstream/fork baseline;
2. profile global D512/GQA8 attention vs SWA layers;
3. sweep GQA `ncols2` and D512 Volta tile config;
4. investigate compact/fused Q8 K/V staging for D512;
5. only then chase dense Q4 matmul and graph/norm fusion.

For **26B-A4B** once downloaded:

1. fair baseline with selective `MMQ=moe` on the fork and matched normal upstream control, plus a dispatch-policy control if needed;
2. separate global-attention time from expert-FFN time;
3. quant-family sweep to find outliers;
4. tune expert MMQ/MXFP4 paths and Gemma router/norm graph;
5. reuse any proven D512/global-attention work if the 26B geometry matches.

## Next steps

1. Prime the fork's own 100k KV state from the same `seq101k.json`.
2. Run matched 100k cached + 1k append + 64 decode on upstream and fork, preferably with mirrored process order to quantify drift.
3. Record fair V100 baseline delta before making Gemma-specific changes.
4. Once Gemma 26B-A4B appears in `/models`, inspect its metadata and benchmark its MoE path with `GGML_CUDA_VOLTA_FORCE_MMQ=moe` using the same fair-build methodology.
5. Profile Gemma-specific hotspots (D512 attention first; then quantized matmul, RMS/norm and MoE for 26B-A4B).
6. For each optimization candidate: benchmark -> record -> keep only if repeatable and correctness-safe; otherwise revert and document.
7. After single-V100 work, test V100 + RTX 2080 Ti as the secondary topology.

## Benchmark correction: use llama-bench depth mode

The first server-slot 100k+1k attempt was INVALID: after `/slots/0?action=restore`, Gemma reprocessed the full ~101k prompt instead of reporting `cache_n=100000,prompt_n=1000`. It was stopped and must not be used as a 100k+1k result.

`llama-bench` has native depth support. Source inspection (`tools/llama-bench/llama-bench.cpp` around the benchmark loop) confirms that with `-d 100000 -p 1000`, it:
1. clears memory,
2. computes the 100000-token depth state before timing,
3. snapshots it with `llama_state_seq_get_data`,
4. restores that state for repeated samples,
5. starts the timer only after the depth state is ready,
6. runs the 1000-token prompt append inside the timed region.

This is therefore the preferred reproducible agentive-workload benchmark. Use identical `-d 100000 -p 1000`, Q8 K/V, batch/ubatch, V100 placement, FlashAttention and repetitions for upstream and fork.

### Canonical 31B upstream V100 100k-depth + 1k append sweep

Model: `/models/Gemma4-31B/gemma-4-31B-it-UD-Q4_K_XL.gguf`
Upstream: `43f3dda62`, fair Release SM70+SM75 build.
Command shape: `llama-bench -d 100000 -p 1000 -n 0 -b 4096 -ub <sweep> -ctk q8_0 -ctv q8_0 -fa on -sm none -ngl 999 -dev CUDA0 -r 5 --no-warmup`.
No fork-only performance env vars were present.

| ubatch | PP tok/s | time for 1k |
|---:|---:|---:|
| 512 | 178.5385 | 5.601 s |
| 1024 | 199.8388 | 5.004 s |
| 2048 | 199.9762 | 5.001 s |
| 4096 | 199.2831 | 5.018 s |

Interpretation: 512 is clearly slow. 1024/2048/4096 are effectively tied; 2048 is nominally best but only +0.07% over 1024. Use the full matched sweep on the fork before choosing a comparison point.

### Gemma 4 26B-A4B downloaded target

`/models/Gemma-4-26B-A4B/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` (17,010,980,576 bytes).
Metadata checked so far: 30 layers, embedding 2816, 16 Q heads, D=512 K/V, KV-head pattern predominantly 8 with every sixth/global layer 2 (therefore local GQA2 and global GQA8), sliding window 1024, context 262144, 128 experts, top-8 routed, expert FFN 704, shared/dense FFN 2112. This makes it a useful MoE counterpart to dense 31B while retaining the same D512/global-GQA8 long-context attention geometry.

### Canonical 31B current-fork V100 100k-depth + 1k append sweep

Same command/configuration as upstream, except binary is current `v100-optimized` `d3c996f22`. All experimental/fork performance env vars explicitly unset.

| ubatch | upstream tok/s | fork tok/s | fork gain |
|---:|---:|---:|---:|
| 512 | 178.5385 | 181.0433 | +1.40% |
| 1024 | 199.8388 | 203.0365 | +1.60% |
| 2048 | 199.9762 | 203.5908 | +1.81% |
| 4096 | 199.2831 | 202.4125 | +1.57% |

Both sides peak nominally at ubatch 2048, but 1024/2048/4096 all execute the 1000-token append in one microbatch and are close. The current fork's Qwen/Ornith optimizations therefore transfer only ~1.5-1.8% to Gemma 31B at the main agentive workload. This is the baseline before Gemma-specific changes.

### D512 attention profiling: first hard bottleneck

Exact microbenchmark added in experimental worktree for Gemma 31B global attention: D512, 4 KV heads, GQA8, KV=101120, Q=1000, Q8 K/V.

- Q8 KV: 328.747 ms, 20.16 TFLOP/s
- F16 KV: 327.966 ms, 20.21 TFLOP/s

Therefore whole-cache Q8->F16 conversion is only ~0.2% in this isolated operation; it is not the main bottleneck. The D512 attention kernel itself is slow.

Nsight Compute baseline (`ncu-d512-gqa8-q8-basic.ncu-rep`):
- kernel `flash_attn_ext_f16<512,512,8,8,...>`
- block 256 threads, grid 500
- 255 registers/thread
- 84.1 KiB dynamic shared memory/block
- registers limit: 1 block/SM
- shared-memory limit: 1 block/SM
- theoretical and achieved occupancy: 12.5% (8 active warps/SM)
- SM compute throughput: 22.09%
- DRAM throughput: 53.83%

This is the primary measured bottleneck.

#### Query-tile sweep
Profiling-only env `GGML_CUDA_VOLTA_GEMMA_D512_NCOLS1` was added in the experimental worktree.
- ncols1=1: unsupported on sm70 (`ncols1*ncols2 < 32`, no device code); not a regression.
- ncols1=2: unsupported for the same reason; not a regression.
- ncols1=4 (total ncols=32): 428.625 ms / 15.46 TFLOP/s, ~30.8% slower than baseline.
- ncols1=8 (total ncols=64): 327.765 ms / 20.22 TFLOP/s, best/current baseline.

Conclusion: simply reducing the query tile is not the fix. Need to reduce the D512 kernel's per-thread accumulator/register footprint or otherwise restructure output accumulation.

### Q8 model-weight opportunity
Source inspection confirms that large dense Q8_0 prompt matmuls on V100 do not use MMQ at N~1000. `ggml_cuda_should_use_mmq()` falls through to cuBLAS because V100 has fast FP16 hardware and the batch is above the small DP4A-MMQ threshold. `ggml_cuda_mul_mat_cublas_impl<F16>` allocates a full FP16 `src0` scratch matrix, dequantizes the entire Q8 weight tensor into it, then runs cuBLAS.

This makes fused/tile-local Q8 -> FP16 Tensor-Core GEMM a plausible independent optimization target for Gemma's very wide FFNs. Synthetic exact Gemma Q8 matmul/MoE cases were added to the experimental `test-backend-ops` perf suite for profiling before implementing this.

### 26B Q8 expert dispatch microbenchmark

Synthetic Gemma 4 26B-A4B exact expert shapes, 128 experts/top-8, N=1000:

| expert op | default | `GGML_CUDA_VOLTA_FORCE_MMQ=moe` | speedup |
|---|---:|---:|---:|
| 2816 -> 704 | 5.52 TFLOP/s (5.744 ms) | 13.79 TFLOP/s (2.301 ms) | 2.50x |
| 704 -> 2816 | 5.62 (5.644 ms) | 13.09 (2.422 ms) | 2.33x |
| 2816 -> 1408 fused gate+up-shaped | 8.32 (7.621 ms) | 15.86 (4.001 ms) | 1.90x |

Selective MoE MMQ is therefore mandatory for the 26B V100 baseline and should be tested as a possible Gemma/V100 automatic policy after full-model correctness/performance validation.

### Dense Q8 weight controls
31B N=1000 exact representative shapes:
- FFN up/gate 5376 -> 21504: Q8 cuBLAS path 3.804 ms / 60.77 TFLOP/s; native F16 3.400 ms / 68.01 TFLOP/s (~11.9% Q8 overhead).
- FFN down 21504 -> 5376: Q8 3.994 ms / 57.89; F16 3.623 ms / 63.82 (~10.2% overhead).
- Q projection 5376 -> 16384: Q8 2.965 ms / 59.42; F16 2.653 ms / 66.40 (~11.8% overhead).

A profiling-only dense-Q8 MMQ force switch was added. Forced existing DP4A MMQ is much worse:
- up/gate 7.415 ms / 31.18 TFLOP/s
- down 7.635 ms / 30.28
- Q projection 5.667 ms / 31.09

Conclusion: do NOT replace cuBLAS with existing MMQ. The promising Q8 optimization is a new fused/tile-local Q8 decode -> FP16 Tensor-Core GEMM that preserves the fast Tensor-Core arithmetic but avoids materializing the whole FP16 weight matrix.

### D512 staging breakthrough and register-pressure diagnosis

After Nsight identified the D512/GQA8 kernel as the dominant hotspot, the 64-column Volta config was swept. Baseline config was effectively `nbatch_fa=32, K2=128, V2=128, combine=64` and the exact Q8 global-attention microbenchmark was ~328 ms.

Measured staging experiments on D512/GQA8, KV=101120, Q=1000, Q8 KV:
- baseline 32/128/128/64: ~328.7 ms (~20.16 TFLOP/s)
- 64/64/64/64: 285.2 ms (~23.24 TFLOP/s)
- 128/32/32/64: **219.7-220.5 ms (~30.1 TFLOP/s), ~33% faster than baseline**
- 256/16/16/64: 285.9 ms (~23.18 TFLOP/s), regression
- 96/32/32/64: 269.1 ms (~24.63 TFLOP/s), regression
- ncols32 query tile remained much slower (~428.6 ms in the earlier sweep).

The 128/32/32/64 point passed the D512/GQA8/Q8 CPU-reference correctness test and reproduced at 219.72 / 220.32 ms.

Important: Nsight showed the gain did NOT come from higher occupancy. The D512 8x8 kernel remains 255 registers/thread and one CTA/SM. `cuobjdump --dump-resource-usage` additionally reports ~752 bytes/thread stack for the SM70 D512 8x8 kernel, versus only tens of bytes for representative D256 kernels. This strongly indicates accumulator/register spill pressure.

Why register reduction alone is insufficient: a 64-column D512 Q tile already consumes ~65 KiB shared memory, so even with dramatically fewer registers a second CTA cannot reside on a V100 SM. A second-stage optimization must lower both live accumulator state and shared-memory footprint (or otherwise partition work).

A split-D feasibility experiment is staged in the experimental worktree: benchmark D512 QK with D256 V/output. Two half-width passes deliberately repeat QK but halve the live V/output accumulator. It is only promising if 2x half-pass time beats the current ~220 ms full-D best. This provides a measurement before implementing complicated full-output stride/glue logic.

### Full-model validation of D512 staging fix

Current-fork baseline at the canonical main workload (`-d 100000 -p 1000`, V100, Q8 K/V, ubatch 1024): **203.0365 tok/s**. Upstream control: **199.8388 tok/s**.

With only the experimental D512/ncols64 Volta staging changed to `nbatch_fa=128, K2=32, V2=32, combine=64`, the same full Gemma 4 31B benchmark produced:

- **273.1870 tok/s**
- 3.6605 s per 1000-token append
- 5 samples: 273.575, 271.678, 273.347, 273.753, 273.582 tok/s
- sd 0.856 tok/s
- gain vs current fork baseline: **+34.55%**
- gain vs upstream: **+36.70%**

This matches the ~33% isolated global-attention gain and confirms the D512/GQA8 global attention path dominates the 100k+1k agent workload. This candidate is worth productionizing after the remaining structural experiment.

### 26B-A4B Q4_K_XL tensor-type mix

The downloaded single model is already a useful multi-quant stress test (658 tensors):
- F32: 392 tensors, ~0.043 GiB (norms/scalars etc.)
- Q8_0: 207 tensors, ~2.610 GiB (token embedding and many attention/dense projections)
- Q5_1: 29 tensors, ~5.140 GiB (expert-down weights)
- Q4_K: 29 tensors, ~7.710 GiB (expert gate+up weights)
- Q5_K: 1 tensor, ~0.325 GiB (one gate+up expert tensor)

Therefore the 26B Q4_K_XL full model naturally exercises Q8 dense/attention, Q4_K expert gate+up, Q5_1 expert-down, and a Q5_K case in one run. Quant-specific profiling should use these actual tensor types rather than only synthetic Q8 experts.

### Actual 26B expert quant dispatch

Exact synthetic expert shapes using the tensor types present in downloaded `UD-Q4_K_XL`, N=1000, 128 experts/top-8:

| tensor/path | default | selective `MMQ=moe` | speedup |
|---|---:|---:|---:|
| Q4_K gate+up 2816 -> 1408 | 7.22 TFLOPS / 8.784 ms | 14.39 / 4.408 ms | **1.99x** |
| Q5_1 expert-down 704 -> 2816 | 5.36 / 5.919 ms | 12.37 / 2.564 ms | **2.31x** |
| Q5_K gate+up outlier | 7.48 / 8.477 ms | 13.71 / 4.627 ms | **1.83x** |

Thus selective MoE MMQ is strongly beneficial for the actual quant families in the downloaded 26B model, not only synthetic Q8 experts. Full-model 26B baselines should use the same selective expert policy on both upstream-control and optimized builds.

### Naive split-D feasibility result

Synthetic D512 QK + D256 V/output, ncols32 occupancy-oriented path: 478.386 ms for **one** half-width pass (10.39 TFLOPS), while the optimized full D512 path is ~220 ms. Resource usage did reduce stack pressure (SM70 ~176 B/thread vs ~752 B/thread full-D) but stayed at 255 registers/thread and throughput collapsed. Two such passes would be hopelessly slower. This confirms that register relief is necessary but absolutely not sufficient: work/tile organization must remain efficient. Do not implement this ncols32 split-D design.

### Actual 26B-A4B mixed-quant expert MMQ results

Exact `UD-Q4_K_XL` routed expert shapes, V100, N=1000, 128 experts/top-8:
- Q4_K gate+up (2816 -> 1408): default 7.22 TFLOP/s (8.784 ms), selective `MMQ=moe` 14.39 TFLOP/s (4.408 ms), **1.99x**.
- Q5_1 down (704 -> 2816): default 5.36 TFLOP/s (5.919 ms), selective `MMQ=moe` 12.37 TFLOP/s (2.564 ms), **2.31x**.
- Q5_K gate+up outlier (2816 -> 1408): default 7.48 TFLOP/s (8.477 ms), selective `MMQ=moe` 13.71 TFLOP/s (4.627 ms), **1.83x**.

Therefore selective MoE MMQ benefits every major routed expert quant type present in this real 26B file. It should be part of the production V100 configuration unless the full-model benchmark contradicts it.

### Split-D feasibility result

The profiling-only D512-QK / D256-V half-width kernel ran at **478.39 ms per half pass** and still used up to 255 registers/thread (SM70 resource report; stack reduced relative to D512 but not enough). Two such passes would be far slower than the ~220 ms optimized full-D kernel. The naive split-D strategy is rejected.

This reinforces the design constraint: register pressure must be reduced without throwing away the `FA128/K32/V32` dataflow or repeating the expensive QK work. A future structural kernel should stream/retire subsets of the V accumulator inside one attention pass, or otherwise reduce live accumulator lifetime while preserving KQ reuse.

### 26B-A4B full-model validation of D512 staging fix

Canonical V100 agent workload: `-d 100000 -p 1000`, Q8 K/V, ubatch 1024, with identical selective `GGML_CUDA_VOLTA_FORCE_MMQ=moe` expert policy in both fork arms.

- current `v100-optimized` baseline: **723.4352 tok/s**, 1.382322 s / 1k append
- D512 `nbatch_fa=128, K2=32, V2=32` candidate: **959.4843 tok/s**, 1.042284 s / 1k append
- candidate samples: 958.781, 946.203, 961.817, 965.137, 965.482 tok/s
- gain vs current fork: **+32.63%**

This is the downloaded `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf`, so the result includes its real mixed Q8_0/Q4_K/Q5_1/Q5_K weights and routed experts. The near-match to the 31B +34.55% full-model gain confirms the same D512 global-attention staging problem is a major bottleneck in both dense and MoE Gemma 4.

### 26B-A4B full-model decomposition, canonical 100k+1k

Gemma 4 26B-A4B `UD-Q4_K_XL`, V100-only, Q8 K/V, ubatch 1024, 5 reps:

| implementation | tok/s | vs upstream |
|---|---:|---:|
| upstream `43f3dda62`, default dispatch | 689.5175 | baseline |
| clean fork + D512 staging only | 905.8393 | **+31.37%** |
| current fork `d3c996f22` + selective `MMQ=moe` (no D512 change) | 723.4352 | **+4.92%** |
| clean D512 staging + selective `MMQ=moe` | 959.4843 | **+39.15%** |

The D512 staging change remains the dominant improvement. Once attention is fixed, selective MoE MMQ adds another **+5.92%** over the D512-only result. Relative to the old fork+MMQ configuration, the D512 change adds **+32.63%**.

### Clean D512 candidate short/medium regression sweep

One-line candidate only (`D512/ncols64: FA 32->128, K2/V2 128->32`), V100, Q8 K/V, ubatch 1024:

| model | prompt | upstream | candidate | delta |
|---|---:|---:|---:|---:|
| Gemma 4 31B | 1k | 723.6448 | 736.2451 | **+1.74%** |
| Gemma 4 31B | 16k | 603.4149 | 655.2311 | **+8.59%** |
| Gemma 4 31B | 100k depth + 1k | 199.8388 | 273.1870 | **+36.70%** |
| Gemma 4 26B-A4B | 1k | 2023.2862 | 2077.6201 | **+2.69%** |
| Gemma 4 26B-A4B | 16k | 1850.0905 | 1955.0735 | **+5.67%** |
| Gemma 4 26B-A4B | 100k depth + 1k | 689.5175 | 905.8393 | **+31.37%** |

The improvement scales with context length and no tested prompt length regressed.

Clean candidate backend correctness: `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=512'` completed with **22/22 supported D512 cases passed**, including the existing D512/GQA2/SWA case. Earlier exact D512/GQA8/Q8 CPU-reference test also passed.

### Experimental dense Q8 fused-Tensor-Core prototype v1

Ported the proven NInfer SM70 W8 fused-dequant MMA design to llama `Q8_0` behind `GGML_CUDA_VOLTA_Q8_TC_GEMM=1`. llama Q8_0 is layout-compatible in quant semantics (32 signed int8 values + one FP16 scale/group) but interleaves scale+codes rather than separate planes. Initial backend correctness: 19/19 matching Q8 MUL_MAT cases passed.

Real Gemma 26B Q8 dense shapes at N=1000 were **much slower** than stock dequant->FP16+cuBLAS: ~14.8-15.4 TFLOP/s fused vs ~48.1-56.5 TFLOP/s stock. This prototype is not a candidate yet.

Nsight Compute on m=2048,k=2816,n=1000: ~746.98 us kernel, 80 regs/thread, 10.24 KiB static shared/block, theoretical occupancy 37.5%, achieved 34.0%, compute throughput ~38%, DRAM throughput only ~8%, L2 hit ~96.8%. Therefore the loss is not a register-spill problem; the current 32-token fused schedule underutilizes the SM / overpays shared-load+issue work relative to cuBLAS while repeatedly revisiting weight tiles from cache. Profile before any redesign.

### Dense Q8 fused-Tensor-Core prototype conclusion

Successive SM70 fused-dequant Q8_0 x F32 prototypes were profiled and measured on real Gemma 26B N=1000 projection shapes:
- v1: 32 output rows x 32 tokens, 4 warps: ~14.8-15.4 TFLOP/s; 80 regs/thread.
- v2: 64 output rows x 32 tokens, 8 warps: ~16.2-17.0 TFLOP/s; 72 regs/thread.
- v3: 64 output rows x 64 tokens, reuse each decoded weight tile across two token groups: ~21.7-23.9 TFLOP/s; 80 regs/thread.
- v4: 64 output rows x 128 tokens, four accumulator groups: ~25.8-28.0 TFLOP/s; 119 regs/thread.

Stock llama.cpp dequant->FP16 + cuBLAS on the same shapes is ~48.1-56.5 TFLOP/s. Nsight on v1/v3 showed the fused route is L1/TEX/shared-issue limited rather than DRAM- or spill-limited: ~61% no-eligible-warp cycles, large shared-store bank conflicts, 57% excessive global sectors, and very high L2 hit rates. Increasing token reuse improves throughput monotonically, but by 128 tokens register pressure is already 119 regs/thread and performance is still only about half of cuBLAS. Further tile growth would trade away occupancy aggressively.

Conclusion: reject this fused Q8 dense-GEMM prototype for N~1000. Keep stock dequant+cuBLAS. A future viable design would need a more fundamental layout/coalescing solution or a CUTLASS-style fused dequant mainloop that approaches cuBLAS tensor-core efficiency; do not merge the current prototype.

### RTX 2080 Ti Ornith regression investigation

Model: `Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf` (~21 GiB), single RTX 2080 Ti 22 GiB, ctx 67,584, Q8 K/V, cached prefix 65,536 + 1,000 append, ubatch 512. The model and 67,584 context load fully; startup leaves ~203 MiB free. Saved prefix size is ~745 MiB.

Fair normal-build ABBA: upstream ~1308.59 tok/s vs fork ~1272.41 tok/s (-2.76%).
Initial global-FORCE_MMQ builds also showed ~1310.28 vs 1276.06, but investigation found the fork's compile-time `GGML_CUDA_FORCE_MMQ` check occurred *after* the Turing dense crossover, so FORCE_MMQ did not actually override Turing dense N>=256 routing. Moving the compile-time FORCE check before the Turing block fixes option semantics; after rebuilding with that correction, fair ABBA still shows upstream 1324.66 vs fork 1283.93 tok/s (-3.08%). Therefore the remaining regression is not expert/dense MMQ selection.

Ornith attention geometry from GGUF: D256, 16 Q heads, 2 KV heads => GQA8. Commit `2bb7aca44` changed generic Turing D256 ncols32/64 `nbatch_fa` from 64 to 32 for Qwen. Turing generic dispatch always uses a 32-column tile; Ornith GQA8 therefore inherits the Qwen tuning despite not being the GQA6 target. Testing upstream-style nbatch_fa=64 is the next isolation step. Do not merge that global revert until Qwen regression is checked.

### Qwen3.8 regression gate for Ornith SM75 work

Do not merge an Ornith/Turing optimization unless Qwen3.8-27B remains non-regressive. Candidate dispatch is deliberately gated to SM75 + D256 + Q8 K/V + 2 KV heads + GQA8 + long KV + 128..999 query tokens, while existing Qwen3.8 uses the separate GQA6 long-prompt route. Required before merge: candidate normal (non-FORCE_MMQ) build vs current v100-optimized on the retained RTX 2080 Ti 65,536 cached + 1,000 append state, plus V100 Qwen control. Reject or narrow the dispatch if Qwen moves outside measurement noise.

### RTX 2080 Ti Ornith/Qwen3.5-MoE investigation

Fair builds use Release, CUDA graphs ON, SM70+SM75, and global `GGML_CUDA_FORCE_MMQ=ON` on both upstream `43f3dda62` and fork. Found and fixed an ordering bug where the fork's Turing dense cuBLAS crossover ran before the `GGML_CUDA_FORCE_MMQ` check, so a nominal force-MMQ build was not actually globally forced on SM75.

Baseline fully-resident Ornith `AD-Q5_K-Q4_K`, RTX 2080 Ti 22 GB, Q8 KV, 65,536 restored + 1,000 append, ctx 67,584, batch/ubatch 4096/512, ABBA: upstream ~1324-1332 PP/s; fork before GQA8 attention adaptation ~1284-1288 PP/s (~-3%).

Component isolation showed the recent Qwen/Turing work already helps Ornith: GATED_DELTA_NET 32x128 PP1024 is ~1.43 ms fork vs ~2.16 ms upstream (~1.51x faster); exact D256/GQA8/2-KV-head/Q8 attention at KV=65,536,Q=1000 is 30.11 ms fork vs 37.58 ms upstream (~24.8% faster). Therefore neither GDN nor generic long-Q8 attention explains the whole-model regression.

Qwen3.8's largest SM75 long-Q8 optimization is hard-gated to D256/GQA6/2 KV heads and dispatches to a 16x2 specialization that enables INT8 Tensor-Core QK. Ornith has the same D256/2-KV/Q8 geometry but GQA8, so it missed this path. Added an experimental strict GQA8-only dispatch to 16x2. Exact attention improves 30.11 -> 26.05 ms (~15.6%) and the 65,536x1000 CPU-reference test passes 1/1.

Full-model global-MMQ ABBA with GQA8 candidate: upstream **1331.51 tok/s**, fork **1359.90 tok/s**, **+2.13%**; wall TTFT 784.69 -> 773.08 ms (-1.48%). This turns the previous regression into a gain. Qwen3.8 GQA6 dispatch predicate is separate and unchanged; exact GQA6 kernel still measures ~40.35 TFLOP/s with the GQA8 patch present. Full Qwen model regression test is still required before merge.

### Ornith RTX 2080 Ti natural-layout INT8 QK candidate

Exact backend shape: D256, 2 KV heads, GQA8, Q8 K/V, KV=65,536, Q=1,000. Extending the existing SM75 D256 INT8-QK kernel to the natural 4x8 GQA8 layout passes CPU-reference correctness. Isolated FA: FP16-QK 30.001 ms/op (35.79 TFLOP/s) -> INT8-QK 25.581 ms/op (41.97 TFLOP/s), ~17.3% faster. Full Ornith 65,536 cached + 1,000 append, single RTX 2080 Ti 22 GB, global FORCE_MMQ on both upstream and fork: upstream 1328.62 PP/s, candidate 1355.25 PP/s (+2.00%); TTFT wall 788.35 -> 769.37 ms (-2.41%). Candidate preserves Ornith's natural 4x8 GQA8 dispatch rather than rerouting to Qwen's 16x2 GQA6 path.

Qwen3.8 regression gate remains mandatory before merge: compare candidate normal build to clean pushed 352827268 on RTX 2080 Ti 65,536+1k and V100 controls. Do not merge if Qwen moves outside measurement noise.

Next requested: V100 + RTX 2080 Ti Ornith fair FORCE_MMQ comparison and profile remaining bottlenecks with Nsight before further tuning.

### Qwen3.8 regression gate for natural GQA8 INT8 candidate

Compared exact pushed `352827268` in a clean detached worktree against the candidate normal (FORCE_MMQ=OFF) build, using retained saved KV states and mirrored ABBA process arms. Output hashes matched in every retained sample.

- RTX 2080 Ti 22 GB, 65,536 cached + 1,000 append, Q8 K/V, production Qwen long-Q8 settings: 494.16 -> 493.33 PP/s (-0.17%).
- V100 32 GB, 100,000 cached + 1,000 append, Q8 K/V, production Qwen settings: 456.28 -> 454.69 PP/s (-0.35%).

Both deltas are within run-to-run variance. Candidate does not alter Qwen's GQA6 dispatch; it only broadens the SM75 INT8-QK template eligibility to natural 32-column GQA8 layouts. Treat Qwen regression gate as passed for this candidate, subject to final post-merge smoke if merged.

### Ornith V100 + RTX 2080 Ti 100k+1k FORCE_MMQ sweep

Model: `Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf`, native ctx 131072, shared retained 100,000-token slot, Q8 K/V, MTP off, layer split `CUDA1,CUDA0` = RTX 2080 Ti,V100 with `--tensor-split 14,35`, global `GGML_CUDA_FORCE_MMQ=ON` compiled identically in upstream and fork, batch=2048. Mirrored upstream/candidate/candidate/upstream process arms; first request per process discarded; output hashes match.

- ubatch 256: upstream 814.50 PP/s, candidate 849.36 PP/s, +4.28%.
- ubatch 512: upstream 869.77 PP/s, candidate 989.02 PP/s, +13.71%. This is the best matched setting for both sides and the fair headline candidate.
- ubatch 1000: upstream 818.49 PP/s, candidate 985.01 PP/s, +20.35%. Do not use this as the headline gain because upstream itself regresses versus ubatch 512; it is useful diagnostically because the fork tolerates the wide microbatch much better.

Next: Nsight Systems/Compute at ubatch 512 to attribute the remaining runtime and explain why the fork scales better. Profile GDN, routed MoE MMQ, full-attention GQA8/Q8 and inter-GPU copies before proposing further changes.

### Dual Ornith ubatch=512 Nsight attribution

Production-like server traces with a restored 100k slot reproduce the benchmark behavior under Nsight: second request ~838.8 PP/s upstream vs ~985.7 PP/s candidate. There is one compute stream per GPU and no same-GPU kernel overlap (sum of kernel durations == interval union), so the longer individual MMQ kernels in the candidate trace are not caused by concurrent kernels on the same GPU.

Upstream kernel-time mix is dominated by D256 FlashAttention (~57% aggregate GPU kernel time) and routed quantized MMQ (~38%); GatedDeltaNet is only ~3.6%. Candidate attention is substantially faster: the V100 D256 path is ~28.8% faster and RTX natural GQA8 `4x8` INT8-QK is ~35.7% faster than upstream's corresponding FP16-QK path. GDN x4 also improves ~32.7%, but is too small a share to dominate total PP.

Per-device production trace showed candidate Q5_K/Q6_K kernel durations apparently longer than upstream. Exact isolated routed-expert microbenchmarks prove this is not an intrinsic kernel regression:

- V100 Q5_K gate/up exact shape (`256 experts, top8, K=2048, M=512, N=512`): upstream 2514.12 us vs candidate 2508.60 us, both ~3.42 TFLOP/s.
- V100 Q6_K down (`K=512, M=2048, N=512`): upstream 2564.86 us / 3.35 TFLOP/s vs candidate 1508.28 us / 5.70 TFLOP/s. The current Volta Q6 configuration is a major win and must be preserved.
- RTX 2080 Ti Q5_K: upstream 1523.52 us vs candidate 1519.27 us (~5.64-5.65 TFLOP/s).
- RTX Q6_K: upstream 1728.91 us vs candidate 1727.10 us (~4.97 TFLOP/s).

Therefore do not patch Q5/Q6 arithmetic based on the raw full-model trace. The next hypothesis is routing-sensitive J tile selection: generic MMQ selects J against the total 512-token microbatch even though 4096 routed rows are distributed across 256 experts (~16 rows/expert average). Q5_K J128 uses very high register counts (V100 ~221 regs/thread; RTX ~255). Smaller J may reduce register pressure and wasted per-expert tile work.

Historical worktree `/workspace/llama-ornith-sm70-moe-mmq` already found on the older Ornith setup that per-expert load controls the best J: gate/up J32 through ~24 rows/expert and J40 around ~32; down J32 <=8, J40 <=16, J64 above. Revalidate on the current Q5_K gate/up + Q6_K down artifact and on both SM70/SM75 rather than copying this rule blindly.

### Routed-MMQ J-tile sweep and dual-GPU result

Nsight showed the remaining major cost after the GQA8 INT8 attention fix is routed Q5_K/Q6_K MMQ. Exact `MUL_MAT_ID` shape tests at ubatch 512 reveal the generic J128 choice is badly mismatched to Ornith's sparse routed workload.

Routing dump on the real 100k+1k append: 256 experts, top-8, 4096 routed rows from a 512-token microbatch. Average load is 16 rows/expert but distribution is very skewed: median 1-3 rows, p90 ~38-50, p95 ~54-98, p99 ~110-233, max 170-334 depending layer/device. Thus J128 wastes work for most experts.

Exact gate/up shape Q5_K: K=2048, M=512, 256 experts, top-8. V100 best observed J24 ~10.58 TFLOP/s vs J128 ~3.40. RTX 2080 Ti best observed J32 ~10.94 TFLOP/s; J24 ~10.79, vs J128 ~5.56. A common J24 is therefore near-optimal on both GPUs.

Exact down shape Q6_K: K=512, M=2048, 256 experts, top-8. J48 is best/near-best on both: V100 ~6.74 TFLOP/s vs J64/J128 ~5.6; RTX 2080 Ti ~7.26-7.38 TFLOP/s vs J128 ~4.9.

Fair dual-GPU ABBA with common `Q5_K J24 + Q6_K J48`, layer split RTX,V100 = 14/35, global FORCE_MMQ on both upstream and fork, Q8 KV, 100k restored + 1k append, ubatch 512: upstream **865.49 PP/s**, candidate **1096.89 PP/s**, **+26.74%**. Candidate retained samples 1093.55-1102.21 PP/s, upstream 864.07-866.81. Output SHA identical. This is ~10.9% above the previous optimized candidate (~989.02 PP/s) at the same matched ubatch.

### Ornith routed-MMQ J-tile sweep and full-model result

Real ubatch=512 routing is highly skewed despite an average 16 routed rows/expert. Observed per-layer examples: p50 1-3 rows, p90 ~38-50, p95 ~54-98, p99 ~110-233, max ~170-334. Therefore selecting J from total microbatch width is inappropriate for 256-expert top-8 Ornith.

Exact `MUL_MAT_ID` shapes from the AD-Q6_K-Q5_K GGUF:
- Q5_K gate/up: 256 experts, top-8, `K=2048, M=512`, broadcast activation, N=512.
- Q6_K down: 256 experts, top-8, `K=512, M=2048`, non-broadcast activation, N=512.

Exact-shape J sweeps:
- V100 Q5_K: J24 **0.812 ms / 10.58 TFLOP/s**; J128 2.526 ms / 3.40 TFLOP/s. J24 is ~3.11x faster.
- RTX 2080 Ti Q5_K: J32 0.785 ms / 10.94 TFLOP/s, J24 0.796 ms / 10.79 TFLOP/s; J24 is only ~1.4% behind the RTX optimum and is much better on V100.
- V100 Q6_K: J48 **1.274-1.280 ms / ~6.72 TFLOP/s** vs current J64 ~1.52 ms, ~19% faster.
- RTX 2080 Ti Q6_K: J48 **~1.164 ms / 7.38 TFLOP/s** vs J128 ~1.75 ms, ~50% faster.

Best shared dual-GPU pair at N=512 is therefore Q5_K J24 + Q6_K J48.

Full real dual-GPU 100k cached + 1k append, ubatch=512, FORCE_MMQ build, layer split RTX:V100=14:35, Q8 KV, MTP off, mirrored upstream/candidate/candidate/upstream and identical output hashes:
- upstream: **865.49 PP/s**
- candidate with GQA8 INT8 attention but default MMQ J: prior fair result **989.02 PP/s**
- candidate + Q5 J24 + Q6 J48: **1096.89 PP/s**
- gain from J tuning over candidate default: **+10.91%**
- gain vs upstream: **+26.74%**

Do not hard-code J24/J48 globally yet. Next: sweep exact Ornith shapes at smaller microbatch N={64,128,177,256,512} on both GPUs and derive a narrow automatic policy; preserve Qwen because the rule must require routed experts and exact Ornith-like expert projection geometry.

### Ornith routed-MMQ J retile result

Current 25 GiB `AD-Q6_K-Q5_K`, dual V100+RTX, ubatch 512 routing is highly skewed: average 16 rows/expert, but p50 only 1-3 rows, p90 ~38-50, p95 ~54-98, p99 ~110-233 and max ~170-334 depending layer/device. Generic MMQ selects J128 from the total 512-token microbatch, which is a poor match for most experts and carries very high register pressure.

Temporary per-quant J overrides show exact expert microbench optima near Q5_K J24 (V100 ~10.58 TFLOP/s; RTX J24-32 ~10.79-10.94) and Q6_K J48 (V100 ~6.74 TFLOP/s; RTX ~7.26). Real dual 100k+1k candidate-only sweep confirms `Q5 J24 / Q6 J48` is the best tested pair: 1105.46 PP/s; J24/Q6=64 1099.76, Q5=40/J48 1092.53, Q5=32/J48 1089.01, Q5=16/J48 1083.11, J24/Q6=40 1083.09, J24/Q6=32 1060.61.

Fair mirrored ABBA with Q5 J24 / Q6 J48: upstream **865.49 PP/s** vs candidate **1096.89 PP/s**, **+26.74%**, identical output hashes. This is ~+10.9% over the prior candidate without J retile (~989 PP/s). Treat the per-type J overrides as experimental until converted to a strict architecture/geometry/batch gate and regressed on V100-only, RTX-only, Qwen, Gemma and shorter append sizes.

### Automatic routed-MMQ selector validation

Moved the Ornith J policy out of `mmq.cuh` and into the routed `ncols_opt` hint in `mmq.cu`; MMQ device templates are unchanged. The generic selector therefore chooses the measured J while preserving its normal validity/shared-memory checks.

With no J override environment variables, exact N=512 perf reproduces the manual winners:
- V100: Q5_K 812.53 us / 10.57 TFLOP/s; Q6_K 1237.74 us / 6.94 TFLOP/s.
- RTX 2080 Ti: Q5_K 777.30 us / 11.05 TFLOP/s; Q6_K 1161.70 us / 7.39 TFLOP/s.

Temporary exact CPU-reference cases for `MUL_MAT_ID` (256 experts, top-8, Q5_K gate/up and Q6_K down) pass 2/2 on V100 and 2/2 on RTX 2080 Ti. Temporary test fixtures were restored immediately afterward; they are not production changes.

### Ornith dual 64-token generation regression check: first GQA8 INT8 version rejected for decode

Dual V100 + RTX 2080 Ti, `AD-Q6_K-Q5_K`, FORCE_MMQ on both sides, restored 100k prefix + 1k append + 64 generated, Q8 K/V, ubatch 512, MTP off. Before restricting the new GQA8 INT8-QK path to prefill, mirrored arms measured roughly **840.60 -> 1075.38 PP/s (+27.93%)** and **54.49 -> 54.77 TG tok/s (+0.50%)**. However upstream produced the same 64-token SHA on every restored run while the fork produced different token SHAs between otherwise identical restored runs.

Throughput was non-regressive, but the per-run output instability is not acceptable as a release validation result. The likely cause is the new approximate GQA8 INT8-QK path also being selected for single-token decode. Decision: keep Qwen3.8 GQA6 behavior untouched, but require `Q_src->ne[1] >= 128` for the new GQA8 INT8-QK route. Ornith decode then falls back to the existing FP16-QK path. Rebuild and rerun TG before merge.

### Dual-GPU layer-split sweep after attention + routed-MMQ tuning

Candidate, FORCE_MMQ, 100k cached + 1k append, ubatch 512, Q8 KV. Coarse sweep showed the old `14:35` split at ~1106 PP/s, 1:1 ~1302, 3:2 ~1373, 2:1 ~1327, 3:1 ~1303; 4:1 OOMed during the request. Narrow sweep around the peak showed 7:5 ~1379, 3:2/8:5/5:3 around 1330-1341 in a single-pass run. A mirrored `3:2 -> 7:5 -> 7:5 -> 3:2` confirmation removes drift: 3:2 averages ~1368.03 PP/s, 7:5 ~1378.00 PP/s (+0.73%).

VRAM after load: 7:5 uses ~16,266 MiB RTX / 11,873 MiB V100, leaving ~5.7 GiB RTX and ~20.6 GiB V100. 3:2 uses ~16,870 MiB RTX / 11,265 MiB V100. The discrete 7:5 layer boundary is the current production-placement winner.

### Final fair dual-GPU placement comparison

After selecting the discrete `7:5` RTX:V100 layer split, ran mirrored upstream/candidate/candidate/upstream with identical FORCE_MMQ builds, AD-Q6_K-Q5_K, 100k restored + 1k append, Q8 KV, ubatch 512, same saved state, and no J override env variables. All retained one-token output hashes match.

- upstream: **1135.54 PP/s**
- optimized: **1380.29 PP/s**
- improvement: **+21.55%**
- optimized prompt time: ~724.5 ms vs upstream ~880.6 ms.

The old 14:35 split gave upstream ~871.8 and optimized ~1104.9. Thus the placement improvement is important to both implementations; use the same 7:5 split for fair headline comparisons. Optimized 7:5 is ~24.9% faster than optimized 14:35 and ~58.3% faster than upstream 14:35, but the fair code-branch gain is +21.55% at identical placement.

TG64 control at old 14:35: upstream 54.49 TG/s, fork 54.77 TG/s (+0.50%), while PP improved ~27.9%. Long-generation token hashes were stable within upstream but varied between fork requests despite greedy seed=1234; record this as a determinism investigation item and do not claim long-generation output identity from that test.

### Final Ornith SM70/SM75 candidate and merge decision

Production source changes retained for merge:
1. Turing D256 long-Q8 INT8-QK template eligibility is generalized from only 16x2 to 32-column layouts with ncols2={2,4,8}. Existing Qwen3.8 GQA6 remains on its unchanged 16x2 dispatch. Ornith GQA8 can use its natural 4x8 layout only for long Q8 **prefill with Q >= 128**.
2. Routed Ornith/Qwen3.5-MoE MMQ gets a narrowly gated host-side `ncols_opt` policy for 256 experts, top-8, exact 2048<->512 projection geometry, N<=512, SM70/SM75. Q5/Q6 and the 21 GiB RTX Q4/Q5 quant use the measured J ranges; N<64 remains on J8 so TG/tiny appends are not retiled.
3. Compile-time `GGML_CUDA_FORCE_MMQ` is checked before the Turing dense cuBLAS crossover so a FORCE_MMQ build actually forces MMQ on SM75.

Final Qwen3.8 regression gates against clean pushed `352827268`, normal builds, retained cached states, mirrored ABBA, identical hashes:
- RTX 2080 Ti 65,536+1k: **498.67 -> 498.99 PP/s (+0.06%)**.
- V100 100k+1k: **455.18 -> 455.58 PP/s (+0.09%)**.
No measurable Qwen regression.

Final single RTX Ornith, `AD-Q5_K-Q4_K`, 65,536 cached + 1k append, ctx 67,584, Q8 K/V, MTP off, global FORCE_MMQ identically compiled on both engines, ubatch 512, mirrored ABBA:
- upstream **1327.01 PP/s**
- optimized **1500.26 PP/s**
- **+13.06%** PP
- TTFT **790.14 -> 699.88 ms (-11.42%)**.

Final dual Ornith topology screen showed tensor split is substantially better than layer split. Candidate screen at 100k+1k:
- tensor 1:1, ub512: ~1199.37 PP/s
- tensor 1:1, ub1024: ~1453.32 PP/s
- tensor 4:5, ub1024: ~1093.89 PP/s
- tensor 5:4, ub1024: ~1335.28 PP/s
The 1:1 split also leaves ample VRAM on both GPUs (~8.1 GiB free RTX, ~18.4 GiB free V100 in the screen).

Fair final tensor 1:1 / ubatch 1024 ABBA, same 25 GiB `AD-Q6_K-Q5_K`, 100k cached + 1k append, Q8 K/V, MTP off, global FORCE_MMQ on both upstream/fork, internal CUDA all-reduce:
- upstream **1247.39 PP/s**
- optimized **1458.85 PP/s**
- **+16.95%** PP
- TTFT **856.31 -> 732.63 ms (-14.44%)**
- all retained one-token PP output hashes identical.
A clean candidate-only rerun after removing all profiling GDN instrumentation reproduced **1449.62 PP/s** at tensor 1:1 / ub1024.

### Dual Ornith 64-token decode determinism follow-up

A 100k restored + 1k append + 64-token greedy decode test shows roughly neutral/slightly positive TG throughput (~54.4 upstream vs ~54.7 optimized tok/s), but repeated optimized runs do not reproduce identical 64-token hashes while upstream does. This variation persists when the new GQA8 INT8-QK path is disabled, when the pre-existing SM70/75 x4 GatedDeltaNet prefill path is disabled, and when internal all-reduce is disabled. Therefore the issue was **not isolated to either new Ornith optimization**. The new GQA8 INT8 path is nevertheless kept prefill-only (Q>=128) to minimize decode surface area. Continue investigating the broader mixed-GPU/hybrid-state determinism separately; do not claim deterministic dual-Ornith decode in README results yet.

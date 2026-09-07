# Nsight profiling of parallel Ornith — 7 September 2026

## Scope and status

This is the first **actual Nsight capture for the parallel-serving experiment series**. Earlier work under `parallel-experiments-0907` used timing and correctness tests, not Nsight. Both **Nsight Systems 2025.1.3** and **Nsight Compute 2025.2.1** were used here. No engine or production setup behavior was changed.

The profiled binary is the immutable `parallel-experiments-0907/final-bin` snapshot used by research branch `63cccacea0`. It serves real Ornith AD-Q6_K-Q5_K, Shisa Q5_0 MTP, Q8 target and draft KV, CPU vision projector, on V100-SXM2 32 GB (200 W) + RTX 2080 Ti 22 GB (250 W). Layer split is 14:35, batch/ubatch 2048/256, draft ubatch 128, pipeline copies 1. Physical KV reservation is 1.4M and four logical caps are 400k. **Actual histories are four 100k histories sharing 98k, not four full 400k contexts.** The existing synthetic repetitive-text fixture is used, not a coding-quality benchmark.

Each server warms up, restores the four saved histories, processes 1k new tokens per agent plus one generated token (PP range), and then generates 128 per agent with at most two new input tokens (TG range). Startup, restore and warmup are excluded using explicit NVTX ranges in the driver. The TG range includes HTTP/request admission and the small transition into generation; it is not a pure CUDA-kernel benchmark.

`parallel.nsys-rep` compares ordinary MTP3, compact MTP3, compact direct-Q8 pack4, and compact without MTP. The first capture left CUDA-event completion tracing at its default. `attributed.nsys-rep` explicitly disables it and repeats compact MTP3 with temporary LD_PRELOAD NVTX markers around target decode, MTP drafting, and MTP process/flush. Every TG CUDA kernel in that run matched a launching API correlation ID and decode scope.

Node-level graph tracing and interposition add profiling overhead. These captures diagnose execution; they do not replace the earlier unprofiled performance tests. GPU-kernel duration sums, CPU API duration sums, and wall time are distinct quantities. In particular, time in `cudaStreamSynchronize` overlaps real GPU execution and is **not all removable overhead**.

## 1. TG is not keeping both GPUs busy

In the attributed TG window (4.235 s), counting the union of recorded kernel and memcpy intervals:

| Activity | Fraction of the profiled window |
|---|---:|
| V100 has recorded kernel/copy activity | 38.71% |
| RTX 2080 Ti has recorded kernel/copy activity | 27.34% |
| Both GPUs active simultaneously | 0.30% |
| Neither has recorded kernel/copy activity | 34.25% |

These are timeline occupancy numbers, not SM utilization or hardware-counter occupancy. Memset-only intervals are not included in this accounting. The first capture independently showed similarly low overlap (0.03%) and an inactive fraction of 34.59% for compact MTP3.

The attributed TG range contains **16,649 cudaStreamSynchronize calls**, with 2.364 s of summed CPU API duration. That does not prove a synchronization bug or imply that 2.364 s can be removed. It does establish that launch/synchronization orchestration and the serial two-GPU schedule deserve investigation, rather than assuming the pair is already saturated.

A useful next experiment is preserving complete MTP verification groups while testing small pipelined agent microbatches with limited additional buffering. This must be checked against the tight VRAM budget and recurrent rollback constraints. Another is locating repeated host readbacks/synchronization call sites, then keeping sampling/verification data on-device where semantics allow. No such change was enabled here.

## 2. MTP drafting is not the dominant GPU component

The tagged TG kernel-duration sum is 2.675 s:

| Component | Summed GPU kernel duration | Share of kernel-duration sum |
|---|---:|---:|
| Target forward/verification | 2.231 s | 83.41% |
| MTP draft generation | 0.353 s | 13.19% |
| Draft process/cache maintenance | 0.091 s | 3.40% |

The MTP/drafting CPU scopes total 0.579 s over 39 calls; MTP/process scopes total 0.122 s over 40 calls. These CPU scopes do not form an additive wall-time decomposition of the asynchronous workload. The target component includes ordinary model work and work required by speculative verification, not just speculative overhead.

The conclusion is not that MTP cannot improve. It is that **target verification efficiency and orchestration are larger targets than draft-head compute alone**. A depth controller cannot solve an expensive target path by maximizing acceptance in isolation. The existing fixed MTP3 remains unchanged.

## 3. The Q8 prototype saves conversion time but loses more inside attention

The initial Systems capture separates these kernel families during TG:

| Kernel family, GPU-duration sum | Existing attention, compact | Direct-Q8 pack4, compact |
|---|---:|---:|
| Attention and its reduction kernels | 443.9 ms | 1,076.9 ms |
| Q8-to-FP16 conversion kernels | 335.4 ms | 203.3 ms |
| Those two families combined | 779.2 ms | 1,280.2 ms |

Thus conversion work drops by about **132 ms**, but attention/reduction work grows by about **633 ms**. The net ~501 ms GPU-work increase is consistent with the traced TG window increasing from 4.172 to 4.711 s. This is a kernel-family attribution, not a one-to-one paired launch experiment. Q8 conversion outside the replaced path remains included, and the synthetic outputs match.

This corrects the hypothesis that avoiding full-cache dequantization by itself is enough. The particular kernel implementation is the problem; the direct-Q8 idea remains a candidate for a different resource layout.

## 4. Nsight Compute identifies the prototype's resource and issue stalls

These are **isolated 4,096-KV, 16-query operator tests** on the V100, not long-context serving measurements. The pack4 report captures one four-query tile launch, whereas pack16 captures all sixteen in one launch; individual kernel durations must not be compared as equal work. Profiler replay and unfixed GPU clocks further prevent using these durations as production speedups.

| Counter / resource | Pack4 kernel | Pack16 kernel |
|---|---:|---:|
| Threads per block | 128 | 512 |
| Registers per thread | 255 | 128 |
| Dynamic shared memory / block | 33,920 B | 84,992 B |
| Resident block limit from registers/shared memory | 2 | 1 |
| Theoretical warp occupancy | 12.50% | 25.00% |
| Achieved warp occupancy | 10.72% | 24.43% |
| Compute throughput relative to peak | 6.66% | 3.43% |
| DRAM throughput relative to peak | 13.73% | 40.29% |
| Scheduler cycles with no eligible warp (separate stall capture) | 92.27% | 96.96% |

For pack4, the dominant reported stall is the local/global memory instruction queue being full (~56.3% of average cycles between issued instructions). Pack16 additionally spends ~61.9% waiting on L1TEX scoreboard dependencies, plus ~30.9% on local/global queue throttling. These are reported ratios from a separate replay capture; they are not predicted application speedups.

The register/shared-memory limits and instruction-issue stalls argue for smaller live accumulator sets, shorter register lifetimes, inspection of local-memory placement/spills, better memory instruction packing and a more balanced division of QK/PV work. **Register spills themselves have not been directly counted here.** Wider packing alone made the resource problem worse; lowering the register limit blindly could increase spills.

## 5. Nsight supports the compact-layout performance explanation, not its quality safety

For the main V100 prefill attention kernel, ordinary versus compact restore changes cumulative time from **3.701 s to 1.951 s**, over 133 launches on both sides. The corresponding main RTX attention kernel changes from **1.458 s to 0.767 s**. This is consistent with the existing highest-occupied-cell attention-span issue. However, physical extent was not separately instrumented, and timing evidence does not resolve the known numerical differences from compact restore. That feature remains experimental.

## Revised priorities

1. **For TG/MTP:** target verification plus synchronization/readback and two-GPU scheduling. First identify which waiting is necessary dependency work, which is host overhead, and whether a small safe pipeline improves overlap.
2. **For PP:** resolve compact-layout numerical equivalence, then reduce excess attention span and mask work. The Nsight capture supports this being a substantial compute saving.
3. **For direct-Q8 kernels:** redesign around the measured register pressure and memory-instruction stalls before adding larger query packs. Avoid multiplying experiments whose bottleneck is already visible.
4. **For interactive service:** keep evaluating tile-aligned prefill admission as a responsiveness option; earlier tests did not establish a universal throughput win.

Production `v100-optimized` and `local-llm-setup` remain unchanged. Raw `.nsys-rep`, `.ncu-rep`, and SQLite exports are retained in the persistent Development Sandbox under `/workspace/oai-qwen38-pp-lab/results/nsight-parallel-0907/`. `trace-manifest.json` records report sizes and hashes; large trace binaries are not committed.

## Reproduction

The harnesses require the existing local GGUF and cache fixtures. `markers.c` is a driver-only NVTX helper; `mtp_markers.cpp` is an interposition library used only for profiling. Neither is linked into the production server. Compile the latter with `-Iinclude -Iggml/include -I/usr/local/cuda/include -shared -fPIC -ldl`.

Systems: `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-event-trace=false --cuda-graph-trace=node -o attributed python3 profile-attributed.py`, then export SQLite and run `analyze.py attributed` and `attribute.py`.

Compute: select only the V100, set `GGML_CUDA_VOLTA_Q8_MULTI=1` and pack 4 or 16, and run `ncu --kernel-name-base demangled --kernel-name 'regex:ggml_q8multi.*partial_kernel' --launch-count 1` around `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=2,nr23=\[8,1\],kv=4096,nb=16,'`. Collect LaunchStats/Occupancy/SpeedOfLight and WarpStateStats/SchedulerStats/MemoryWorkloadAnalysis separately. Use `test`, not `perf`: the targeted cases are in the correctness-case list; the initial perf invocation selected no kernel and supplied no evidence.

NVIDIA references: [Systems graph-tracing overhead](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [NVTX projection and post-collection interpretation](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html), [Compute profiling, replay and hardware metrics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

# Flash-Next offload research notes - 2026-09-05

This note is a design gate before further performance implementation. Do not add a new expert scheduler/cache until the manual dual-GPU baseline and profiles below are complete.

## Target constraints

- Qwen3.8-Flash-Next UD-IQ4_XS in llama.cpp.
- Native 262144-token context, q8_0 KV.
- Hardware: CUDA0 V100-SXM2 32GB (SM70), CUDA1 RTX 2080 Ti 22GB (SM75), PCIe Gen3, no useful P2P/NVLink path.
- Measured H2D: about 13.1 GB/s per GPU, about 26.1 GB/s aggregate when both transfer concurrently.
- N-gram/PLE table should remain mmap/lazy where possible rather than consuming VRAM. FreeToken itself currently pins the Flash-Next PLE table in host RAM, so that part should not be copied blindly.
- Preserve routed-expert semantics first. Lossy expert substitution/deferral is a separate experiment.

## FreeToken

Paper: https://arxiv.org/abs/2608.16157
Code inspected locally: /workspace/freetoken-feasibility-0905, af71ba4

Current FreeToken explicitly supports Qwen3.8-Flash-Next FP8 and NVFP4, but not this Unsloth mixed GGUF directly. Its mechanisms are still highly relevant.

### Prefill

Long prefill activates almost all experts in each layer, so decode-style sparse demand fetching is the wrong model. FreeToken reserves two full expert-layer buffers and overlaps copying layer L+1 on a dedicated CUDA stream while computing layer L. Current source uses CUDA events to protect buffer reuse.

The newer source can split prefill rows into cache hits and misses: resident rows are gathered D2D, while only missing expert-id runs cross PCIe via batched async copies. It also has a driver-specific safeguard for small mixed copy entries that can accidentally make cudaMemcpyBatchAsync synchronous.

Implication for us: first test whole-layer double buffering. The more complex hit-D2D prefill path is only worth implementing if profiles show a useful persistent hit region and the extra synchronization remains hidden on SM70/SM75.

### Decode

FreeToken maintains a shared GPU LRU expert-slot cache. Its hybrid backend does not insist on fetching every miss. It fetches a fraction over PCIe and computes the remaining misses on CPU concurrently.

Its bandwidth calibration chooses approximately:

    fetch_fraction = PCIe_bandwidth / effective_CPU_expert_bandwidth

and rounds the number of fetched misses so GPU transfer and CPU overflow work finish as close together as possible.

Implication for us: measure *effective quantized expert execution throughput* on the EPYC and actual pinned H2D throughput, rather than choosing a transfer threshold from theory. With two GPUs, host DRAM is shared by CPU expert reads and both H2D streams, so calibration must include contention.

### Other useful implementation details

- Cache metadata/routing stays on GPU to avoid per-layer D2H synchronization.
- Persistent slots have explicit expert->slot and slot->expert maps.
- Decode statistics accumulate on device rather than forcing host reads per token.
- The source separates prefill double buffers from the persistent decode cache lifecycle even though storage may be borrowed from the same pool.

## llama.cpp expert-cache work

RFC / measurements: https://github.com/ggml-org/llama.cpp/discussions/24528
Current PR lead: https://github.com/ggml-org/llama.cpp/pull/27861
Weight-prefetch lead: https://github.com/ggml-org/llama.cpp/pull/21067

Important lessons from the RFC and follow-up measurements:

- Synchronously moving every CPU MoE miss to GPU can regress decode badly. Residency is required; PCIe transfer alone is not a win.
- Earlier cache prototypes lost performance from per-layer CPU/GPU rendezvous even with very high cache hit rates.
- A direct-resident path on Qwen3.6 avoided copying cache hits into another scheduler buffer and improved throughput. GPU-resident routing IDs and incremental source-pointer metadata reduced CPU preparation overhead further.
- Experiments reported that smarter eviction/prediction often improves hit rate much less than reducing the *cost of a miss*. This is consistent with our older predictor work: prediction is not the first mechanism to revive.
- Prefetch is complementary to residency: large-ubatch/prefill wants latency-hidden transfer; ubatch-1 decode wants hot weights not to be retransferred.

## Other papers / systems

### Fiddler
https://arxiv.org/abs/2402.07033

Key idea: for misses, use CPU computation to avoid expensive CPU->GPU expert movement. Relevant to the CPU side of FreeToken's hybrid policy.

### MoE-Infinity
https://arxiv.org/abs/2401.14361

Uses request-level expert activation traces for caching and prefetch. Relevant evidence that routing reuse exists, but prediction/cache policy should come after transport and residency overhead are under control.

### HOBBIT
https://arxiv.org/abs/2411.01433

Combines token-level expert loading, layer-level adaptive prefetch, sequence-level caching, and mixed-precision miss handling. Its low-precision substitutions can change numerical behavior, so treat that as a later quality/speed tradeoff, not baseline optimization.

### KTransformers (SOSP 2025)
https://doi.org/10.1145/3731569.3764843
https://ktransformers.net/en/docs/technical-work/heterogeneous-inference

Maps model components by arithmetic intensity and uses asynchronous CPU/GPU scheduling. Expert Deferral increases overlap but is allowed a small accuracy change, so the scheduling principle is relevant while deferral itself is not first-line for exact-output work. Its current long-context docs also use chunked/layerwise prefill, which reinforces separating long-prefill scheduling from decode.

### SP-MoE
https://arxiv.org/abs/2510.10302

Speculative-decoding-aware expert prefetch. Uses draft/target structural correspondence, a cutoff-layer policy to avoid over-prefetch, async prefetch threads, and batched I/O. Relevant only after ordinary decode/offload is efficient and when MTP is re-enabled.

### MoE-SpeQ
https://arxiv.org/abs/2511.14102

Uses a small on-device draft model to predict future expert needs and an amortization/roofline governor. This is the closest literature analogue to our older predictor idea, but it reinforces that prediction must be evaluated against transfer lead time and amortization, not prediction accuracy alone.

### MoEpic
https://arxiv.org/abs/2509.08342

Vertically splits experts so a hot segment of more experts can be resident under the same VRAM budget, then predicts/prefetches the remainder. Interesting if full-expert residency coverage is poor; requires nontrivial GGUF/kernel layout work and should come after a conventional cache baseline.

### Kernel-managed tiering study
https://arxiv.org/abs/2608.12103

Reports that simple kernel/page-cache LRU can be surprisingly competitive with model-specific frequency policies at large scales, while router lookahead gave little benefit in its regime. This is another warning against spending early effort on sophisticated eviction/prediction policies.

## Ordered mechanism plan for this hardware

1. Finish upstream merge and correctness validation.
2. Establish the baseline with `--fit off`, explicit `--device CUDA0,CUDA1`, explicit `--tensor-split`, explicit `--n-cpu-moe`, full 262144 context and q8 KV. Sweep placement manually; do not use auto-fit as the reference.
3. Measure best manual placement both with ordinary kernels and the already-existing Volta MoE MMQ switch. This defines the no-new-runtime-code baseline.
4. Validate 261632 prompt + 512 generation capacity on the chosen placement.
5. Profile prefill and decode separately: GPU kernels, GPU idle gaps, CPU expert time, H2D bytes/time, synchronization, routing readbacks, and per-GPU VRAM headroom.
6. Measure isolated/contended CPU expert bandwidth and pinned H2D bandwidth to both GPUs.
7. First new mechanism: prefill whole-layer double-buffered transfer if transfer is exposed on the critical path.
8. First decode mechanism: persistent direct-resident expert slots with GPU-side routing metadata; on a miss, compute on CPU by default rather than synchronously forcing a GPU upload.
9. Add FreeToken-style calibrated CPU-vs-PCIe miss splitting only after the basic direct-resident path is correct.
10. Only then test prediction/prefetch, MTP-aware SP-MoE-style lookahead, vertical expert splitting, or lossy mixed-precision/deferral ideas.

## Acceptance

Every claimed improvement needs controlled repeated runs, identical model/context/workload, output/hash checks, VRAM/RAM/transfer accounting, and separate PP/TTFT/TG results. Once MTP is enabled, record acceptance counts too.

## 2026-09-05 follow-up: current llama.cpp cache implementations

Current upstream work also includes PR #26824 (WackMall-style expert cache) and PR #27861 (GPU-resident LRU cache). #26824 exposes explicit expert slots, move/copy modes, mmap-page pinning, heatmaps/sidecars, hysteresis, dwell time, sync cadence and fill-only/no-evict modes. These are useful implementation leads, especially for instrumentation and avoiding churn, but they do not override the gate above: first measure the manual two-GPU baseline and the miss/synchronization costs on this exact Qwen4/SM70+SM75 system.

The relevant literature converges on the same ordering:
- FreeToken: hardware-calibrated heterogeneous execution and cache/offload policy.
- Fiddler: for small decode batches, moving tiny activations and computing misses on CPU can beat moving whole expert weights.
- KTransformers: asynchronous CPU/GPU scheduling matters as much as placement; current public serving is primarily Ampere+, so use it as a systems-design reference rather than a drop-in SM70/75 runtime.
- SP-MoE / MoE-SpeQ: prediction is valuable when it creates enough *lead time* to hide transfers; speculative verification can otherwise increase the union of experts and PCIe pressure.
- MoEpic: vertical expert splitting can increase effective cache coverage, but it is a later layout/kernel experiment.
- HOBBIT: mixed-precision miss handling is explicitly a quality/performance tradeoff and remains outside the exact-output baseline.

## Manual layer split detail

`--tensor-split` with `--split-mode layer` assigns repeating layers by normalized split fraction, not by their actual byte sizes. `--n-cpu-moe N` then keeps the routed expert tensors of the first N layers on CPU; it does not reduce the later layers placed on CUDA1. Therefore a nominal VRAM-ratio split such as 3:2 overloads the 22 GB RTX with the same final-layer block regardless of `n-cpu-moe`.

For this 48-layer model the output acts as the 49th placement unit. Prefer explicit count-like ratios such as `35,14`, `36,13`, etc., then measure actual VRAM. `per_layer_token_embd.weight` is created with `TENSOR_READ_LAZY`; keep an explicit CPU override in benchmark commands so the intended PLE placement is reproducible and remains separate from the expert-placement sweep.

## Hardware-specific implication of the selected manual split

The explicit full-context frontier currently selected for the 1000-token append is:

    --device CUDA0,CUDA1 --split-mode layer --tensor-split 36,13
    --n-cpu-moe 18 --ubatch-size 1024 --fit off

This is important for interpreting FreeToken rather than copying its single-GPU policy literally:

- Only routed experts in layers 0..17 are host-resident; later routed experts remain statically GPU-resident.
- The full routed-expert pool is ~55.432 GiB, or ~1.155 GiB per layer and ~2.31 MiB per layer/expert.
- Therefore the host-resident routed-expert pool in this baseline is about 20.79 GiB (18 layers), not the whole 55.4 GiB.
- At the measured ~13.1 GB/s H2D rate of one PCIe Gen3 link, streaming all 18 host expert layers once is roughly a 1.7 s best-case transport budget (GiB->GB included), before synchronization/copy setup. This is small enough that full-layer prefill double buffering can plausibly hide much of it behind the target 1k prefill, but it must be profiled.
- The measured ~26.1 GB/s aggregate two-GPU H2D rate cannot simply be substituted into that estimate: in stock layer-split execution, layers 0..17 belong to CUDA0/V100, so their expert fills target one device. Exploiting both links would require a real multi-device expert executor/cache, not merely `--tensor-split`.
- Decode touches about 18 * 10 * 2.31 MiB ~= 416 MiB of host-resident routed weights per token before locality/cache effects. That makes direct CPU execution strongly bandwidth-sensitive and makes FreeToken's measured B_H/B_P policy relevant.
- The RTX is otherwise idle while early V100-owned layers execute. A later multi-device cache design could potentially use spare RTX residency/compute for early-layer hot experts by moving tiny activations rather than whole expert weights. This is an inference/design hypothesis, not a measured optimization yet.

The immediate measurements required before implementing that hypothesis are: effective quantized CPU expert bandwidth, H2D bandwidth under simultaneous CPU reads, per-device idle timelines for PP/TG, and available decode-time VRAM after the full-context buffers are allocated.

### FreeToken Qwen3.8-Flash-Next compatibility boundary

Current FreeToken `docs/models.md` explicitly supports Qwen3.8-Flash-Next FP8 and NVFP4 HF checkpoints and exposes `offload`, `cpu`, and calibrated `hybrid` MoE backends. It does **not** list the mixed Unsloth GGUF used here; FreeToken only documents native GGUF loading for Gemma-4. Its Qwen3.8-Flash-Next path also pins its 47.7 GiB PLE n-gram table in host RAM, whereas the inspected UD-IQ4_XS GGUF contains a ~26.82 GiB lazy PLE tensor. Therefore reuse the scheduling mechanisms, not the loader/storage assumptions.

Reference: https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md

### llama.cpp #21067: prefetch evidence close to our regime

The current weight-prefetch PR has unusually relevant data: Laguna-S-2.1, 48 layers, 256 experts, top-10, 2x RTX 3090 without NVLink, 28 CPU expert layers, and 262K context. Whole-layer prefetch improved 10K/90K prefill by ~14%, but short-prompt TTFT regressed badly because prefetch forfeits selective-expert copying. A batch-size gate recovered TTFT; the reported crossover was roughly 250-1040 MoE batch. Decode still remained slightly below baseline because the worst-case double buffer permanently consumed ~15% of VRAM.

This is directly relevant to our 1000-token append and very tight full-context VRAM frontier. Do not reserve a permanent two-layer prefetch buffer in the first implementation. If profiling validates the transport hypothesis, prefer phase/safe-point memory reuse (or cache slots temporarily borrowed during PP), and retain selective-expert copy below the measured crossover. Also avoid the PR's global `--no-mmap` requirement: pin/register only the CPU expert transfer pool so the large PLE table can remain mmap/lazy.

Reference: https://github.com/ggml-org/llama.cpp/pull/21067

### llama.cpp #25859: serial H2D and mmap page-locking

A July 2026 Qwen3.6-35B-A3B report (`-ncmoe 26`, pp/ub2048) profiles the current offload path as serial H2D followed by expert compute. Selectively page-locking the mmap-backed CPU expert weights improved PP ~21%; adding a second transfer stream / staging slots to upload layer N+1 while layer N computes improved 1143 -> 1880 tok/s (~64%) with token-identical output. This is not evidence that our exact SM70/SM75 setup gets the same gain, but it separates two low-risk mechanisms worth measuring independently: registration/pinning overhead and exposed copy/compute serialization.

Reference: https://github.com/ggml-org/llama.cpp/issues/25859

### Static residency versus a same-VRAM dynamic cache

The selected static baseline keeps complete routed-expert pools for layers 18..47 on GPU: 30/48 layer-equivalents (~62.5% of all routed-expert bytes), while layers 0..17 always take the host path. This means roughly 37.5% of routed layer invocations are forced through the CPU/offload path independent of actual expert locality.

A future cache comparison should therefore be *same VRAM*: convert the V100's static all-or-nothing expert layers into per-(layer,expert) slots rather than simply adding cache memory. The V100 owns layers 0..35 and currently has about 18 layer-equivalents of expert storage there, i.e. ~50% capacity across its owned expert pool. The RTX owns layers 36..47 and can currently keep those expert pools fully resident.

FreeToken reports that at equal cache capacity its per-miss LRU substantially beats routing-blind static placement on Qwen3.6, but that is not a prediction for Qwen3.8 Flash-Next. Before implementing the cache, record/replay this model's actual `(layer,expert)` route stream from the 100k+1k+512 workload and compute the theoretical LRU miss curve at the exact V100 slot budget. If the trace does not show enough locality, do not build the cache.

### FreeToken implementation cross-check (current repository)

The current FreeToken code matches the paper's phase split, not just the prose. `OffloadMoeCache._init_prefill_overlap_buffers()` borrows the first `2 * num_experts` slots of the unified MoE cache as two complete layer buffers. `OffloadMoELayer._wait_prefill_overlap()` requests layer L and L+1; `prefetch_prefill_layer()` copies a whole layer on a dedicated CUDA stream, and ready/release events fence compute consumption and ping-pong reuse. Decode is a different path: LRU/on-demand, with the hybrid mode capping PCIe misses and sending the remainder to the CPU executor.

For the Qwen3.8 Flash GGUF baseline, the first experimental port intentionally uses one *weight-tensor* lookahead rather than two full layer buffers because `36,13 / n-cpu-moe=18 / u1024` has only about 1 GiB of V100 headroom. This is a memory-feasibility probe, not a claim that the narrower pipeline is equivalent to FreeToken. If overlap is too shallow, the next controlled candidate should move one additional expert layer to CPU (freeing ~1.15 GiB) and spend that VRAM on a true whole-layer ping-pong buffer, keeping total VRAM approximately constant.

Source inspected locally: `/workspace/freetoken-feasibility-0905/python/freetoken/moe/offload_cache.py` and `python/freetoken/layers/moe.py`.

### Measured outcome of the first FreeToken-inspired port

A low-VRAM one-weight-ahead variant is effective on this exact SM70/SM75 setup. With `n-cpu-moe=18`, `ubatch=1000`, pinned host expert buffers and Volta MoE MMQ, two repeated 100k-cache + 1k runs reached ~224.5 PP versus ~212.5 PP for the same placement/load mode without prefetch. Nsight on the earlier n19 variant directly measured ~483 ms of V100 H2D/kernel overlap. Thus transfer/compute serialization is a confirmed bottleneck, not only a paper-derived hypothesis.

The full-layer FreeToken/PR-21067 shape is not a good direct fit under the 262k VRAM budget: its extra live buffers force more expert layers to CPU or smaller ubatches. On this workload the narrower pipeline preserves enough residency and the single 1k microbatch to win end-to-end. Decode remains a separate problem; do not generalize the prefill mechanism into a decode cache policy.

## Mechanism ranking after broader 2026 review (2026-09-05)

The selection criterion is end-to-end speed on this exact V100 32 GiB + RTX 2080 Ti 22 GiB, PCIe Gen3, 262144-context Qwen3.8-Flash-Next deployment, not predictor accuracy in isolation.

### 1. Prefill: explicit stream-loading/ring buffers (highest priority)
FreeToken and Wang et al. OSDI'26 independently converge on streaming expert layers during prefill rather than decode-style sparse caching. OSDI'26 uses a manually managed Experts Ring Buffer whose depth is dynamic: more look-ahead when transfer dominates, shrinking to two ping-pong slots when long-context KV pressure dominates. That is a closer fit to our 262k constraint than scheduler/gallocr lifetime extension. Our compact one-weight experiment already proves the underlying opportunity: ~0.483 s of V100 H2D/kernel overlap was created and 1k PP improved from ~212.5 to ~224.5 tok/s at identical n-cpu-moe=18/u1000, while much H2D remains exposed.

Implementation target: explicit reusable device buffers, pinned host source, copy stream + ready/release events, phase/density-aware depth. Do not infer that full-layer streaming is unpromising from the failed gallocr-fusion prototype; that prototype paid general graph-lifetime VRAM, unlike a fixed ring.

### 2. Decode: exact GPU expert cache + CPU fallback (highest TG priority)
llama.cpp PR #27861 / commit bccbacdb8 was motivated directly by Qwen3.8-Flash-Next (512 experts, top-10): LRU-64 reportedly achieved ~67% temporal hit rate on a mixed workload and improved warm decode 15.5 -> 19-20 tok/s on its 2x3090 host. It splits each decode layer exactly: cached experts execute from compact GPU cache slots, uncached experts execute on CPU, and the results sum exactly. A port onto the merged branch is isolated at /workspace/llama-flashnext-moecache-0905 (2420a1372).

For IQ4_XS one complete gate+up+down expert is only ~2.222 MiB, so 18 host MoE layers cost ~0.625 GiB at 16 slots/layer, ~0.937 GiB at 24, and ~1.25 GiB at 32. Sweep placement and cache size jointly rather than assuming K=64.

### 3. Decode miss scheduling: FreeToken/Fiddler hybrid
If a useful cache still leaves misses, benchmark the real expert-shaped CPU kernel bandwidth and PCIe gather bandwidth, including concurrent contention. Divide misses so CPU and PCIe/GPU finish at about the same time rather than transferring every miss. Fiddler provides the same general lesson: choose CPU-vs-GPU work to minimize max(CPU work, GPU/transfer work), not to maximize GPU residency.

### 4. DualDeadline component staging
DualDeadline observes two exact deadlines in a gated expert: gate/up are needed first; down can arrive while gate/up arithmetic executes. Its llama.cpp research branch has a persistent per-tensor LRU and staged-vs-monolithic A/B. The existing prototype synchronously reads routed IDs to host per MoE layer and disables CUDA graphs, so port the *staging principle*, not that synchronization bottleneck verbatim. It is most attractive as a refinement to an efficient device-side cache/miss path.

### 5. Prediction only after cache mechanics are efficient
FATE/ProMoE/pre-attention/MoE-Infinity/DALI can extend the overlap window, but prediction is useless if the cache churns before use or if cache bookkeeping costs exceed the avoided CPU/PCIe work. First measure Qwen3.8's actual 512-token routing trace. Then test training-free temporal/cross-layer prediction; only train a predictor if the measured ceiling warrants it.

### 6. MTP/speculation later: MoE-SpAc
MoE-SpAc uses speculative execution as an expert-demand look-ahead signal and utility-ranked asynchronous cache. This is unusually relevant because Flash-Next has MTP, but it should be evaluated only after ordinary decoding has a good cache/miss executor. Otherwise speculative expert over-activation can increase work faster than accepted-token speedup.

### Additional exactness rule
Backend performance claims keep native router selections and arithmetic exact. Approximate fallback ideas (SPICE low-rank surrogates, HOBBIT low-precision miss replacement, stall-free APEX) belong to a separate quality/performance track and must never be mixed into the exact baseline.

### Qwen3.8 coding-workload route trace: cache/prediction reality check

A 512-token decode route trace was collected for the exact 100k-restored + 1k coding workload, covering the 18 host-resident MoE layers of the selected `36,13 / n-cpu-moe=18` placement. The last 511 complete token cycles contain 91,980 expert activations (18 layers x top-10 x 511 tokens).

Measured same-layer overlap with the immediately preceding token is only 1.339/10 experts on average (median 1). It declines materially with depth: layer 1 is ~2.29/10 while the final host layers are ~0.72-0.84/10. Simple previous-token prediction therefore has only ~13.36% recall at ~10 candidates/layer; unioning the previous two tokens reaches ~19.09% recall at ~18.6 candidates. Cross-layer reuse is negligible (~1.93% recall from the preceding layer).

Trace-replayed per-layer LRU hit rates with unlimited admission are approximately: K16 17.3%, K24 21.2%, K32 24.6%, K48 31.0%, K64 37.5%. With the current llama.cpp cache PR's two-insert/layer/step throttle the same replay gives only ~11.5%, 15.1%, 18.3%, 24.7%, and 30.3%. Therefore the PR's published ~67% LRU-64 mixed-workload hit rate must not be used as a performance assumption for this coding workload.

A global phase-shared replay also shows that merely replacing per-layer LRU with global LRU is not enough. At ~0.94 GiB of IQ4_XS expert capacity (432 complete expert slots), global LRU is ~19.2% while an offline Belady ceiling is ~45.7%; at ~2.22 GiB (1024 slots), LRU is ~33.1% while Belady is ~60.8%. This gap says the interesting cache research direction is lookahead-aware admission/eviction (SpecMD/utility scheduling), not a generic global LRU.

Artifacts: `qwen38-route-trace.log`, `qwen38-route-analysis.json`, `qwen38-global-cache-sim.json`, `analyze_routes.py`, and `sim-global-cache.py`.

### Decode transfer regime: zero-copy/UVA versus explicit staging

Forcing `GGML_OP_OFFLOAD_MIN_BATCH=1` on the pinned-host configuration produces ~14.64 TG at 100k context, essentially tied with ordinary CPU-expert decode. Nsight, however, records only ~0.23 GB V100 H2D + ~0.12 GB RTX H2D during a profiled 64-token generation, orders of magnitude below the routed expert-byte volume. The CUDA kernels are therefore predominantly reading pinned host expert memory in place across PCIe/UVA rather than staging every selected expert slice into device memory.

That distinction matters for DualDeadline and FreeToken. An explicit compact staging cache is not merely trying to overlap an already-efficient memcpy path; it can replace scattered GPU reads over PCIe with coalesced asynchronous H2D transfers into local VRAM, then overlap the down projection's transfer deadline with gate/up compute. The faithful DualDeadline prototype must still be measured because its per-layer host readback/synchronization and CUDA-graph disable can offset this benefit.

### Long-context priority

Full-capacity execution is now proven at `261632 prompt + 512 generation` with context shifting explicitly disabled. The monolithic capacity request reported only 3.427 tok/s, but a reusable 261632-token state plus the append-only restored-hybrid-cache fix reproduces the original continuation bit-for-bit at 8.247 tok/s for 511 tokens. Therefore 3.427 tok/s was not the intrinsic max-context decode floor. Even the corrected 8.25 tok/s remains far below the ~14.4 tok/s 100k-context rate, so Qwen4 QSA/indexer scaling is still a first-class optimization target. Standalone sparse FlashAttention is neutral at both ~100k and max context; the likely win is the broader 4-head Lightning Indexer + incremental pooled-key cache + selected-cell gather chain.

## 2026-09-05 synthesis: frequency placement outranks churn-heavy LRU for this workload

The Qwen3.8 route trace and independent SGLang results both show a strong distinction:
- adjacent-token temporal reuse is modest (our CPU-layer trace ~1.34/10 experts), so LRU must churn to capture much;
- long-window expert-frequency skew is strong (our trace top-171 ~82%, matching the independent SGLang campaign; top-320 ~98% in-sample and ~96% on a chronological holdout after 256-token adaptation).

A complete UD-IQ4_XS expert slice is ~2.2217 MiB. Keeping K=320 experts in every one of the 48 layers consumes ~33.33 GiB of routed-expert VRAM, approximately the same raw expert budget as the current 30 fully GPU-resident expert layers. Therefore an equal-budget *per-expert* placement can, in principle, replace whole-layer residency without increasing expert-weight VRAM.

This changes the preferred decode architecture:
1. Put all routed-expert source tensors in pinned CPU memory (exact source of truth).
2. Allocate fixed per-layer GPU slots from the same VRAM currently spent on whole expert layers.
3. Build the hot set from prompt routing statistics and/or the first 64-256 decode tokens.
4. Freeze residency for a useful epoch; GPU computes hits, CPU computes misses and the exact outputs are summed (reuse the exact split-graph mechanism from the llama.cpp expert-cache PR).
5. Rebalance only with hysteresis / measured benefit, not every miss. A transfer must earn back its cost over expected future hits.
6. FreeToken/Fiddler bandwidth balancing remains useful for misses: some misses can be fetched if the copy engine is otherwise idle, the rest compute on CPU.
7. Prediction (FATE/ProMoE/MTP lookahead) becomes optional: use it to improve the *next epoch's* hot set or opportunistic miss transfers, not as a requirement for correctness or every-layer progress.

This also agrees with earlier local experiments where continuous expert replacement was costly and freezing admissions after an early warmup was more promising.

Before implementing, collect routed IDs for all 48 layers (current trace covers the 18 host layers only) and redo train/holdout coverage. K-only QSA/indexer storage can provide the allocator margin needed to make an equal-budget K~300-320 placement practical at 262144 context.

## 2026-09-05: gather-QSA validation changes the long-context ranking

A controlled max-context A/B now proves that Qwen3.8's QSA selection only becomes useful when the selected original K/V cells are physically compacted before attention. From the same 261632-token restored state, the dense/masked path is 8.2474 TG while gather-QSA is 11.3756 TG over two 511-token reps (+37.93%), with identical output hash and final cache position.

This moves QSA/indexer work ahead of further expert-cache work for native-context TG. The remaining QSA depth cost is the indexer: current llama.cpp still reconstructs historical block-summary keys every token. SGLang's Qwen3.8 design and the `qwen4exp-spec-mtp` research branch both keep one persistent compressed index key per 4-token block and update only newly completed blocks. That mechanism is now the highest-probability exact-output follow-up, followed by overlapping indexer work with the main Q/K/V projection on a separate CUDA stream.

## 2026-09-05 correction: pooled indexer state was not validated

The first persistent completed-block QSA index-cache result was invalid: the experimental fast path was gated by `!ubatch.is_pos_2d()`, while ordinary Qwen3.8 text uses multi-position M-RoPE and therefore reports `is_pos_2d()==true`. The apparent 11.77 TG result was the established gather-QSA fallback plus run-to-run variance, not a pooled-cache gain. Removing that bad gate really activates the generic persistent `SET_ROWS` implementation, but that implementation stalls in live graph setup/execution on the heterogeneous V100 + RTX 2080 Ti setup. Treat the *mechanism* as promising but completely unvalidated here. A future attempt must use a purpose-built incremental block-summary store rather than the old generic external `SET_ROWS` design.

The remaining verified QSA problem is still context-sized work: block summaries/scores are repeatedly rebuilt and block scores are expanded to per-cell scores before top-k. The corrected block-first experiment must prove activation in Nsight (the old test had the same `is_pos_2d()` gating mistake) and must compare output against the dense reference, not against an accidentally inactive experimental path.

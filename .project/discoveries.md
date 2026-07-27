# Discoveries Log

This file records experimental findings, successful ideas, dead ends, and hypotheses so work can continue across sessions.

## 2026-07-26

### 1. Vertical cache locality is real, but complete-layer locality is the correct objective

Finding:
- Individual expert residency reached ~75% after warmup.
- Complete 8/8 coverage remained much lower (~20–23%).
- Zero tokens achieved a fully GPU-resident MoE path across all dynamic layers.

Conclusion:
Optimize 8/8 completion, not average expert hit rate.

---

### 2. Prefill learning matters

Tried:
- Added multi-token prefill observation.
- Added bounded warm-start.

Result:
- Decode starts substantially warmer.
- Locality improves immediately.

Conclusion:
Prompt observations should seed the cache before measured decode.

---

### 3. Fixed-admission policy is too conservative

Observed:
- Cache required most of the decode to fill.
- Large numbers of candidates rejected by admission limits.

Conclusion:
Replace admission-count limits with byte/time/queue budgets.

---

### 4. LRU-style protection is preferable to pure score decay

Implemented:
- Protected/probationary metadata.
- Recency tracking.

Reason:
Previous policy churned aggressively.

Next:
Measure churn reduction on long traces.

---

### 5. Predictor direction

Implemented:
- Online same-layer transition predictor.

Finding:
Useful for asynchronous prefetch.

Next:
Predict the final missing expert (7/8 -> 8/8) instead of maximizing popularity.

---

### 6. Biggest architectural discovery

A/B experiments showed:
- Dynamic graph overhead dominates before cache quality does.
- Removing CPU fallback alone improves speed only modestly.
- Removing the cold branch from graph construction improves further.

Conclusion:
Execution topology is now the primary bottleneck.

---

### 7. CPU routing callback

Trace discovery:
- One CPU routing callback per dynamic MoE layer per decode token.
- Prevents larger CUDA graph fusion.

Current prototype:
- GPU-side route remapping inside CUDA MUL_MAT_ID.

Status:
Prototype under development.

---

### 8. GPU-only experiments

Unsafe experiments:
- Force GPU-only execution.
- Remove CPU branch.

Outcome:
Performance improves but output diverges.

Use only as an architectural probe.

---

### 9. Capacity

Finding:
Per-layer reallocation helps less than expected.

Much larger gains come from:
- warm-start
- better replacement
- removing scheduler overhead.

---

## Ideas backlog

- Freeze cache after warmup for architecture benchmarking.
- Device-side route observation.
- GPU route remapping using persistent slot maps.
- Deadline-aware predictor.
- Pipeline states instead of cache states:
  cold -> predicted -> copying -> ready -> protected -> aging -> evictable.
- Persistent cache across coding-agent turns.
- Forced-token replay for deterministic comparisons.

---

### 10. CUDA graphs are topology-dependent, not universally beneficial

Matched 64/64 coding matrix:

- baseline, graphs off: 21.50 t/s;
- baseline, graphs on: 20.60 t/s;
- fragmented GPU-only graph, graphs off: 13.34 t/s;
- fragmented GPU-only graph, graphs on: 14.00 t/s;
- forced-rebuild GPU route-map, graphs off: 20.25 t/s;
- forced-rebuild GPU route-map, graphs on: 20.22 t/s.

Conclusion:

- CUDA graph capture does not fix a fragmented scheduler topology.
- Capture helped the fragmented GPU-only probe modestly, but the graph remained far slower than baseline.
- Once the CPU route callback was removed and topology stabilized, capture became nearly neutral on this V100 workload.
- The best setup is likely phase-specific rather than a single global graph setting.

Recommended phase model:

1. Prefill: ordinary execution; capture choice determined separately by prompt throughput.
2. Cache population or refresh: capture disabled or invalidated while slot state changes.
3. Steady decode: freeze cache/slot-map addresses, rebuild once, then optionally capture.
4. Refresh boundary: leave steady graph, promote/replace, rebuild once, resume.

---

### 11. Removing arithmetic is not enough; removing submissions is decisive

Finding:

The fragmented GPU-only graph performed less expert arithmetic but remained at 13–14 t/s because it still issued thousands of small CPU/GPU scheduler submissions.

The forced-rebuild route-map probe recovered approximately 20–21 t/s by removing the per-layer CPU route callback from the measured phase.

Conclusion:

The primary speedup came from reducing host intervention and stabilizing graph topology, not merely from dropping expert GEMMs.

---

### 12. Explicit graph phase markers are required

Finding:

Whole-run cache and submission summaries mix warmup and measured phases. This made route-map traces initially appear to retain CPU callbacks even though the callback count exactly matched warmup tokens only.

Action:

Added explicit trace markers around warmup completion and graph rebuild so future analysis can separate phases without inference.

Conclusion:

Every benchmark that changes graph or cache policy after warmup should emit a machine-readable phase marker.

---

### 13. Correctness remains unresolved for GPU route-map probes

Observation:

Some unsafe route-map runs generated the same output bytes as baseline even though uncached expert contributions may be omitted.

Conclusion:

Matching free-running text is not evidence of numerical equivalence. Forced-token logit replay remains mandatory before any correctness claim.

---

### 14. Exact post-rebuild topology

Explicit phase markers plus top-level scheduler tracing established the measured topology after warmup:

- one scheduler compute call per measured token;
- graph size: 4,471 nodes and 983 leaf tensors;
- 68 backend splits;
- zero CPU route callbacks;
- zero dynamic split-prepare events;
- zero scheduler-level split transfers during the measured phase;
- zero explicit target-sync events during the measured phase.

The same topology appeared with CUDA graph capture enabled and disabled.

Measured scheduler-return time over eight tokens:

- graphs disabled: mean 47.63 ms;
- graphs enabled: mean 48.40 ms.

Conclusion:

The CPU routing callback has genuinely been removed from the measured route-map phase. The next execution bottleneck is not route mapping or transfer—it is the remaining 68 static backend splits and the work inside the 4,471-node graph.

Important correction:

The earlier absence of dynamic split events was initially attributed to CUDA graph replay. The graphs-disabled trace showed the same absence. The correct interpretation is that the rebuilt route-map topology no longer enters the dynamic split paths being traced; top-level scheduler tracing is required.

---

### 15. Best graph setup so far

For the short V100 coding workload:

- ordinary baseline prefers CUDA graphs disabled by a small margin;
- fragmented GPU-only topology benefits modestly from CUDA graphs but remains slow;
- rebuilt GPU route-map topology is nearly indifferent to CUDA graphs;
- graphs disabled is currently the simplest and slightly fastest measured setting for the rebuilt route-map topology.

Recommended experimental setup:

1. Run warmup with dynamic cache population.
2. Freeze admissions and prediction.
3. Force one scheduler graph rebuild.
4. Use GPU route mapping with persistent slot-map addresses.
5. Keep CUDA graphs disabled on V100 unless a longer repeated benchmark proves otherwise.
6. Re-enable capture only after reducing the 68 backend splits and repeating the matrix.

---

### 16. Correction: the forced-rebuild GPU route-map result was not a valid cache measurement

Later graph-build diagnostics invalidated conclusions 10, 11, 14, and 15 where they describe the 20–21 t/s measured phase as an active resident-expert GPU route-map graph.

At the warmup boundary, `llama_context_force_graph_rebuild()` creates a new backend scheduler. Existing expert-cache allocations still consume VRAM, but their cache state is not inherited by the replacement scheduler. The replacement scheduler can calculate a zero expert-cache budget, plan zero dynamic slots, and build the ordinary graph instead.

The final `profiling/graph-build-gates-2026-07-26/` diagnostic showed:

- zero measured graph-build records with `gpu-route-map=1`;
- zero records with `force-gpu-only=1`;
- dynamic layers built with `feature=0`, `split-requested=0`, and `slots=0`;
- a 4,471-node, 68-split decode graph consistent with the ordinary horizontally placed topology.

Corrected conclusion:

- The valid fragmented resident-only mode remains 13.34–14.00 t/s versus the 21.50 t/s baseline.
- The 20.25/20.22 t/s route-map numbers do not prove that resident-only GPU route mapping approached baseline.
- The next experiment must use a startup-static frozen resident map or otherwise preserve cache ownership/state across graph rebuilding, with hard measured-phase assertions.
- See `.project/HANDOFF.md` for the authoritative next experiment.

---

### 17. Proven: startup-static frozen vertical resident-only is faster than horizontal auto-fit

The experiment specified in `.project/HANDOFF.md` was implemented and run on 2026-07-26. Full results are in `profiling/static-vertical-ceiling-2026-07-26/RESULTS.md`.

Three interleaved 64-warmup / 256-measured-token repetitions on the V100 produced:

- horizontal auto-fit: 20.69, 20.63, 20.79 t/s; mean **20.70 t/s**;
- frozen vertical resident-only: 42.11, 41.77, 42.43 t/s; mean **42.10 t/s**;
- skeleton-only control: 68.04, 67.84, 67.22 t/s; mean **67.70 t/s**.

The vertical arm is **2.03x** the horizontal baseline (+103.36%). All hard assertions passed in all repetitions:

- 48 static-mapped layers;
- 144 compact component buffers attached;
- 4,678 READY slots and zero COPYING slots;
- decode graph built with feature/split/force-GPU/GPU-route/static-map enabled;
- zero measured dynamic expert-copy groups and bytes;
- zero CPU cold expert calls;
- zero CPU route-decision synchronization calls;
- zero allocation failures.

Batch-one scheduler splits fell from 66 in the horizontal baseline to 7 in the frozen vertical graph.

This proves the mechanism-level performance ceiling but not correctness: nonresident routes are intentionally omitted. The next task is an exact, low-fragmentation miss path with forced-token/full-logit replay.

---

### 18. Exact CPU miss fallback restores route completeness but destroys the vertical speedup

The frozen static cache was extended with an exact complementary CPU miss branch that does not use the old host route-map callback. GPU `MUL_MAT_ID` remaps resident canonical IDs to compact slots; CPU `MUL_MAT_ID` reads the immutable slot map and internally skips resident IDs, computing only nonresident selected experts. The partial tensors merge on GPU before router weighting and reduction.

Three interleaved 64-warmup / 256-measured-token repetitions produced:

- horizontal auto-fit: **20.54 t/s** mean;
- ordinary `--cpu-moe`: **15.89 t/s** mean;
- static GPU hits + exact CPU misses: **8.55 t/s** mean;
- static GPU hits with misses omitted: **42.09 t/s** mean.

All exact assertions passed: 48 static layers, 144 attached component buffers, 4,678 READY slots, zero measured expert-weight copies, zero route-decision synchronization callbacks, and active CPU fallback islands.

The exact graph performs 49 CPU and 49 GPU scheduler submissions per token, with 56 H2D and 96 D2H transfer groups per token. Batch-one scheduler splits remain 98, versus 7 for resident-only. Exact throughput is 58.35% below horizontal and 46.16% below ordinary `--cpu-moe`.

A 64-step full-vocabulary teacher-forced coding replay showed:

- ordinary `--cpu-moe` vs exact split: 64/64 top-1 agreement;
- horizontal vs exact split: 62/64;
- horizontal vs ordinary `--cpu-moe`: 62/64.

The horizontal comparisons disagreed on the same two steps and selected the same alternatives. This supports algorithmic route completeness; remaining logit differences are mixed-backend numerical drift, not omitted expert routes.

Correct conclusion: a per-layer CPU miss island is mathematically exact but structurally incompatible with the proven vertical speedup. The next experiment must service misses on GPU, preferably with reusable spill slots and staged blocking uploads.

Full report: `profiling/static-exact-cpu-fallback-2026-07-26/RESULTS.md`.

---

### 19. Long-chat warming is real; cross-layer prediction improves cache quality but not yet speed

A 1,024-token chat-style generation with the existing dynamic policy showed a genuine warming curve:

- tokens 3-130: 4.72 traced t/s, 66.25% resident-route coverage, 16.13% complete layers;
- tokens 259-386: 6.23 traced t/s, 80.98% resident-route coverage, 40.84% complete layers;
- later windows settled near 6.0 t/s as routing drift reduced coverage.

Overall resident coverage was 76.48% and complete 8/8 layer coverage was 32.16%. The cache remained dynamic throughout: 10,209 admissions and 5,578 evictions occurred.

Offline prequential analysis showed adjacent-layer prediction is substantially stronger than the existing same-layer-across-token predictor. At K=8:

- marginal popularity: 27.54% route recall;
- same-layer temporal: 37.75%;
- layer L-1 conditional prediction of L: 50.45%.

Layer L-2 alone was almost as good as L-1 (50.07% versus 50.84% on the common target set). Combining L-2 and L-1 with a sparse joint interaction reached 55.64% recall at K=8 and 73.45% at K=16.

A two-stage live predictor was implemented:

- L-2 routes issue early predictions for L;
- L-1 routes issue later refinements;
- normalized conditional counts are learned online without look-ahead.

A forced-history 512-token A/B comparison produced:

- complete layers: 32.86% same-layer versus 33.55% cross-layer;
- mean CPU-dependent layers: 32.225 versus 31.894;
- 169 additional complete GPU layers;
- 116 fewer admissions and evictions;
- approximately 451 MiB less H2D traffic;
- identical aggregate traced throughput, 5.8043 t/s.

The largest gain was early warming (tokens 67-130): complete layers rose from 26.53% to 29.95% and traced throughput from 5.09 to 5.38 t/s.

Crucially, neither L-1 nor L-2 predictions became READY before the target layer in the same token. Median promotion latency was 7.64 ms for L-1 and 12.27 ms for L-2, and same-token READY count was zero. Around 61-62% were used later, with median first hot use about 40-41 tokens after admission. The current predictor improves future cache composition; immediate exact miss service must bypass the background promotion queue.

Full report: `profiling/dynamic-real-chat-ab-2026-07-26/RESULTS.md`.

### 20. Lossless compressed expert tier is useful only selectively

All 12,288 expert bundles (47.03 GiB) were scanned:

- zstd level 1: 41.23 GiB, saving 5.80 GiB / 12.32%;
- LZ4 fast: 42.21 GiB, saving 4.81 GiB / 10.24%.

Compression is highly layer-dependent:

- layers 0-1 save 72.77%;
- layers 0-15 save 28.54%;
- most layers after 30 save only 1-4%;
- layer 46 is effectively incompressible.

The canonical model is already IQ3/IQ4/Q5 quantized and host memory is plentiful on this machine, so a compressed host tier is unlikely to improve promotion speed. A selective compressed VRAM tier for early layers may be useful if decompression occurs on GPU into exact spill slots. A universal compressed tier is not justified.

---

### 21. Promotion latency is mostly software at one-expert scale; route-only predictor has a measured upgrade path

A direct CUDA benchmark on the V100's PCIe Gen3 x16 host link measured:

- normal 4.08 MB expert bundle: 0.3215 ms pinned, 0.4408 ms pageable;
- large 5.51 MB bundle: 0.4302 ms pinned, 0.5594 ms pageable;
- eight normal bundles: 2.5507 ms pinned.

Current cross-layer admission-to-READY latency is 7.64 ms median for L-1 prediction and 12.27 ms for L-2 prediction. The large difference is architectural: prediction reserves a slot, but the three component transfer jobs are only created later when scheduler graph attachment reaches the target layer. The background worker is therefore unsuitable for immediate spill service even though its raw copy engine is adequate for one predicted expert.

The existing predictor is an online conditional co-occurrence table:

    score_L,d(e) = sum_{s in selected(L-d)} C_L,d[s,e] / N_L,d[s]

with minimum observations, resident filtering, per-layer/global budgets, and prequential updates.

Stronger route-only variants on the same trace achieved K=8 recall:

- current conditional: 50.45%;
- sliding 64-token conditional: 52.50%;
- within-source pair interactions: 53.42%;
- L-2/L-1 interaction: 55.09%;
- all interactions combined: 55.49%.

PMI, rank weighting, and positive-feature naive Bayes did not improve accuracy. The next predictor should combine sparse interactions with recency and optimize deadline-aware miss utility rather than route recall alone.

Full note: `profiling/latency-predictor-research-2026-07-27/RESULTS.md`.

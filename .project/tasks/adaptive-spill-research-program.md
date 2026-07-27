# Adaptive spill research program

Date: 2026-07-27

## Objective

Restore exact dynamic MoE inference without the 48 permanent CPU fallback islands, while retaining the proven resident-only vertical-cache speed ceiling.

## Experimental decomposition

### Group A — isolate and run separately

These variables affect different mechanisms and must be measured independently before combining them.

1. **Predictor quality, offline**
   - current conditional co-occurrence;
   - exponential/sliding recency;
   - same-layer source-pair interactions;
   - joint L-2/L-1 interactions;
   - confidence/abstention and utility scoring;
   - next-router oracle/approximation if router inputs can be recorded.
   - Metrics: route recall, complete 8/8 prediction, bytes per correct prediction, calibration, phase-change recovery.

2. **Copy-service latency, synthetic**
   - pageable versus pinned source;
   - immediate issue versus queued worker;
   - one bundle event versus component synchronizations;
   - gate/up-first pipelining;
   - 1/2/4/8 simultaneous bundles.
   - Metrics: issue latency, H2D completion, p50/p95, effective GiB/s.

3. **VRAM capacity/reserve**
   - reserve 8192/7168/6144/5120 MiB;
   - no spill initially.
   - Metrics: slots, hit rate, minimum free memory, OOM/fragmentation, long-context safety.

4. **Compression tier**
   - direct canonical copy;
   - host zstd/LZ4 decompress then H2D;
   - selective early-layer compressed tier;
   - do not combine with predictor until latency wins independently.
   - Metrics: end-to-end bytes-to-READY, RAM/VRAM saving, CPU cost.

5. **Stable/adaptive segmentation policy**
   - protected fraction and promotion threshold;
   - no urgent spill in this experiment.
   - Metrics: churn, phase-change recovery, long-window hit rate.

### Group B — variables that should be tested together

1. **Predictor + immediate spill issue**
   - A predictor cannot improve current-token exactness unless upload begins immediately.
   - Test current conditional and stronger recency/interaction predictor against the same direct spill service.

2. **Spill capacity + predictor budget**
   - Sweep 1/2/4/8 spill bundles and matching confidence-controlled prediction budgets.
   - Extra predictions without spill capacity only waste traffic.

3. **Permanent-cache reserve + spill capacity**
   - VRAM should be allocated jointly between residents and spill buffers.
   - Use a fixed total memory safety target.

4. **Long forced-chat history + dynamic policy**
   - Cache warming, phase changes and prediction reuse require 2048/4096-token histories.
   - Replay identical tokens for fair A/B comparisons.

5. **Exactness + performance**
   - Every end-to-end spill configuration requires forced-token full-logit replay and throughput measurement.

### Group C — defer until prerequisites pass

1. GPU compressed L2: only after direct uncompressed spill is exact and measured.
2. Learned hidden-state/router-logit predictor: only after route-only predictor and spill timing establish useful deadlines.
3. Full neural path predictor: only after a large training trace exists and predictor overhead is budgeted.
4. CUDA graph capture: only after staged exact execution has stable topology.

## Ordered execution plan

1. Recover the source-pointer/component registration path and implement urgent bundle requests.
2. Add direct spill/prefetch issue at prediction time with pinned staging and bundle-level completion.
3. Run synthetic and live same-token READY tests at capacities 1/2/4/8.
4. Implement recency and sparse interaction scoring behind separate flags.
5. Run forced 512-token development A/B; select candidates.
6. Run 2048-token forced chat on reserve/spill matrix.
7. Run full-logit replay on finalists.
8. Investigate selective compressed early-layer L2 only if upload bandwidth remains limiting.

## Pass criteria

- At least one predicted normal expert READY before its target layer.
- Normal bundle prediction-to-READY below 0.8 ms p50 and 1.5 ms p95 in isolation.
- No backend-wide synchronization in urgent path.
- Exact replay agrees with ordinary CPU-MoE at decision level and within established logit-drift envelope.
- End-to-end exact throughput exceeds 15.89 t/s ordinary CPU-MoE, then targets 20.54 t/s horizontal.
- No OOM and at least 1 GiB measured free-memory safety margin at target context.

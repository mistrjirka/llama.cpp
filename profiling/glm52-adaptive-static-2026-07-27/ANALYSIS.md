# GLM-5.2 Static Core and Low-Frequency Adaptation Analysis

**Date:** 2026-07-27
**Repository:** `/workspace/llama-mainline-cache`
**Model:** GLM-5.2 UD-IQ2_XXS
**Hardware:** NVIDIA Tesla V100-SXM2 32 GB

## Purpose

This analysis answers three questions:

1. What exactly did the successful GLM static map achieve?
2. Can adaptation occur much less often than the current mutable policy while retaining useful responsiveness?
3. Does a manually inspected mutable trace expose bugs, avoidable churn, or transfer-path improvements?

The adaptive-map results are route-residency simulations. They do not model graph publication, copy overlap, or end-to-end throughput. The mutable runtime trace is an actual diagnostic run.

---

## 1. Exact GLM static-map evidence

The static map is stored at:

```text
profiling/glm52-scale-2026-07-27/static-frequency-map.txt
```

Its held-out analysis is:

```text
profiling/glm52-scale-2026-07-27/static-frequency-map.analysis.json
```

The map was learned from absolute decode steps 1–64 of `routes96.jsonl` and evaluated on steps 65 and later.

The trace contains two incomplete early steps that record only layer 77, followed by 95 complete 75-layer steps. This matters when reproducing the exact train/evaluation split.

### Capacity

- 75 routed layers, layers 3–77.
- 1,262 resident experts total.
- 71 layers have 17 slots.
- Three layers have 14 slots.
- One layer has 13 slots.

### Held-out coverage

- Held-out routes: 19,800.
- Held-out hits: 5,708.
- Exact static held-out route coverage: **28.8283%**.

Per-layer coverage ranges from approximately **8.33% to 52.27%**.

### Throughput

| Configuration | Throughput | Decode splits |
|---|---:|---:|
| GLM horizontal | 4.87–4.97 tok/s | about 150 |
| GLM frozen exact vertical, original prompt | 5.33–5.43 tok/s | 152 |
| GLM frozen exact vertical, different prompt | 5.1647 tok/s | 152 |

The static map is valuable despite only about 29% route coverage because it preserves a stable 152-split graph.

---

## 2. Low-frequency adaptive-tail simulation

The simulator is:

```text
profiling/analyze_epoch_adaptive_static_map.py
```

It preserves most of each layer's measured static map as a protected core and allows only 1, 2, or 4 tail slots to change. Changes occur at token epochs, under a global migration budget, hysteresis ratio, and minimum-residence interval.

### Exact original held-out split

The aligned result is:

```text
profiling/glm52-adaptive-static-2026-07-27/epoch-adaptive-static-original-prompt-aligned.json
```

| Policy | Transfers/token | Coverage | Gain over static | Incremental hits/transfer |
|---|---:|---:|---:|---:|
| Static only | 0 | 28.828% | — | — |
| 2 tail slots, epoch 16, 1 migration | 0.0606 | 28.889% | +0.061 pp | 6.0 |
| 4 tail slots, epoch 8, 2 migrations | 0.2424 | 29.040% | +0.212 pp | 5.25 |
| 4 tail slots, epoch 8, 4 migrations | 0.4848 | 29.247% | +0.419 pp | 5.19 |

The original short held-out region is only 33 complete tokens, so these gains are directionally useful but statistically limited.

### Different-prompt trace

The same static map was applied to `routes-alt96.jsonl`.

| Policy | Transfers/token | Coverage | Gain over static | Incremental hits/transfer |
|---|---:|---:|---:|---:|
| Static only | 0 | 27.667% | — | — |
| 2 tail slots, epoch 16, 1 migration | 0.0526 | 27.942% | +0.275 pp | 31.4 |
| 4 tail slots, epoch 8, 2 migrations | 0.2316 | 28.604% | +0.937 pp | 24.27 |
| 4 tail slots, epoch 8, 4 migrations | 0.4632 | 28.986% | +1.319 pp | 17.09 |

### Longer different continuation

The measured static map covers 19.480% of the longer 1,023-step trace, indicating a larger distribution shift.

| Policy | Transfers/token | Coverage | Gain over static | Incremental hits/transfer |
|---|---:|---:|---:|---:|
| Static only | 0 | 19.480% | — | — |
| 4 tail slots, epoch 32, 1 migration | 0.0302 | 19.762% | +0.282 pp | 55.93 |
| 4 tail slots, epoch 16, 1 migration | 0.0615 | 20.006% | +0.525 pp | 51.22 |
| 4 tail slots, epoch 16, 4 migrations | 0.2461 | 21.151% | +1.671 pp | 40.74 |

### Interpretation

A semi-static design can adapt at approximately **0.03–0.48 expert bundles/token**, instead of the current mutable trace's **9.02 admissions/token**.

That is roughly 19× to 300× fewer admissions, depending on the policy point.

The simulation supports:

- preserve a large static core;
- reserve a small adaptive tail;
- batch map changes every 8–32 tokens;
- permit only 1–4 migrations globally per epoch initially;
- require hysteresis and minimum residence;
- expand adaptation only after measured phase shift.

The simulation does not prove tokens/s gains. Its purpose is to show that useful adaptation does not require per-token, per-layer churn.

---

## 3. Actual mutable diagnostic run

Artifacts:

```text
profiling/glm52-adaptive-static-2026-07-27/mutable-diagnostic-80.jsonl
profiling/glm52-adaptive-static-2026-07-27/vertical-free-r1.summary.json
```

Configuration:

- 16 warm-up tokens;
- 64 measured tokens;
- 79 complete decode tokens in the trace;
- minimum hot routes: 4;
- maximum admissions/token: 16;
- 302 decode splits.

### Runtime outcome

- Throughput: **4.24 tok/s**.
- Resident route hit rate: **30.16%**.
- Executed GPU route rate: **12.49%**.
- Full GPU expert layers: **1 of 5,927**.
- CPU-dependent layers: **5,926 of 5,927**.
- Full GPU expert-path tokens: **0 of 79**.
- Mean CPU-dependent layers/token: **74.9873 of 75**.

The cache achieves slightly more route coverage than the frozen map, but almost every layer still remains CPU-dependent and the graph remains twice as fragmented.

---

## 4. Manual trace findings

### 4.1 Warm-start burst creates a transfer storm

At decode step 3, the runtime admits 592 warm-start experts across layers 3–76. Layer 77 receives no warm-start event.

This produces:

- a maximum transfer queue depth of 795 jobs;
- a 795-job worker batch;
- maximum queue delay around 470 ms;
- mean queue delay around 90–93 ms/component.

The warm start should be a prefill-time planned operation or a throttled staged fill, not a decode-time transfer storm.

The missing layer-77 warm start should be investigated. It may reflect the two incomplete early route steps or graph callback ordering, but the trace establishes that layer 77 is treated differently.

### 4.2 The runtime continues admitting almost every token

After warm start:

- 1,630 total expert admissions;
- 592 warm-start admissions;
- 1,038 ordinary admissions;
- 368 evictions in 79 decode tokens;
- approximately 9.02 admissions/token over the full trace.

There are admissions on 79 distinct steps. The policy therefore never settles into a mostly stable map.

### 4.3 Many admitted experts are evicted before proving useful

Among 368 evictions:

- 168 victims had zero hits;
- 148 had one hit;
- median admission-to-eviction lifetime: 28 tokens;
- zero-hit victims had median lifetime: 23 tokens;
- 52 protected-segment experts were eventually evicted.

This indicates that the admission threshold is too permissive relative to transfer cost and that protection decays too quickly for a design whose strongest result is a stable static core.

### 4.4 Adaptation is not targeted by static-map weakness

The correlation between a layer's measured static held-out coverage and its number of evictions is approximately **0.05**.

In other words, the current mutable policy churns high-coverage and low-coverage layers similarly. It does not concentrate adaptation on layers where the static map is weak or where a phase shift is detected.

### 4.5 The global cap rejects many layers, but traffic remains excessive

Reject reasons:

- 4,538 `global_admission_cap`;
- 193 `candidate_not_better`.

Many `global_admission_cap` events report `admissions_this_step=0` because the rotating layer eligibility mask rejected the layer before the counter reached the limit. This trace field is not itself evidence of a counter bug, but it is easy to misread.

The rotating cap controls fairness, not utility. It may admit a low-value candidate from an eligible layer while rejecting a much higher-value candidate from an ineligible layer.

A global priority queue ranked by expected latency saved per byte would be preferable.

### 4.6 One expert bundle component bypasses pinned staging

Default staging configuration:

- 96 MiB total;
- 24 staging slots;
- approximately 4 MiB per slot.

Observed GLM bundle components:

- component 0: 3.09 or 3.84 MiB, staged;
- component 1: 3.09 or 3.84 MiB, staged;
- component 2: 4.59 or 6.38 MiB, never staged.

Counts:

- 3,260 staged component transfers;
- 1,630 non-staged transfers;
- every component-2 transfer bypassed staging.

Median H2D API time:

- staged components: about 8 microseconds;
- non-staged component 2: about 626 microseconds.

This is a concrete configuration mismatch. Increasing staging slot size above the largest GLM component, or staging complete bundles in suitably sized buffers, should reduce host issue overhead.

It will not fix the 302-split topology, but it is a real transfer-path improvement.

### 4.7 Promotions are not predictions and are issued too late

All 1,630 completed bundles report:

- `prediction_distance = 0`;
- `prediction_step = 0`;
- `queued_mono_ns = 0` in `ready_after_completion`;
- no urgent jobs or urgent batches.

The current path is demand/admission driven, not future prediction driven. Promotion work is encountered through graph inputs and the worker queue, so queue delay dominates raw copy time.

This explains why a physically fast 1 ms bundle copy can still become ready hundreds of milliseconds after admission.

### 4.8 Worker batching has pathological startup outliers

Most completed batches contain one or two jobs, but startup includes batches of:

- 24 jobs;
- 72 jobs;
- 144 jobs;
- 360 jobs;
- 411 jobs;
- 795 jobs.

The 795-job batch takes approximately 383 ms.

Batching should have hard limits in both jobs and bytes, with urgent/deadline work preempting background warm fill.

### 4.9 Resident hits are poorly converted into GPU execution

The run reports:

- 14,299 resident hits;
- only 5,921 executed GPU routes;
- 30.16% resident hit rate;
- 12.49% execution hit rate.

A large fraction of useful residency is lost because the minimum-hot threshold and mixed CPU/GPU branch topology prevent those hits from becoming profitable execution.

The frozen exact result proves that partial resident routes can be valuable when represented through a stable graph.

---

## 5. Bug and improvement classification

### Confirmed implementation/configuration problems

1. **Staging-slot mismatch:** 4 MiB staging slots cannot hold GLM component 2, so one-third of component transfers bypass pinned staging.
2. **Unbounded startup burst:** warm-start admissions can create hundreds of queued component jobs and a 795-job batch.
3. **No real prediction lead:** all promotions are distance zero and no transfer is marked urgent.
4. **Map never stabilizes:** admissions and evictions continue through nearly every decode token.
5. **Protection is too weak for the proven static-core regime:** protected entries become eviction candidates after limited recency aging.
6. **Adaptation is not layer-targeted:** churn has almost no relationship to measured static-map weakness.
7. **Trace accounting gap:** `ready_after_completion` records zero queued timestamps and zero prediction-to-ready time for all bundles, although `worker_transfer_timing` contains the real queue timestamps. This weakens bundle-level diagnostics.

### Suspected issue requiring focused validation

- Layer 77 receives no warm-start event while layers 3–76 do. This may be callback ordering caused by incomplete early trace steps rather than a placement bug. Add an assertion and layer-completeness counter before changing behavior.

### Architectural bottleneck, not a small bug

- The 302-split mutable topology remains the dominant demonstrated throughput loss. Transfer improvements alone cannot make the current graph competitive with the 152-split frozen path.

---

## 6. Recommended semi-static runtime policy

### Static core

- Initialize from a corpus/request static ranking.
- Protect at least 75–85% of each layer's slots indefinitely for the request epoch.
- Do not age the core into the ordinary LRU victim pool.

### Adaptive tail

Start with:

- 2 adaptive slots/layer for stable requests;
- 4 slots/layer after detected phase shift;
- no immediate per-layer replacement.

### Publication cadence

- Collect observations continuously.
- Evaluate replacements every 16 tokens initially.
- Commit at most 1–4 complete expert migrations globally per epoch.
- Publish one new route-map epoch after all selected bundles are ready.
- Coalesce multiple layer changes into the same map publication.

### Admission requirements

A candidate should satisfy all of:

- sufficiently more useful than the tail victim;
- minimum observation support;
- minimum expected residence/reuse;
- positive expected latency saved after transfer and eviction cost;
- no deadline conflict with urgent current-token work.

### Phase-shift response

Increase migration budget only when one or more of these persist:

- static-core coverage drops materially from its calibrated baseline;
- predictor residual error rises;
- tail candidate advantage remains high for multiple epochs;
- request class changes.

Decrease the budget after stability returns.

---

## 7. Best next engineering steps

1. Implement the versioned A/B GPU route map and retain the frozen 152–160 split topology.
2. Load the measured static map as a genuinely protected core.
3. Add a 2–4 slot adaptive tail and coarse epoch publication.
4. Replace rotating layer admission with a global utility-ranked migration queue.
5. Limit background batches by jobs and bytes; reserve queue capacity for urgent work.
6. Increase staging slot size to cover the largest GLM component, then benchmark complete-bundle staging.
7. Preserve queue timestamps through `ready_after_completion` and log admission-to-ready, ready-to-first-hit, lifetime hits, and bytes per useful hit.
8. Add layer-77 warm-start completeness assertions.
9. Re-run with migration budgets around 0.05, 0.10, and 0.25 bundles/token.
10. Integrate the +6/+3/+1 predictor only after map publication is stable.

The next target is not maximum route coverage. It is:

> Preserve frozen-map speed while obtaining measurable cross-prompt adaptation with well below one expert migration per token.

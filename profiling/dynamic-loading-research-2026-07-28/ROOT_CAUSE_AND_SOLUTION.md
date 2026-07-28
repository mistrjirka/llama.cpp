# Dynamic Expert Loading: Root Cause and Solution Design — 2026-07-28

## Executive result

The current dynamic path is not primarily slow because expert weights arrive too late. It is primarily slow because mutable residency is exposed to ggml as CPU-side routing and mixed CPU/GPU execution at every MoE layer.

The current graph has four backend segments per MoE layer:

1. CPU router result / mutable route mask and policy;
2. GPU resident-expert branch;
3. CPU nonresident-expert branch and merge preparation;
4. GPU continuation.

For 75 MoE layers this gives `75 × 4 + 2 = 302` scheduler splits. Frozen exact execution uses two segments per layer, giving `75 × 2 + 2 = 152` splits.

The scheduler assignment trace directly measured the dynamic 302-split graph alternating CPU→GPU 151 times and GPU→CPU 150 times. Frozen decode had 76 CPU and 76 GPU segments.

The graph is not rebuilt from scratch for each expert choice. It is already a fixed superset graph over compact GPU slots. The problem is that the current mutable route decision is a CPU custom op and the mutable GPU slot map is not maintained as an asynchronous device-side data structure. Therefore the scheduler must expose extra CPU islands and copies.

## Proof that loading latency is not the dominant current bottleneck

Direct predictor transfer timing:

- median predictor admission-to-READY: about 1.59 ms;
- all tested correct distance-one and distance-two predictions were READY before target use;
- measured READY-before-target rate in the true configuration: 85.56% of admissions.

Prediction improves mutable throughput from about 4.33 to 4.55 tok/s, roughly 5.1%. It helps, but leaves a much larger gap to frozen static at roughly 5.8–5.9 tok/s.

A worse-coverage prompt-derived cache map, frozen before decode, ran 32.59% faster than mutable execution. Since that map had lower route coverage, the gain cannot be explained by better expert selection. Stable execution structure was the main changed factor.

A future-aware fixed-map oracle increased frozen throughput by only 1.50%. This bounds the value of perfect fixed placement under the frozen topology and further demonstrates that cache quality alone is not responsible for the mutable/frozen gap.

## CUDA Graphs A/B

Crossed two-repeat test, identical forced output hash:

| Path | CUDA Graphs on | CUDA Graphs off | On versus off |
|---|---:|---:|---:|
| Dynamic | 4.567523 tok/s | 4.329824 tok/s | **+5.49%** |
| Frozen static | 5.802923 tok/s | 5.805299 tok/s | **−0.04%** |

Turning CUDA Graphs off is harmful to dynamic execution and neutral for frozen static. CUDA Graphs reduce launch overhead inside each GPU segment, but they cannot remove the CPU/GPU scheduler boundaries. Current ggml captures individual GPU subgraphs; the host scheduler still submits and transfers between 302 segments.

## MLP cost and evidence

A single-thread C++ microbenchmark evaluated a sparse 32-hidden, 256-output MLP, including top-16 selection, for all 75 MoE layers in:

- mean: 755.2 microseconds per token;
- median: 754.0 microseconds;
- p95: 760.2 microseconds.

That is about 0.34% of the current approximately 220 ms/token dynamic latency. Arithmetic cost is not a reason to reject a small MLP.

However, the MLP already tested on route, temporal, and token features was less accurate than the sparse learned perceptron:

| Predictor | Budget-16 precision |
|---|---:|
| Balanced token-aware MLP, hidden 32 | 39.65% |
| Route-temporal perceptron | 62.31% |
| Token-aware perceptron | 64.36% |

The existing MLP architecture/training objective is therefore not a performance candidate. A new MLP should consume richer intermediate activations or a compressed hidden-state projection, and use a ranking/useful-prefetch objective.

## Transfer microbenchmarks

Actual GLM expert bundle sizes were benchmarked on V100:

- direct registered file-backed source: about 0.874–0.880 ms per 11.3 MiB bundle;
- current hybrid component pattern: about 1.164 ms;
- all write-combined staging: about 1.049 ms;
- naive ordinary pinned staging: about 1.91 ms when CPU copy and H2D are serialized;
- two-slot write-combined staging pipeline: about 0.883 ms per bundle.

This establishes useful transfer-level optimizations, but even eliminating approximately 0.3 ms per transferred bundle cannot recover the roughly 45–50 ms/token topology gap by itself.

## Correct solution: mutable data, fixed topology

The highest-value design is to generalize the frozen exact route-map topology to mutable slots.

### 1. Persistent device slot map

Allocate a stable device array for each layer:

```
slot_by_expert[layer][256]
```

The pointer never changes. Only contents change when a promotion or eviction completes. The existing CUDA `MUL_MAT_ID` remap kernel already consumes this type of map, but the present implementation initializes it once for immutable static maps.

### 2. Canonical route IDs for both branches

Avoid the CPU `llm_graph_moe_dynamic_mask` custom op. Feed canonical router IDs to both branches:

- GPU branch remaps expert IDs through the current device slot map; absent experts become `-1`;
- CPU branch uses the host registry map internally to skip resident experts;
- merge resident and nonresident outputs on GPU.

This is the same structure already validated by frozen exact CPU fallback. Generalizing it to mutable maps should remove the extra CPU route-mask island and reduce dynamic decode toward 152 splits.

### 3. Event-driven slot lifecycle

Use a load stream and compute stream with per-slot or per-bundle events.

Promotion sequence:

1. wait for the previous compute-use event before overwriting a victim slot (WAR protection);
2. asynchronously copy gate/up/down components to the physical slot;
3. update the device slot-map entry on the same load stream;
4. record a slot-ready event;
5. compute stream waits on the slot-ready event only when the selected route needs it (RAW protection).

No host `cudaStreamSynchronize()` is required in the normal path.

### 4. Keep CUDA Graphs enabled

Because slot-map pointers and compact weight-buffer pointers remain stable, changing map contents does not change graph topology or node pointers. CUDA Graph instances can remain reusable.

If explicit graph customization is needed, installed CUDA 12.9 exposes:

- `cudaGraphNodeSetEnabled()`;
- `cudaGraphExecKernelNodeSetParams()`;
- `cudaGraphExecMemcpyNodeSetParams()`;
- `cudaGraphExecEventWaitNodeSetEvent()`;
- `cudaStreamWaitEvent()`.

A simpler first implementation should update persistent map contents outside the captured graph and issue an event wait before launching the affected GPU segment.

### 5. Predictor role

After topology reduction, use a better predictor to increase the fraction of routes that use GPU slots.

Recommended predictor inputs:

- current token routes at L−1, L−2, and optionally L−3;
- previous token routes at target layer L;
- compressed layer input or pre-attention activation;
- token embedding as a bounded correction;
- residency, transfer deadline, and reuse features.

Recommended objective:

```
P(selected) × P(nonresident) × P(ready before use)
× P(executed on GPU) × reuse value
− transfer bytes − interference − branch cost
```

A small MLP is affordable, but it should be trained on multiple prompts and compared against the stronger sparse perceptron. Prediction should run one or two layers ahead and in parallel with inference.

## Can dynamic overtake static?

Not with the current 302-split topology. Dynamic needs roughly 30% more throughput to reach frozen static, while the current predictor contributes about 5.1% over no predictor and admission ranking contributes about 0.94% over the previous predictor.

After reducing mutable execution to approximately the frozen 152-split topology, overtaking static becomes plausible. The dynamic cache could then retain the static core while using accurate prefetch to replace low-value residents. Expected gains would come from higher useful residency, not from rebuilding a different graph per token.

The sequence should be:

1. mutable device slot map;
2. canonical GPU remap plus CPU dynamic miss mask;
3. GPU merge, reducing to two segments per layer;
4. event-driven promotion publication;
5. keep CUDA Graphs enabled;
6. then implement and evaluate the richer MLP.

## Similar system designs

The design is consistent with recent MoE systems:

- FluxMoE keeps the compute graph static while dynamically materializing parameters through stable virtual addresses and cross-stream CUDA event barriers.
- ProMoE uses a learned predictor, stride prefetching, chunked transfer, early preemption, and reordered inference to hide prediction/loading latency.
- MoE-Infinity combines activation-aware caching with request-level expert tracing and prefetching.
- Pre-attention expert prediction reports high accuracy using lightweight learned functions over intermediate activations, suggesting richer activations are more useful than token ID alone.

# Mutable Expert Cache with Fixed Decode Topology — Results and Code

Date: 2026-07-28

Repository: `/workspace/llama-mainline-cache`

Branch: `expert-cache-mainline`

Starting HEAD: `ccb9468648ef`
Status: opt-in, uncommitted prototype

## Executive answer

Yes: the large dynamic-path penalty is mainly **CPU/GPU coordination and execution topology**, not the expert-weight upload itself.

The legacy dynamic graph exposes four scheduler segments per MoE layer:

1. CPU mutable route-policy/snapshot work;
2. GPU resident-expert branch;
3. CPU nonresident-expert branch and merge preparation;
4. GPU continuation.

For 75 MoE layers this becomes `75 × 4 + 2 = 302` splits. The fixed exact topology uses two segments per layer, `75 × 2 + 2 = 152` splits. A scheduler trace measured 151 CPU→GPU and 150 GPU→CPU transitions in the legacy dynamic graph.

This “communication” includes:

- host scheduler submission and synchronization at backend boundaries;
- copying tiny expert-ID/map data between CPU and GPU;
- moving the token activation into the CPU cold branch and the cold result back to the GPU merge;
- mutable registry locks, route masking, admission, and readiness publication.

It is **not the same operation as uploading expert weights**. Expert weights are copied by a separate promotion worker. Existing direct-prefetch measurements put a correct promotion near 1.59 ms and usually READY before use; that helps residency, but it cannot remove 150 extra scheduler boundaries.

The concrete fixed-topology prototype proves the structural diagnosis:

| Controlled configuration | Fixed topology | Legacy topology | Change | Output |
|---|---:|---:|---:|---|
| One dynamic layer, CUDA Graphs off | 4.729537 tok/s | 4.579198 tok/s | **+3.28%** | exact same SHA-256 |
| 75 layers, one resident/layer, Graphs off | 5.065196 tok/s | 3.594574 tok/s | **+40.91%** | exact same SHA-256 |
| 75 layers, one resident/layer, Graphs on | **6.339055 tok/s** | 4.059881 tok/s | **+56.14%** | exact same SHA-256 |

The fixed path therefore can exceed the previously measured frozen-static range in this controlled stable-map configuration. It is not yet a production dynamic-cache victory because decode-time multi-slot mutation still has a correctness failure.

A final rerun on the retained source reached **6.349746 tok/s** across 32 measured tokens, exit `0`, with SHA-256 `e165b05c0568da0b099159b3045190e4f33ffbd97698d0a9881b982ead61f514`.

## What was planned

The original plan is in `IMPLEMENTATION_PLAN.md`. The implementation sequence was:

1. Build an opt-in graph with the frozen path's fixed topology.
2. Keep a persistent mutable `expert -> compact slot` map.
3. Route canonical expert IDs through a GPU remap kernel.
4. Make the CPU branch skip currently resident experts.
5. Merge complementary CPU and GPU route outputs on GPU.
6. Keep CUDA Graph pointers and shapes stable while only map contents change.
7. Validate exact output, split count, asynchronous publication, and throughput.

## Implemented data flow

### Legacy dynamic path

```text
GPU router
    ↓ IDs copied to CPU
CPU mutable route split / policy
    ↓
GPU resident experts
    ↓
CPU nonresident experts + merge preparation
    ↓
GPU route weighting / reduction / next layer
```

### Fixed-topology prototype

```text
GPU router ──────────────┐
                        │ canonical IDs
CPU snapshot + cold MMID│
    │                   │
    │ current published │
    │ slot map skips    ↓
    │ resident routes   GPU ID→slot remap + hot MMID
    └──────────────┬───────────────┘
                   ↓
              GPU exact merge
                   ↓
          weighting / reduction / next layer
```

The CPU cold computation remains. What moves to the GPU is the **resident-route lookup and compact-slot mapping**, allowing route coordination to share the same CPU→GPU boundary already needed by the cold/hot merge.

## Code changes

### `src/llama-graph.cpp`

Adds opt-in `GGML_MOE_DYNAMIC_FIXED_TOPOLOGY` graph construction.

Key behavior:

```cpp
// CPU snapshot for policy and exact cold masking.
ggml_tensor * snapshot_source = ggml_cont(ctx0, selected_experts);
ggml_backend_sched_set_tensor_backend(sched, snapshot_source, backend_cpu);

ggml_tensor * snapshot_ids = ggml_map_custom1(
    ctx0,
    snapshot_source,
    llm_graph_moe_dynamic_snapshot,
    1,
    split_params);
ggml_backend_sched_set_tensor_backend(sched, snapshot_ids, backend_cpu);

selected_hot  = selected_experts; // canonical GPU router IDs
selected_cold = snapshot_ids;     // stable CPU policy snapshot
```

The fixed graph constructs the CPU branch first and the GPU branch second:

```cpp
cold_out = build_branch(
    up_exps, gate_exps, down_exps,
    selected_cold,
    backend_cpu,
    /* mask_resident_routes = */ true,
    "ffn_moe_dynamic_cold_out");

hot_out = build_branch(
    hot_up_weights, hot_gate_weights, hot_down_weights,
    selected_hot,
    backend_gpu,
    /* mask_resident_routes = */ false,
    "ffn_moe_dynamic_hot_out");

split_experts = ggml_add(ctx0, cold_out, hot_out);
ggml_backend_sched_set_tensor_backend(sched, split_experts, backend_gpu);
```

This is what collapses decode toward two backend segments per MoE layer.

### `ggml/src/ggml-backend.cpp`

The mutable registry now has a separately published map:

```cpp
std::vector<int32_t> published_slot_by_expert;
uint64_t published_slot_map_generation = 0;
```

Only READY slots are published:

```cpp
std::vector<int32_t> snapshot(n_expert, -1);
for (int32_t slot_index = 0; slot_index < layer.n_slots; ++slot_index) {
    const auto & slot = layer.slots[slot_index];
    if (slot.state == GGML_BACKEND_MOE_SLOT_READY) {
        snapshot[slot.expert] = slot_index;
    }
}

if (snapshot != layer.published_slot_by_expert) {
    layer.published_slot_by_expert = std::move(snapshot);
    layer.published_slot_map_generation = registry.next_slot_map_generation++;
}
```

A critical cache-buffer rebinding fix clears stale CUDA metadata before attaching a graph tensor to the persistent cache buffer:

```cpp
tensor_copy->buffer = nullptr;
tensor_copy->data   = nullptr;
tensor_copy->extra  = nullptr;

ggml_backend_tensor_alloc(
    entry->buffer,
    tensor_copy,
    ggml_backend_buffer_get_base(entry->buffer));
```

Before clearing `extra`, the first real positive compact-slot read could use stale arena allocation metadata and fault. This fix made one real promoted slot per layer execute correctly.

### `ggml/src/ggml-cuda/ggml-cuda.cu`

Maintains persistent per-layer CUDA state:

```cpp
struct ggml_cuda_moe_slot_map_state {
    int32_t * slot_map;
    int32_t * mapped_ids;
    int32_t * slot_map_host;       // pinned
    cudaEvent_t slot_map_upload_event;
    uint64_t generation;
};
```

The device pointer remains stable. When the generation changes, only 256 integers are updated:

```cpp
cudaMemcpyAsync(
    state.slot_map,
    state.slot_map_host,
    n_expert * sizeof(int32_t),
    cudaMemcpyHostToDevice,
    stream);

cudaEventRecord(state.slot_map_upload_event, stream);
state.generation = generation;
```

A CUDA kernel maps canonical IDs to compact slots; missing experts become `-1`. The existing `MUL_MAT_ID` operation then executes against compact cache tensors without changing graph shapes or pointers.

### `ggml/src/ggml-cpu/ggml-cpu.c`

The CPU MMID branch reads the current host slot map and skips routes already published as resident:

```c
if (dynamic_miss_mask && dynamic_slot_map[i02] >= 0) {
    has_masked_routes = true;
    continue;
}
```

This preserves the exact invariant:

```text
each selected route executes exactly once:
resident -> GPU
nonresident -> CPU
```

### `ggml/src/ggml-cuda/set-rows.cu`

A debugging validator was implicitly enabled by fixed-topology mode and performed D2H readback plus `cudaStreamSynchronize()` during CUDA graph capture. NVIDIA explicitly prohibits host synchronization on a captured stream.

It is now strictly opt-in:

```cpp
const bool validate_set_rows =
    getenv("GGML_MOE_DYNAMIC_VALIDATE_SET_ROWS") != nullptr &&
    atoi(getenv("GGML_MOE_DYNAMIC_VALIDATE_SET_ROWS")) != 0;
```

This concrete fix allowed asynchronous warm-start plus CUDA Graph replay to run correctly.

### `profiling/run_glm52_scale_case.sh`

The runner now exposes and records:

- fixed topology;
- maximum admissions per token;
- warm-start limits;
- asynchronous promotion;
- predictor budget and ranking;
- urgent direct upload versus deferred worker path.

This made the failure isolation reproducible rather than relying on one-off shell changes.

## What worked

### 1. Fixed topology reaches the target split count

One-layer A/B:

| Mode | Decode splits | Throughput | SHA-256 |
|---|---:|---:|---|
| Fixed | **152** | 4.729537 | `56654198...ddc1` |
| Legacy | 154 | 4.579198 | `56654198...ddc1` |

For all 75 MoE layers, scheduler tracing confirms the intended 152-split structure rather than the legacy 302-split structure.

### 2. Structure, not loading speed, accounts for the large gain

The strongest controlled comparison used the same synchronously promoted experts and disabled all post-warmup admission. Therefore expert selection and upload work were matched.

CUDA Graphs off:

| Mode | Throughput |
|---|---:|
| Fixed topology | **5.065196 tok/s** |
| Legacy topology | 3.594574 tok/s |

Gain: **+40.91%**.

CUDA Graphs on, 16 warmup plus 32 measured tokens:

| Mode | Throughput |
|---|---:|
| Fixed topology | **6.339055 tok/s** |
| Legacy topology | 4.059881 tok/s |

Gain: **+56.14%**.

Both comparisons produced exact matching output hashes. No faster expert upload can explain this result because the promoted expert set was the same.

### 3. Asynchronous worker publication works for a stable map

After making `SET_ROWS` validation capture-safe:

- one asynchronously promoted expert per layer;
- CUDA Graphs enabled;
- sampled component readback verified byte-for-byte;
- exit `0`;
- throughput **6.1863 tok/s**.

This proves that asynchronous H2D upload and fixed CUDA Graph replay can coexist when the published map is stable before measured decode.

## What did not work

### Multi-slot and decode-time mutation

The following all failed with exit `134`:

- predictor-driven admissions with urgent direct upload;
- the same admissions with urgent upload disabled;
- the same admissions with CUDA Graphs disabled;
- two synchronously warmed slots per layer;
- the generic safe MMID fallback.

Therefore the blocker is not:

- the urgent queue;
- CUDA Graph replay;
- asynchronous promotion alone;
- a particular optimized MMID kernel.

### Sanitizer result

`compute-sanitizer --tool memcheck` localized the visible fault to:

```text
Invalid global write in k_set_rows<float, int, __half>
ERROR SUMMARY: 257 errors
```

The expert promotion copies themselves did not write outside cache allocations; sampled copies matched source bytes exactly. The `SET_ROWS` failure occurs later, meaning an upstream fixed-topology hot/cold result or routing tensor becomes numerically invalid and supplies bad row indices to a later model operation.

This is why the prototype remains opt-in and is not safe for full dynamic mutation.

### Reverted attempts

Two speculative changes were tested and reverted:

1. Feeding the CPU custom-op route snapshot directly back into GPU MMID. The scheduler could not safely use that custom-op tensor as a GPU branch input.
2. Deduplicating repeated route IDs in cache scoring. It did not fix the fault; repeated IDs were a downstream symptom rather than the root cause.

## Interpretation

### Is the issue “a lot more CPU communication”?

Yes, with an important distinction.

The dominant penalty is not repeatedly sending 10–14 MiB experts to the GPU. It is repeatedly crossing execution domains for every layer:

```text
CPU policy -> GPU hot compute -> CPU cold compute -> GPU merge
```

Each boundary can require:

- scheduler bookkeeping;
- waiting for prior work;
- copying route IDs or activations;
- launching another backend segment;
- breaking CUDA Graph capture into smaller pieces.

The fixed topology removes the standalone CPU policy island and folds route lookup into a stable GPU slot-map operation. The CPU still computes cold experts, but coordination no longer creates four segments per layer.

### Can the coordination be moved to the GPU?

Mostly, yes:

- `expert ID -> slot` lookup: moved to GPU;
- route masking for the hot branch: moved to GPU;
- hot/cold merge and router weighting: kept on GPU;
- slot-map pointer: persistent on GPU;
- slot-map contents: updated asynchronously.

Some CPU responsibilities remain:

- cold expert computation;
- admission/eviction policy;
- issuing weight transfers;
- publishing READY state.

These can stay on CPU without recreating the 302-split graph, as long as they update stable device data and communicate through events rather than inserting new graph islands.

## CUDA/documentation alignment

The implementation follows CUDA's documented constraints and mechanisms:

- synchronizing a stream during capture is prohibited;
- `cudaStreamWaitEvent()` provides device-side cross-stream ordering;
- event-wait graph nodes and graph-exec parameter updates allow mutable data with stable graph structure;
- stable pointers plus changing map contents avoid graph reconstruction.

Relevant NVIDIA documentation:

- CUDA Graphs and prohibited capture operations: `https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html`
- Asynchronous execution and stream synchronization: `https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html`
- `cudaStreamWaitEvent`: `https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html`
- CUDA graph management APIs: `https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html`

The architecture is also consistent with recent MoE serving work that keeps compute topology stable and overlaps expert materialization with inference, including FluxMoE and predictive-prefetch systems such as ProMoE.

## Retention decision

Retained:

- opt-in fixed-topology graph builder;
- persistent published slot map and generation;
- GPU canonical-ID remapping;
- exact CPU miss masking;
- cache-buffer metadata rebinding fix;
- capture-safe set-rows validation;
- reproducible runner controls and diagnostics.

Not enabled by default:

- fixed topology;
- multi-slot mutation;
- predictor-driven replacement on the fixed graph.

Safe validated checkpoint:

```text
fixed topology + stable one-slot-per-layer published map
```

Production-ready status:

```text
No — structural performance is proven, but full mutable numerical correctness is unresolved.
```

## Next minimal engineering step

Do not spend the next iteration on a larger predictor yet. First identify the first corrupt fixed-topology tensor.

Recommended instrumentation:

1. Add an opt-in finite/range check immediately after each layer's `hot_out`, `cold_out`, and merged expert tensor.
2. Run fixed and legacy graphs on the same forced token stream.
3. Record the first layer/token where outputs diverge.
4. Compare the corresponding route IDs, slot map, gate/up/down component outputs, and router weights.
5. Repair numerical parity for two or more simultaneous resident routes.
6. Re-run compute-sanitizer until zero errors.
7. Add per-slot compute-use events before permitting slot replacement.
8. Then attach the learned predictor/MLP to this 152-split topology.

Once multi-slot correctness is fixed, the MLP has a realistic path to beating frozen static: the structural penalty is already removed, so improved contextual residency can become a net advantage instead of being hidden under coordination overhead.

## Validation commands

```bash
git diff --check
bash -n profiling/run_glm52_scale_case.sh
cmake --build build-v100 --target llama-completion test-moe-split-backends -j 16
build-v100/bin/test-moe-split-backends
```

Observed exits:

- build: `0`;
- focused GPU test: `0`;
- stable-map fixed-topology runs: `0`;
- current full multi-slot mutation: `134`;
- compute-sanitizer failing multi-slot run: `99` with 257 invalid writes localized downstream in `SET_ROWS`.

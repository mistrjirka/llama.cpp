# Experiment Log

## 2026-07-26 — Graph-mode matrix and continuous-GPU decode

### Question

Can decode return to a stable, mostly continuous GPU graph after the expert cache is populated, and which graph mode produces the best throughput without hidden topology changes or graph-capture failures?

### Background

Measured results established that dynamic splitting loses most of its performance before cache quality becomes the primary limitation:

- baseline short coding decode: approximately 20–21 t/s;
- dynamic graph with hot execution disabled: approximately 8.3 t/s;
- unsafe GPU-only with scheduler-side cold suppression: approximately 10.7 t/s;
- unsafe GPU-only graph with cold branch removed from graph construction: approximately 13.7 t/s;
- the same GPU-only topology with CUDA graphs enabled: approximately 14.9 t/s;
- clean forced-rebuild GPU route-map probe: approximately 19.4 t/s over 16 measured tokens, but correctness remains unsafe because uncached experts are omitted.

Tracing identified one CPU route-map callback and one or more GPU submissions per dynamic layer as the central source of fragmentation.

### Hypotheses

1. CUDA graph capture helps only after topology and buffer addresses become stable.
2. Disabling CUDA graphs may outperform capture while cache admissions or graph topology are changing.
3. A forced graph rebuild immediately after warmup should outperform incidental delayed rebuilding.
4. GPU route mapping should eliminate the per-layer CPU route callback and reduce submission count.
5. A startup-static topology should avoid rebuild cost but may initialize routing state before promotions are READY.
6. A frozen cache is necessary for stable CUDA graph replay unless slot-map data can be updated safely without changing addresses.
7. The best eventual setup may be phase-specific rather than one global graph mode:
   - prefill: ordinary graph execution with low-overhead observation;
   - cache warmup: CUDA graphs disabled or capture invalidated;
   - steady decode: frozen slot map plus captured GPU graph;
   - cache refresh: leave captured mode, update/promote, rebuild once, resume capture.

### Planned matched comparisons

Use the short coding workload first:

- prompt: `profiling/coding-sanity-prompt.txt`;
- context: 4096;
- warmup: 64 tokens;
- measured: 64 tokens;
- seed: 1;
- temperature: 0;
- same V100 and CPU thread configuration.

Graph modes:

1. `baseline`, CUDA graphs disabled.
2. `baseline`, CUDA graphs enabled.
3. `evolved`, CUDA graphs disabled.
4. `evolved`, CUDA graphs enabled.
5. `evolved-gpu-only`, CUDA graphs disabled.
6. `evolved-gpu-only`, CUDA graphs enabled.
7. `evolved-gpu-map-rebuild`, CUDA graphs disabled.
8. `evolved-gpu-map-rebuild`, CUDA graphs enabled.

For unsafe modes, throughput is an architectural ceiling only. Output hashes and correctness status must be recorded.

### Primary measurements

- measured tokens/s;
- GPU utilization;
- backend submit calls and cumulative submit-return time;
- CPU dynamic route-map graph count;
- dynamic hot/cold graph count;
- source synchronization time;
- graph reserve/rebuild messages;
- VRAM used/free and allocation failures;
- output hash.

### Acceptance criteria for the graph experiment

The experiment is useful when it identifies:

- whether CUDA graphs help each topology;
- whether forced rebuild removes delayed mixed-mode execution;
- whether CPU routing graphs disappear in the measured phase;
- whether the GPU route-map topology is stable for at least 64 measured tokens;
- the next submission or synchronization bottleneck after CPU callback removal.

### Correctness warning

GPU-only and GPU-route-map modes currently omit uncached expert contributions. They are intentionally unsafe probes and cannot be treated as model-quality improvements.

### Results

| Mode | CUDA graphs | Measured t/s | Interpretation |
|---|---:|---:|---|
| Baseline | Off | 21.50 | Best baseline point in this matrix. |
| Baseline | On | 20.60 | Capture slightly regressed this short V100 workload. |
| Fragmented GPU-only | Off | 13.34 | Less arithmetic did not overcome submission fragmentation. |
| Fragmented GPU-only | On | 14.00 | Capture helped modestly but topology remained poor. |
| Forced-rebuild GPU route-map | Off | 20.25 | Removing the CPU route callback recovered near-baseline speed. |
| Forced-rebuild GPU route-map | On | 20.22 | Capture was effectively neutral after topology stabilization. |

### Accepted conclusions

- The desired experiment succeeded architecturally: a warmup boundary followed by a rebuilt GPU route-map graph can recover approximately baseline throughput.
- CUDA graphs should be controlled by execution phase and topology stability.
- Graph capture is not the primary solution; eliminating CPU intervention and small graph submissions is.
- The cache should be frozen during the captured steady-state phase until a safe mutable slot-map design exists.

### Rejected or weakened hypotheses

- Rejected: CUDA graphs should always be enabled for decode.
- Rejected: doing less expert arithmetic automatically produces higher throughput.
- Weakened: CUDA graphs provide a large benefit once routing is on-device. In this matrix they were nearly neutral.

### Experimental problems encountered

1. Late topology switching initially required an extra ~2.36 GiB graph allocation and exhausted VRAM. A larger reserve was needed.
2. Canonical expert IDs were initially interpreted as compact slot IDs by selective-copy scheduling.
3. Temporary CUDA pool buffers were invalid during CUDA graph replay; persistent per-layer buffers fixed the illegal access.
4. Static admissions initially remained `COPYING`; they had to be connected to the dedicated promotion path.
5. Persistent cache identity based on graph-tensor pointers failed across graph rebuilds; stable tensor name plus backend identity was required.
6. The first forced-rebuild hook was a no-op because the scheduler was not marked dirty.
7. Whole-run summaries mixed warmup and measured topology; explicit JSON phase markers were added.

### Next experiment

Run a trace with explicit `warmup_complete_before_graph_rebuild` and `measurement_graph_rebuild_complete` markers and measure only the post-rebuild phase:

- CPU route-map submissions per token;
- GPU submissions per token;
- graph node count;
- capture/replay behavior;
- slot-map upload count;
- exact measured-phase transfer volume.

Then design a correctness-preserving miss path that retains on-device route mapping and launches CPU fallback only for actual missing routes, without inserting a blocking CPU callback into every layer.

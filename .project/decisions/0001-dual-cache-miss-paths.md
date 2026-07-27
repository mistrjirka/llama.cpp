# Decision: Explore both exact CPU fallback and blocking GPU streaming for cache misses

## Decision

Treat two exact cache-miss paths as first-class experimental options:

- **Option A — CPU fallback:** compute uncached selected experts from canonical RAM-resident weights on CPU, transfer/merge the small partial activation result, and optionally promote the expert asynchronously for future tokens.
- **Option B — blocking GPU stream:** transfer the complete missing expert bundle from RAM to a reserved GPU slot, wait until it is ready, then compute that expert on GPU for the current token.

Do not choose one globally in advance. Instrument both and select per miss or per layer using measured cost and expected reuse.

## Reason

Option A pays CPU compute and a per-layer CPU dependency but avoids blocking on multi-megabyte weight transfers. Option B avoids CPU expert compute and can preserve a GPU-only layer, but pays PCIe/staging latency before the layer can continue. The crossover depends on expert size, CPU kernel time, transfer source type, PCIe bandwidth, route coverage, queue state, and whether the transfer completes an 8/8 GPU route set.

Coding workloads may provide enough locality for Option B or asynchronous promotion to amortize transfer cost, but random-token behavior is not the primary target.

## Alternatives considered

- Always use CPU fallback. Simple and safe, but may retain CPU synchronization in nearly every MoE layer.
- Always block and stream every missing expert. Keeps computation GPU-side but can move hundreds of MiB or more per token and stall on PCIe.
- Approximate or substitute experts. Rejected because exact routing and model behavior must be preserved.

## Consequences

- Add profiling that separates CPU cold-branch time, GPU hot-branch time, activation copy/merge time, expert staging time, H2D time, queue delay, and complete-route coverage.
- Implement Option B behind an explicit runtime switch before any automatic policy.
- A later policy may value a miss more highly when resolving it changes a layer from 7/8 to 8/8 GPU coverage.
- Preserve canonical CPU weights and exact CPU fallback even when Option B is available.
- Performance conclusions require deterministic coding traces and repeated interleaved comparisons against tuned default llama.cpp.

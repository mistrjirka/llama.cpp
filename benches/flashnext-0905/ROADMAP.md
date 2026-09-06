# Qwen3.8 Flash-Next optimization roadmap

Baseline for all new work: `v100-optimized` at or after `d83475883`, V100 32 GiB + RTX 2080 Ti 22 GiB, PCIe Gen3, Qwen3.8-Flash-Next UD-IQ4_XS, q8_0 K/V, native context 262144, MTP off unless a task explicitly tests MTP. Keep the PLE/n-gram tensor lazy/CPU-backed. Every performance branch must preserve exact output unless it is explicitly placed in the separate lossy track.

## Current accepted state

- Final public branch: `v100-optimized`.
- Manual placement: `36,13`, `n-cpu-moe=18`, `ubatch=1000`, `--fit off`, PLE on CPU/lazy.
- One-weight-ahead MoE prefetch: about 192.59 -> 202.48 PP/s (+5.13%) on the final V100 base, identical output.
- Gather-QSA at ~262k: 7.974 -> 9.399 TG/s (+17.88%) on the final V100 base, identical output.
- A newer upstream-synced experimental stack previously reached ~11.38 TG/s at the same native-context boundary, but syncing that code also regressed Ornith PP by ~2.4%. Do not merge the whole upstream stack just to recover that number.
- MTP has not yet been tested on Flash-Next on this machine. There is currently no Flash-Next MTP GGUF under `/models`.

## Required acceptance gates

For every candidate that can touch generic CUDA/backend code:

1. Flash-Next: 100k restored + 1k PP + 64/512 TG and native-context decode from the reusable 261632-token state.
2. Exact generated-token hash and final cache position.
3. Ornith Q5/Q4 100k + 1k control with expert-only Volta MMQ; reject a repeatable >1% regression unless the change is intentionally model-gated.
4. Dense Qwen3.8 control; generic paths should remain neutral.
5. If generic attention/backend code changes, run a small unrelated model sanity check (Gemma4 is available locally).
6. Record PP, TTFT, TG, VRAM, and for transfer work H2D bytes/time/overlap. For MTP also record drafted/accepted counts.

## P0 - Re-profile the final base and recover known 9.40 -> 11.38 TG headroom

**Branch:** `perf/flashnext-qsa-recover-0906` from current `v100-optimized`.

This is the highest-value first task because the faster result already exists in measured history. Isolate the small set of changes between the safe base and the old synced experiment instead of merging upstream wholesale.

Test independently:

- CUDA FA XOR-swizzle commit `e4b9af007`.
- Qwen4exp-only indexer head reduction `09412af38` (primarily a PP candidate; upstream reported 2170 -> 2366 PP/s at 55k on different hardware).
- Qwen4exp correctness/graph changes from `0eadefebd` and `36b101543` only if needed by later QSA work.
- Treat sparse-FA commit `8e93a9773` as high risk: it rewrites generic FA machinery and conflicts with our Volta/Turing tuning. Extract mechanisms or minimal hunks rather than merging it blindly.

Run Nsight on both 100k and ~262k after each meaningful step. The goal is to identify exactly why compact gather-QSA is ~21% faster on the old synced stack while keeping Ornith at the current fast level.

**Success:** recover most of 9.40 -> 11.38 TG with <1% unrelated-model regression and exact output.

## P1 - True block-first QSA top-k

**Branch:** `perf/flashnext-qsa-blocktopk-v100-0906` from the best accepted P0 result.

The current QSA indexer has about 65k compressed blocks at 262k context, but the graph expands their scores back to ~262k cell scores before selecting ~2k cells. Nsight on the old accepted gather path attributed roughly 15 ms/generated token across QSA layers to the large `get_rows` score expansion. Raw CUDA TOP_K itself changes little between the block and cell shapes, so eliminating score materialization matters more than merely sorting fewer entries.

Port the corrected WIP from `perf/flashnext-qsa-blocktopk-0906`, but do not reuse its old performance conclusion. The old benchmark did not activate because text M-RoPE was incorrectly rejected by `is_pos_2d()`.

Correct semantics for ratio 4 / indexer top-k 2048:

- select 512 complete blocks;
- expand only those selected blocks to 2048 physical cells;
- append the true incomplete causal tail (0-3 cells);
- pad the FA width (currently 2304) with masked slots rather than selecting extra valid low-score cells;
- use the existing gathered K/V attention path.

**Proof of activation:** Nsight must show that the ~262k-cell `k_get_rows_float` score expansion disappears. Add a one-time diagnostic if needed. Compare against the dense-reference generated hash.

**Success:** exact output and a measurable TG gain at 262k. If the context-sized expansion disappears but TG does not improve, profile the new graph before rejecting the mechanism.

## P2 - Persistent compressed QSA index keys

**Branch:** `perf/flashnext-qsa-index-cache-v100-0906` from the best accepted QSA branch.

Today each decode step re-reads historical raw indexer K, groups cells in fours, mean-pools, RMS-normalizes and RoPE-rotates historical block summaries again. This remains O(context) even after K/V attention is compacted. The intended architecture stores one compressed index key per completed four-token block and only computes a new summary when a block completes.

Use `remotes/apepojken/qwen4exp-spec-mtp` commit `472b75842` as a design reference, not as an unquestioned cherry-pick. The previous local `qsa-pooled` experiment is rejected: its apparent speedup never activated, and the truly activated generic external `SET_ROWS` store stalled.

Implement a purpose-built cache/update path with:

- one persistent normalized/rotated summary per completed block;
- append-only update for the newest completed block;
- correct M-RoPE/text/image handling;
- state save/restore, rollback, sequence copy/remove, and context-shift semantics;
- no generic graph-time `SET_ROWS` mutation of an external tensor if that recreates the old stall.

**Success:** exact state round-trip + exact tokens; lower context slope in TG. The strongest signal is a much flatter 32k/64k/128k/262k decode curve.

## P3 - Fuse the QSA selection/gather path

**Branch:** `perf/flashnext-qsa-fused-v100-0906`.

The accepted gather path is still generic:

`q8 KV cache -> ggml_get_rows -> F32 -> reshape/cast F16 -> FlashAttention`.

Explore two implementations, benchmarking the smaller one first:

1. dedicated q8 selected-row gather/dequant directly to compact F16 K/V; or
2. a Volta-compatible sparse FA/indexed-load path that consumes selected cell IDs directly and never materializes compact F32 K/V.

Do not import upstream sparse FA wholesale unless it passes the Ornith/V100 FA gates; it overlaps heavily with our custom SM70 attention code.

**Success:** reduce selected-K/V gather/cast launch and bandwidth cost without changing QSA semantics.

## P4 - Indexer/graph small-kernel cleanup

**Branch:** `perf/flashnext-indexer-graph-v100-0906`.

After P1/P2, re-profile before porting small optimizations. Candidate references:

- upstream `09412af38`: permute-free/sliced indexer head reduction;
- `remotes/apepojken/qwen4exp-spec-mtp` `6634bfde7`: permute-free indexer scoring for batched prefill;
- `b6d995d50`: skinny inject matmul through mat-vec path;
- `51c0d9c3`: contiguous GDN conv concat operand;
- `9b09f26...`: graph reuse / shared QSA input.

These are primarily PP/launch-overhead candidates. Cherry-pick/test one mechanism at a time; do not claim another branch's hardware numbers.

## P5 - Better MoE prefill streaming

**Branch:** `perf/flashnext-prefetch-ring-v100-0906`.

The accepted one-weight-ahead scheduler proves exposed PCIe serialization exists (+5.13% PP on the final base), but it is deliberately shallow. The earlier full-layer gallocr-lifetime prototype is not a fair test of FreeToken-style ring buffering because it paid general graph allocator lifetime/VRAM costs.

Explore an explicit fixed device ring:

- pinned host expert source;
- reusable device buffers independent of graph allocation lifetime;
- copy stream + ready/release events;
- dynamic depth (1/2/3 lookahead) based on available VRAM and measured copy/compute ratio;
- preserve the one 1000-token microbatch;
- if one extra expert layer must move to CPU to fund the ring, compare total end-to-end time at the same VRAM budget.

**Success:** materially more H2D/kernel overlap and >5% incremental PP gain over the current prefetch path, not just a microbenchmark win.

## P6 - MTP for Flash-Next

**Branch:** `perf/flashnext-mtp-v100-0906`.

Port/validate native qwen4exp NextN/MTP only after the ordinary decode path is stable enough to measure. Upstream/community work shows that MTP can be a large independent lever on Flash-Next, but there are also current reports of multi-slot state/content contamination, so start conservatively with `--parallel 1` and exact output checks.

Tasks:

- obtain a compatible Flash-Next MTP GGUF (none is currently downloaded);
- validate draft-head loading and target/draft state alignment;
- benchmark n-max 1/2/3 and draft placement (likely test RTX 2080 Ti first);
- test with and without any prompt-defer optimization separately;
- record acceptance and rejected-work cost at 100k and 262k;
- only then combine MTP with expert-prefetch/cache ideas.

**Success:** exact target output and an end-to-end TG gain after accounting for draft overhead. Ornith's +17.7% MTP result is evidence that the hardware can benefit from speculation, not an estimate for Flash-Next.

## P7 - Equal-budget per-expert hot placement for decode

**Branch:** `perf/flashnext-moe-hotset-v100-0906`.

Current whole-layer placement is structurally wasteful for decode: layers 0-17 have no GPU-resident experts while later layers are fully resident. The existing 511-token coding trace shows weak immediate temporal locality (previous-token overlap ~1.34/10; LRU64 ~37.5%) but strong longer-window frequency skew (top-320 ~98% in-sample and ~96% chronological holdout on the traced host layers).

Before implementation, collect routed expert IDs for **all 48 layers** on the final branch. Then compare at exactly the same expert-weight VRAM budget:

- current whole-layer residency;
- frozen per-layer frequency hot set;
- global/equal-budget variants;
- LRU only as a control.

Reuse an exact split-expert execution design: GPU hits + CPU misses, sum outputs exactly. Avoid continuous churn initially; freeze the hot set for an epoch.

**Success:** significant TG gain at 100k and 262k without more expert VRAM than the current placement.

## P8 - Hybrid CPU/PCIe miss scheduling and staged experts

**Branches:** `perf/flashnext-moe-hybrid-v100-0906`, then `perf/flashnext-dualdeadline-v2-0906`.

If P7 leaves a meaningful miss fraction:

- measure effective CPU expert bandwidth and compact pinned H2D bandwidth under contention;
- send some misses to CPU and some to staged GPU slots so the two paths finish at roughly the same time;
- then revisit DualDeadline: gate/up must arrive first, while down can transfer during gate/up compute.

Do not reuse the old DualDeadline prototype's host router readback / CUDA-graph-disable design; that implementation benchmarked the synchronization bottleneck as much as the staging idea.

## P9 - PLE / cold-prefill I/O only if cold prompts matter

**Branch:** `perf/flashnext-ple-cold-v100-0906`.

Keep mmap/lazy as the production warm-path baseline. Local direct `pread` testing was worse for warm decode. Current upstream reports show direct/lazy PLE reads can improve large *cold* prefill substantially on other storage systems, so only revisit this with a strict cold-cache benchmark and keep it separately gated. Never pin or copy the whole ~26.8 GiB PLE table into VRAM.

## Separate lossy/research track

Do not mix any of these into the exact branch:

- lower-precision expert substitutions;
- expert deferral/drop;
- approximate router prediction;
- reduced KV precision below the chosen q8_0 baseline;
- lossy PLE/indexer approximations.

Use branches under `exp/flashnext-lossy-*` and report quality/output divergence explicitly.

## Recommended execution order

1. `perf/flashnext-qsa-recover-0906` - recover known safe headroom and make fresh 100k/262k profiles.
2. `perf/flashnext-qsa-blocktopk-v100-0906` - remove the proven ~262k score expansion.
3. `perf/flashnext-qsa-index-cache-v100-0906` - make historical compressed keys persistent.
4. `perf/flashnext-qsa-fused-v100-0906` - remove generic gather/dequant intermediates.
5. `perf/flashnext-indexer-graph-v100-0906` - clean remaining small-kernel/graph overhead.
6. `perf/flashnext-mtp-v100-0906` - add speculation on top of the optimized ordinary path.
7. `perf/flashnext-prefetch-ring-v100-0906` - deepen the proven PP overlap mechanism.
8. `perf/flashnext-moe-hotset-v100-0906` - equal-budget decode residency.
9. hybrid miss scheduling / DualDeadline.
10. PLE cold-I/O and lossy ideas only if the production workload makes them worthwhile.

The order is intentionally QSA-heavy: at native context, decode still scales far more with context than the architecture intends, and both our profiles and current llama.cpp reports identify QSA/indexer reconstruction/selection as the dominant structural hole. Once that slope is flattened, re-profile before committing time to MoE prediction or more exotic kernels.

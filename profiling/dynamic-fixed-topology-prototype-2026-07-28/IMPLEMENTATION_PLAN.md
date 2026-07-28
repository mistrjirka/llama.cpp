# Mutable Expert Cache with Fixed Decode Topology — Implementation Plan

Date: 2026-07-28
Repository: `/workspace/llama-mainline-cache`
Branch: `expert-cache-mainline`
Starting HEAD: `ccb9468648ef`

## Objective

Prototype the highest-impact structural fix identified by the bottleneck investigation:

- keep mutable expert-slot contents;
- keep a persistent expert-to-slot map;
- route canonical expert IDs through the existing GPU remap path;
- use exact CPU fallback for current misses;
- remove the extra CPU mutable route-mask island;
- reduce dynamic decode from approximately 302 scheduler splits toward the frozen path's 152 splits.

The prototype will be opt-in and preserve the existing dynamic implementation as the control path.

## Why this prototype comes first

Current evidence shows:

- dynamic decode has 302 scheduler splits versus 152 for frozen decode;
- a worse-coverage frozen map was 32.59% faster than mutable execution;
- correct predictions are usually ready before use, so transfer latency is not the primary remaining gap;
- CUDA Graphs improve dynamic throughput by 5.49%, so they should remain enabled;
- better prediction can help residency but cannot remove the extra CPU/GPU boundaries.

## Proposed implementation

### Phase 1: Reuse the fixed exact graph structure

1. Identify the frozen exact builder that feeds canonical expert IDs to:
   - a compact GPU expert tensor through a slot map;
   - the CPU fallback path for nonresident IDs;
   - a GPU merge.
2. Add an opt-in dynamic-fixed-topology graph mode that uses this structure with the mutable cache buffers.
3. Retain stable tensor pointers and graph shapes.

### Phase 2: Mutable slot-map publication

1. Extend the mutable layer registry to publish its current `expert -> slot` map.
2. Keep the device slot-map allocation stable.
3. Update map contents after promotions/evictions.
4. Start with a correctness-first synchronized update if necessary.
5. If Phase 2 is correct, replace broad synchronization with stream/event publication.

### Phase 3: Exact CPU miss masking

1. Make the CPU fallback skip experts currently present in the mutable slot map.
2. Ensure each selected route executes exactly once:
   - resident route on GPU;
   - nonresident route on CPU.
3. Preserve numerical/output parity.

## Safety and fallback

- New behavior will be gated by an environment variable.
- Existing dynamic behavior remains the default until the prototype passes correctness and performance tests.
- No destructive repository cleanup or unrelated artifact changes.

## Validation gates

1. `git diff --check`.
2. Build `llama-completion` and `test-moe-split-backends`.
3. Run `test-moe-split-backends` with exact exit code.
4. Forced-token smoke with exact SHA-256 match.
5. Scheduler trace:
   - record graph nodes and split count;
   - compare against 302-split current dynamic and 152-split frozen static.
6. Crossed throughput A/B:
   - current dynamic control;
   - fixed-topology dynamic prototype;
   - at least two repeats when the prototype is correct.
7. Record cache promotions, misses, route conversion, transfer bytes, and CUDA/runtime failures.

## Success criteria

Minimum success:

- exact output parity;
- no CUDA or scheduler failure;
- split count materially below 302.

Strong success:

- decode topology reaches or approaches 152 splits;
- throughput improves over the current dynamic predictor path.

## Stop conditions

If the existing backend APIs cannot safely expose a mutable map without invasive ABI changes, the work will stop at a compiled instrumentation/prototype checkpoint and document the exact blocker and the smallest next code change. A non-working path will not be presented as a performance improvement.

## Final report contents

After implementation and testing, `RESULTS_AND_CODE.md` will document:

- exact files and functions changed;
- data-flow and synchronization behavior;
- build and test commands with exit codes;
- split-count and throughput results;
- correctness hashes;
- what worked, what failed, and why;
- retained or reverted code paths;
- next implementation step.

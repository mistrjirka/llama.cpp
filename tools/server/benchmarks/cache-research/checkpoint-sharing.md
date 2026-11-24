# Immutable recurrent checkpoint experiment

The active slot and parked prompts share an immutable checkpoint object when they inherit the same checkpoint. The target, draft, and speculative byte vectors do not change. Independently recomputed checkpoints are not compared or merged.

Set `LLAMA_SERVER_CHECKPOINT_SHARING=0` before starting the server to deep-copy parked checkpoint objects as before. Unset the variable, or set it to `1`, to enable sharing. The variable changes checkpoint ownership only. It does not change model weights, context size, cache precision, or speculative decoding.

## Bounds

- The parked-cache budget counts a shared checkpoint once across all parked entries. It includes vector capacity, the checkpoint object, and a conservative allocation charge.
- Admission includes checkpoint references from the incoming active prompt before evicting existing entries. A reference remains charged when its old cached owner is removed.
- The separate active-slot checkpoint budget still counts every active checkpoint in full. A checkpoint shared between the active slot and the parked cache can be charged by both budgets. This is conservative: actual allocated bytes cannot exceed their sum.
- Publishing uses `shared_ptr<const common_prompt_checkpoint>`. Creating or importing a checkpoint finishes before publication. The separate speculative rollback checkpoint keeps its existing reusable-vector implementation.
- Dropping the final reference frees the vectors. No global reference index or stale weak-reference list is required.

## Tests

`tools/server/tests/integration/test_shared_checkpoints.cpp` verifies immutable identity sharing, byte-exact deep-copy control, unique accounting, admission with an incoming shared reference, eviction ownership, and final-reference release. Link against the server/common libraries from the same build. Both changed implementation files also pass syntax checks with the hub build configuration.

Runtime RAM, latency, cache-capacity, and output-equality comparisons are still required before selecting this candidate. Up to three approximately 88 MiB checkpoint copies can be avoided for a save that inherits all three checkpoints. This is an upper bound per save, not a measured process-RAM reduction.

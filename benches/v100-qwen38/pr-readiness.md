# Fork code-quality and upstreaming notes

This document is an engineering checklist for splitting the deployment fork into small upstream candidates. It is not a pull-request description.

## Baseline reviewed

- Fork: `b403a781e` plus local documentation/style cleanup.
- Fork base: upstream `0e1d9185c`.
- Fresh upstream master reviewed and built: `bb4caa754` (`llama.cpp 0.2.0-dev`, 2026-08-21).
- Master is 20 commits ahead of the fork base.
- None of those 20 upstream commits modify a source file changed by the fork.
- `git diff --check` passes.

The local cleanup follows `AGENTS.md` and `CONTRIBUTING.md`: comments explain non-obvious invariants in one or two lines, source additions are ASCII, braces are used for new conditionals, and new code keeps the surrounding snake_case/4-space style.

## Fresh master performance check

Exact real Qwen3.8 continuation: 100,000 cached tokens + 1,000 new prompt tokens + 64 generated tokens, V100 + RTX 3060 Ti, split `64,2`, q8_0 target KV, F16 draft KV, MTP `n=2`.

| implementation | PP tok/s | TG tok/s | MTP accepted | output |
|---|---:|---:|---:|---|
| vanilla `bb4caa754` | 313.98 | 26.17 | 37 / 52 | reference |
| fork `b403a781e` | 447.11 | 26.20 | 37 / 52 | identical |

PP gain: **+42.40%**. TG difference: +0.12%. Generated token IDs and content are identical.

## Candidate PR split

### 1. Server prompt-cache absolute-prefix selection

Status: strongest small bug-fix candidate.

Scope:

- `tools/server/server-task.cpp`
- `tools/server/tests/unit/test_kv_keep_only_active.py`

Why it is suitable:

- Replaces a two-ratio winner condition with absolute reusable-prefix ranking.
- Keeps the existing minimum keep/similarity viability guards.
- Small behavioral change with a focused regression test.
- Related to the recurrent/long-agent cache problem area tracked in upstream issue #22746.

Required before submission:

- Rebase this isolated commit onto current master.
- Run the new server regression test and the surrounding cache-idle-slot file.
- Keep the PR limited to this cache-selection bug.

### 2. Volta 256x256 FlashAttention base tuning

Status: good backend-only performance candidate if split before adaptive occupancy.

First PR should contain only:

- sm70 256x256 config with Q staged in shared memory;
- K96/V64 scratch split;
- forward split-K traversal needed to preserve MMA accumulation order.

Why it is suitable:

- No model-name check; guarded by CUDA architecture and tensor geometry.
- No public API or CLI additions.
- Ampere and other FA configurations keep the upstream path.
- Existing q8_0 256x256 backend cases pass on both GPUs.
- Strong isolated and whole-model speed evidence already exists.

Do not include the adaptive 2-CTA scheduling change in the first FA PR. It makes review substantially harder because it changes dynamic shared-memory launch occupancy and contains a separate synchronization concern.

### 3. Volta scalar-gate GatedDeltaNet x4-column prefill

Status: good independent CUDA performance candidate.

Scope:

- `ggml/src/ggml-cuda/gated_delta_net.cu`

Guard:

- NVIDIA CUDA, not HIP/MUSA;
- sm70 only;
- `S_v == 128`;
- scalar gate (`!KDA`);
- prompt processing (`n_tokens > 1`).

The kernel groups four independent state columns per warp so q/k/g/beta loads are shared while each column keeps its original recurrence/reduction ordering. Ampere and other shapes stay on the upstream kernel.

Required before submission:

- Keep this independent from FA.
- Add or promote a correctness case that exercises a realistic long prefill (`n_seq_tokens >= 64`) in the normal correctness test set. The current default correctness set exercises the specialized path at four tokens; the 64/256/512/1024 shapes currently live in the perf-case set.
- Include the existing isolated performance measurement, not the full fork benchmark history.

### 4. Adaptive Volta 2-CTA FA occupancy

Status: follow-up only.

Reason for separation:

- Changes launch shared-memory footprint from the legacy footprint to the compact footprint on a restricted Q range.
- Occupancy can alter Stream-K partitioning and KV-pruning execution order.
- The implementation needs a uniform block barrier in the compact metadata-combine path.
- The current implementation duplicates a sizeable metadata-combine block, which is the largest maintainability issue in the CUDA diff.

Before upstreaming, try to reduce that duplication without changing the arithmetic or synchronization path. Any such refactor is performance/numerics sensitive and must rerun the Qwen3.8 100k benchmark plus sanitizer checks.

### 5. Independent MTP draft ubatch

Status: plausible generic speculative-decoding PR, but needs tests.

The change lets the draft context process target prompt data in draft-sized chunks. It preserves token positions and target hidden rows.

Tests still needed:

- target ubatch larger than draft ubatch;
- chunk boundary exactly at and around the draft ubatch;
- multiple sequences with contiguous rows;
- multi-head MTP chaining;
- output equality against the unchunked path.

### 6. Prefill-reuse and pipeline-copy controls

Status: useful fork feature, weak upstream API shape in current form.

Concerns:

- Adds two fields to public `llama_context_params`, which is passed by value in public libllama APIs.
- Adds public `ggml_backend_sched_new_ex` solely to expose scheduler-copy tuning.
- Adds two new CLI knobs that are primarily useful for a specific hardware/memory configuration.
- llama.cpp contribution rules set a high bar for new public API and CLI surface.

The prefill-reuse CUDA implementation itself is localized, but an upstream design should preferably make the optimization automatic/internal or justify a generic API independent of one machine.

Mechanical cleanup already applied locally:

- destination byte offset no longer relies on non-standard `void *` arithmetic;
- argument validation uses normal braced conditionals;
- help text no longer claims the mechanism is universally lossless.

### 7. Recurrent checkpoint retention/eviction policy

Status: relevant problem, not yet small enough for upstream.

Concerns:

- More than 100 lines of server policy in the current fork.
- Value formula contains heuristic weights for replay boundaries and observed hits.
- Needs focused tests for invalidation, branch changes, exact-prefix refresh, capacity eviction, and draft-state restore behavior.

Important baseline finding: `test-recurrent-state-rollback` on Qwen3.8 fails the dirty-context check on both this fork and current vanilla master with the same mismatch at position 6 (`3.36596 != 6.54106`). This is not a fork regression and should be treated as upstream bug evidence/known limitation while working on #22746-related behavior.

### 8. PFlash proxy

Status: fork-only.

It is approximate, depends on an external Lucebox scorer workflow, and changes prompt content. It should remain separate from lossless llama.cpp CUDA/server PRs.

## Current validation

- Full fork source rebuild after style cleanup: passes.
- `git diff --check`: passes.
- New benchmark shell scripts: `bash -n` passes.
- 256x256 q8_0 FA backend correctness: 10/10 on V100 and 10/10 on RTX 3060 Ti fallback.
- GatedDeltaNet `head_size=128` default correctness: 2/2 on V100 and 2/2 on RTX 3060 Ti; the V100 case includes a multi-token invocation of the specialized path.
- Qwen3.8 dirty-context recurrent rollback: fails identically on fork and current master; known upstream limitation, not a fork regression.
- Server cache test file `unit/test_kv_keep_only_active.py`: **3/3 passed** through llama.cpp's official pytest harness after making the new long-prefix case explicitly use `n_ctx=1024`.

Raw local validation logs are under `../results/pr-readiness-20260821/` in the development workspace, not in the source tree.

## Submission policy reminder

Before any upstream submission, the human contributor should re-read current `AGENTS.md`, `CONTRIBUTING.md`, and the PR template. The project requires AI-usage disclosure and restricts AI-generated PR/reviewer text. The human contributor is responsible for reviewing every submitted line and should write the final PR description and reviewer replies themselves.

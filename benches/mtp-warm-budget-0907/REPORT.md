# Request-local warm MTP pause and resume

Research branch `perf/mtp-warm-budget-0907`, based on `5e5ee0440` (validated K/V-only draft refresh and zero-budget fixes). This extension does not change target attention, token sampling, model weights, cache precision or context capacity. The request-control feature is experimental and off unless `LLAMA_EXPERIMENT_MTP_REQUEST_BUDGET=1`.

## What is implemented

An individual request can set `speculative_n_max` (alias `speculative.n_max`) from zero to the server's configured MTP maximum. This is a ceiling, not a new context allocation. The server keeps the existing loaded MTP head and per-sequence state, updates its K/V through the separately enabled `LLAMA_EXPERIMENT_MTP_KV_ONLY=1` path, and skips speculative prompt-copy/checkpoint preparation at a zero ceiling. A later request can resume drafting immediately without replaying the full target history. Missing ceilings inherit the process default, not another agent's setting.

The field is available only when the configured speculative implementations are MTP (optionally accompanied by the no-op `none` entry). It rejects negative/out-of-cap integers, booleans, fractions, strings and values outside integer capacity. The global context's proposal capacity is never increased by a request. This is **warm pause**, not removal of MTP memory or suspension with a hidden-state journal, and not a new automatic depth-selection controller.

## Build isolation

Only three server translation units change: `server-context.cpp`, `server-schema.cpp`, and `server-task.cpp`. No task layout or public engine ABI changes. The private benchmark binary recompiles all three and relinks a private server archive against the immutable K/V-refresh engine; `build-manifest.json` records compiler/linker commands and binary hashes. A normal complete CMake build of this worktree is also supported. The separately maintained production checkout is not used as the benchmark executable.

## Validation design

A sequential 3 → 0 → 3 request sequence is run with ordinary and K/V-only draft refresh, from the same four real 100k histories. Compare generated tokens, accepted/drafted counts, all target/draft/speculative snapshot bytes and a restored continuation. Resume must process only new input tokens, not rebuild the 100k cache.

The concurrent test assigns ceilings 0/1/3/0 to four agents, verifies that the zero agents actually produce no drafts while positive agents do, then checks that an omitted ceiling resumes the configured default. This tests request isolation rather than relying on an echoed configuration field alone.

The persistence test uses one GPU execution slot and five distinct 4096-token histories: park four in RAM, save the parked bank, exit the entire server, start a fresh server, restore the bank and continue the first history with drafting enabled. Compare tokens and MTP acceptance with keeping that history live.

The first transition harness accidentally extended its shared input list between arms. The check exposed different first-turn prompt lengths (38 versus 182), invalidating that run before attributing differences to the engine. The corrected driver copies the immutable prompt and checks the expected first-turn token count. Earlier request parsing also exposed numeric coercion of booleans/fractions; the final field validates types before conversion. These invalid trials are retained locally and excluded from correctness/performance evidence.

## Completed correctness checks

The corrected ordinary-versus-K/V-only 3 → 0 → 3 schedule passed all comparisons: generated tokens, acceptance counts, target/draft/speculative snapshot bytes and the restored continuation. The second/third turns processed just one additional input token, not the 100k history. Mixed 0/1/3/0 budgets and default-depth reactivation passed. All six malformed/out-of-range ceiling inputs were rejected with HTTP400.

The one-slot/five-history bank test saved and reloaded four parked sessions across a complete process exit. Its bank was 460,427,588 bytes; the load endpoint reported 104.99 ms in this single diagnostic, not a performance benchmark. Restoring the first session reused all 4096 cached tokens, processed only the 38-token suffix, and matched the live control's 64 generated tokens and MTP counts (43 accepted out of 57 proposed). The prefill requests explicitly request one token; the last sampled token is not appended to the stored history used for the control/restore comparison. This avoids relying on this baseline's n_predict=0 first-token behavior.

`validation-summary.json` preserves timings, checks and state/output hashes. Large raw responses, token fixtures and serialized states remain in Development Sandbox. These are focused state-consistency checks, not a broad coding-quality evaluation.

A second-architecture check on Qwen3.8-27B with built-in MTP also passed 0/1/3/0 concurrent ceilings and reactivation of a paused request with the default depth. All four reused the 2000-token parent; reactivation required one new input token and resumed proposing drafts. Q8 target/draft KV remained enabled. This short-context functional check is not a Qwen speed benchmark; see `qwen-summary.json`.

## Performance design

Compare genuine MTP-off with request ceilings 0/1/3 while **all three capped arms use the same process-level MTP3 allocation**, including the same recurrent rollback capacity. This distinguishes dynamic warm pause from launching a separate process configured at depth zero or one. The 1.4M physical Q8 KV pool and four 400k logical caps do not change.

Actual histories are four 100k C++ source histories sharing roughly 98k. Each request appends 38/42/44/41 tokens and generates 128. The V10032GiB + modified RTX2080Ti22GiB configuration uses real Ornith AD-Q6_K-Q5_K, Shisa Q5_0, Q8 target/draft K/V, CPU vision projector, target ubatch256 and draft ubatch128. No full-350k/400k timing is implied.

Process order is off/0/1/3/3/1/0/off, with one excluded warmup and two retained turns per process: four retained turns per arm. Snapshot restoration and startup are excluded; HTTP admission, prompt append and generation are included. No profiling, compilation or other GPU task runs during timing. Report per-request TG separately from aggregate outputs divided by whole-turn elapsed time. Generated trajectories can depend on proposal depth/batch shape; timing is not a coding-quality evaluation.

## Measured request-budget results

All capped arms retain the same MTP3 context allocation and use K/V-only refresh. Four measured turns per arm follow the mirrored design above.

| Mode | Four-agent turn | Observed range | TG per active agent |
|---|---:|---:|---:|
| Genuine MTP off | 5.685 s | 5.647–5.721 s | 26.28 tok/s |
| Warm pause, request ceiling 0 | 5.865 s | 5.783–5.906 s | 25.46 tok/s |
| Request ceiling 1 | 5.999 s | 5.828–6.178 s | 25.36 tok/s |
| Request ceiling 3 | 6.469 s | 6.276–6.616 s | 24.18 tok/s |

Warm pause reduces mean whole-turn latency by **9.33% versus continuing to use MTP3** in this workload, while preserving an immediately resumable draft. It remains **3.16% slower than genuine MTP-off**. These are screening means over four retained observations, not a universal guarantee. Ceiling1 reduces latency relative to ceiling3 but does not beat the off control here.

Do not substitute the earlier fixed-process zero/one-depth numbers: those also alter MTP context/output/rollback configuration. The prior global-zero guard screen was within ~1% of off; the practical pause in an already-MTP3-capable process is ~3% behind off in this matched test. This experiment separates the configurations but does not attribute the remaining gap to a single component.

The request control enables externally chosen budgets; **no improved automatic choice policy is claimed**. In particular, we do not claim that MTP now beats normal decoding with four 100k agents, or recommend disabling it purely by agent count at 350k. The one-agent and fixed-depth1 cache-refresh gains are separate results in `../mtp-kv-refresh-0907/REPORT.md`.

## Scope remaining

This implements neither draft-only windowed attention nor accepted-prefix-only refresh. It also does not automatically decide which agents should pause. Those policies must be judged against genuine off and include transition cost. Lowering the target's Q8 cache precision is not part of this work.


## Reproduction

`MTP_BUDGET_RESULTS_DIR` relocates the run/fixture directory. The scripts reference the retained sibling source fixture and model paths; `manifest.json` and `build-manifest.json` identify the exact tested inputs and binaries. Rebuild new candidate snapshots under a separate name, rather than modifying an executable used by active tests.

A normal build uses the repository's CMake configuration for SM70/SM75. The retained `build-server.py` describes the faster server-only rebuild used for this test; it needs the matching base build artifacts. Run `validate.py` for the complete transition/mixed-agent/bank suite, then `bench.py` under the exclusive GPU resource lock. `bank.py` resumes only the persistence section after a retained successful transition result. No benchmark should run during compilation or profiling.

Example native completion request, on an explicitly enabled research server:

```json
{"prompt":"Continue this task.","speculative_n_max":0,"n_predict":128}
```

A later request can set `speculative_n_max` to 1 or 3, or omit it to inherit the configured maximum. The flag and API ceiling do not lower the Q8 target/draft cache precision or discard context.

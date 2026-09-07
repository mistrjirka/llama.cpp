# MTP rollback and warm-pause integration — 7 September 2026

## User-visible changes

This update integrates the measured indexed-tail-removal and unused-prompt-copy improvements into the existing `v100-optimized` runtime. It also adds the independently tested per-request MTP ceiling, enabled by default. Existing K/V-only refresh, consolidated readbacks, exact-prefix sharing and parked-session persistence remain enabled. Target/draft weights, context limits, attention kernels and KV precision are unchanged.

The implementation builds on previous production `5a1e654a2`; its predecessor `f9d1c3a2b` has identical runtime code and supplies the immutable baseline executable. The full combined candidate is built normally with CMake, rather than linking an isolated research object into a different engine. Source/binary/fixture identity is recorded in `manifest.json`.

## Included changes and defaults

| Change | Integrated commits | Effect |
|---|---|---|
| Indexed sequence-tail removal | `d61c6b5d8`, `c30407d98` | Uses the existing ordered per-sequence index, retaining all cache metadata and allocation semantics; visits the requested interval instead of the full reserved pool. Generic for sequence-specific KV removal, not Ornith-specific. |
| No unused MTP prompt copies | `d61c6b5d8`, `c30407d98` | Only a sole MTP implementation can skip copying the prompt; other draft methods and combinations retain their original behavior. |
| Request-local MTP ceiling | `bdeb7a7fe`, `c30407d98` | An integer from zero through the configured server maximum. Does not resize the model, change target sampling, or overwrite another request's setting. |

With `--spec-type draft-mtp --spec-draft-n-max 3`, native completion requests may use `speculative_n_max: 0`, `1`, `2`, or `3`. `speculative.n_max` is an alias. Missing/null ceilings keep the process default. Primary-name precedence is retained. Invalid negative, fractional, boolean, string and oversized inputs are rejected. Zero keeps the draft cache current while suppressing proposals and unnecessary preparation; returning to a positive cap needs no full-history replay. The public internal draft-limit helper continues to treat negative overrides as unset; HTTP input is deliberately stricter.

This is **warm pause**, not a draft-model unload, hidden-state journal or reduced-memory mode. It does not add an automatic occupancy controller, change the model head, truncate draft attention or import the experimental target-attention/grouping/compact-layout changes. Clients can choose request budgets; defaults do not override the server's configured draft depth.

Rollback switches, read before process start: `LLAMA_KV_INDEXED_RM=0`, `LLAMA_MTP_SKIP_PROMPT_COPY=0`, `LLAMA_MTP_REQUEST_BUDGET=0`. Their `LLAMA_EXPERIMENT_*` aliases remain recognized only when the corresponding production variable is absent. Unset or `1` enables the feature; other explicit values disable it. Older refresh/readback switches are unchanged.

## Integration validation

The model-free CTest suite covers sampling, speculative cap semantics, batch allocation and indexed removal. The random removal test compares every metadata field, ownership set, logical position and selected free-cell index with the old scan through 100,000 mutations; the separately retained sanitizer run is in [the Gantt report](../mtp-gantt-0907/REPORT.md).

The default, disabled and previous-production binaries are compared on identical four-slot 100k histories. Separate transition tests use 3 → 0 → 3, compare ordinary versus cache-only refresh, then save/erase/restore target, draft and speculative state. Mixed 0/1/3/0 requests verify isolation, zero proposals for paused requests and default-depth reactivation. A one-slot/five-history bank test restores four parked sessions into a new process and compares continuation against the always-live control. It requires all 4096 cached tokens to be reused and only the 38-token suffix to be processed.

Qwen coverage includes four-slot built-in MTP and the actual local setup's single-slot tensor-parallel profile: 4:5 device split, Q8 target cache, FP16 draft cache, existing vocabulary shortlist, target ubatch2048 / draft512. These are short-context compatibility tests, not 400k performance measurements. A real image request checks that the normal multimodal path still completes. Validation interposers are never used for serving timings.

One initial Qwen baseline-driver assertion compared a symlink spelling to the canonical shared-library mapping and stopped before generation. The helper now compares resolved paths; the original baseline executable is unchanged. No failed trial is counted as a performance or correctness pass.

### Completed gates on the combined build

All four model-free CTest targets pass. The old-production/default/disabled comparison matches all four 64-token outputs, acceptance counts, raw target/draft/speculative snapshot bytes and a 32-token restored continuation. The 3 → 0 → 3 comparison matches the same state/output/counter criteria. All six malformed ceiling inputs are rejected; concurrent 0/1/3/0 and omitted-limit resume work on Ornith and Qwen.

The final bank test saves four parked histories (460,427,588 bytes), restarts the entire server, loads them, and resumes the selected history with 4096 cached tokens plus its 38-token suffix. Its 64 output tokens and 43/57 MTP acceptance match the always-live control. The reported 104.14 ms bank-read time is a single diagnostic and is not included in the serving gain.

The actual Qwen tensor-parallel/FP16-draft/shortlist profile matches baseline output and raw snapshots, with the independent cache and sampling-view validators passing. The combined Ornith validation checks 12 draft-cache shapes, 425 sampling views and 311 hidden rows; its real-image request completes 16 output tokens without server error. These are focused correctness/compatibility gates, not a broad model-quality evaluation. `checks.json` and the raw/summary files preserve their evidence.

## Final full-build benchmark

The main fixture is real Ornith AD-Q6_K-Q5_K plus Shisa Q5_0, with Q8 target and draft K/V, CPU projector, V100 32 GiB at 200 W + modified RTX 2080 Ti 22 GiB at 250 W. The 1.4M physical pool exposes four logical 400k slots, but the measured histories are **100k C++ source tokens per agent**, sharing approximately 98k. Requests append 38/42/44/41 tokens and generate 128 each. Four histories stay resident even with one active agent.

Each configuration compares old/new/new/old process arms with one excluded warmup and two retained turns per process: four observations per side. GPU clocks are not locked. No profiling, compiler or concurrent GPU job runs during timings. Startup and state restoration are excluded, while HTTP admission, prompt append and generation are included. Per-agent TG is the server's generation rate, not aggregate throughput. Both sides use the earlier cache-refresh/readback improvements; the new percentages must not be added to older separate experiments.

| Workload | Previous build: mean (range) | Integrated build: mean (range) | Turn-latency change | TG per agent, old → new |
|---|---:|---:|---:|---:|
| One active agent, MTP3 | 2.659 s (2.653–2.674) | 2.500 s (2.475–2.514) | -5.99% | 53.27 → 56.96 |
| Four active agents, MTP3 | 6.276 s (6.211–6.399) | 5.711 s (5.518–5.990) | -9.00% | 24.60 → 27.25 |
| Four active agents, MTP off | 5.709 s (5.655–5.779) | 5.750 s (5.682–5.810) | +0.72% | 26.14 → 26.00 |

At unchanged MTP3, the final combined build reduces mean turn time by **6.00% with one active agent** and **9.00% with four**. Per-agent TG increases by **6.94%** and **10.76%**, respectively. MTP-off mean latency increases by 0.72% with overlapping ranges, so this small screen does not establish an MTP-off regression or improvement. The tests are repeatable controlled measurements, not universal speed guarantees.

The four-agent MTP3 mean (5.711 s) is approximately equal to the separately measured old true-off control (5.709 s), and close to the new off control (5.750 s). The prior clear MTP3 penalty is removed in this fixture; do not treat the sub-percent new-MTP3-versus-new-off difference as a broadly established speculative-decoding advantage.

The original ablation identifies indexed cache removal as the principal gain and contains the actual before/after Nsight timelines: [MTP CPU-stall investigation](../mtp-gantt-0907/REPORT.md). Its measurement matrix is historical evidence, distinct from this full-build validation. The CPU-removal timing and GPU-idle percentage are profiled diagnostics, not user-facing token rates.

## Reproduce and audit

Persistent result root: `/workspace/oai-qwen38-pp-lab/results/mtp-final-integration-0907`. The committed harnesses reference the original immutable binaries and model/source fixtures; adjust those paths explicitly and record new hashes on another machine. They do not reconstruct prompts, silently use a different head, or overwrite production binaries.

```sh
cmake -S . -B build-sm70-75 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;75" -DCMAKE_BUILD_TYPE=Release
cmake --build build-sm70-75 --target llama-server test-kv-indexed-remove test-mtp-draft-budget test-sampling test-batch-alloc -j 12
ctest --test-dir build-sm70-75 --output-on-failure -R '^(test-sampling|test-mtp-draft-budget|test-kv-indexed-remove|test-batch-alloc)$'
```

Use `parity.py`, `transitions.py`, `qwen-budgets.py`, `qwen-tp.py` and `regression.py interposer` as separate correctness suites. Run `bench.py --suite four`, `--suite one`, and `--suite off` only under an exclusive GPU lock and without profilers. Reports keep output/state hashes and counters; multi-GB KV snapshots and raw traces stay outside Git.

# Validated MTP serving integration — 7 September 2026

This integration starts from production `7571e2e17`, not from the full research branch. It selects the independently validated cache refresh, readback and draft-budget changes. Target attention kernels, batch grouping, compact-first restore, adaptive depth, draft windowing and suspend/resume experiments are not imported. The existing shared-prefix/parked-cache behavior, context limits, quantization and configured MTP depth are preserved.

## Included code

- Synchronized sampling-output view and one dense target-hidden read per MTP refresh batch, originating in `5615ab1a5`.
- Early zero-budget checks before the first draft call and before server prompt/checkpoint preparation, originating in `575ca6ea7`.
- The no-output, single-layer MTP K/V update graph from `32dabfcb4`. It preserves the existing projection, normalization, RoPE, optional rotation and quantized cache-write operators, but does not construct an attention mask or run attention/FFN/vocabulary projection during refresh.
- The per-request positive draft cap from `9b579df8a`, extracted without its scheduler or attention experiments.
- An integration correction for the public `n_max = -1` convention: a negative request cap means no override, not no proposals. The same resolver is used before drafting and after each draft token; `test-mtp-draft-budget` covers boundary values and integer extremes without loading a model.

## Defaults and rollback

The validated readback paths are now automatic. Cache-only refresh defaults on only for a single-layer `qwen35` or `qwen35moe` MTP head with separate target/draft memory. Unsupported heads retain normal refresh. The target forward and proposing graph are unchanged.

Set any of these to `0` before starting the process to restore its original path:

```sh
LLAMA_MTP_KV_ONLY=0
LLAMA_MTP_BULK_HIDDEN=0
LLAMA_SAMPLING_VIEW=0
```

Unset or `1` enables the path; other explicit values disable it. The old `LLAMA_EXPERIMENT_*` aliases are retained only for compatibility and are consulted when the corresponding production variable is absent. No experimental controller becomes enabled by this integration. Existing local-llm-setup launch profiles use the new defaults after rebuilding the engine; their model/context settings need not change.

## Validation methodology

The old production revision and the integrated candidate are freshly built in the persistent sandbox and copied into separate immutable binary directories. Launchers pin matching shared libraries. GPU tests are serialized through `gpu:all`; profiler/interposer runs are separate from timings. Another worktree's ongoing research and production's unrelated untracked files are left untouched.

Main fixture: real Ornith AD-Q6_K-Q5_K with Shisa Q5_0 MTP, Q8 target and draft K/V, CPU vision projector, V100-SXM2 32 GiB plus modified RTX 2080 Ti 22 GiB. Four logical 400k slots use the same 1.4M physical KV pool. Actual histories contain 100k C++ source tokens each; this is not a four-full-400k or new near-350k benchmark.

The parity test compares old production, integrated optimizations disabled, and integrated defaults. It checks target probability records on identical input histories, sequential continuations, and raw target/draft/speculative snapshot hashes plus a restored continuation. Independent validation libraries compare the new sampling view to the established accessors on the same buffers and compare all logical draft K/V bytes against an ordinary refresh from the same saved input state. They are never linked into production or used for timing.

A real-image request checks the multimodal fallback. A second-architecture Qwen3.8-27B test exercises its built-in MTP with Q8 target/draft KV. The no-proposal test requires zero drafted tokens and a successful continuation. The parked-bank test replaces conversations through normal automatic slot selection, persists the bank, starts a new server process and verifies that a parked history resumes without full prefill and with the same output.

The initial bank harness pinned `id_slot=0`, bypassing automatic replacement/parking, and correctly saved an empty bank. Its assertion failed before testing persistence. The corrected test omits the physical-slot override; this is a test-driver correction, not a change to cache semantics.

Final test results, source/binary hashes and the benchmark summary are recorded alongside this report. Timings are within-fork old-production versus integrated comparisons, not new upstream-versus-fork measurements. Small timing differences and variations are not reported as guaranteed gains.

## Integration results

The final source is `f9d1c3a2b`; the comparison baseline is the actual production `7571e2e17`. The full integration is tested as a bundle, not inferred by adding separate research percentages.

- All three model-free test targets pass, including 48 draft-cap boundary combinations plus the default negative-override case.
- Final old-production/default/disabled parity checks pass: target probability records and four 64-token outputs match; enabled-versus-disabled acceptance counters, all raw snapshot components and restored continuation match.
- The integration candidate's same-state validators pass 12 Ornith K/V shapes and 426 sampling views; Qwen's Q8 draft test passes 12 shapes and 392 views. The later final change only corrects negative request overrides and has its own unit/parity rerun.
- The real-image smoke request completes, and the zero-budget server completes four requests with no drafted or accepted speculative tokens.
- A final-build parked bank containing two conversations (227,905,962 bytes) reloads into a newly started server. The selected conversation resumes with 4,000 cached tokens and the exact always-live control output. This tests real process restart, not just erase/restore inside one process.

### Final unprofiled serving comparison

For each depth, A/B/B/A has two process lifetimes per arm, each with one discarded warmup and two retained turns: four measured turns per condition. Each turn appends 38/42/44/41 source-review prompt tokens to four 100k cached histories and generates 128 tokens per agent. Loading/restore is excluded; request admission and prompt processing are included. Per-agent TG is the server-reported generation rate, not aggregate throughput. GPU clocks are not locked and these small samples do not establish universal speedups.

| Draft depth | Old production turn | Integrated turn | Mean turn reduction | Old / new TG per agent |
|---|---:|---:|---:|---:|
| Off | 5.731 s | 5.711 s | 0.35% | 26.14 / 26.18 tok/s |
| MTP1 | 5.936 s | 5.786 s | 2.52% | 25.56 / 26.21 tok/s |
| MTP3 | 6.443 s | 6.368 s | 1.16% | 24.31 / 24.38 tok/s |

MTP1's measured ranges (5.883–5.961 s old; 5.746–5.828 s integrated) do not overlap in this screen. MTP-off and MTP3 differences overlap run variation. The results do not establish that four-agent MTP3 beats ordinary decoding; it still does not in this workload. Configured production MTP depth has deliberately not been changed.

The older isolated cache-refresh experiment's one-agent result is not rerun here and is not added to the integrated percentages. Its 96.5% reduction describes only refresh-stage GPU work, not total inference latency.

### Actual Qwen launch-profile compatibility

The final binary also passes a short-context test using the local setup's tensor split (RTX:V100 4:5), its existing vocabulary shortlist, target Q8 K/V, **FP16 draft K/V**, target ubatch2048 and draft ubatch512. Baseline and integrated defaults produce the same 64-token output and byte-identical target/draft/speculative snapshot files. Same-state cache and sampling-view validators pass in a separate instrumented run. See `qwen-production-profile.json`. This checks the actual split/quantization path, not a new 400k capacity benchmark.

## Reproduce

The retained Python drivers refer to immutable binaries and existing fixtures under `/workspace/oai-qwen38-pp-lab/results/mtp-production-merge-0907`. They intentionally do not recreate downloaded models or overwrite recorded binary snapshots. `manifest.json` identifies the exact baseline, candidate, final binaries and source-token fixture. Adapt these explicit paths for another host and record its hashes anew.

Build the engine and model-free tests normally, then run:

```sh
cmake --build build-sm70-75 --target llama-server -j 12
cmake --build build-sm70-75 --target test-mtp-draft-budget test-sampling test-batch-alloc -j 12
ctest --test-dir build-sm70-75 --output-on-failure -R '^(test-sampling|test-batch-alloc|test-mtp-draft-budget)$'
```

The two validation interposers compile separately with `c++ -shared -fPIC -O2 -std=c++17 -Iinclude -Iggml/include SOURCE.cpp -ldl -o OUTPUT.so`. They must never be preloaded during benchmark timing or production serving. `regression-final.py` has separate `parity`, `zero` and `perf` suites; `bank-check.py bank` tests automatic parking; `qwen-production-profile.py` tests the local Qwen layout. Main `regression.py interposer` and `regression.py qwen` retain the earlier integration-candidate byte checks.

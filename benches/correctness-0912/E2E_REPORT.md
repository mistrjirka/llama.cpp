# INT8 Stream-K end-to-end validation

Date: 2026-09-12.

Base revision: `229c45494`. The production change forwards `int8_qk` to the final partial Stream-K tile in `ggml/src/ggml-cuda/fattn-mma-f16.cuh`.

## Cause and scope

The two complete-tile calls forwarded the packed INT8 Q/K flag; the last partial-tile call omitted it and used the default `false`. The tail then interpreted packed data and byte strides as the ordinary FP32-Q/FP16-K layout. This can read beyond the allocation. The 65k threshold enables the affected long-Q8 specialization.

The fix keeps the original GQA6/GQA8 dispatch, INT8 arithmetic, Stream-K launch policy, VMM allocator, MMQ tuning, batch sizes, and context capacity. Earlier tile-layout and Stream-K-disable workarounds are not included.

## Crash and lifetime checks

| Test | Result | Details |
|---|---|---|
| Exact boundary restore/append | Pass | Appends: 128, 384, 512, 640, 1000, 512; verified 65,024 cached tokens before every request. |
| Exact boundary under Compute Sanitizer | Pass | Application and sanitizer exit 0; `ERROR SUMMARY: 0 errors`. |
| Fresh 100k, MTP off | Pass | 0 cached + 100,000 fresh tokens; 1028.19 PP tok/s; 16 generated tokens. |
| Fresh 100k, Q4 draft MTP | Pass | 0 cached + 100,000 fresh tokens; 1020.84 PP tok/s; 16 generated tokens. |
| Four slots, Q4 MTP with head reuse | Pass | 10 erase/restore/generate rounds, four 100k slots, 128 generated tokens per request. |
| Four slots, MTP off | Pass | 10 erase/restore/generate rounds, four 100k slots, 128 generated tokens per request. |
| Four slots, Q8 MTP without head reuse | Pass | 10 erase/restore/generate rounds, four 100k slots, 128 generated tokens per request. |

Stress tests check successful completion, expected cache reuse, unchanged binary hashes, and clean process exit. They do not treat different generated-token hashes as a pass for deterministic decoding.

## Numerical regression tests

All ten new D256/Q8 cases passed against the CPU reference on the RTX 2080 Ti, and all ten passed on the V100. They cover GQA6 and GQA8, query sizes 128/129/512, and KV lengths immediately below, at, and above 65,536. The original matched CUDA library crashes on the new GQA8 / 65,536 KV / 128-query test with an illegal memory access (exit 134). The fixtures therefore detect the original defect rather than only exercising already-safe shapes.

Evidence: [RTX 10/10](e2e-results/numerical-rtx-ten.log), [V100 10/10](e2e-results/numerical-v100-ten.log), [original failing control](e2e-results/numerical-baseline-gqa8-128.log).

Focused Compute Sanitizer racecheck on GQA8 / 65,536 KV / 128 queries also passed on both GPUs: 1/1 numerical case per device, zero errors and zero warnings. This is a focused kernel check, not proof that full-model nondeterminism is resolved. [RTX racecheck](e2e-results/racecheck-rtx-gqa8-128.log), [V100 racecheck](e2e-results/racecheck-v100-gqa8-128.log).

## Matched performance checks

ABBA process order, one warm-up plus three retained 1,000-token append timings per process, and one 64-token generation per process. Both engines use the same saved prefix and explicit launch settings. Normal and FORCE_MMQ pairs share all non-CUDA binaries; the CUDA libraries share immutable unchanged objects, and the three INT8-capable translation units are rebuilt from pinned pre-fix/fixed sources.

| Workload | Build | Before PP | Fixed PP | PP change | Before TG | Fixed TG | TG change |
|---|---|---:|---:|---:|---:|---:|---:|
| ornith-dual | force-mmq | 1085.61 | 1083.33 | -0.21% | 53.37 | 53.61 | +0.45% |
| ornith-rtx | force-mmq | 1489.93 | 1469.63 | -1.36% | 58.91 | 58.96 | +0.09% |
| qwen-rtx | normal | 494.31 | 493.84 | -0.09% | 17.30 | 17.28 | -0.09% |
| qwen-v100 | normal | 454.16 | 454.18 | +0.00% | 16.40 | 16.41 | +0.08% |

PP and TG are tokens/second. RTX-only workloads use 65,536 cached tokens; dual Ornith and V100 Qwen use 100,000. The dual Ornith regression control uses the established 14:35 RTX:V100 layer split, not the README tensor-split headline configuration. These comparisons isolate the crash fix; they do not replace the historical upstream-versus-fork graph.

## Single-RTX throughput confirmation

A separate reverse-order BAAB run retained ten prefill samples per engine. Baseline: **1488.65 PP tok/s** (SD 1.75); fixed: **1464.94 PP tok/s** (SD 3.62); change: **-1.59%**. Decode: 58.66 -> 58.64 tok/s (-0.03%). Together with the first ABBA (-1.36% PP), this confirms a small **1.4-1.6% prefill regression**, not just within-arm noise. The fix is retained because it removes the reproducible invalid memory access while preserving the fast path. No claim of zero performance regression is made.

[Reverse-order raw results](e2e-results/confirm-ornith-rtx.json).

## Remaining checks and limitations

Repeated greedy four-slot runs still produce different output hashes, including with MTP disabled. The older mixed-GPU determinism issue remains open; this crash repair is not claimed to fix it. Numerical backend-reference tests passed as described above; they do not establish full-model greedy determinism.

Dense Qwen MTP target-head reuse is not exercised by the no-MTP Qwen throughput controls. The roughly 2.9 GiB draft compute arena is unchanged. Those are separate coverage/optimization items, not reasons to disable the crash fix.

An additional tensor-split Ornith ABBA did not finish: its last baseline arm returned a content-format parser HTTP 500. It is not included as a completed performance result. The four reported profile comparisons and the reverse-order RTX confirmation completed normally.

## Reproduction and evidence

`validate_streamk_e2e.py` runs the boundary, cold-prefill, and four-slot tests. `benchmark_streamk_regression.py` runs matched ABBA controls. Use the Development Sandbox `gpu:all` lock; do not benchmark while another workload uses these GPUs.

Original logs, binary hashes, and full records remain under `/models/.bench-ornith-mtp4/int8-streamk-validation/e2e/`. The following completed result records are copied next to this report:

- [fixed-boundary-series](e2e-results/fixed-boundary-series.json)
- [fixed-boundary-memcheck](e2e-results/fixed-boundary-memcheck.json)
- [fixed-cold100k-off](e2e-results/fixed-cold100k-off.json)
- [fixed-cold100k-mtp-q4](e2e-results/fixed-cold100k-mtp-q4.json)
- [fixed-stress-q4-head](e2e-results/fixed-stress-q4-head.json)
- [fixed-stress-off](e2e-results/fixed-stress-off.json)
- [fixed-stress-q8-plain](e2e-results/fixed-stress-q8-plain.json)
- [bench-ornith-dual](e2e-results/bench-ornith-dual.json)
- [bench-ornith-rtx](e2e-results/bench-ornith-rtx.json)
- [bench-qwen-rtx](e2e-results/bench-qwen-rtx.json)
- [bench-qwen-v100](e2e-results/bench-qwen-v100.json)

[Compute Sanitizer log](e2e-results/fixed-boundary-memcheck.log).

No production service or host launch configuration is changed by these validation scripts. Commit/push status must be checked independently; these records alone do not establish deployment.

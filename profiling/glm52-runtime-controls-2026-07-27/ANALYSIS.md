# GLM-5.2 CPU/runtime control sweeps — 2026-07-27

## Decision

Keep the established production settings for the frozen exact vertical path:

- decode threads: 40;
- batch threads: 48;
- polling: 50;
- no strict CPU affinity;
- graph-build diagnostics unchanged from the existing runner.

None of the tested thread, affinity, polling, or graph-log variants produced a repeatable throughput improvement. The benchmark runner now exposes these settings so future tests can vary them without editing the command line.

All runs in these sweeps preserved:

- exact output SHA-256 `e165b05c0568da0b099159b3045190e4f33ffbd97698d0a9881b982ead61f514`;
- 152 decode graph splits;
- 14540.62 MiB accepted cache allocation;
- no CUDA OOM.

## Host topology

The test host has one AMD EPYC 7352 socket with 24 physical cores and 48 SMT threads in a single NUMA node.

## Decode-thread sweep

Single-pass screening results:

| Decode threads | tok/s |
|---:|---:|
| 20 | 5.6925 |
| 24 | 5.8786 |
| 32 | 5.4375 |
| 36 | 5.8283 |
| 40 | 6.0858 |
| 42 | 6.0680 |
| 44 | 6.1099 |
| 46 | 6.0975 |
| 48 | 3.9066 |

Crossed repetitions showed that the apparent 44-thread lead was noise:

| Threads | Run 1 | Run 2 | Mean |
|---:|---:|---:|---:|
| 40 | 6.0858 | 5.8878 | 5.9868 |
| 44 | 6.1099 | 5.8633 | 5.9866 |

The 40–46 region is a flat performance plateau. Using all 48 SMT contexts causes a severe collapse, so 40 remains the safer default.

## Affinity and polling

| Configuration | tok/s |
|---|---:|
| Unpinned, poll 50 | 5.9855 |
| Strict 0–39 decode / 0–47 batch, poll 50 | 5.9444 |
| Unpinned, poll 100 | 5.9603 |
| Strict 0–39 decode / 0–47 batch, poll 100 | 5.9169 |

Strict placement and maximum polling did not improve this workload. The default unpinned poll-50 configuration remains preferred.

## Graph-build diagnostic logging

The vertical runner emitted 1575 `moe-graph-build` lines per run. A crossed five-run A/B compared the existing diagnostic mode with those lines disabled.

| Mode | Mean tok/s | Median tok/s | Trimmed mean tok/s | Pooled token median | stderr size |
|---|---:|---:|---:|---:|---:|
| Diagnostics on | 6.0153 | 6.0185 | 6.0192 | 164.383 ms | 1136.4 KiB |
| Diagnostics off | 5.9410 | 5.9434 | 5.9617 | 165.812 ms | 430.9 KiB |

Disabling diagnostics greatly reduces output volume, but did not improve throughput and had more isolated scheduler stalls in this sample. This is not evidence that logging is beneficial; it is evidence that logging overhead is below the run-to-run noise and should not be claimed as a speed optimization.

The existing runner default is retained for continuity. `TRACE_GRAPH_BUILD=0` remains available for quieter production-style logs.

## Runner controls added

`profiling/run_glm52_scale_case.sh` now accepts:

- `THREADS`;
- `BATCH_THREADS`;
- `POLL`;
- `CPU_RANGE` and `CPU_STRICT`;
- `BATCH_CPU_RANGE` and `BATCH_CPU_STRICT`;
- `TRACE_GRAPH_BUILD`.

These values are also recorded in compact summary JSON files.

## Next priority

CPU tuning cannot close the remaining architectural gap. The next test should target stable-graph GPU launch overhead by comparing CUDA graphs on/off for the frozen exact route-map path. If CUDA graphs remain ineffective, effort should move to reducing the exact CPU fallback islands or their scheduler boundaries rather than tuning host worker placement.

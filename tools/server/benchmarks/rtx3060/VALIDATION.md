# Validation of the selected RTX 3060 configuration

**Status: runtime stability, cache limits, and basic vision checked. The 100K answer check failed. Original-baseline comparison and deployment verification remain pending.**

Tested settings: native MTP2, draft Q4_0 KV, target Q8_0 KV, logical batch 2048, microbatch 128, configured context 110,000, cache budget 4,096 MiB, three checkpoints with a 384 MiB checkpoint cap. The context allocation is 110,080 tokens. Tests used the clean release prepared from `1af6247`.

Directory names matter: `soak-b512` contains the earlier rejected batch-512 candidate. `final-soak` contains the selected batch-2048 run.

## Full-cache pressure and vision

The selected run completed **83 total requests**: two rounds of 20 catalog branches, 40 interleaved auxiliary requests, two cold comparisons, and one image request.

| Measurement | Observed |
| --- | ---: |
| Peak accounted parked-cache storage | 4,084.972 MiB of 4,096 MiB |
| Entries at peak accounted storage | 17 |
| Logical snapshot bytes represented at that point | 7,079.433 MiB |
| Token references represented at that point | 130,145 |
| Maximum token references, at a different point | 131,161 |
| Kernel peak process RSS | 6,869.406 MiB |
| Sampled peak process PSS | 6,783.139 MiB |
| Minimum available host RAM | 5,472.375 MiB |
| Peak reported GPU memory | 11,089 MiB |
| Process swap / cgroup OOM events | 0 / 0 |
| New zram failed writes | 0 |

The 17 entries include retained prefixes and auxiliary prompts. Token references count shared prefixes more than once. They are not 130K unique tokens or a larger active-context limit. Cache sharing lets these logical snapshots fit within the parked-cache budget. The full process still needs RAM beyond that budget.

Repeated responses were identical, but the expected-answer check failed for branch 4 and branch 13 in both rounds. This establishes cache and memory behavior for the trace, not perfect model accuracy. The independent branches had no identical checkpoint objects to share at their peak. The separate fork experiment remains the evidence for checkpoint-sharing savings.

The image check passed with `left color: red, right color: blue.` It took **83.67 seconds** after the cache-pressure workload. This checks a simple two-color image, not general vision quality.

Sources: [final-soak.json](results/2026-11-24/final-soak.json), [configs.json](results/2026-11-24/configs.json), [traces.json](results/2026-11-24/traces.json), and [runtime-limits.json](results/2026-11-24/runtime-limits.json). Raw captures are under `/home/jirka/.cache/rtx3060-experiments-20261124/final-soak`.

## Context length, reuse, and restoration

The selected run completed 15 requests across three context lengths. Each length has a cold request, immediate repeat, changed followup, auxiliary request that parks the long state, and restoration of the original request.

Times below are **seconds to first visible response content**, not full-response times. Each cell is one observation, not a repeated-trial statistic.

| Actual cold prompt tokens | Cold | Immediate repeat | Changed followup | Restore after parking | Cold input tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8,058 | 11.845 | 0.324 | 0.828 | 1.052 | 680.63 |
| 32,059 | 52.105 | 0.337 | 0.870 | 2.104 | 615.50 |
| 100,052 | 216.830 | 0.406 | 1.139 | 5.427 | 461.56 |

- **8K and 32K:** every checked cold, repeat, followup, and restored answer matched the expected JSON values.
- **100K:** all four answer/format checks failed. Each response used the 96-token limit and did not produce the required JSON.
- At 100K, the cold, repeated, and restored requests returned byte-identical text. This supports correct reuse of that tested state despite the answer failure.
- The followup changes the prompt. Judge its answer against the expected JSON, not against the cold-response text. Its different text is not by itself a cache error.

This is an arithmetic catalog with a predictable rule. It tests runtime behavior and a small answer check, not difficult long-context retrieval or broad quality.

The complete context-length run recorded a 6,225.801 MiB kernel peak RSS, 6,139.924 MiB sampled peak PSS, at least 6,127.781 MiB available host RAM, zero process swap, zero cgroup OOM events, and no new zram failed writes.

Sources: `/home/jirka/.cache/rtx3060-experiments-20261124/long-text/selected/{config,results,long-context-input}.json` and `long-text/runtime-limits.json`. The long-text result archive is awaiting its repository refresh.

## Remaining decision

The original binary's 100K control is running under `long-control/original`. **Do not attribute the selected configuration's failed 100K answer check to the optimization until that comparison completes.** Record whether the original also fails, the exact request/configuration differences, and its response hash.

The selected final soak and long-text run stayed within the cache budget without swap-write failures or OOM events. This is evidence from the tested workloads, not a claim that every historical crash is fixed.

Deployment is not verified by this document. Before marking it complete, record the running service and binary, selected flags, installed commit, and post-restart health check.

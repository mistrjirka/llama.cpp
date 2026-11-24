# Validation of the final RTX 3060 settings

**Final choice: draft Q4 with microbatch 256. The 100K comparison is complete and all eight final 8K/32K answer checks pass. The new final pressure run including vision, and deployment verification, remain pending.**

Settings: native MTP2, draft Q4_0 KV, target Q8_0 KV, logical batch 2048, microbatch 256, configured context 110,000, cache budget 4,096 MiB, three checkpoints with a 384 MiB checkpoint cap. The context allocation is 110,080 tokens. The clean release was prepared from `1af6247`.

Archives: `soak-b512` is the rejected batch-512 run. `soak-ub128` is the rejected microbatch-128 run. The new `final-soak` uses the final microbatch-256 settings.

## Completed 100K comparison

The original and final cold requests contain **100,052 prompt tokens**. Times below are seconds to first visible response content. Each cell is one observation, not a repeated-trial statistic.

| Configuration | Cold | Immediate repeat | Restore after parking | Cold input tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Original, draft Q8, microbatch 256 | 205.602 | 0.418 | 5.613 | 486.759 |
| Final, draft Q4, microbatch 256 | 205.669 | 0.418 | 5.441 | 486.596 |
| Rejected, draft Q4, microbatch 128 | 216.830 | 0.406 | 5.427 | 461.560 |

**All five final response hashes match the corresponding original hashes:** cold, repeat, changed followup, auxiliary prompt, and restored request. The final changed followup reached first visible content in 1.104 seconds, versus 1.102 seconds originally.

Both original and final configurations nevertheless fail the expected-answer check on all four catalog requests. They return:

| Field | Expected | Original and final |
| --- | ---: | ---: |
| first | 640 | 640 |
| middle | 77,082 | 7,707 |
| last | 154,116 | 54,116 |

They produce valid JSON in 36 generated tokens. The answer error therefore exists in the original setup. The draft-Q4 change preserves the tested response behavior. Matching incorrect answers is not a quality pass.

The rejected microbatch-128 configuration instead produces prose, reaches the 96-token response cap, and fails the required JSON format. Its cold first-content time is about 5.5% longer. Its additional 62 MiB GPU saving does not justify this tradeoff.

A changed followup includes the preceding answer. When earlier answers differ, followup prompts can differ too. Judge a followup against its expected JSON, not against the cold text or a different configuration's followup hash alone.

This is an arithmetic catalog with a predictable rule. It tests runtime behavior and a small answer check, not difficult long-context retrieval or broad quality.

Sources under `/home/jirka/.cache/rtx3060-experiments-20261124`:

- `long-control/original/{config,results,long-context-input}.json`
- `long-draft-q4/draft_q4_only/{config,results,long-context-input}.json`
- `long-text/selected/{config,results,long-context-input}.json` for the rejected microbatch-128 comparison

## Completed final 8K/32K checks

The final settings completed ten requests, including two auxiliary prompts. All eight cold, repeat, changed-followup, and restored answer checks passed. Times are single observations to first visible content.

| Actual cold prompt tokens | Cold | Immediate repeat | Changed followup | Restore after parking |
| --- | ---: | ---: | ---: | ---: |
| 8,058 | 11.203 s | 0.325 s | 0.818 s | 1.023 s |
| 32,059 | 49.483 s | 0.332 s | 0.853 s | 2.121 s |

Source: `long-final-short/selected/{config,results,long-context-input}.json` under the same experiment directory. This uses the final microbatch 256, unlike the earlier `long-text` run.

## Historical pressure evidence, not final-setting results

The earlier microbatch-128 run completed **83 total requests**, including one image request. Its measurements remain useful for diagnosing cache growth, but must not be labeled as the final microbatch-256 result.

| Earlier microbatch-128 measurement | Observed |
| --- | ---: |
| Peak accounted parked cache | 4,084.972 MiB of 4,096 MiB |
| Entries at that point | 17 |
| Logical snapshot bytes represented | 7,079.433 MiB |
| Token references at peak bytes | 130,145 |
| Maximum token references, at a different point | 131,161 |
| Kernel peak process RSS | 6,869.406 MiB |
| Sampled peak process PSS | 6,783.139 MiB |
| Minimum available host RAM | 5,472.375 MiB |
| Peak reported GPU memory | 11,089 MiB |
| Process swap / cgroup OOM / new zram failed writes | 0 / 0 / 0 |

Entries include retained prefixes and auxiliary prompts. Token references count shared prefixes more than once. They do not represent 130K unique tokens or a larger active-context limit. The process needs RAM beyond the 4 GiB parked-cache budget.

That run's repeated responses were identical, but branch 4 and branch 13 failed the expected-answer check in both rounds. Its basic two-color image check passed in 83.67 seconds. Its earlier 8K and 32K tests passed all expected-answer checks. These results do not replace validation of the final settings.

Raw pressure evidence is under `soak-ub128/selected`. The earlier batch-512 run is separately archived under `soak-b512`. Repository evidence is collected under [results/2026-11-24](results/2026-11-24).

## Final gates still pending

| Gate | Required result |
| --- | --- |
| Final pressure, `final-soak` | 83 total requests including vision, cache-budget maximum, retained capacity, memory plateau, swap/OOM counters, image answer and latency |
| Deployment | Running service and binary, selected flags, installed commit, and post-restart health check |

No claim of final deployment is made here. The short-suite final draft-Q4 result saves 108 MiB of GPU memory. The 56 MiB short-suite CPU RAM reduction belonged to rejected microbatch 128. Checkpoint sharing's separate 446.70 MiB fork peak-RSS saving remains supported by its own workload.

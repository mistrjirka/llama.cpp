# Validation of the final RTX 3060 settings

**Runtime validation is complete for draft Q4 with microbatch 256. All eight final 8K/32K answer checks pass. The 100K output matches the original, including its known wrong answer.**

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

## Completed final pressure and vision test

The final microbatch-256 configuration completed **83 requests**, including one image request. All recorded parked-cache accounting stayed within 4,096 MiB. Repeat and cold-control responses matched their corresponding earlier responses. Four catalog answers, branches 1 and 4 in each round, failed the expected-answer check. Those branches also failed in the original short cache baseline. This is a memory/cache consistency pass, not an all-answers-correct claim.

| Final configuration measurement | Observed |
| --- | ---: |
| Peak accounted parked cache | 4,082.173 MiB of 4,096 MiB |
| Entries at that point | 17 |
| Logical snapshot bytes represented | 7,076.833 MiB |
| Token references at peak bytes | 130,146 |
| Maximum token references, at a different point | 131,161 |
| Kernel peak process RSS | 6,929.727 MiB |
| Sampled peak process PSS | 6,843.313 MiB |
| Minimum available host RAM | 5,378.551 MiB |
| Peak reported GPU memory | 11,149 MiB |
| Process swap / cgroup OOM / new zram failed writes | 0 / 0 / 0 |

Request-end maximum PSS was 6,779.674 MiB in round one and 6,630.838 MiB in round two. Cache entries were evicted and reused within the budget. This finite trace supports bounded behavior here, not a promise that every future workload has no leak.

Entries include retained prefixes and auxiliary prompts. Token references count shared prefixes more than once. They are not unique tokens or a larger active-context limit. The 4 GiB cache budget is separate from total process RAM.

The image test correctly returned left red and right blue in 86.09 seconds. The full-precision image projector stayed on the CPU. Its latency was not an optimization target in this release.

Evidence: [final-soak.json](results/2026-11-24/final-soak.json), [validation-checks.json](results/2026-11-24/validation-checks.json), [runtime-limits.json](results/2026-11-24/runtime-limits.json), and the preserved host logs under `final-soak/selected`.

## Deployment verification

The runtime validation is complete. The service restart and installed-binary record will be added after deployment.

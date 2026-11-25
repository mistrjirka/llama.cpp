# Cache hit diagnosis and checkpoint fix

## Deployed result

The RTX 3060 hub was reloaded successfully at **2026-11-25 01:18:09 CET** with source commit **dafcd5a627664491f4c95e2f8a21f612df722d74**.

The fixed policy preserves useful older checkpoints when a new checkpoint is close to an existing one. It also handles replacement and rejected allocations without evicting unrelated state. MTP remains enabled with two draft tokens and Q4 draft KV. The cache budget remains 4096 MiB. The checkpoint limit remains three and 384 MiB.

[Deployment and loaded binary hashes](results/2026-11-25/deployment.json) verify the new process, nine mapped executable/library files, runtime arguments, health, and memory counters.

## Observed cache reuse before the fix

These are token-weighted reuse fractions, not the fraction of requests with any cache hit. Repeated prefixes are counted on each request.

| Observation | Reused input tokens | Other finding |
|---|---:|---|
| 133 completed requests | 95.76% | Median prompt processing 1.34 s |
| Latest 20 completed requests | 97.68% | Median prompt processing 1.86 s |
| Stalled request at 00:42:45 CET | About 20,570 of 83,004 | 62,434 processed tokens in 130.59 s |

The stalled request shared approximately 63,762-63,788 prefix tokens with the previous prompt. A usable checkpoint at token 63,643 had been evicted at 00:39:05 CET. The server fell back to the protected base checkpoint around token 20,570.

There was no parked-cache load before this stall. Selection to launch took 1.83 ms. The continuation predicate was false, so slot-local continuation bookkeeping does not explain this event. Request-arrival timestamps are absent, so queue wait cannot be measured.

The common-prefix interval comes from rounded similarity logs. Historical input and reused-token counts are reconstructed from completion counters and can differ by up to two tokens when generation stops inside an accepted MTP group. Exact eviction positions come directly from the log. These uncertainties do not affect the approximately 43,000-token excess replay.

[Sanitized diagnosis and numeric observations](results/2026-11-25/cache-diagnosis.json)

## Confirmed retention bug

The policy considered only existing checkpoints before selecting a victim. At capacity, it discarded the distant checkpoint at 63,643 while retaining a checkpoint at 72,470 and creating a new checkpoint around 72,798. The last two were only about 328 tokens apart, well below the configured 8192-token spacing.

The deterministic regression uses those positions:

| State after admitting the new checkpoint | Checkpoint positions |
|---|---|
| Previous policy | 20,570 / 72,470 / 72,798 |
| Fixed policy | 20,570 / 63,643 / 72,798 |

The new policy makes room only when the incoming checkpoint exceeds count or byte capacity. It first replaces a nearby checkpoint from a previous request. If none qualifies, it removes an older redundant point or the oldest non-base point. It stops as soon as the incoming checkpoint fits.

Exact-position replacement happens before capacity eviction. An oversized incoming checkpoint is rejected before any existing checkpoint is removed. Byte checks still count vector capacity, including target, draft, and speculative state.

The system-prefix preference and current-request proximity exemption remain. The new checkpoint is captured only after space is available. Serialization, immutable sharing, model weights, KV types, and inference kernels are unchanged.

Keeping checkpoint 63,643 would have reduced processing in the recorded request from about 62,434 to 19,361 tokens, avoiding about 43,073 token evaluations. At the observed mean prefill rate, that corresponds to roughly 90 seconds. This is a counterfactual estimate, not a measured replay of the original user request.

## Validation

The original retention ordering was extracted into the same testable helper before applying the fix. It failed eight assertions. The corrected helper passed all 62 server component tests and 221 assertions, including 30 checkpoint assertions.

Coverage includes the observed close-pair eviction, exact refresh, rejected allocations, spare capacity, byte-capacity accounting, base/current-request protection, and stopping eviction when the incoming checkpoint fits.

- [Before test output](results/2026-11-25/unit-before.log)
- [After test output](results/2026-11-25/unit-after.log)

After the reload, four synthetic chat requests used the existing checkpoint-fork fixture. Each request contained 10,162 input tokens and generated the same six-token JSON response.

| Request | Reused tokens | Processed tokens | Total request time |
|---|---:|---:|---:|
| Cold conversation A | 0 | 10,162 | 15.20 s |
| Repeat A | 10,158 | 4 | 0.48 s |
| New branch B | 539 | 9,623 | 14.80 s |
| Return to A | 10,158 | 4 | 1.82 s |

All four responses matched the expected object and had identical response hashes. MTP drafted four tokens and accepted three in each request. The repeated prompt reused 99.96% of its input.

The new branch still required substantial processing. Three checkpoints cannot cover every earlier branch while retaining the system prefix and current-request tail points. This fix prevents avoidable loss of useful state. It does not guarantee a checkpoint at every possible edit location.

These are single observations, including cache-transfer and request overhead. They are a deployment smoke test, not a broad quality or throughput benchmark.

[Smoke-test results](results/2026-11-25/cache-smoke.json)

## Other speed and memory observations

Generation slowed as conversations grew and draft acceptance fell. Across 127 completed generation runs, ten pairs matched for context, draft acceptance, and output length averaged 28.52 tok/s earlier and 28.22 tok/s later. Five stricter pairs averaged 28.25 and 28.53 tok/s. This does not show a large cumulative generation slowdown within the original process.

Before the reload, the server had zero swap use, no OOM events, no sustained memory pressure, and normal active GPU clocks. The zram failed-write count remained 47,732,577. After the reload and smoke test, swap and OOM events remained zero, and that counter was unchanged.

The 4096 MiB setting limits parked cache storage. It is not a limit on total process RAM. Model data, active checkpoints, inference buffers, and the image projector also use RAM. No lower total-RAM claim is made from the newly restarted process.

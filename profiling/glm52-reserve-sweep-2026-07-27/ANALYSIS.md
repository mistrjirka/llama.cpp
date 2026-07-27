# GLM-5.2 static vertical reserve sweep — 2026-07-27

## Result

The best tested configuration is the frozen static vertical map with a 3072 MiB cache reserve.

| Placement | Runs | Mean decode tok/s | Decode splits | Minimum free VRAM |
|---|---:|---:|---:|---:|
| Default horizontal | 3 | 4.9233 | n/a in compact summaries | 4538 MiB |
| Static vertical, 4096 MiB reserve, aligned prompt | 2 | 5.3805 | 152 | 1988 MiB |
| Static vertical, 3072 MiB reserve | 3 | 5.9630 | 152 | 1364 MiB |
| Static vertical, 2048 MiB reserve | 1 | 5.7766 | 152 | 482 MiB |

The 3072 MiB configuration is:

- 21.12% faster than the three recorded default-horizontal runs;
- 10.83% faster than the two aligned 4096 MiB static-vertical runs;
- topology-preserving at 152 decode graph splits;
- deterministic across all three repetitions by output SHA-256;
- above the required 512 MiB measured VRAM safety guard by 852 MiB.

Measured 3072 MiB repetitions:

- 6.0254 tok/s;
- 5.9994 tok/s;
- 5.8641 tok/s.

All three produced:

`e165b05c0568da0b099159b3045190e4f33ffbd97698d0a9881b982ead61f514`

## Why 3072 MiB wins

Reducing the reserve from 4096 to 3072 MiB raises the static capacity from approximately 17 to 18 experts in most routed layers. Cache allocation rises to 14540.62 MiB, with 1337 ready slots. The additional coverage is useful enough to reduce CPU-side expert work without increasing graph splits.

Reducing the reserve again to 2048 MiB fits approximately 19 experts in most layers and raises allocation to 15343.78 MiB, but throughput falls to 5.7766 tok/s. The extra hot-branch GPU computation outweighs the saved CPU work, and measured free VRAM falls to 482 MiB, below the safety requirement.

Therefore maximum occupancy is not the objective. The useful operating point balances resident-route coverage against the cost of evaluating a larger hot branch.

## Next engineering priorities

1. Keep the frozen/static topology and 3072 MiB reserve as the new performance baseline.
2. Reuse one canonical-to-slot GPU remap per routed layer. The current graph performs equivalent remapping separately for up, gate, and down tensors, creating roughly 150 redundant tiny kernel launches per decode token.
3. Move static bundle loading out of the promotion-worker queue. Even the static run reports 4011 jobs, 170 batches, and multi-second aggregate queue delay during initialization. This does not affect steady-state decode in the short benchmark, but it increases startup and complicates future map publication.
4. Implement versioned A/B route-map publication with updates disabled first. It must retain 152 splits and at least 98% of this 5.9630 tok/s baseline before any adaptive policy is enabled.
5. Add only a very small protected adaptive tail after topology parity. Per-token mutation and admission churn are not competitive with horizontal placement.

## Scope and caveats

The horizontal results are existing compact summaries from the same worktree/build lineage, not newly rerun in this sweep. Full-logit parity was not recomputed here; this sweep verified identical generated output hashes under the forced-token workload. The 2048 MiB configuration was sampled once because it already failed both the throughput trend and the 512 MiB headroom gate.

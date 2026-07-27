# GLM-5.2 frozen static bulk-upload experiment — 2026-07-27

## Decision

Reject the packed static-upload implementation.

It reduced the physical H2D API transfer count from 4011 to 225 while preserving correctness, topology, and VRAM usage, but it increased end-to-end wall time. The additional full-size pageable-to-pinned memcpy pass outweighed the reduction in API calls and queue depth.

## Baseline

The accepted runtime baseline remains:

- frozen static vertical expert map;
- 3072 MiB cache reserve;
- 5.9630 mean decode tok/s over the established three-run set;
- 152 decode graph splits at batch size 1;
- 1364 MiB minimum free VRAM;
- deterministic output SHA-256 `e165b05c0568da0b099159b3045190e4f33ffbd97698d0a9881b982ead61f514`.

## Implementation tested

For each frozen layer/component, the experiment gathered the arbitrary canonical expert slices selected by the static map into one contiguous pinned staging allocation and issued one contiguous H2D transfer into the compact destination slot tensor.

The implementation was gated by:

```sh
GGML_MOE_DYNAMIC_STATIC_BULK_UPLOAD=1
```

Setting it to `0` selected the legacy worker path in the same binary.

## Correctness validation

Build:

```sh
cmake --build build-v100 --target llama-completion test-moe-split-backends -j 16
```

Result: exit code 0.

Focused split-backend test:

```sh
build-v100/bin/test-moe-split-backends
```

Result: exit code 0.

A verified exact GLM run used:

```sh
GGML_MOE_DYNAMIC_VERIFY_PROMOTIONS_PER_SHAPE=1
```

The worker checked nine representative packed destination shapes with zero failures. The run also produced the exact accepted output hash, 152 decode splits, and 1364 MiB minimum free VRAM.

## Crossed A/B results

| Mode | Run | Wall time | Decode tok/s | Physical transfers | Staging memcpy | Worker sync |
|---|---:|---:|---:|---:|---:|---:|
| Packed | 1 | 33.5979 s | 6.0001 | 225 | 1786.754 ms | 1170.606 ms |
| Legacy | 1 | 33.1685 s | 6.0130 | 4011 | 1100.557 ms | 34.150 ms |
| Legacy | 2 | 32.8923 s | 5.9676 | 4011 | 1111.955 ms | 34.165 ms |
| Packed | 2 | 33.5948 s | 5.9268 | 225 | 1785.924 ms | 1170.068 ms |

Means:

| Mode | Mean wall time | Mean decode tok/s | Mean reported load time |
|---|---:|---:|---:|
| Packed | 33.5964 s | 5.9634 | 17227.64 ms |
| Legacy | 33.0304 s | 5.9903 | 17239.53 ms |

Packed mode was 0.5660 seconds, or 1.71%, slower in end-to-end wall time. The small decode-rate difference is ordinary run variation; initialization changes should not affect steady-state decode.

## Why it lost

The legacy path stages only transfers that fit its 4 MiB pinned slots. It staged 8293.69 MiB and allowed the remainder to use the backend's direct source path. Its reported host-issue time already includes much of the transfer-side blocking, leaving only about 34 ms in final synchronization.

The packed path staged the entire 14540.62 MiB cache image. That added approximately 680 MiB-equivalent milliseconds of host memcpy work and then exposed roughly 1170 ms of H2D completion time in synchronization. Fewer API calls did not reduce total bytes moved or the memory-bandwidth cost.

Queue-delay totals also fell sharply, but that metric sums delay over logical jobs and therefore overstates user-visible benefit. The end-to-end wall clock is the correct gate.

## Next priority

Do not optimize transfer count in isolation. The next investigation should target steady-state decode's CPU cold branch and synchronization boundaries:

1. quantify per-token CPU cold-branch time and GPU overlap from existing dual-path traces;
2. identify layers where the CPU branch extends beyond the GPU hot branch;
3. reduce CPU work or protect additional experts only in those critical layers;
4. retain the 3072 MiB reserve, exact output parity, and 152-split topology gates.

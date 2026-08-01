# Selective-copy cache pivot — Qwen/V100

**Status date:** 1 August 2026
**Hardware:** Tesla V100-SXM2-32GB
**Model:** Qwen3.5-122B-A10B UD-Q3_K_XL
**Goal:** Preserve the ordinary upstream graph and cache repeated expert-slice transfers instead of splitting every MoE layer into CPU and GPU branches.

## Result

The pivot is **not applicable to the current single-GPU Qwen CUDA path**.

Qwen overflow expert tensors are normally stored in `CUDA_Host`. CUDA executes `MUL_MAT_ID` directly against that host-mapped storage. The scheduler therefore performs no per-token expert-slice staging copies that a persistent copy cache could skip.

When `--no-host` is used, the expert tensors become ordinary `CPU` buffers, but the scheduler places the corresponding MoE operation on CPU instead of staging selected slices into a CUDA tensor. This also produces zero selective-copy events.

The required middle path does not exist on this configuration:

```text
CPU expert tensor
    -> selected expert slices copied to persistent CUDA scratch
    -> normal CUDA MUL_MAT_ID
```

Current behavior is instead one of:

```text
CUDA_Host expert tensor -> CUDA kernel reads host-mapped weights directly
```

or:

```text
CPU expert tensor -> MoE operation executes on CPU
```

## Experiments

### 1. Equal-VRAM auto-fit sweep

Script:

```text
run_qwen_equal_vram_sweep.sh
```

Cache arenas of 256, 512 and 1,024 MiB were paired with controls using the same fit target and identical placement.

| Nominal cache | Control | Cache arm | Difference | Selective inputs | Hits/misses |
|---:|---:|---:|---:|---:|---:|
| 256 MiB | 33.1788 tok/s | 32.7341 tok/s | -1.34% | 0 | 0 / 0 |
| 512 MiB | 32.5046 tok/s | 32.7350 tok/s | +0.71% | 0 | 0 / 0 |
| 1,024 MiB | 32.5298 tok/s | 32.2351 tok/s | -0.91% | 0 | 0 / 0 |

These are no-op controls, not cache results. All output hashes and placements matched. The differences are run noise plus the effect of allocating unused cache buffers.

### 2. Explicit CPU expert override with mmap

Command shape:

```text
-ngl 99 \
-ot '\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU'
```

The loader converted the override to `CUDA_Host`:

```text
tensor blk.0.ffn_down_exps.weight ... buffer type overridden to CUDA_Host
```

Result over 64 forced tokens:

| Arm | Throughput | Selective inputs | Attach attempts |
|---|---:|---:|---:|
| Control | 24.4472 tok/s | — | — |
| 1 GiB cache | 24.3113 tok/s | 0 | 0 |

### 3. Explicit CPU expert override with `--no-mmap`

The loader still selected `CUDA_Host`. Result over 32 forced tokens:

| Arm | Throughput | Selective inputs | Attach attempts |
|---|---:|---:|---:|
| Control | 24.3943 tok/s | — | — |
| 1 GiB cache | 24.2312 tok/s | 0 | 0 |

### 4. Explicit CPU expert override with `--no-host`

The loader retained ordinary CPU buffers:

```text
tensor blk.0.ffn_down_exps.weight ... buffer type overridden to CPU
```

But the MoE operation followed the CPU tensor to the CPU backend. No scheduler staging copies occurred.

Result over 32 forced tokens:

| Arm | Throughput | Selective inputs | Attach attempts |
|---|---:|---:|---:|
| Control | 24.4876 tok/s | — | — |
| 1 GiB cache | 24.6261 tok/s | 0 | 0 |

Again, this is a no-op comparison.

## Degradation bug analysis

The research found several distinct issues. They should not be conflated.

### Bug 1: disabled dynamic scheduler polluted the ordinary path — fixed

The patched scheduler previously scanned graph names, nodes and dynamic state even when no expert cache was attached.

Measured Qwen regression before the fix:

```text
clean upstream:       about 33.31 tok/s
patched-disabled:     about 24.91 tok/s
regression:           -25.22%
```

An early upstream-equivalent scheduler path restored Qwen to within 0.10% of clean upstream and warm GLM to roughly 0.8%.

### Bug 2: predictive suffix was not actually protected — fixed

Warm start filled all regular slots, including the nominal predictive suffix. Oracle admission then evicted useful warm residents.

A new opt-in rule reserves the suffix as empty:

```text
GGML_MOE_DYNAMIC_WARM_START_EXCLUDE_FLEX=1
```

With a frozen protected core, exact completion packets improved throughput by 8.19% relative to the matching inactive layout. This proved the protection bug was real.

### Bug 3: cache allocation occurred without decode usefulness — fixed experimentally

The full-layout expert cache allocated hundreds of MiB during prompt processing even when measured decode later reported:

```text
selective-inputs=0
hits=0
misses=0
```

The profile reset intentionally cleared counters at the benchmark boundary but left prompt-time allocations resident. This wasted VRAM without providing decode reuse.

An opt-in decode-only allocation guard was added:

```text
GGML_EXPERT_CACHE_DECODE_ONLY=1
```

It prevents cache attachment during batched prompt operations and allocates only after observing an eligible single-token selective-copy input.

Validation with a nominal 1,024 MiB cache:

```text
entries=0
allocated=0.00 MiB
selective-inputs=0
throughput=33.3351 tok/s
```

Before the guard, the same inapplicable cache configuration could allocate up to 996 MiB despite zero measured decode hits or misses.

### Architectural limitation: fixed CPU fallback remains in the graph

The fixed dynamic topology contains a CPU snapshot prefix and CPU cold-compute suffix. The route residency decision becomes available only after the snapshot executes. Consequently, `GGML_MOE_DYNAMIC_SKIP_COLD_BRANCH=1` cannot cheaply remove the CPU segment before scheduling it.

Two bypass prototypes were tested:

| Prototype | Result |
|---|---:|
| Synchronize snapshot then inspect actual route | -3.31% |
| Oracle-prevalidated asynchronous prefix | -2.28% |

Both changed transfer/route timing and reduced useful completion coverage. They remain disabled.

This is the main reason expert-route coverage did not translate into performance. It requires a different execution architecture, not another predictor.

## Correct interpretation of the active oracle

A repeated perfect-route oracle increased fully GPU-covered layers from 5,599.7 to 8,896.3 per 512-token run and removed 6.45 CPU-dependent layers/token.

It still tied the reactive control:

```text
reactive mean:         19.0066 tok/s
perfect-oracle mean:   18.8763 tok/s
mean difference:       -0.69%
median difference:     +0.14%
```

The oracle added about 44.71 MiB/token and 4.48 ms/token of staging-copy time while removing only about 2.11 ms/token of measured median CPU cold work.

A better predictor cannot solve that cost equation because the oracle already had perfect future routes.

## Recommended next architecture

The viable continuation is a **kernel-level mixed-residency cache**, not a scheduler selective-copy cache.

The ordinary CUDA `MUL_MAT_ID` graph should remain intact. The CUDA implementation would need to resolve each selected expert from either:

1. A compact GPU-resident expert cache, or
2. The existing `CUDA_Host` source tensor.

Conceptually:

```text
router IDs
    -> GPU residency lookup
    -> hot experts: device pointer/cache slot
    -> cold experts: CUDA_Host pointer
    -> one CUDA MoE operation
```

This avoids:

- CPU/GPU graph splitting
- CPU fallback graph construction
- Partial-output merging
- Scheduler-level expert transfers for every layer

The first experiment should be an exact-route, fixed-residency CUDA kernel prototype on one Qwen layer. Continue only if it beats direct `CUDA_Host` access for that layer after including lookup and cache-fill cost.

## Directory map

```text
profiling/selective-copy-cache-2026-08-01/
├── README.md
├── run_qwen_equal_vram_sweep.sh
├── qwen-equal-vram-sweep/
│   └── results.json
├── qwen-cpu-buffer-smoke/
├── qwen-cpu-buffer-nommap-smoke/
└── qwen-no-host-smoke/
```

Large stdout/stderr files are retained locally for audit but need not be committed when summary files are sufficient.

## Validation

- `llama-completion` build: exit 0
- `test-moe-online-mlp`: exit 0
- `git diff --check`: exit 0
- Forced-token hashes matched in every A/B pair
- Placement matched in the equal-VRAM sweep

Matching forced tokens does not prove identical logits or hidden states; it proves only that the same forced sequence was consumed.

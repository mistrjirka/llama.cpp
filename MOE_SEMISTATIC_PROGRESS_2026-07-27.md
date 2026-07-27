# Large-MoE semi-static cache progress — 2026-07-27

## Repository state

- Workspace: `/workspace/llama-mainline-cache`
- Branch: `expert-cache-mainline`
- Base: `555881ebc` (`upstream/master` at the original checkpoint)
- Current HEAD at document creation: `bd3d5ab10`
- Persistent sandbox session: `s_e45422ab9bfbbdd1`
- Model/workload: GLM-5.2 UD-IQ2_XXS, exact forced-token decode benchmark on CUDA0/V100

The current tracked source tree contains the accepted large-MoE cache runtime checkpoint. Experimental source changes from unsuccessful follow-up tests were reverted. At document creation, the only tracked uncommitted change was benchmark-runner plumbing that makes CPU and batch thread counts configurable.

Large raw traces, `.out`, `.err`, GPU telemetry, and profiler artifacts remain untracked intentionally. Compact summaries and analyses are committed when an experiment reaches a decision.

## Objective

Beat default llama.cpp horizontal stacking without sacrificing exact output, VRAM safety, or graph topology.

Required gates:

1. exact deterministic output under the forced-token workload;
2. 152 decode graph splits at batch size 1, or no material topology regression;
3. no CUDA OOM and at least 512 MiB measured free VRAM;
4. repeatable throughput improvement, not a single-run fluctuation;
5. no adaptive publication of incomplete expert bundles.

## Accepted performance baseline

The accepted configuration is:

- frozen static vertical expert map;
- automatic equal-byte layer planner;
- 3072 MiB expert-cache reserve;
- approximately 18 resident experts in most routed layers;
- 1337 ready slots across 75 routed layers;
- 14540.62 MiB cache allocation;
- exact CPU cold fallback retained;
- no per-token cache mutation.

Established three-run result:

| Metric | Result |
|---|---:|
| Mean decode throughput | **5.9630 tok/s** |
| Individual runs | 6.0254, 5.9994, 5.8641 tok/s |
| Decode graph splits | **152** |
| Minimum free VRAM | **1364 MiB** |
| Output SHA-256 | `e165b05c0568da0b099159b3045190e4f33ffbd97698d0a9881b982ead61f514` |

The recorded default-horizontal mean is approximately 4.9233 tok/s. The accepted static vertical baseline is therefore approximately **21.1% faster** than the recorded horizontal configuration.

The 2048 MiB reserve was rejected: it fitted more experts but fell to 5.7766 tok/s and left only 482 MiB free VRAM. Maximum residency is not the optimum because a larger GPU hot branch also performs more work.

Detailed result:

`profiling/glm52-reserve-sweep-2026-07-27/ANALYSIS.md`

## Completed commits

| Commit | Purpose | Decision |
|---|---|---|
| `1a012a088` | Checkpoint the dynamic/static MoE cache runtime and compact evidence | Accepted checkpoint |
| `a4700633c` | Tune the frozen static vertical reserve to 3072 MiB | Accepted performance improvement |
| `858675fa8` | Record one-remap-per-layer experiment | Rejected; no demonstrated gain |
| `3295eefa5` | Record frozen static bulk-upload experiment | Rejected; slower wall time |
| `bd3d5ab10` | Record per-layer slot-count plan experiment | Rejected; neutral performance |

## Follow-up experiment decisions

### 1. CUDA route-remap reuse

Hypothesis: gate, up, and down projections redundantly launch the same canonical-expert-to-slot remap kernel. Reusing one mapped-ID buffer could remove approximately 150 tiny launches per decode token.

Validation:

- build passed;
- `test-moe-split-backends` passed;
- exact output hash preserved;
- 152 splits preserved;
- 1364 MiB minimum free VRAM preserved.

Three-run mean: 5.9963 tok/s versus the 5.9630 baseline, a nominal +0.56% difference smaller than run-to-run noise.

Decision: **reject and revert**. The kernels are probably hidden behind larger GPU/CPU work, and the shared mutable buffer would add an execution-order assumption without demonstrated value.

Details:

`profiling/glm52-remap-reuse-2026-07-27/PROGRESS.md`

### 2. Packed frozen-static bulk upload

Hypothesis: replace 4011 small promotion-worker transfers with one contiguous transfer per cached layer/component, reducing the physical H2D transfer count to about 225.

Result:

- transfer count fell from 4011 to 225;
- exact output, 152 splits, and VRAM usage were preserved;
- packed mode mean wall time: 33.5964 seconds;
- legacy mode mean wall time: 33.0304 seconds;
- packed mode was 0.5660 seconds, or 1.71%, slower.

The packed path copied the complete 14540.62 MiB cache image into pinned staging memory. The extra pageable-to-pinned pass outweighed the reduction in API calls.

Decision: **reject and revert**. Transfer count alone is not the correct optimization objective.

Details:

`profiling/glm52-static-bulk-ab-2026-07-27/ANALYSIS.md`

### 3. Per-layer slot-count allocation

Hypothesis: redistribute nearly the same cache bytes toward layers with greater marginal route frequency per MiB.

Result over five crossed repetitions:

| Mode | Mean tok/s | Median tok/s | Trimmed mean tok/s |
|---|---:|---:|---:|
| Equal-byte baseline | 5.9592 | 5.9872 | 5.9806 |
| Per-layer override | 5.9407 | 6.0219 | 5.9896 |

All ten runs preserved exact output, 152 splits, allocation safety, and no OOM. Robust per-token latency was effectively identical.

Decision: **reject and revert**. Route coverage is not a sufficient planner objective because it does not identify CPU work on the token critical path.

Details:

`profiling/glm52-slot-plan-2026-07-27/ANALYSIS.md`

## CPU-thread tuning in progress

The benchmark runner currently has an uncommitted change that replaces fixed `-t 40 -tb 48` arguments with configurable `THREADS` and `BATCH_THREADS` environment variables and records them in each compact summary.

Completed first-pass single runs:

| Decode threads | Decode tok/s | Wall time | Output/topology |
|---:|---:|---:|---|
| 20 | 5.6925 | 33.3466 s | exact hash, 152 splits |
| 24 | 5.8786 | 32.9357 s | exact hash, 152 splits |
| 32 | 5.4375 | 34.0636 s | exact hash, 152 splits |
| 36 | 5.8283 | 32.9992 s | exact hash, 152 splits |
| 40 | 6.0858 | 32.9143 s | exact hash, 152 splits |
| 42 | 6.0680 | 32.7994 s | exact hash, 152 splits |
| 44 | **6.1099** | **32.8690 s** | exact hash, 152 splits |
| 48 | 3.9066 | 42.0789 s | exact hash, 152 splits |

These are preliminary single-run observations. The 44-thread result is the best first pass, but it is not yet accepted because the spread seen in prior experiments is large enough to require crossed repetitions. Forty-eight threads clearly oversubscribe or otherwise disrupt the CPU cold path.

At document creation, persistent job `j_47cb25b0a419e95e` was continuing the sweep with:

1. 42 threads, repetition 1 — completed at 6.0680 tok/s;
2. 46 threads, repetition 1;
3. 44 threads, repetition 2;
4. 40 threads, repetition 2.

Thread-sweep artifacts:

`profiling/glm52-thread-sweep-2026-07-27/`

## Current technical conclusion

The vertical placement advantage is real and repeatable. The remaining performance gap is not dominated by:

- GPU route-map remap launch count;
- promotion-worker API transfer count;
- global route coverage or an offline frequency-per-MiB slot objective.

The exact static path always evaluates a GPU hot branch and an exact CPU cold branch. Additional resident experts help only when they remove CPU work that extends token completion latency. Work already hidden behind the GPU branch does not improve throughput when removed.

The next planner therefore needs a **critical-path latency objective**, not a route-hit objective.

## Recommended next steps

### Immediate

1. Finish crossed CPU-thread repetitions around 40–46 threads.
2. Compare means, medians, trimmed means, and pooled per-token latency.
3. Commit the runner parameterization and compact summaries only if a stable thread setting is demonstrated.
4. Otherwise revert the runner change and retain a negative-result analysis.

### Higher-impact engineering

1. Add low-overhead sampled timing for the exact CPU cold branch by layer.
2. Measure when each layer's CPU branch finishes relative to its GPU hot branch and final merge.
3. Count distinct uncovered experts and estimate cold arithmetic per layer/token.
4. Rank a cache slot by observed reduction in token completion latency, not by route frequency.
5. Protect additional experts only in layers repeatedly shown to extend the critical path.
6. Preserve the frozen topology while evaluating any new planner.

### Adaptive work remains deferred

Do not return to per-token mutable admission yet. Earlier mutable runs reached approximately 302 graph splits and about 4.24 tok/s despite higher resident-route coverage. Any future adaptive design should use:

- a large protected static core;
- a very small epoch-updated tail;
- complete expert bundles only;
- versioned A/B route-map publication;
- no graph rebuild or publication every token.

## Resume checklist

1. Inspect `git status -sb` and do not add raw profiling output accidentally.
2. Check persistent job `j_47cb25b0a419e95e` and collect its remaining summaries.
3. Analyze the thread sweep before modifying runtime source.
4. Rebuild `llama-completion` and `test-moe-split-backends` after any source change.
5. Run `build-v100/bin/test-moe-split-backends`.
6. Gate every GLM run on exact SHA-256, 152 splits, no OOM, and at least 512 MiB free VRAM.
7. Use `git diff --check` and inspect the explicit worktree diff before committing.

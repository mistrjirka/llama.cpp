# Verification-group pipelining — 7 September 2026

Research branch: `perf/parallel-group-pipeline-0907`, based on `221ddc9f4`. No change in this report is enabled in production. The experiments hold Ornith AD-Q6_K-Q5_K target weights, Shisa Q5_0 MTP3, Q8 target/draft KV and the CPU projector constant. Actual histories are four 100k C++-source histories sharing a 98k prefix, not four fully populated 400k contexts. The server reserves the same 1.4M physical KV pool and advertises four logical 400k contexts.

## Implementation, not a global small-batch approximation

The splitter accepts an optional maximum number of independent sequence sets per microbatch. The hybrid-memory path enables it only for small, all-output token batches with recurrent rollback state. It retains the existing recurrent-tail rule, so a speculative verification tail cannot be split across microbatches. Large prefills retain their ordinary batch size. Coupled sequence inputs and embedding inputs are excluded. This uses the existing asynchronous multi-GPU backend scheduler; it is not a new scheduler or arbitrary overlap of dependent steps.

`LLAMA_EXPERIMENT_VERIFY_GROUP_SEQS=1` or `2` enables grouping. Unset/zero/invalid values preserve the original splitting behavior. `--pipeline-copies` independently selects backend input-buffer duplication. These two dimensions are tested separately, not conflated.

Optional diagnostics expose the *effective* scheduler copies and the emitted verification group shapes. `LLAMA_EXPERIMENT_VERIFY_GROUP_TRACE` and `LLAMA_EXPERIMENT_VERIFY_SCHED_DIAGNOSTIC` are used only for diagnostic runs, not timed benchmarks.

## An important invalid comparison caught before interpretation

At the original 256-token target ubatch, requesting two pipeline copies causes a target scratch allocation failure (about 4.1 GiB on the constrained device). The server then starts successfully **after disabling pipeline parallelism**. The initial fit/smoke tests are retained, but they cannot establish the effect of active two-copy pipelining.

The controlled pipeline experiment therefore uses target ubatch **128 in every arm**. The physical KV pool and quantization do not shrink. Diagnostic logs confirm target `requested_copies=2 effective_copies=2 pipeline=1 ubatch=128`. Draft/vision contexts can correctly report one effective copy because they are not the two-GPU target context. Startup alone is not treated as evidence that a requested feature actually ran.

## Correctness and isolation

The batch allocator test suite passes 31 tests and 26,780 assertions, including 768 combinations of unequal 1–4-token verification lengths, one/two/four sequence-set limits and nonmonotonic sequence arrival. Every token appears once, every position/output flag is preserved, every complete verification sequence remains within one microbatch, and resets work.

A separate C++ test loads the same target states and evaluates six identical 16-token verification batches: four teacher-forced tokens for each of four agents per step. It compares **all 248,320 logits for each of 96 output rows**, not only generated strings or a truncated top-k distribution. The full-distribution comparison reports normalized squared error, maximum logit/probability differences, Jensen–Shannon divergence and greedy disagreements.

At ubatch128:

- Original grouping with one versus two pipeline copies: all logits identical.
- Two-agent grouping with one versus two copies: all logits identical.
- Single-agent grouping with one versus two copies: all logits identical.
- Repeated original control: all logits identical.
- Changing grouping relative to the original four-agent batch: four of 96 greedy selections differ for two-agent groups, one for single-agent groups. Maximum probability changes are 0.1395 and 0.1126 respectively.

Thus the double-buffering mechanism passes this focused exact-equivalence test, whereas changing the arithmetic batch shape still fails a strict numerical-equivalence gate. These differences are not attributed to a race or harmless rounding without evidence. Grouped serving remains experimental. This test is target-only and does not replace whole-server MTP, cache-lifetime or general quality evaluations.

## Nsight proof that the intended mechanism executes

The captured TG intervals exclude model load and cache restoration. The same real-code fixture is used. Recorded GPU kernel/copy interval unions give:

| Configuration | Concurrent activity on both GPUs |
|---|---:|
| Original grouping, one copy | 0.29% |
| Two-agent groups, one copy | 9.18% |
| Two-agent groups, two copies | 10.30% |
| Single-agent groups, two copies | 19.15% |

This is concurrent recorded device activity, **not SM utilization or a guaranteed speedup**. Most of the overlap change comes from the grouping, not from adding another buffer copy alone.

The smallest groups also perform more work. Summed Q8-to-FP16 conversion time in this capture rises from about 1.23 s for two-agent/two-copy grouping to 2.16 s for single-agent/two-copy grouping. More overlap does not imply greater end-to-end efficiency. Nsight changes execution overhead; unprofiled measurements are reported separately.

The raw trace remains in the persistent sandbox; `traces.json` gives its path, size and SHA256. `profile-summary.json` records effective scheduler configuration, actual group shapes, kernel-family totals and overlap. A full timeline-derived summary is retained locally for drill-down.

## Unprofiled mirrored comparison

All arms use target ubatch128, the same 1.4M Q8 pool, four restored 100k real-code histories and MTP3. Each agent appends its task/template (38/42/44/41 tokens) and generates 128 tokens. The ten process lifetimes run A/B/D/C/E/E/C/D/B/A, with one excluded warmup and two retained turns per lifetime: four measured turns per arm. These are within-fork comparisons, not an upstream comparison. State restoration is excluded from whole-turn timing. No profiler or compilation runs alongside these timings.

| Arm | Complete four-request turn | Observed range | Mean TG per request | Wall-time change vs A |
|---|---:|---:|---:|---:|
| Original grouping, one copy | 6.523 s | 6.444–6.594 s | 24.30 tok/s | +0.00% |
| Two-agent groups, one copy | 6.422 s | 6.254–6.497 s | 24.01 tok/s | -1.54% |
| Original grouping, two copies | 6.505 s | 6.366–6.572 s | 24.31 tok/s | -0.28% |
| Two-agent groups, two copies | 6.360 s | 6.236–6.499 s | 24.13 tok/s | -2.49% |
| Single-agent groups, two copies | 7.539 s | 7.467–7.603 s | 19.61 tok/s | +15.57% |

The two-agent/two-copy arm averages about 2.5% lower whole-turn latency than the control, but observed ranges overlap and generation speed per request is essentially unchanged. The incremental effect of adding the second buffer to two-agent grouping is only about 1% in this screen. No robust general speedup is established. Single-agent groups are clearly worse here despite substantially greater GPU overlap.

Do not count the profiled 0.3% → 10.3% overlap change as a 34× throughput improvement. The unprofiled timing table above is the application outcome. Changing grouping also changes output probabilities, so the grouped arms remain research-only. Double buffering alone matches all tested logits but yields no useful demonstrated speedup here.

## Reproduction and acceptance boundaries

`metadata.json` records the base revision, source patch and immutable library hashes. Large GGUFs, serialized KV, token fixtures, raw full logits and profiler binaries are intentionally not committed. The standalone comparison source and test driver are included.

All GPU workloads use the Development Sandbox `gpu:all` lock. Do not run Nsight, compilation or another model during timing. Every timing arm must use the same ubatch and physical KV capacity, and two-copy arms must reject an allocation fallback. Do not combine ubatch256 fallback measurements with ubatch128 active-pipeline measurements.

The acceptance target is not merely a faster synthetic string. It includes fixed-input distribution checks, preserved speculative state, whole-turn latency, generation throughput, stable behavior with changing request counts, and memory margins. No claim of all-model coverage, long-running agent quality, near-350k performance or production readiness is made here.

### Build the validation programs

From this worktree, after the CUDA build completes:

```sh
cmake --build build-sm70-75 --target llama-server test-batch-alloc -j 12
build-sm70-75/bin/test-batch-alloc
B="$(pwd)/build-sm70-75/bin"
R=/workspace/oai-qwen38-pp-lab/results/group-pipeline-0907
c++ -O2 -std=c++17 -Icommon -Iinclude -Iggml/include -Isrc -Ivendor \
  benches/group-pipeline-0907/teacher-groups.cpp \
  "$B/libllama-common.so" "$B/libllama.so" "$B/libggml.so" \
  -Wl,-rpath,"$B" -pthread -ldl -o "$R/teacher-groups"
c++ -O2 -std=c++17 benches/group-pipeline-0907/compare-logits.cpp \
  -o "$R/compare-logits"
```

`run128.py` pins the immutable `validated128-bin` snapshot used in the results. To test a new build, copy it to a separately named immutable directory and adjust that path deliberately; do not silently replace a recorded binary snapshot. The teacher program additionally uses `GROUP_FIXTURE`, `GROUP_SNAPSHOTS` and `GROUP_LOGITS`; the included driver sets them. Full float-output files are validation artifacts and are not required for serving.

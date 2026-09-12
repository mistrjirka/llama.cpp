# SSM convolution snapshot correctness

Date: 2026-09-12. Fix: `580168936`, pushed to `origin/v100-optimized`. Baseline: `a70ee26ae` (the already-validated INT8 Stream-K crash repair).

## Cause

`ssm_conv_try_split_nc4` recognized a CONCAT input and replaced reads from the materialized convolution input with reads from `concat->src[0]` and `src[1]`. Those ancestor buffers are not inputs of the SSM_CONV operation. They can be updated or reused after CONCAT has completed. Reading them later changes the operation's meaning and can feed overwritten data into the recurrent model.

This was introduced in `9662bccd4` on September 9, 2026. The fault affects the single-sequence, kernel-size-four layout, on both tested GPUs. It is not specific to PXQ weights, despite originating in that optimization experiment.

The fix removes that shortcut and reads the materialized input. Existing CUDA SSM_CONV, bias/SiLU fusion, FlashAttention, INT8, GQA8, Stream-K, GatedDeltaNet, MMQ tuning, context sizes, and production launch settings are retained. The convolution implementation is back to upstream `43f3dda62` behavior, with a comment explaining the lifetime constraint.

## Regression test

[`test-ssm-conv-snapshot.cpp`](../../../tests/test-ssm-conv-snapshot.cpp) first computes CONCAT, overwrites the original state/token tensors, then evaluates SSM_CONV against the unchanged materialized snapshot. Expected values come from the original inputs. Shapes cover one/two sequences; 1, 2, 31, 32, 33 and 128 tokens; plain convolution, SiLU and bias-plus-SiLU graphs.

- Before: 72/108 cases pass. All 18 single-sequence cases on each GPU fail;
  the CPU and two-sequence controls pass. Errors reach about 30, not rounding noise.
- Fixed: 108/108 cases pass on V100, RTX 2080 Ti and CPU.
- Compute Sanitizer memcheck: 108/108, zero errors.

Evidence: [before](results/snapshot-before.log), [fixed](results/snapshot-fixed.log), [memcheck](results/snapshot-memcheck.log). The initial test-harness draft omitted COMPUTE flags on manually added graph nodes and was discarded; these records use the corrected harness, including a passing CPU reference.

To run the registered test after building:

```sh
cmake --build build --target test-ssm-conv-snapshot
ctest --test-dir build --output-on-failure -R '^test-ssm-conv-snapshot$'
```

## Model-output checks

Three identical greedy 64-token Ornith requests on the pre-fix RTX build produced three different sequences, even with CUDA graph replay disabled. A separate fusion-disabled attempt ended in a content-parser HTTP 500 and is not counted as a completed comparison.

Removing only the convolution shortcut produced identical 64-token sequences across three restored-prefix requests and three fresh 1k-prefix requests. All six matched the actual upstream server's output token-for-token. These probes also checked that the server applied temperature zero. They used the same model, explicit batch sizes, token IDs and saved states. Probability-reporting overhead in those probes is excluded from the throughput comparisons below.

## Matched throughput

Before/fixed/fixed/before process order. Each process has one warm-up followed by three retained 1,000-token append measurements and one 64-token decode. Only `ssm-conv.cu.o` differs between each paired CUDA library; the three fixed INT8 Stream-K objects and all other inputs are identical immutable snapshots.

| Workload | Before PP | Fixed PP | PP change | Before TG | Fixed TG | TG change |
|---|---:|---:|---:|---:|---:|---:|
| Ornith RTX, 65,536 cached | 1485.17 | 1582.86 | +6.58% | 59.31 | 59.29 | -0.04% |
| Ornith dual, 100k cached | 1087.15 | 1136.08 | +4.50% | 53.55 | 53.64 | +0.17% |
| Ornith V100, 100k cached | 926.24 | 975.28 | +5.29% | 56.86 | 57.01 | +0.26% |
| Qwen RTX, 65,536 cached | 500.41 | 497.46 | -0.59% | 17.37 | 17.34 | -0.21% |
| Qwen V100, 100k cached | 453.62 | 453.51 | -0.02% | 16.39 | 16.38 | -0.06% |

Rates are tokens/second. RTX/dual Ornith use matched FORCE_MMQ builds; V100 Ornith and Qwen use normal builds with the same Volta MoE-MMQ environment. Dual Ornith uses the established 14:35 RTX:V100 layer split, not the README's tensor-split headline. Every completed process exited cleanly and retained its original binary hash.

Qwen's observed changes are small compared with sample scatter (RTX PP standard deviation: 6.31 before and 3.34 fixed); this is not a claim that every workload has exactly zero performance cost. Fixed greedy output differs from the broken baseline, so decode timing can include different expert routes. These are end-to-end workload measurements, not isolated convolution speedups.

All fixed arms produced a single repeatable 64-token output hash per profile. Before-fix Ornith hashes varied. Qwen hashes were repeatable before and after, but changed after the repair; stable hashes alone were therefore not enough to prove correctness of the old computation.

[Raw performance records and hashes](results/summary.json) include all five completed comparisons and links/paths to their retained evidence. These checks isolate the correctness repair and do not replace historical upstream-versus-fork README graph rows.

## Additional Qwen output check

The repaired V100 Qwen build matched actual upstream for all 64 generated tokens in both repeated runs. The RTX Qwen build matched the first 63 tokens, then produced token 22 (`7`) rather than upstream token 23 (`8`) at position 63. Both results were repeatable. Disabling `GGML_CUDA_TURING_INT8_QK` left the repaired RTX output unchanged, so this control does not attribute the difference to INT8 QK. The measured probability gap is substantial; it is not characterized as a proven insignificant tie. The remaining cross-engine numerical discrepancy needs separate isolation. This SSM repair is justified by its direct snapshot reference test; it is not a claim of universal output identity with upstream.

See `qwen_upstream_identity` and the retained RTX INT8-off probe in [the summary](results/summary.json). The probe files retain raw token IDs, sampling settings, timing, and source-file hashes.

## State and integrated checks

Both normal and FORCE_MMQ main builds completed. The registered snapshot CTest passed, and all ten long-Q8 FlashAttention cases passed again on each GPU (20/20) with the final main build.

Fresh 100k prefill completed with MTP off (981.87 PP tok/s) and Q4 draft MTP (975.78 PP tok/s). Those are successful cold-prefill checks, not matched before/ after cold-throughput comparisons. The six exact 65,024-prefix boundary appends also passed.

All 30 four-slot erase/restore/generate rounds completed: 120 requests, each restoring 100k tokens and generating 128 tokens. Mean aggregate throughput was 103.39 tok/s for Q4 draft MTP/head reuse, 90.96 for MTP off, and 104.87 for Q8 draft MTP without head reuse. These three arms are stability checks, not an ABBA comparison against the pre-fix program.

Concurrent output invariance remains unproven: unique 128-token hashes per slot were [7,6,7,7] with Q4 MTP, [6,5,6,6] without MTP, and [7,5,7,6] with Q8 MTP. Successful completion is not treated as a determinism pass. The follow-up controls completed four rounds each (64 tokens/request):

| Control | Unique hashes per slot |
|---|---|
| Fixed build, requests executed serially | [1,1,1,1] |
| Fixed build, concurrent, INT8 QK disabled | [4,4,3,1] |
| Actual upstream, concurrent, MTP off | [3,3,3,1] |

Thus the repaired serial path repeats, while concurrency-related variation is also observable without our INT8 path and in native upstream. This does not prove that every source of concurrent variation is identical; it does rule out calling all differing concurrent hashes a new regression unique to this fork. The upstream control omits fork-only prefix deduplication, per-slot-limit and pipeline-copy flags; it restores the same four 100k target files into a 1.4M physical pool. It is an output-repeatability control, not a performance comparison.

Evidence: [serial/INT8-off controls](results/batch-controls.json), [native upstream control](results/upstream-batch.json).

See [the validation summary](results/summary.json) for retained configurations, cache-count checks and binary hashes. Only completed records are included; an absent test is not a pass.

## Operational scope

Saved states created through the faulty path are not retroactively corrected. Regenerate affected target/draft snapshots for trusted use rather than treating a successful file restore as proof that their numerical contents are valid. Existing files were not deleted or overwritten during these tests.

The actual full-model checks cover specific fixtures and execution layouts; they do not establish universal batch-size-independent bitwise determinism. Dense Qwen MTP target-head-reuse coverage and shrinking the draft's approximately 2.9 GiB compute arena remain separate tasks. No production service was restarted, and no host launcher or model quantization setting was changed.

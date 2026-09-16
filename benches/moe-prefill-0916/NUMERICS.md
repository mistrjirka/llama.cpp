# Numerical correctness deep check

September 16, 2026

## Conclusion

The tests strongly support reduced-precision attention arithmetic, amplified by later model selection decisions, as the source of the native-sparse versus control differences in the tested requests. They found no incorrect selected-index list, cached-prefix corruption, cross-query contamination, or CUDA memory error in the exercised paths.

However, the review also found a concrete inherited precision-policy mismatch: the model requests FP32 accumulation, while the active CUDA attention implementations retain the running value-weighted sum in FP16. This is present in the pre-experiment source as well as the candidate. Therefore this report does not certify that everything is merely harmless rounding or that the implementation fully honors its documented precision requirement.

No production source, frozen runtime, model, launch default, commit, or branch was changed by this audit. Native sparse attention remains opt-in.

## Scope and provenance

Hardware: V100 SXM2 32 GB plus RTX 2080 Ti with 22 GB. Model: Qwen3.8-Flash-Next UD-IQ4_XS. KV cache: Q8_0. Main model-level cases restore the same existing 100,000-token prefix, then append 512 English-prose tokens or 1,000 C++ tokens. The prefix remains the repetitive fixture; this is not diverse-prefix quality certification.

A matching all-layer diagnostic campaign was already present in the shared Sandbox. This continuation inspected its implementation and results, independently rechecked its saved evidence, and added new GPU relocation and precision-setting probes rather than rerunning the same campaign. The all-layer campaign is in `/workspace/moe-layer-first-0914/numerics-deep-0916/`. The independent additions, source evidence, consolidation script and this report are in `/workspace/moe-layer-first-0914/numerics-relocation-audit-0916/`.

Frozen candidate runtime identities:

- `libllama.so.0`: `c0c184bf041e198e180e7b8b63cc19527a59644458e6b3cec8c7d5250a1c6e6b`
- `libggml-cuda.so.0`: `daa9dcafeb9645f37a6d8d43873baca391f7985f716f906ea02017de058509fc`

Actual loaded SONAMEs were checked for the added relocation campaign. The relevant live attention source hashes match the saved candidate source hashes. The pre-experiment Git base is `d67e66b0b9e9d2746e9faf8b08a8b48bf3eaa829`; the substantial existing dirty worktree was preserved.

## 1. Where the difference starts

For the first main attention operation, the control and native sparse path receive the same inputs. Their outputs differ by about 0.14% relative RMS over the entire operation: 0.1437% in the English case and 0.1387% in the C++ case. This is a local tensor difference, not a perplexity change.

The difference is amplified downstream. At the second main attention operation, at least one selected history entry differs for 456 of 512 English queries and 711 of 1,000 C++ queries. That does not mean those queries select completely different histories: many entries remain shared. It demonstrates that the divergence is not limited to a continuously perturbed final score; later discrete selection decisions change too.

All 12 attention layers were captured in both cases. Their restored cached-prefix K/V bytes remained identical across the two variants: 2,611,200,000 bytes checked over 24 layer pairs. This excludes cached-prefix modification in those comparisons, not every possible cache bug on other workloads.

Evidence: `intervention-prose-01.layer-differences.json`, `intervention-cpp-01.layer-differences.json` in the all-layer campaign.

## 2. Causal intervention through the complete model

The diagnostic executes the sparse path, then replaces only each main attention output with its saved control counterpart before proceeding. It does not replace final logits or the model's other operations.

Across the two cases, all 24 main attention substitutions checked their 96 input tensors against the corresponding control tensors. Every input matched byte-for-byte. After substitution, the sampled complete-vocabulary prompt scores, all-token losses/top choices, and eight fixed continuation score rows were restored byte-for-byte to the control. Separate dense and sparse repetitions also matched their respective originals.

The capture runs match the earlier uninstrumented saved predictions. This reduces concern that observation itself changes the reported result, although synchronization instrumentation is not a universal race proof.

This localizes the observed sparse/control discrepancy to attention outputs and their downstream consequences. It does not, by itself, prove the numerical accuracy of attention; the reference and invariant tests below address that separately.

Evidence: `intervention-prose-01.intervention.json`, `intervention-cpp-01.intervention.json`, `prepare-model.py` and saved per-round score files. The independent `consolidate.py` rehashed the intervention/control files.

## 3. Independent index and numerical-reference checks

| Check | Result |
|---|---|
| Captured attention replays | 48: both variants, all 12 attention layers, two requests |
| Replay versus corresponding full-model captured output | Byte-identical in every case |
| Compacted CUDA index entries versus CPU mask scan | 74,426,688 entries matched |
| Selected future cache entries in checked active masks | Zero |
| Higher-precision reference query/head rows | 16,128, each with 256 output components |
| Sparse rows closer to reference after matching query rounding | 16,027 / 16,128 (99.374%) |
| Worst sampled relative RMS error, control | 1.2991% |
| Worst sampled relative RMS error, sparse | 0.6480% |

The reference uses double-precision dot products, normalization and value accumulation on the same dequantized K/V values rounded to the FP16 representation consumed by these CUDA kernels. A second reference also rounds and scales Q as the kernel does. The table uses the latter, so input conversion is not mistaken for accumulation error.

The index-entry total includes repeated query/padding rows across captures; it is not a count of unique context tokens. Numerical rows are selected across query boundaries and all 24 heads, not every possible row or context.

Evidence: `replay-deep.cpp`, 48 `replay-*.json` files, `replays-all-summary.json`; independent recomputation in `verified-summary.json`.

## 4. Tests for unintended dependencies

The all-layer campaign completed 72 operator-level checks over 24 captured attention cases:

- Replacing all later query vectors left the first query unchanged on both paths.
- Replacing every K/V row masked for the first query, including old unselected entries and future cells, left that query unchanged on both paths.
- Swapping the first and last query and its mask swapped sparse outputs exactly.

These are fixed-shape attention-operator checks. They must not be described as a complete end-to-end proof of future-token independence throughout the MoE and recurrent executor.

Evidence: `invariants-main.cpp`, `invariants-all-summary.json`, per-case outputs.

## 5. Added storage-relocation experiment

The new test repeats a captured first query at the original full query geometry, then preserves the selected K/V bytes, their order and mask values while moving the selected rows to three different physical layouts: packed at the front, packed at the end, and evenly spaced through the cache. The mask is updated to describe exactly the same mathematical attention problem.

Eight captured cases were tested: early and late attention on both GPUs, for both the 512- and 1,000-token request geometries. Across all 24 relocations, every sparse output remained byte-identical. All 24 control comparisons changed numerically.

For example, one V100 case changed from 0.330% reference error with the original slots to 2.035% when evenly spaced. Sparse stayed at 0.136% and was bit-identical. The selected values and their order were unchanged. This directly demonstrates sensitivity of the previous implementation to how work is split into tiles, while the selected-entry path is invariant under the tested address changes.

This supports the arithmetic explanation and tests selected K/V address handling independently. It does not make either implementation a full-precision reference and does not establish general model quality. The relocation is an operator storage diagnostic, not a change to the model's causal attention rule.

Evidence: `relocation-main.cpp`, `run-relocation.py`, eight `relocation-*.json` files, `all-summary.json`.

## 6. Memory, initialization and synchronization checks

Eight isolated sanitizer runs passed: memcheck, initcheck, synccheck and racecheck on each GPU, with F16 and Q8_0 attention cases. Those isolated cases use 17 queries, 8,192 cache entries and a 2,051 selected-entry bound.

The unfiltered whole-model CUDA memcheck also completed with exit code zero and `ERROR SUMMARY: 0 errors`. It exercised restored 100k context plus 512 English tokens, dense then sparse, with eight fixed continuation steps each. Observation callbacks were enabled, without output substitution or capture writes.

Sanitizers check specific error classes and executed paths. A clean run is not proof against every race, host-side bug or untested shape. The larger C++ model request did not receive a whole-model sanitizer run in this campaign.

Evidence: `sanitizers-summary.json`, eight sanitizer logs, `whole-model-memcheck-01.log`, its command and exit records.

## 7. Precision-policy mismatch found in source

The checked source explicitly documents `GGML_PREC_F32` as requiring FP32 accumulation (`ggml/include/ggml.h:1453-1465`). `ggml_prec_set_acc()` accepts attention and stores the request (`ggml/src/ggml.c:3302-3324`). The model graph makes that request at `src/llama-graph.cpp:2666`, and real captured attention operations contain precision value 10, which is `GGML_PREC_F32`.

However, the active Turing and Volta attention implementations use an FP16 `half2` accumulator tile for the value-weighted sum: `T_C_VKQ` in `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1310-1326` and `1388-1397`. The corresponding matrix instructions in `mma.cuh` use half-precision result/accumulator operands. QK dot products and normalization also involve FP32 calculations; this is not a claim that the entire operation is FP16.

The model-level FP32 request therefore does not ensure that the running value sum is retained in FP32 throughout attention. This is a precision-contract mismatch in the checked implementation, not evidence of a wrong index or newly dropped attention entries.

A new matched probe requested default, FP16 and FP32 accumulation on identical real inputs, on both GPUs and both attention variants. All requests were accepted, and all three settings produced byte-identical outputs in all four cases. Equality alone would not prove a violation, because a higher-precision implementation can legally satisfy a lower precision minimum. Here it accompanies direct source evidence that the running accumulator is FP16.

The same FP16 accumulator declarations and documented FP32 contract already exist at the pre-experiment Git base. Thus the native sparse patch did not introduce this mismatch. Calling the observed difference simply harmless ordinary FP32 rounding would nevertheless be misleading.

Evidence: `precision-source-evidence.json` contains exact excerpts, line numbers, hashes and the checked base commit; `precision-probe-main.cpp`, `run-precision.py`, `precision-summary.json` contain the matched probes.

## Interpretation and next engineering decision

The supported chain is: changed reduced-precision attention execution produces small local differences; those differences change subsequent hidden states and some discrete attention-selection decisions; later predictions can consequently change much more. No new loss of selected entries was found for identical inputs.

The next precision fix should make the accumulator requirement explicit and effective, then compare FP32-accumulating dense and sparse implementations under the same inputs before claiming that remaining differences are only expected rounding. This report does not claim such a kernel fix or precision-controlled collapse experiment has already been implemented.

The earlier 512-versus-1,000 request-length sensitivity, and the new layer-first executor versus original V100-optimized chronological decode, remain separate questions. The current evidence does not certify all of those differences as pure floating-point effects. Wider prefix diversity and longer continuations remain necessary before broad quality approval.

No new performance claim follows from these diagnostic runs. Instrumentation and synthetic relocation change timing conditions. The candidate was not promoted to the production default.

## Reproduction and evidence

From `/workspace/moe-layer-first-0914/numerics-relocation-audit-0916/`:

```sh
python3 run-relocation.py compile
python3 run-relocation.py all
python3 run-precision.py compile
python3 run-precision.py all
python3 consolidate.py
```

GPU runs require the Sandbox `gpu:all` lock. Runners preserve existing result files rather than overwrite them. Raw captures, model weights and frozen libraries remain in the persistent Sandbox and are not included in the small evidence archive. The archive contains diagnostic source, commands, JSON results, source excerpts and hashes.

Relevant general documentation: NVIDIA's Floating Point and IEEE 754 guide explains arithmetic-order and rounding differences. NVIDIA's Compute Sanitizer documentation describes the error classes and limitations of each tool. Neither document substitutes for the project-specific measurements above.

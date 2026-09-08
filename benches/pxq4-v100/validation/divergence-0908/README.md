# PXQ4 divergence investigation — 8 September 2026

## Conclusion

No new PXQ4 indexing, panel-layout or missing-write bug was demonstrated in the tested single-V100 path. The investigation found two invalid comparison assumptions and then isolated genuine numerical differences. The default fast path is unchanged. An optional floating-point PXQ4 decode path is implemented for accuracy experiments and diagnosis; it is not a universal quality fix or a new default.

Repository: `/workspace/llama-pxq4-v100`, branch `exp/pxq4-v100`.
Reference: `/workspace/pxa`, commit `896c18919ebcfa4724d9df5eaa7feb92e017c563`.
Weights: the same Fusion4 PXQ4 file on one Tesla V100-SXM2-32GB. FP16 KV, MTP off.

## Corrected comparison protocol

The user-provided `/models/pxa-server-host-build` has `GGML_CUDA:BOOL=OFF`. Its log explicitly warns that GPU-layer settings are ignored. Earlier comparisons through that build mixed CPU and GPU execution. GPU-specific ablations on that executable cannot identify CUDA causes.

The same patched sources were rebuilt using the existing CUDA-enabled `/models/pxa-fullprobs-build` (sm_70). Neither the original release nor the CPU test build was changed.

The earlier HTTP requests evaluated complete 401/512-token prefixes, whereas the stored port logits used 256+128 prefill followed by individual forced tokens. Those are different operation shapes, particularly for this recurrent MoE model. The new C++ helper feeds exactly the same token IDs, batch schedule and requested output rows to both libraries: chunk 256, score step 1; four 512-token fixtures scoring their last 128 predictions.

Full-vocabulary logits are read through each build's matching C API. This avoids sampling, token-string ambiguity and HTTP probability rounding. There are 248,320 vocabulary entries per scored prediction. Log-softmax and comparison reductions use double precision in the analysis.

## Matched PXA comparisons

| Test | Candidate | Top-token agreement | Mean KL, PXA to candidate | Sample perplexity ratio |
|---|---|---:|---:|---:|
| 512 short prose/code predictions | Fast port | 498/512 (97.266%) | 0.002508137 | 0.997380 |
| Same 512 predictions | Direct-FP32 PXQ decode | 499/512 (97.461%) | 0.002207729 | 0.997053 |
| Independently computed 101k code prefix, 64 forced predictions | Fast port | 63/64 (98.438%) | 0.000760631 | 1.003225 |
| Same independent long prefix | Direct-FP32 PXQ decode | 63/64 (98.438%) | 0.001009232 | 1.005155 |

The long run constructs the full 100k+1k prefix separately in each process using chunk 2048. It does not reuse a checkpoint between engines, unlike the older validation. PXA sample perplexity was 1.650331; the fast port measured 1.655654 (+0.323%), and direct FP32 measured 1.658839 (+0.516%). These 64 positions are not a broad model-quality benchmark, and a higher numerical precision path is not guaranteed to be closer to PXA's own approximate execution.

As another control, changing PXA itself from its default to `PXA_REFERENCE=1` changed 18/512 top-token choices, with mean KL 0.00323779. This demonstrates that exact agreement is not invariant even between precision/optimization modes of PXA. It does not, by itself, excuse an incorrect port.

## Where differences arise

### Arithmetic before PXQ4

Layer traces show matching embeddings and several initial projections. Tiny differences appear in the linear-attention path before the first routed PXQ4 expert computation. For example, the first softplus result differs at approximately 4.45e-8 relative RMS, and the first layer's linear-attention output differs at approximately 4.97e-5 relative RMS.

An apparently enormous difference in a recurrent-state tensor was a layout comparison mistake: the two graphs expose transposed last axes. Transposing the axes reduced the relative RMS difference from about 1.414 to about 4.97e-5. Raw storage order must not be mistaken for different logical state.

Trace callbacks force graph boundaries and can inhibit fusion. Their purpose here is localization, not performance measurement or a claim that every uninstrumented kernel takes the identical route.

### Additional rounding in fast PXQ4 decode

The wire-format PXQ4 codebook contains floating-point constants. The DP4A decode path approximates them with signed bytes divided by 127, and quantizes activations to Q8_1. It therefore does not compute exactly the same dot product as directly decoded PXQ4 weights and FP32 activations. PXA also has an S8/Q8 MMVQ path, but the overall dispatch, fusion and arithmetic differ between engines.

The new `GGML_CUDA_PXQ4_MMV_F32=1` path directly evaluates the floating-point codebook and FP32 activations without materializing full FP16 expert matrices. It retains compressed weights and fused gate/up support. Prefill, attention, non-PXQ tensors and original model weights are not changed.

An independent double-precision CPU oracle passed all 320 direct-FP32 decode cases, including routing, broadcasting and fusion. The final run's maximum peak-normalized error was 1.40735e-6. This test checks the canonical floating-point decode computation, rather than only reproducing the fast kernel's integer approximation.

### Cumulative drift and routing

In an inspected decode trace, initial routing matched. One earlier difference only reordered the same selected experts. A genuine selected-set change occurred at layer 39: one of eight experts changed, at a near-boundary router score. Thus small arithmetic differences can change later discrete routing decisions; matching the architecture and quantization format alone does not guarantee identical generated continuations.

The optional FP32 path does not remove numerical differences in the rest of the model. It also does not make our engine bit-identical to PXA.

## Same-state experiment isolating decode

`shadow_logits.cpp` restores the exact same reference state before each candidate forward. It disables CUDA graph caching because environment-controlled kernel dispatch is changed within the process. A reference-to-reference restore control must be byte-identical or the experiment aborts.

Final rerun: four fixtures, first 32 predictions after a 384-token context, 128 scored predictions total. All 128 restore controls were byte-identical. Mean KL against the explicit FP32-compute reference was:

- Fast integer decode: 0.000964933.
- Direct-FP32 PXQ decode: 0.000560715.

The reduction is about 42%. Both modes agreed with the reference top token on 126/128 predictions. This isolates a real contribution from additional decode rounding, but it is not a 42% improvement in model quality and does not predict the accumulated long-context comparison.

## Performance and default behavior

The recorded short-context test measured a median 112.83 TG/s with the original integer path and 100.28 TG/s with direct FP32, approximately 11% slower. These are diagnostics on that benchmark, not a universal penalty across contexts.

The default remains unset/0 for `GGML_CUDA_PXQ4_MMV_F32`. A fresh default-path run was byte-identical to all 127,139,840 logits from the previous 512-position validation baseline. Production `v100-optimized` was not changed.

## Additional checks and API limitations

Final compact CUDA memcheck for the FP32 path: 20 cases, zero reported errors. Earlier recorded compact racecheck, initcheck and synccheck logs are retained. These tests are evidence, not a proof over every possible shape or GPU.

A diagnostic test explicitly forcing `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` passed four routed/dense dispatch cases at T=1/8 with zero error. The same expectation is not valid when the normal FP16 fallback is selected; an exploratory log showing a 0.00116 difference without the FP32 override does not establish a dispatch bug.

The patched CUDA server returned HTTP 200 and all 248,320 probability entries for both tested requests. This version's response uses `prob` and `tok_str`, not token IDs. The probability sums were approximately 1.000824 and 1.000008. Therefore raw C-API logits, not string-mapped HTTP probabilities, are the comparison ground truth. The HTTP 500 fix is working; small probability-normalization error and lack of IDs are separate interface limitations.

## Reproduction and evidence

`run_matched.sh` builds the two ABI-matched helpers and runs the matched short comparisons. Set `PXQ_VALIDATE_LONG=1` to compute the long prefixes independently. The wrapper was syntax-checked; the constituent recorded runs were executed separately. Use the Development Sandbox `gpu:all` lock.

`run_followup.sh` and `test_mmvf.cu` contain the direct-FP32 numerical and sanitizer procedure. `shadow_logits.cpp`, `analyze_traces.py`, and the summarizers provide focused diagnostic controls.

`summary.json` contains condensed metrics, fixture hashes, API results and configuration. Full numerical metrics, traces and logs are retained in this directory; large logits/tensor captures remain outside git under `/models/llama-pxq4-build/divergence-0908`. `final-sha256.txt` records source and library hashes. Diagnostic captures and model weights were not deleted.

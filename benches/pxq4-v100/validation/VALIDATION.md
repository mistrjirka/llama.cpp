# PXQ4 implementation validation — 8 September 2026

## Verdict

The implemented **single-V100 Fusion4 PXQ4 path passes the tested kernel, memory-safety,
layout and regression checks after three edge-case fixes**. These results are evidence
of implementation correctness within that scope, not proof of model-quality equivalence
or support for every GGML operation/device. Native DP4A intentionally uses a signed-int8
approximation of the PX16 codebook; it is not bit-exact to the FP16/cuBLAS reference.

Repository: `/workspace/llama-pxq4-v100`
Branch: `exp/pxq4-v100`
Starting commit: `7779c7370a3de283f917cd82c9c286043d4bb5dc`
Build: `/models/llama-pxq4-build`
Model: `PXA-Fusion4-35B-PXQ4.gguf`, unchanged and read-only.
GPU: one Tesla V100-SXM2-32GB; RTX 2080 Ti excluded.

## Bugs reproduced and fixed

1. **Repeated expert IDs in prefill left output entries unwritten.** The old map kept
   only the first occurrence of each expert within a token. The corrected map includes
   every `(token, slot)` occurrence, preserves order and allocates sufficient capacity.
   Unique top-k routing used by Fusion4 is unchanged. The failing original case and
   passing expanded regression are retained in `prefill-duplicate-before.log` and
   `duplicates-final.log`.
2. **Unaligned panel views were advertised as supported.** A view starting one logical
   row into a 64-row panel was accepted despite not describing a valid panel origin.
   The backend now validates panel origin, dimensions, bounds and storage strides.
   Whole aligned panel views still work; unsupported views are declined. Before/after
   results are in `dispatch-before.log` and `dispatch-final.log`.
3. **The explicit FP32 cuBLAS fallback aborted.** Only the FP16 panel conversion existed.
   A float conversion now honors the FP32 fallback, with a nonzero known-answer matrix
   multiplication check. See `f32-before.log` and `f32-final.log`.

These fixes are isolated to the experimental branch. Production `v100-optimized` and
model weights were not changed by this validation work.

## Implementation test results

| Check | Result | Coverage |
|---|---|---|
| Independent CPU reference vs exported decode launcher | 320/320 pass | Scalar nibble extraction and double accumulation; routed/dense, channel/sample broadcast, padded strides, 1–8 tokens, fused/unfused GLUs |
| Original prefill numerical checks | 24/24 pass | FP16-snapped operands, independent direct accumulation |
| Expanded prefill edge shapes | 60/60 pass | K=32/96, token tails, 1–512 experts, shared/per-slot activations |
| Repeated-expert prefill cases | 60/60 pass | All expert occurrences written, including repeated IDs |
| CUDA memory/race/initialization/synchronization checks | Zero reported errors or warnings | All four tools on decode subset and expanded repeated-ID prefill suite |
| View support regression | Pass | Unaligned view rejected; aligned panel view accepted |
| Explicit FP32 reference fallback | Pass | Known nonzero result, no abort |
| GGUF regression suite | 128/128 pass | Final rebuilt binary |
| Stock quantization regression | Pass | Successful exit |

The independent decode test does not call the GPU lookup/dot helper for its reference.
Its largest observed absolute output error divided by peak reference output was
**3.19 × 10⁻⁷**. The original prefill checks remained below **6 × 10⁻⁶** on that measure.
These are peak-normalized errors, not maximum elementwise relative errors near zero.

Sanitizer tools: `memcheck`, `racecheck`, `initcheck`, `synccheck`. Decode sanitizer runs
cover 20 compact cases; the unsanitized independent numerical suite covers all 320.
Prefill sanitizer runs cover 60 repeated-ID cases. Passing sampled tests is not a
formal proof that no memory bug exists in untested configurations.

## Model-level fixed-token comparison

The harness supplies the **same predetermined token IDs**, without sampling, and
records every logit in the **248,320-token vocabulary** at each scored position.
Thus the comparison is not confused by different generated continuations or routing
caused by different sampled tokens. All logits examined were finite.

The reference below is **our PXQ4-to-FP16/cuBLAS control**, not the PXA binary.
K/V precision is FP16, Flash Attention is enabled and speculation is absent.

| Test | Scored positions | Top-token agreement | Mean KL (reference → native), nats | Reference sample perplexity | Native sample perplexity |
|---|---:|---:|---:|---:|---:|
| Four short prose/code samples; single-token scoring | 512 | 96.88% | 0.002385 | 6.0191 | 5.9982 |
| Same samples; 32-token batched scoring | 512 | 97.46% | 0.002032 | 6.0191 | 6.0049 |
| 100k cached + 1k new input + 64 forced tokens | 64 | 98.44% | 0.000785 | 1.6568 | 1.6788 |

Short samples comprise two Wikitext excerpts, pinned C++ grammar code and pinned Python
GGUF reader code. Each uses 512 input positions and scores the final 128. The roughly
0.35% lower short-sample perplexity is **not evidence of a general quality improvement**.

The long test restores the **same 100,000-token cache state** in both modes, then checks
1,000 new input tokens and 64 predetermined continuation tokens. It isolates incremental
work at long context; it does **not** compare two independently computed entire 100k
prefixes. Agreement is 63/64 predictions; sample perplexity is about **1.32% higher**
for native. This is a small sample, so neither zero quality loss nor a broad regression
can be established from it. Model-quality acceptance remains a separate judgment.

`native-vs-reference.json`, `batched-vs-reference.json` and
`long-native-vs-reference.json` retain per-position errors, total-variation distance,
maximum errors and score differences, not only averages.

## Normal-path regression and speed

The hardened build reproduces **all 127,139,840 recorded normal-path logits byte for
byte** against the pre-hardening native build (512 positions × 248,320 logits).
This was checked both with the numerical comparison and with binary `cmp`.
See `hardening-numerical-regression.json`.

Short-context +256-token generation remains **112.93 tokens/s**, compared with the
previous 113.06 tokens/s. These measurements show no meaningful observed short-decode
regression; they do not establish unchanged speed for every prefill shape. Full previous
long-context throughput was not re-benchmarked as part of this validation pass.

## PXA comparison limitation

A full-vocabulary distribution request to the official PXA release returned HTTP 500:
`[json.exception.type_error.316] incomplete UTF-8 string; last byte: 0xE2`.
The attempted comparison did not pass, and no PXA-wide numerical-equivalence claim is
made. `compare_pxa_api.py`, `pxa-api-error.txt` and the server log preserve the attempt.
Binary-matched headers for the release's internal commit were also not obtainable from
the public repository, so the test did not assume C++ ABI compatibility.

## Remaining scope limits

CPU PXQ execution, other PXQ formats, quantization/export, embedding GET_ROWS, MTP,
multi-GPU splitting and other GPU generations remain outside this validation. BF16
fallback overrides are not validated. Unsupported panel views are rejected rather than
silently interpreted as conventional rows. This report is specific to the tested
single-V100 Fusion4 implementation, with the stated numerical contract.

## Reproduce and evidence

Run under the sandbox `gpu:all` lock, from `/workspace/llama-pxq4-v100`:

```bash
bash benches/pxq4-v100/validation/run_validation.sh
PXQ_VALIDATE_LONG=1 bash benches/pxq4-v100/validation/run_validation.sh
```

The script requires the existing CUDA build configuration, local model, Wikitext and
pinned code fixture. It writes new `recheck-*` logs separately from recorded results.
The combined wrapper is syntax-checked; its underlying build/test commands were executed
individually for the recorded run. No automatic global model-quality pass threshold is
assigned to perplexity or KL results.

Raw logits and shared cache state are in
`/models/llama-pxq4-build/validation-logits/`; these large generated artifacts are not
committed. Smaller results, regression tests, fixture hashes and this report are kept
in `benches/pxq4-v100/validation/`. `summary.json`, `final-source.sha256` and
`final-binaries.sha256` identify the measured artifacts. The existing
`results/profiled.json` was unrelated and left untouched.

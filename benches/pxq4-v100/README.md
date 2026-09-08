# PXQ4 on v100-optimized: native CUDA port and benchmark

Experimental branch: `exp/pxq4-v100`, based on `4154e79f7`.
Implementation/measurements: 2026-09-08. Not merged into `v100-optimized`.

## Validation update

The follow-up validation reproduced and fixed repeated-expert prefill omissions,
unaligned-panel view acceptance and the explicit FP32 fallback crash. Independent
CPU decode references, expanded prefill checks, all four CUDA sanitizers and 128/128
GGUF regression tests pass. Full-logit teacher-forced probes quantify the intentional
native/reference numerical differences; this is not a claim of bit-exact model quality.
See [the validation report](validation/VALIDATION.md) for results and limitations.

## Measured results

Both engines load the **same Fusion4 PXQ4 GGUF file**. One V100-SXM2-32GB,
200 W limit; RTX 2080 Ti excluded; FP16 K/V; Flash Attention enabled;
131072 context; one slot; batch/ubatch=2048; 16 CPU threads; no speculation/MTP.
PXA reference is official release `v2026.09.07-rc1` / `d8b0def5`.
PXA automatic speculation and sampler selection are explicitly disabled.

Each long-context cell restores **100000 tokens**, processes **1000 additional
tokens**, and emits exactly 64 or 512 tokens. One warmup plus two retained runs.
The fork additionally reports `cache_n=100000` for every measured request.
Rates below are tokens/second; TG is normalized as `(n_predict-1)/predicted_seconds`,
not each engine's differently-counted displayed TG figure. Raw wall-clock request
latencies are also retained, excluding slot restore and initial cold prefill.

| Prefix | Engine | PP +64 | TG 64 | PP +512 | TG 512 |
|---|---|---:|---:|---:|---:|
| synthetic | fork | 997.8 | 85.17 | 994.8 | 85.58 |
| synthetic | pxa | 840.0 | 51.77 | 837.4 | 51.43 |
| code | fork | 947.7 | 84.77 | 949.5 | 85.81 |
| code | pxa | 803.3 | 51.85 | 802.8 | 51.34 |

The synthetic prefix is the earlier repeated pangram (12 distinct token IDs).
The separate **code prefix has 3868 distinct token IDs**, taken from pinned llama.cpp
source contents; both engines use identical token arrays and prefix hashes.
`results/code-fixture-manifest.json` records the exact source files and content hashes.
The code corpus is more varied, but is not a general model-quality evaluation.

Short prompt + 256 output tokens, batch/ubatch=512, median of three retained runs:
original two-row native kernel **101.66 TG/s**; coalesced native kernel **112.84 TG/s**;
PXA **120.42 TG/s**. The final hardened build recheck measured **113.06 TG/s**. Common normalized timing is used throughout. Coalesced/PXA
256-token continuations match on this smoke prompt. The synthetic long continuations
also match, but the code-prefix continuations diverge between engines (first differing
character at index 79 in the recorded outputs). The code-prefix TG tests therefore use
identical inputs/lengths, not identical generated-token routing. This is a performance
comparison, not proof of model-output parity. Long-context speed gains do not
imply that the port wins at every context length or on every model.

## What changed

* GGML/GGUF type 252 and its **two extra bytes per logical row** are accounted for.
  Payload is interleaved in 64-row panels, not ordinary independently stored rows.
  Python GGUF reader exposes an opaque byte payload, including all anchors.
* Native Q8_1/DP4A decode uses four-warps-per-block, adjacent-row code loads,
  byte-permutation table lookup, shared-memory subscale lookup, and unbiased
  up/gate/GLU fusion. No full FP16 weight conversion is done for native decode.
* Grouped Volta WMMA prefill builds stable expert maps/tile descriptors entirely
  on the GPU, decodes weights directly into shared tiles, uses FP16 inputs with
  FP32 accumulation, and scatters results straight to the original token/slot.
  It removes the old host expert loop, per-expert synchronizations, and full
  FP16 expert-weight materialization. Prefill up/gate are currently separate.
* PXQ code lives in its own CUDA translation unit, so tuning does not recompile
  every stock MMVQ instantiation. Existing stock quant dispatch is retained.
* Reversed expert/token activation strides were fixed. The non-MoE broadcast
  address calculation was corrected as well. Temporary synchronous diagnostics
  and the per-projection debugging mask were removed.

Larger legacy row tiles lost performance: 2/4/8 rows measured 101.66/88.90/85.23
TG/s. The improvement comes from thread/memory mapping, not simply larger tiles.
The isolated coalesced kernels are roughly 2x faster on single-token routed-expert
shapes, with larger gains on some multi-token shapes.

## Numerical contract and validation

Native DP4A decode follows PXA's **signed-int8 image of the PX16 codebook**, with
`1/127` folded into the scale, and standard Q8_1 activations. It is **not bit-exact**
to an FP16-dequantize/cuBLAS path. WMMA prefill uses the original frozen PX16/SUB16
values and FP16 rounding; FP32 accumulation order differs from cuBLAS.

`test_decode.cu` compares 96 tile/shape configurations (both expert projections,
T=1/2/5/8, fused/unfused) with the prior corrected Q8_1 reference kernel. Maximum
observed error relative to peak output is below 3e-7.
`test_prefill.cu` checks 24 K/token/channel shapes, ragged tile tails and unbalanced
expert routing, against direct accumulation of exactly FP16-snapped operands.
Maximum observed error relative to peak is below 6e-6. Outputs are NaN-initialized
to detect missing writes. Both suites were re-run successfully on V100 after loader/fuser hardening.

The GGUF regression suite passed **92/92** after hardening, stock quantization
regression exited successfully, and the dedicated tensor-layout/CPU guard test
passed. The Python reader read all **120 PXQ4 tensors / 17175674880 payload bytes**.
A single deterministic text match is a smoke test, not proof of model-quality parity;
teacher-forced full-logit/perplexity probes are now recorded in the validation report.
Broader downstream task-quality gates remain open.

## Reproduce

Current build directory: `/models/llama-pxq4-build` (not model weights).
From `/workspace/llama-pxq4-v100`:

```bash
bash benches/pxq4-v100/run_build.sh
bash benches/pxq4-v100/build_test.sh
CUDA_VISIBLE_DEVICES=GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79 \
  /models/llama-pxq4-build/test-pxq4-decode
bash benches/pxq4-v100/build_test_prefill.sh
CUDA_VISIBLE_DEVICES=GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79 \
  /models/llama-pxq4-build/test-pxq4-prefill
python3 benches/pxq4-v100/bench_long.py --engine fork --fixture synthetic --label wmma-long
python3 benches/pxq4-v100/bench_long.py --engine pxa --fixture synthetic --label pxa-long
python3 benches/pxq4-v100/bench_long.py --engine fork --fixture code --label fork-code-long
python3 benches/pxq4-v100/bench_long.py --engine pxa --fixture code --label pxa-code-long
python3 benches/pxq4-v100/summarize.py
```

Use the sandbox `gpu:all` lock around GPU benchmarks. Harnesses clean up their own
server processes and use separate ports. Large slot caches, profiler databases,
and generated token arrays are not committed.

## Diagnostic controls

| Variable | Default | Effect |
|---|---|---|
| `GGML_CUDA_PXQ4_NATIVE` | 1 | 0 forces reference decode and disables native decode fusion |
| `GGML_CUDA_PXQ4_PREFILL` | 1 | 0 disables grouped WMMA prefill |
| `GGML_CUDA_PXQ4_COALESCED` | 1 | 0 selects the original native decode kernel |
| `GGML_CUDA_PXQ4_RPB` | 2 | Legacy-only 2/4/8-row tile |
| `GGML_CUDA_PXQ4_WARP_ROWS` | Auto | Experimental 2/4 rows per warp override |
| `GGML_CUDA_PXQ4_VDR` | 4 | Experimental 2/4 code words per lane |

Set both NATIVE=0 and PREFILL=0 for the reference dequantize-to-FP16/cuBLAS control.
The optimized defaults need no environment overrides. Final reference-control smoke
(`NATIVE=0`, `PREFILL=0`) passed and measured 16.17 TG/s; the final native recheck
measured 113.06 TG/s on the short benchmark. Final GPU kernel tests also passed.

## Scope and remaining work

Validated end-to-end model: Fusion4 PXQ4, fully GPU-resident on **one V100**.
This is not yet a production-complete quantization ecosystem port. CPU PXQ4 execution
is explicitly declined rather than being falsely routed through a row-dot callback.
The port does not yet implement PXQ quantization/export, CPU fallback, PXQ4HQ or
PXQ1/2/3/6/PXQU, arbitrary unaligned panel-slicing views, embedding GET_ROWS, distributed
execution, or a broad downstream task-quality suite. Dense broadcasts now have independent
kernel coverage; MTP still needs end-to-end validation. FP32 cuBLAS fallback is tested;
BF16 overrides are not validated. Routed prefill now handles repeated expert IDs too.

The short-context profile now spends more GPU kernel time in the stock **MXFP4
backbone** than in the PXQ4 expert kernels. That and router/activation overhead are
better next profiling targets than further blind PXQ row-tile expansion.

## Provenance

Format constants and byte-permutation lookup are adapted from the MIT-licensed PXA
source at `6d7dbfff653288e5f83fad5f7ee443329b77ebfc`, especially
`ggml/include/ggml-pxq6-tables.h`, `ggml/src/ggml-cuda/pxa/pxq-mmvq.cuh`,
and the grouped-WMMA layout described in `pxq6.cuh`. See repository `LICENSE-PXA`
for the preserved license/copyright notices. The experimental dense `pxq4-v70.cuh`
path was not transplanted: its own notes report regressions.

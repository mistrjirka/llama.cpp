# PXQ family on v100-optimized: native CUDA port and benchmark

Status: PXQ CUDA runtime support is integrated into `v100-optimized`; this directory retains the implementation notes, validation harnesses and benchmark history. Original implementation work began on `exp/pxq4-v100` from `4154e79f7`.

## PXQ family expansion (2026-09-09)

The CUDA runtime now supports **PXQ1, PXQ2, PXQ3, PXQ4, PXQ4-HQ and PXQ6**, plus mixed per-tensor **PXQU/PXQ_UNIVERSAL** models. This includes panel-aware loading/GGUF reading, exact dequant fallback, native decode, mixed gate/up fusion and V100 grouped MoE prefill. PXQ quantization/export remains in the PXA quantizer. See [`results/PXQ-FAMILY-0909.md`](results/PXQ-FAMILY-0909.md) for the all-tier validation and performance table.

For RTX 2080 Ti / SM75 prompt processing, the latest exact dense-SwiGLU fusion and P1k-P16k matched-size results are in [`results/TURING-PREFILL-1K-16K-0909.md`](results/TURING-PREFILL-1K-16K-0909.md).

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
| code | fork exact pair | 945.9 | 86.59 | 945.9 | 87.50 |
| code | pxa | 803.3 | 51.85 | 802.8 | 51.34 |

The synthetic prefix is the earlier repeated pangram (12 distinct token IDs).
The separate **code prefix has 3868 distinct token IDs**, taken from pinned llama.cpp
source contents; both engines use identical token arrays and prefix hashes.
`results/code-fixture-manifest.json` records the exact source files and content hashes.
The code corpus is more varied, but is not a general model-quality evaluation.

Short prompt + 256 output tokens, batch/ubatch=512, one V100. A fresh post-validation
comparison measured **113.13 TG/s** for the established port and **120.93 TG/s** for PXA.
Two additional V100 optimizations were then tested together: direct indexed reads of the
recurrent GDN state (avoiding its GET_ROWS materialization) and a four-warp N=1 MXFP4
MMVQ specialization. This **exact pair** is opt-in and measured **116.67 TG/s** over a
20-run retained soak: +3.13% over the established port and 3.52% below PXA. All 20 runs
produced the same 256-token output hash. A 512-position/full-vocabulary teacher-forced
probe produced a **508,559,360-byte logit file byte-for-byte identical** to the established
path (top-1 100%, KL 0, RMSE 0). The GATED_DELTA_NET backend suite passed 40/40, and
Compute Sanitizer memcheck completed cleanly with CUDA graphs disabled.

An additional Q/K-normalization-in-GDN option reaches **118.68 TG/s** (+4.90% over the
established port, 1.86% below PXA), but it deliberately changes floating-point reduction
order. On the 512-position probe it measured mean KL **0.00227**, 97.46% top-1 agreement,
and about +0.14% sample perplexity versus the established path. It therefore remains a
separate non-bit-exact option rather than part of the exact configuration.

The final clean varied-code 100k-cache + 1k prompt benchmark with the exact pair measured
**945.88 PP / 86.59 TG** for the 64-token arm and **945.89 PP / 87.50 TG** for the
512-token arm. The matching PXA runs were **803.30 / 51.85** and **802.76 / 51.34**.
That leaves the fork about **17.75–17.83% faster in PP** and **67.01–70.44% faster in TG**
at 100k context.

The original two-row native PXQ4 kernel was 101.66 TG/s and the first coalesced version
112.84 TG/s. Common normalized timing is used throughout. PXA and the port need not
generate identical continuations on varied prompts; performance tests use identical inputs
and lengths rather than claiming model-output equivalence.

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
| `GGML_CUDA_GDN_INDEXED_STATE` | 0 | Opt-in exact single-sequence/no-rollback GDN state read; avoids GET_ROWS |
| `GGML_CUDA_MXFP4_LEAN` | 0 | Opt-in Volta N=1 MXFP4 kernel; `4` is the validated winner |
| `GGML_CUDA_GDN_QKNORM_FUSE` | 0 | Opt-in Q/K normalization inside GDN; faster but not bit-exact |

The validated exact performance pair is `GGML_CUDA_GDN_INDEXED_STATE=1` plus
`GGML_CUDA_MXFP4_LEAN=4`. These controls are still opt-in because only the Fusion4
single-V100 workload has received the full model-level validation described above.

Set both NATIVE=0 and PREFILL=0 for the reference dequantize-to-FP16/cuBLAS control.
The optimized defaults need no environment overrides. Final reference-control smoke
(`NATIVE=0`, `PREFILL=0`) passed and measured 16.17 TG/s; the final native recheck
measured 113.06 TG/s on the short benchmark. Final GPU kernel tests also passed.

## Scope and remaining work

Validated end-to-end model: Fusion4 PXQ4, fully GPU-resident on **one V100**.
This is not yet a production-complete quantization ecosystem port. CPU execution of the PXQ
slab family is explicitly declined rather than being falsely routed through stock row-dot callbacks.
CUDA runtime support now covers PXQ1/2/3/4/4-HQ/6 and mixed PXQU on validated sm_70 and sm_75 CUDA paths. Mixed V100 + RTX **layer split** is validated; PXQ **tensor split** still hits a separate meta-buffer packing limitation and is not advertised as supported yet. Remaining ecosystem work
includes PXQ quantization/export inside this fork, CPU fallback, arbitrary unaligned panel-slicing
views, embedding GET_ROWS, full PXQ tensor-parallel support, and a broad downstream task-quality suite.
Dense broadcasts have independent fallback/native coverage; MTP still needs end-to-end validation.
FP32 cuBLAS fallback is tested; BF16 overrides are not validated. Routed prefill handles repeated
expert IDs and mixed per-tensor PXQ tiers.

The short-context profile showed that the PXQ4 expert kernels were no longer the main
gap to PXA. The validated exact gains came instead from the stock **MXFP4 backbone**
and recurrent GDN state plumbing. After those changes the exact configuration is about
3.5% below PXA at short decode while retaining the large 100k-context lead. Remaining
profiled overhead is concentrated in recurrent convolution-state gather/concat/copy and
small activation/routing kernels; a direct in-place convolution-state experiment was
rejected after its smoke test corrupted recurrent behavior and is not present in the code.

## Provenance

Format constants and byte-permutation lookup are adapted from the MIT-licensed PXA
source at `6d7dbfff653288e5f83fad5f7ee443329b77ebfc`, especially
`ggml/include/ggml-pxq6-tables.h`, `ggml/src/ggml-cuda/pxa/pxq-mmvq.cuh`,
and the grouped-WMMA layout described in `pxq6.cuh`. See repository `LICENSE-PXA`
for the preserved license/copyright notices. The experimental dense `pxq4-v70.cuh`
path was not transplanted: its own notes report regressions.


## Numerical divergence investigation (2026-09-08)

The matched raw-logit comparison and layer-level evidence are in
[`validation/divergence-0908/README.md`](validation/divergence-0908/README.md).
PXA commit `896c189` was rebuilt **with CUDA enabled** in
`/models/pxa-fullprobs-build`; its separate `pxa-server-host-build` is CPU-only.
The new comparison uses identical token IDs and decode schedules on one V100.

`GGML_CUDA_PXQ4_MMV_F32=1` is an **opt-in decode accuracy/diagnostic control**.
It directly evaluates the original PXQ4 floating-point codebook and FP32
activations, avoiding the extra S8-codebook/Q8-activation approximation of the
fast DP4A path. It retains compressed PXQ4 weights and fused gate/up support.
It does not change prefill, attention, non-PXQ tensors, or model weights.
It requires native PXQ4 decode to be enabled (the default).

Default: unset/0, preserving the existing fast DP4A implementation. In the
recorded short-context test it measured 112.83 TG/s versus 100.28 TG/s for the
floating-point control. The control is not a universal quality improvement:
it did not improve the independent 100k-prefix comparison with PXA. Do not
enable it as a production fix solely to chase agreement with another engine.

Reproduction: `bash validation/divergence-0908/run_matched.sh` from this
benchmark directory, or invoke that script by its repository-root path.
Set `PXQ_VALIDATE_LONG=1` for independently computed 101k-prefix comparisons.
Run these scripts under the sandbox `gpu:all` lock. Model-distribution metrics
are diagnostics, not an invented universal quality pass/fail threshold.

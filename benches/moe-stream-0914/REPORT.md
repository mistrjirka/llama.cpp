# Bounded MoE expert streaming — implementation and experiments

Date: 2026-09-14. Status: **experimental; not production-accepted or merged**.

Worktree: `/workspace/llama-moe-stream-prefill-0914`

Branch: `experiment/moe-stream-prefill-0914`, based on `v100-optimized` commit `eae5d0ec20767a68217738ec7fbdb08da35eec8b` in the user's fork. The original production worktree and its existing README edits were not changed. No model files were changed.

## Implemented scope

This first implementation streams **one `MUL_MAT_ID` projection at a time**, using the fork's existing quantized MMQ kernels. It is not yet the proposed whole-expert-FFN executor.

An operation-scoped scheduler capability leaves supported expert weights in host memory instead of creating a full-shaped device copy. Two fixed-size CUDA weight slots are reused across all layers of one backend context. One stream transfers selected expert runs, the compute stream processes the current group, and ready/consumed events protect each slot's lifetime. The original expert routing maps, global output positions, quantization and logical-model tile choices are preserved. Empty expert runs are skipped.

The default optional pool is 64 MiB total, split into two 32 MiB slots. Group size defaults to 16 experts but is capped by the byte budget. Pinned model weights transfer directly; pageable model weights use an additional bounded, lazily allocated pinned-host staging pool. The latter requires CPU staging copies and some host waits. Routing metadata incurs one host wait per projection.

The scheduler isolates streamed operations so they are not captured as CUDA graphs or fused with incompatible host-weight operations. The feature is off by default, only enables for sufficiently large prefill batches, and does not add a streamed decode kernel. Retaining the pool can nevertheless affect available decode memory. Nonstreamed paths remain available.

Source locations:

- `ggml/src/ggml-backend.cpp`: operation-scoped host-input capability, copy-allocation bypass, split boundaries.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: dispatch, capture compatibility, private capability registration and context cleanup.
- `ggml/src/ggml-cuda/mmq.cu`: bounded pool, event pipeline and compact physical expert groups.
- `ggml/src/ggml-cuda/moe-stream.cuh`: internal declarations.
- `tests/test-moe-stream.cpp`: real-GGUF operator correctness and timing probe.

## Models and compatibility

**Ornith-1.5-35B-A3B AD-Q5_K-Q4_K:** qwen35moe, 40 layers, hidden size 2048, 256 experts, top-8, expert intermediate size 512. Tested actual Q4_K gate weights and Q5_K down weights.

**Qwen3.8-Flash-Next UD-IQ4_XS:** qwen4exp, 48 layers, hidden size 2560, 512 experts, top-10, intermediate size 640. The model contains a separate trigram PLE module. Its operations are left unchanged; the complete model loads and runs with streaming enabled. Actual expert formats include IQ3_S, IQ4_NL, IQ4_XS and Q8_0, all covered in the operator probes. Full-model distribution equivalence has not been established for Next.

**DeepSeek:** `/models/DeepseekV4` points to `/home/jirka/big/models/DeepSeek-V4-Flash-imatrix-aligned`, which is not accessible through that symlink inside the sandbox. No DeepSeek measurement or compatibility claim is made.

Streaming support is deliberately restricted to NVIDIA SM70/SM75, supported quantized expert matrices with suitable layout/alignment, immutable host weight buffers, and the existing routing helper's size limits. Unsupported operations fall back. HIP/MUSA branches are guarded, but those backends have not been built or tested.

## Correctness evidence

`correctness-v1-v100/results.jsonl` and `correctness-v1-2080ti/results.jsonl` contain **30 passing cases each, 60 total**. Each case asserts actual streaming activation and compares all output elements with a GPU-resident reference using identical real weights and deterministic inputs.

Coverage: six actual expert tensors across both models; gate/broadcast inputs and down/per-expert inputs; token counts 64, 512, 1024 and 4096; dense, sparse and skewed routing; group sizes 7, 16 and 32; final partial groups and multiple ring wraps; pinned and pageable host memory.

Maximum observed absolute operator difference on either device: **4.76837158e-7**. Maximum relative L2: V100 **1.70099597e-7**, RTX 2080 Ti **2.04191984e-7**. Many cases were exactly equal. Inputs are synthetic, not captured hidden states.

`memcheck-ornith.log`: CUDA Compute Sanitizer memcheck, Ornith down Q5_K, 512 tokens, sparse routing, pageable source, group 7: **zero errors**. This is one targeted memory check, not exhaustive race or multicontext validation.

### Full-model numerical qualification — unresolved

A fixed text fixture was processed with the complete Ornith model at context/batch/microbatch 512 and 24 host-expert layers. This is a numerical comparison fixture, **not a representative quality/perplexity benchmark**.

The baseline saved log-probabilities with `llama-perplexity --save-all-logits`. An unchanged baseline replay using `--kl-divergence` reported mean KLD approximately zero and 100% top-choice agreement. The streamed run reported:

- Mean KLD **0.000915**, maximum **0.035105**.
- Top-choice agreement **99.608%**.
- Fixture PPL **1.083749 -> 1.087393** (ratio 1.003362).

Simply disabling existing CUDA fusion without enabling streaming also changes the control distribution: mean KLD **0.000882**, maximum **0.022226**, top-choice agreement 100%. This shows sensitivity to execution/numerical choices, but does **not** prove the streamed discrepancy harmless or establish its precise cause. Log-probability serialization itself introduces a small comparison floor; the repeated control's maximum KLD was 0.000041.

A further matched test disabled fusion in both the saved control and streamed candidate. The discrepancy persisted: mean KLD **0.000977**, maximum **0.058453**, probability RMS difference **1.126%**, with 100% top-choice agreement on this fixture. Thus merely disabling fusion does not resolve the issue. See `ornith-nofusion-save.log` and `ornith-nofusion-compare.log`.

Therefore operator checks do not suffice for model-level acceptance. The feature remains experimental. Raw controls are `ornith-numerical-*.log`; binary comparison fixtures are kept locally and excluded from source artifacts.

## Preliminary end-to-end performance

These are **synthetic `llama-bench` prefill throughput** results on the V100, not real coding prompts or TTFT. Control and candidate use the same newly built executable with streaming disabled/enabled, not a separately rebuilt pristine executable. Both use `GGML_CUDA_VOLTA_FORCE_MMQ=moe`. Existing whole-weight prefetch was not enabled, and a comparison against its best configuration remains pending.

Shared parameters: prompt 2048, batch 2048, microbatch 1024, 16 CPU threads, pinned loading (`-lm none`), Flash Attention on, Q8_0 K/V, layer split, three timed repetitions per process. Placement is held fixed. The GPU lock prevents overlapping GPU benchmarks; binary SHA-256 records are in `tested-binaries-v1.sha256`.

| Model | Host-expert layers | Disabled PP tok/s | Enabled PP tok/s | Change | Evidence |
|---|---:|---:|---:|---:|---|
| Ornith AD-Q5_K-Q4_K | 24 | 888.704 | 972.485 | +9.43% | ABBA, 3 repeats/process |
| Qwen3.8-Flash-Next UD-IQ4_XS | 36 | 204.184 | 303.474 | +48.63% | Initial AB, 3 repeats/process |

Ornith per-process PP: A1 888.677, B1 968.718, B2 976.251, A2 888.731. The model fits the V100 without offload; 24 host layers were deliberately configured to test the requested scenario.

Ornith's short, separate 64-token generation samples were noisy: A1 51.825, B1 47.024, B2 54.461, A2 53.588 tok/s. Pooled means were 52.707 -> 50.743 (-3.73%). This is **not accepted as regression-free**, nor enough to attribute the difference confidently. No Next decode benchmark was completed in this set.

Next's verbose log confirms 864 streamed projection calls and 33.203 GiB of cumulative H2D traffic across the run, with zero pageable staging bytes. Prefill control/candidate raw files: `next-e2e-a1.*`, `next-e2e-b1.*`. Ornith raw files: `ornith-e2e-{a1,b1,b2,a2}.*`. Tool job provenance records the exact environment for the original runs; the nonverbose Ornith logs suppress streaming informational messages.

## Memory accounting

The operator probe's `sched_gpu_mib` excludes the separate streaming pool and all other process allocations. It also deliberately retains a resident reference matrix for correctness, so its total process VRAM is not a deployment memory measurement.

Example: Ornith gate matrix, N=512. Control scheduler allocation 152 MiB (144 MiB weights + output). Streamed scheduler allocation 8 MiB + separate 64 MiB pool = 72 MiB for these categories. Next layer-0 gate: 343.75 MiB matrix; streamed scheduler 12.5 MiB + 64 MiB pool = 76.5 MiB.

In the complete Next run, scheduler compute buffers were **1211 MiB -> 1020 MiB**, but streaming adds 64 MiB externally: net **127 MiB saved in these categories**, not 191 MiB. Model residency, KV/SSM, CUDA runtime and dynamic MMQ scratch must be added. The full intermediate FFN activation footprint is not yet bounded by the expert pool. Peak total memory and automatic `--fit` budgeting are not implemented/validated.

## Reproduction

Build from this experimental worktree:

```sh
cmake -S . -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES='70;75' -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=OFF
cmake --build build --target llama-bench llama-perplexity test-moe-stream -j12
```

Example Ornith benchmark (explicitly V100-only; does not touch server configuration):

```sh
CUDA_VISIBLE_DEVICES=GPU-1d64d3a4-4aea-87ec-9048-33a5217efd79 \
GGML_CUDA_VOLTA_FORCE_MMQ=moe \
GGML_CUDA_MOE_STREAM=1 \
GGML_CUDA_MOE_STREAM_GROUP=16 \
GGML_CUDA_MOE_STREAM_MIB=64 \
GGML_CUDA_MOE_STREAM_TRACE=1 \
./build/bin/llama-bench \
  -m /models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf \
  -ngl 99 -ncmoe 24 -p 2048 -n 64 -b 2048 -ub 1024 \
  -lm none -fa on -ctk q8_0 -ctv q8_0 -t 16 -r 3 -o json -v
```

Set `GGML_CUDA_MOE_STREAM=0` for the control. For Next, use the first shard path and `-ncmoe 36`; the original Next run used `-n 0`. Use exclusive GPU coordination for timings. The example is experimental, not a recommendation for production or the fastest overall model placement.

Operator suites (new unique output tags are required):

```sh
python3 benches/moe-stream-0914/run_probes.py --gpu v100 --suite correctness --tag check-v100-new
python3 benches/moe-stream-0914/run_probes.py --gpu 2080ti --suite correctness --tag check-2080-new
```

## Remaining acceptance work

Resolve full-model numerical differences with captured real hidden states, layer-by-layer comparisons and matched fusion controls. Validate repeated prefill/decode transitions and longer decode samples. Repeat Next in reverse order and with representative prompts. Compare against the best existing prefetch path and an untouched baseline. Account for the pool in the model fitting budget and measure total peak VRAM.

Only then progress to whole-FFN expert groups: reuse routing and compatible input quantization across up/gate/down, bound intermediate activations, and tune group size/microbatch under the same total VRAM cap. No universal best group size or final speedup is claimed.

## Final build/cleanup check

The final source moves stream-state destruction after the backend's pre-existing global capture lock and guards the NVIDIA-specific implementation from HIP/MUSA compilation. The CUDA build succeeded after this cleanup. Targeted V100 pageable/group-7 and RTX 2080 Ti IQ4_NL down-projection checks were repeated and passed exactly. A one-token V100 control with the feature requested took the original nonstreaming path and matched its resident reference exactly. Full 60-case suites and whole-model throughput/numerical figures above refer to the preceding binary; the cleanup does not change the NVIDIA arithmetic path. Both binary sets have recorded hashes. No production merge or remote push was performed.

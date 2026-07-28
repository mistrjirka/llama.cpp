# Reddit draft: fixed-topology MoE expert cache

## Suggested title

Experimental llama.cpp MoE cache: reducing 302 CPU/GPU scheduler splits to 152 improved GLM-5.2 decode by up to 56%

## Draft post

I have been experimenting with running a roughly 222 GiB quantized GLM-5.2 MoE model on a 32 GiB NVIDIA V100 using CPU fallback plus a GPU expert cache.

The surprising bottleneck was not mainly copying expert weights. The legacy mutable cache exposed four CPU/GPU graph segments per MoE layer:

```text
CPU route policy -> GPU resident experts -> CPU cold experts -> GPU continuation
```

Across 75 MoE layers this produced 302 scheduler splits and about 301 CPU/GPU transitions per generated token.

I added an experimental fixed-topology path with a persistent GPU `expert -> compact cache slot` map. It keeps resident-route lookup and hot/cold merging inside a stable graph and reduces decode to 152 scheduler splits.

### Controlled decode A/B

Same model, GPU, context, warmed expert set, forced token stream, and exact output hash:

| Configuration | Fixed topology | Legacy mutable | Change |
|---|---:|---:|---:|
| CUDA Graphs off | 5.065 tok/s | 3.595 tok/s | +40.9% |
| CUDA Graphs on | 6.339 tok/s | 4.060 tok/s | +56.1% |

A final retained-source rerun reached 6.350 tok/s.

### Horizontal llama.cpp baselines

On a separate 256-token request:

- fixed-topology expert cache: **6.40 tok/s**;
- modern llama.cpp `--fit` with tensor-level CPU overrides: **5.04 tok/s**;
- pure whole-layer placement at the maximum fitting `-ngl 11`: **3.59 tok/s**;
- legacy mutable expert cache: **4.17 tok/s**.

These natural greedy runs did not produce identical complete output hashes, so I am treating this as a speed comparison rather than an exact-parity test.

An important nuance: current `--fit` in this branch is not pure whole-layer splitting. It used tensor overrides while keeping 79/80 layers logically offloaded. I separately found the true whole-layer capacity by disabling fit: `-ngl 11` succeeds with a 28.24 GiB model buffer, while `-ngl 12` OOMs on the compute-buffer allocation.

### How long has it run?

- Exact forced-token parity: 48-token benchmark sequence.
- Longer runtime test: a 256-token request completed 255 decode evaluations at 6.40 tok/s versus 4.17 tok/s for the legacy mutable graph.
- The unforced greedy outputs diverged over the longer run, so I am treating that as runtime stability/performance—not exact long-run numerical parity.

### Prompt processing

On a 1,793-token coding prompt:

- modern tensor-level `--fit`: **8.95 prompt tok/s**;
- pure whole-layer `-ngl 11`: **8.47 prompt tok/s**;
- fixed topology: **8.17 prompt tok/s**;
- legacy mutable: **8.15 prompt tok/s**.

So the fixed topology's advantage is decode-specific in the current implementation. Pure whole-layer placement was about **3.67% faster** for this batched prompt workload, and modern tensor-level fit was about **9.55% faster**.

The fixed topology mainly targets token-by-token decode. Batched prompt processing amortizes scheduler overhead differently, so I measured it separately instead of using the forced-token decode harness.

### Important limitation

This is not production-ready dynamic replacement yet.

The validated mode has one asynchronously warmed resident expert per MoE layer and disables decode-time admissions. Multi-slot mutation still has a CUDA correctness issue: Compute Sanitizer eventually sees invalid indices reaching a later `SET_ROWS` kernel. Sampled expert uploads themselves match byte-for-byte, so the unresolved problem appears to be upstream multi-route numerical/routing parity rather than an out-of-bounds weight copy.

The feature is opt-in and the README command explicitly disables unsafe replacement.

### What is in the branch

- fixed-topology graph implementation;
- persistent device expert-slot maps;
- exact CPU miss masking;
- CUDA Graph-compatible asynchronous warm-start;
- run guide and environment variables;
- benchmark methodology and result summaries;
- known failures and sanitizer findings.

The next step is restoring numerical parity for multiple simultaneous resident routes. Once that is correct, the learned expert predictor/MLP can be connected to the 152-split topology instead of paying the old coordination penalty.

# Merge validation: September 16, 2026

The request-wide MoE implementation was snapshotted in `045046939` and merged with the remote `v100-optimized` tip `fad1bf938`. Source merge: `b90d03cb3`. This preserves the newer Volta conversion kernels and upstream sync rather than replacing them with the older experiment tree.

The substantial original experiment worktree and the existing local target worktree were left intact. Only the isolated integration worktree was committed. `results/merge-validation.json` records code hashes; no model weights, saved prefix states, raw logits or profiler databases were added to Git.

## Build and tests

A fresh Release CUDA 12.9 build with SM70 and SM75 targets completed for `llama-server`, `llama-cli`, `llama-bench`, the four targeted test programs and `test-backend-ops`. All four registered checks passed: request admission/boundaries, resident workspace gather/scatter/reuse, short expert-tail arithmetic, and aliased-input fused router execution. See [CTest output](results/unit-tests.txt).

The README's Qwen3.8-27B single-GPU server commands were exercised with the real Q5_K_XL model, a 131,072-token context on V100 and a 32,768-token context on RTX. The test used free local ports instead of 8080 and requested 32 output tokens. Two HTTP chat requests per GPU returned 200 with output tokens; the second reused part of the prompt. The experimental request-wide path was not used in these ordinary 27B launches. See [HTTP results](results/qwen-server-smoke.json).

The portable Flash-Next runner completed 21 cached-prefill requests across standalone V100, standalone RTX and the mixed pair. Eight prompt score rows and eight fixed-continuation score rows repeated byte-for-byte within each attention setting. These are repeatability checks, not proof of equal model quality between settings or hardware layouts.

The README's default 4,096-token test also completed seven requests on V100 with an empty prompt cache. Its default Python fixture is long enough for that input. Checked sparse-off/on scores were identical because the new long-history attention path does not activate there. [Default-run results](results/merged-v100-readme-default/summary.json).

## Fresh timings on merged source

Model: Qwen3.8-Flash-Next UD-IQ4_XS. K/V: Q8_0. A compatible saved 100,000-token prefix is restored, then 1,000 C++ input tokens are processed. Mixer chunk and expert tile are 1,024; context capacity is 131,072. Canonical expert weights stay in host memory, with per-preset resident caches. This is not a fully VRAM-resident model.

Both columns use the same request-wide executor. The comparison changes only the selected-entry attention toggle. It does not compare against upstream or the original chronological executor.

| GPU layout | Previous attention | Selected-entry attention | Sparse input tokens/s | Less processing time |
|---|---:|---:|---:|---:|
| V100 32 GB | 4.796 s | 3.825 s | 261.4 | 20.3% |
| RTX 2080 Ti 22 GB | 5.029 s | 4.159 s | 240.4 | 17.3% |
| V100 + RTX 2080 Ti | 3.971 s | 3.132 s | 319.3 | 21.1% |

Each layout runs off/off/on/on/off/on/off. Medians exclude the first observation of each setting, leaving three control and two sparse measurements. Model loading, saved-prefix restoration, first cache population and eight fixed continuation inputs are outside the reported input-processing interval. Individual timings, loaded-library hashes and commands are in [results](results/).

The README's new Flash-Next chart uses [graph.json](results/graph.json), derived from these merged-source checks, not the earlier portability timings. The older [hardware report](HARDWARE.md) remains an archival result. Small differences between the two campaigns are not a controlled before/after merge comparison. The September 12 upstream graph remains separately dated and backed by its original source data.

## What remains experimental

The request-wide executor and selected-entry attention are opt-in. No default was changed to enable either. The inherited FP32-request/FP16-value-accumulator mismatch, model-quality effects and the third-device base-residency limitation remain unresolved. Three-physical-V100 scaling, broad quality, arbitrary multi-slot Flash-Next serving and all possible context sizes are not certified by this merge.

The earlier sanitizer and all-layer numerical results in [NUMERICS.md](NUMERICS.md) describe the frozen diagnostic runtime. This merge campaign did not rerun that entire sanitizer/reference campaign. The checks here establish that the merged source builds and runs the tested configurations, not that every numerical question is settled.

The benchmark manifests captured the Git revision visible during the build/checks. Some cached runs predate the merge commit object and therefore show `045046939` with a dirty tracked tree. Their binaries were already built from the merged source. The source hashes in `merge-validation.json` identify that source independently of commit-time metadata.

# 2026-09-11 upstream comparison

Current upstream: `b0dcb8192` (`server: fix speculation after an image (#28715)`).
Fork base before this update: `1fb6376b9` on `v100-optimized`.

All Qwen rows use `Qwen3.8-27B-UD-Q5_K_XL.gguf`, Q8 K/V, FlashAttention, MTP off, and one server slot. Cold comparisons use the same `--batch-size 4096 --ubatch-size 2048` on both implementations.

## Qwen cold prompt processing

| hardware | 1k upstream | 1k fork | change | 16k upstream | 16k fork | change |
|---|---:|---:|---:|---:|---:|---:|
| V100 32 GB | 857.45 | 901.59 | +5.15% | 849.80 | 921.07 | +8.39% |
| RTX 2080 Ti 22 GB | 670.78 | 923.28 | +37.64% | 641.91 | 923.07 | +43.80% |
| V100 + RTX 2080 Ti | 982.50 | 1085.45 | +10.48% | 1031.86 | 1239.70 | +20.14% |

## Qwen 100k cached + 1k append

The cached test uses 409600 context with static YaRN (`1.5625`, original context 262144), tensor split `4,5`, V100 + RTX 2080 Ti, and an independently generated 100000-token slot state for each implementation.

| metric | upstream | fork | change |
|---|---:|---:|---:|
| prompt processing | 408.79 tok/s | 691.80 tok/s | +69.23% |
| TTFT | 2.497 s | 1.499 s | -39.97% |
| 64-token decode | 24.02 tok/s | 26.68 tok/s | +11.1% |
| 64-token end-to-end | 5.131 s | 3.867 s | -24.6% |

The first repetition is excluded from means. First-token output is deterministic within each arm. The attention operator was separately checked against the CPU reference at the real D256/GQA6/KV101120/Q1000 geometry before merge.

## Batch/ubatch notes

A short follow-up sweep on the merged fork found:

- V100-only, 16k PP: ub1024 876.49, ub2048 923.31, ub4096 952.18 tok/s.
- RTX 2080 Ti-only, 16k PP: ub1024 890.95, ub2048 923.63, ub4096 930.62 tok/s.
- V100 + RTX 2080 Ti, 400k context: ub1024 1157.40 and ub2048 1222.42 tok/s. ub4096 started faster but left about 1 GiB free on the 2080 Ti, so 2048 remains the safer serving default for this large-context pair.

For the README, `4096/2048` is therefore the conservative default. V100-only users with headroom can test ub4096.

## Ornith control

Model: `Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf`. For this control both upstream and fork were built with global `GGML_CUDA_FORCE_MMQ=ON`.

| hardware | 1k upstream | 1k fork | 16k upstream | 16k fork |
|---|---:|---:|---:|---:|
| V100 32 GB | 1207.05 | 1038.91 | 1273.00 | 1002.24 |
| V100 + RTX 2080 Ti | 1415.37 | 1362.93 | 1754.81 | 1512.97 |

This is a control, not the recommended fork deployment. The normal fork build supports `GGML_CUDA_VOLTA_FORCE_MMQ=moe`, which limits forcing to routed MoE matmuls. The 25 GiB Q6/Q5 model does not fit fully on the 22 GB RTX 2080 Ti alone.

## Files

The directory contains the retained JSON/raw files for the tables above, plus `bench.py` and validation logs. Experimental sweeps not used in the README should not be treated as headline results.

## Final merged-tree check

After making the validated V100/Turing long-Q8 dispatches and SM75 INT8-QK route default-on inside their existing geometry guards, the final merged library was rebuilt for SM70+SM75. With the old performance env switches unset, the 100k+1k run settled at **693.68 tok/s over the last four warm samples** (the first two requests were warmup-sensitive). The exact KV101120/Q1000 CPU-reference test passed on both V100 and RTX 2080 Ti with the final library.

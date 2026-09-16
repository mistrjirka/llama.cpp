# MoE router fusion: cross-model V100 check

Date: September 16, 2026

This isolates the runtime `--moe-router-fusion` switch on a standalone Tesla V100-SXM2-32GB. Each model processes the same 4,096-token text prompt. The first round is discarded and the median of six measured rounds is reported. `GGML_CUDA_VOLTA_FORCE_MMQ=moe` is held constant.

| Model | Fusion off | Fusion on | Throughput change | Fused route exercised? |
|---|---:|---:|---:|---|
| Ornith 1.5 35B-A3B AD-Q6_K-Q5_K | 2560.5 tok/s | 2558.0 tok/s | -0.10% | yes (2240 calls across 7 rounds) |
| Gemma 4 26B-A4B UD-Q4_K_XL | 2595.1 tok/s | 2625.4 tok/s | +1.17% | yes (840 calls across 7 rounds) |
| Gemma 4 E2B Q4_K_M | 7145.6 tok/s | 7203.0 tok/s | n/a | no (0 eligible calls) |

Gemma 4 26B-A4B shows a small but repeatable improvement in this workload. Ornith is effectively neutral in this stricter runtime-flag rerun. Gemma 4 E2B never presents a graph eligible for this fusion, so its small timing difference is noise rather than a router-fusion effect.

For Ornith and Gemma 26B, fusion-on and fusion-off keep the same final top token in every repeated run, but the full final-logit hashes differ between the arithmetic paths. Within each setting the hash is stable across all seven repetitions. Gemma E2B has identical hashes because the fused path is not exercised.

These results measure router fusion only. Request-wide layer-first execution has not yet been ported to Ornith/Gemma, and selected-entry attention is only relevant to models that provide a compatible native sparse-history selection.

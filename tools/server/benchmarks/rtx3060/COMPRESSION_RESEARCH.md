# Compression research for the hub RTX 3060

Reviewed 2026-11-24. The target is SenseNova's existing Q8_0 attention cache and F32 recurrent checkpoints. Lossless means restoring every bit of those representations. It does not mean converting F32 to BF16 or preserving only a benchmark score.

## Recommendation

Screen reversible preprocessing on the saved CPU snapshots first. Prioritize F32 exponent separation and byte/bit shuffling, then separate Q8 scales from integer values. Do not implement a new GPU codec until the measured ratio and total encode/decode cost justify it.

The research does not establish a new production win. The root experiment measured LZ4 with no useful reduction and Zstd-1 with approximately 3.3% Q8 KV reduction and 6.0% recurrent-checkpoint reduction. Encoding was approximately 1 GB/s for Q8 and 0.7 GB/s for recurrent data on the Intel N95. These are local sample results, not paper results. The measured host-to-GPU link is about 0.787 GB/s on PCIe 3.0 x1.

Compressing parked snapshots on the CPU saves RAM. If they are decompressed on the CPU before restoration, the same uncompressed bytes still cross PCIe. Reducing PCIe traffic requires transferring compressed bytes and decoding them on the GPU. Reducing active VRAM further requires attention or recurrent kernels that consume a compressed representation, or a suitable page-management design. These are three different changes.

## Exact methods

| Method and primary source | Published scope | Fit for this hub |
| --- | --- | --- |
| [ZipNN](https://arxiv.org/html/2411.05239v1), especially Sections 3 and 5 | Separates floating-point exponents and byte groups, then entropy-codes them. One ordinary F32 weight sample occupied 83.2% of its original size with ZipNN versus 92.3% with raw Zstd. The speed tests used one Apple M1 Max core, not an N95. | Best small CPU experiment for F32 checkpoints. Weight measurements do not establish recurrent-state compression ratios. |
| [SplitZip](https://arxiv.org/html/2605.01708v2) | Exact BF16 KV transfer codec. A calibrated exponent codebook and sparse escapes produce a 1.324x ratio on its main BF16 workload. Main measurements used H200, Qwen3-32B, and server interconnects. Additional A100 codec tests are reported. | Borrow exponent separation. Its BF16 codec does not directly encode Q8_0 integer blocks, and a full GPU port is premature. |
| [GAAC](https://arxiv.org/html/2610.38210v2) | Exact BF16 cache plus attention that decodes tiles on demand. About 1.405x payload compression, around 26% resident-cache reduction. Main evaluation used DGX Spark/GB10 and Apple M4, with W4A16 models and BF16 KV. | Relevant architecture for a future compressed-attention kernel. Its approximately 11.78 resident bits/value exceed the hub's Q8_0 8.5 bits/value. Switching to that format would increase cache size. |
| [Bitshuffle implementation](https://github.com/kiyo-masui/bitshuffle) | Reversible bit transposition before LZ4 or Zstd. It groups corresponding bits of typed elements. | Small CPU screening experiment for F32. Measure transformation and reversal time too. |
| [fpzip](https://computing.llnl.gov/projects/fpzip) and [zfp reversible mode](https://zfp.readthedocs.io/en/latest/modes.html) | General floating-point compressors with lossless modes. Their usefulness depends on numerical structure and spatial correlation. | Secondary screens if simple preprocessing succeeds. Use full-precision fpzip or explicit reversible zfp, never a tolerance or reduced precision. |

[ZipServ](https://arxiv.org/html/2603.17435v1) and DFloat11-style methods primarily compress BF16 weights. They do not establish further compression of this already Q6_K model, or directly solve Q8_0 cache storage. Their fused-decompression design is relevant, but their weight-size headline should not be applied to this model.

## Approximate methods

These methods can save more memory, but change the target computation unless an unchanged target subsequently verifies the result.

| Method and primary source | What changes and measured conditions | Decision for the exact-memory track |
| --- | --- | --- |
| [KIVI](https://arxiv.org/html/2402.02750v2) | Quantizes keys per channel and values per token to 2 bits, retaining a recent high-precision region. Its 2.35-3.47x throughput result used Llama-2-7B on A100 80GB and larger admitted batches. | Not equivalent to selecting a llama.cpp Q2 type. It needs a different layout/kernels and target-quality validation. Batch-serving gains do not predict single-request speed. |
| [KVQuant](https://arxiv.org/html/2401.18079v3) | Uses pre-RoPE keys, nonuniform quantization and separate outliers. Reports under 0.1 perplexity increase at 3 bits on tested LLaMA/Mistral models. Kernel timings used an A6000. | Approximate and calibration-dependent. Existing Q4_0/Q8_0 kernels do not implement its algorithm. |
| [CacheGen](https://arxiv.org/html/2310.07240v3) | Combines lossy quantization with lossless arithmetic coding of the resulting symbols. The paper also tests entropy coding after existing 8-bit quantization. Its hardware was a four-A40 server, with constrained network bandwidth. | The entropy-only idea can preserve the existing representation. Do not label the full quantization pipeline lossless. Its channel-aware coding motivates testing separated Q8 streams. |
| [KVzip](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4eaa4b8f2d08edb3f0af990d56134ea-Abstract-Conference.html) / [KVzap](https://arxiv.org/html/2601.07891v1) | Remove selected token entries. KVzip reports 3-4x logical-cache reduction and about 2x FlashAttention decode improvement on tested transformer models. KVzap trains a predictor and explicitly leaves engineering actual GPU-memory and wall-clock gains for future work. | Pruned tokens are absent from target attention. Requires quality tests and variable-length per-head storage. No exact byte restoration. |
| [MiniCache](https://arxiv.org/html/2405.14366v1) | Approximately merges adjacent attention layers. The headline 5x throughput used Llama-2-7B, A100 80GB, batch 128 and 4-bit storage against FP16. | Weak fit for sparse attention layers interleaved with 42 recurrent layers. Similarity is not byte identity. |
| [DAMP](https://arxiv.org/html/2608.27513v1) | Mixed FP16/INT8 recurrent state with offline channel selection and fused updates. Reports 69.1% state reduction at 9.9 bits/value on Qwen3.6-35B and Kimi-Linear-48B. Speed measurements use batch sizes 32-256 in SGLang. | A relevant longer-term GDN project, but approximate, not an exact checkpoint codec. Repeated state rounding propagates through later updates. Its model and batch results do not validate SenseNova batch one. |

Lower precision in the MTP draft is a different scope: it changes proposals while the original target verifier remains in place. That does not justify quantizing the target's recurrent state. The root's draft-Q4 experiment is separate from these target-compression proposals.

## Current representation and backend limits

The fork's `block_q8_0` stores one 2-byte FP16 scale and 32 signed 8-bit values in 34 bytes. This is 8.5 bits/value. Scale bytes are only 5.88% of the payload, so compressing scales alone has a small ceiling. The dominant integer stream must also become more compressible for a substantial gain.

F32 recurrent tensors have 1 sign bit, 8 exponent bits and 23 fraction bits. Separating these fields can expose exponent redundancy without changing a value. If only exponents compress, the 24 sign/fraction bits still impose a substantial size floor. Do not apply BF16's compression factor to F32.

The inspected CUDA FlashAttention dispatch supports the existing Q4_0 and Q8_0 layouts. It does not provide the custom GAAC/KIVI/KVQuant/pruned-cache layouts or a DAMP compressed-state executor. A paper implementation in Triton, vLLM or SGLang is not an immediately available llama.cpp setting.

## Bounded offline screening plan

1. Use the existing validated snapshot parser. Separate typed Q8 tensors, F32 recurrent tensors, and metadata. Do not reinterpret whole checkpoint files as floats.
2. F32: compare raw, four byte planes, bit planes, and exact exponent separation. For raw word `u`, use `e=(u>>23)&255` and `r=(u&0x7fffff)|((u>>31)<<23)`. Store `e` and the three bytes of `r`. Reverse these bit operations exactly, including NaNs and signed zero.
3. Q8: separate each 34-byte block into its 2 scale bytes and 32 integer bytes. Compare this with an exact signed-to-unsigned zigzag transform and bit planes. Preserve incomplete block tails raw.
4. Optional XOR prediction must operate on raw integer bit patterns. Floating-point subtraction followed by addition is not guaranteed to restore bits. Avoid dependencies on other checkpoints in this first screen.
5. Use 128 KiB blocks first, then up to 4 MiB if useful. Include headers, per-block modes and raw fallbacks in the ratio. Bound scratch, especially `unpackbits` expansion.
6. Measure preprocessing plus compression, and decompression plus inverse transformation. Report peak RSS and exact byte comparison. Keep per-block raw storage whenever encoding grows the block.
7. Compare hash-identical checkpoint payloads across `a`, `a2`, `b`, `b2` before designing payload interning. Keep `id_task` and other checkpoint metadata separate. Do not assume equal tokens imply equal bytes.
8. Run CPU screens only in the coordinated quiet window, then benchmark the strongest candidate through actual snapshot save/restore. A ratio alone is not a deployment result.

For GPU transfer screening, 110,080 target tokens occupy 3,353,477,120 bytes of Q8 KV. At 0.787 GB/s, reloading them takes at least about 4.26 seconds. GPU-side decompression could reduce this traffic, but CPU-only parked-cache compression does not. Full-runtime measurements must decide whether saved RAM or transfer time offsets codec work.

# Exact compression screening on the RTX 3060 hub

## Result

F32 checkpoint preprocessing improves lossless compression, but the measured gain is modest for the whole server. Byte shuffling or exponent separation followed by Zstd level 1 saved about 14.3% of sampled recurrent tensor bytes. This is an offline candidate, not an enabled runtime feature.

Q8 KV preprocessing saved at most 4.7% and added roughly 1.6 seconds to encode each GiB plus 1.0 second to decode it on the hub CPU. Reject it for the hot cache. The scalar bitshuffle prototype was slower and gave less useful results than the best F32 alternatives. Do not interpret this as a measurement of a tuned SIMD bitshuffle implementation.

## Method

- Device: the hub's Intel N95 CPU. The RTX 3060 stayed idle.
- Inputs: existing local `a.bin`, `a2.bin`, and `aux.bin` slot captures. No private payload bytes were copied into the repository.
- Parsing: known v4 contiguous Q8 KV serialization, standard recurrent state, and SCKP v1 appendix. The parser validates the complete layout and rejects unknown formats. It does not search blindly for magic bytes.
- F32 transforms apply only to parsed F32 tensor payloads. Headers remain outside the transformed samples.
- Q8 samples and blocks align to the 34-byte Q8_0 block: two scale bytes and 32 quantized values.
- Distinct sampled payload: 25.125 MiB of recurrent F32 and 76.144 MiB of Q8 KV. Each comparison uses the same sample windows.
- Blocks: 128 KiB and up to 4 MiB. Recurrent tensor rows in these captures are at most 2 MiB, so that is the actual maximum F32 block size in the larger-block comparison.
- Each block has one warmup and three measured repetitions. Codec contexts are reused. The table sums median block times and normalizes by sampled MiB.
- Timings include preprocessing and its inverse. They exclude file I/O, buffer allocation, and the final byte comparison.
- Compression falls back to the original raw block when the compressed payload plus 16 bytes of metadata would be larger.
- All 120 real-data combinations restored every byte exactly. Additional checks covered random data, all int8 values, F32 NaN bit patterns, infinities, signed zero, subnormals, and partial block tails.
- Peak process RSS: 47.60 MiB. Peak counted scratch plus codec contexts: 24.66 MiB. No runtime codecs or model kernels were changed.

## Selected measured results

All times are milliseconds per MiB and include the transform when present. Saved percentage includes the compressed-block metadata and raw fallback.

| Payload | Maximum block | Transform and codec | Saved | Encode ms/MiB | Decode ms/MiB |
|---|---:|---|---:|---:|---:|
| F32 recurrent | 128 KiB | Raw + Zstd1 | 6.02% | 1.395 | 0.902 |
| F32 recurrent | 128 KiB | Byte shuffle + Zstd1 | 6.25% | 1.631 | 1.299 |
| F32 recurrent | 128 KiB | Exponent split + Zstd1 | 7.59% | 2.279 | 1.371 |
| F32 recurrent | 128 KiB | Scalar bitshuffle + LZ4 | 11.16% | 3.432 | 4.313 |
| F32 recurrent | 2 MiB | Raw + Zstd1 | 6.16% | 1.510 | 0.858 |
| F32 recurrent | 2 MiB | Byte shuffle + Zstd1 | 14.25% | 1.547 | 0.715 |
| F32 recurrent | 2 MiB | Exponent split + Zstd1 | 14.30% | 1.572 | 0.685 |
| F32 recurrent | 2 MiB | Integer XOR + byte shuffle + Zstd1 | 13.37% | 1.514 | 0.691 |
| F32 recurrent | 2 MiB | Scalar bitshuffle + Zstd1 | 12.59% | 3.902 | 4.587 |
| Q8 KV | 128 KiB | Raw + Zstd1 | 3.27% | 1.314 | 0.894 |
| Q8 KV | 128 KiB | Scale split + Zstd1 | 3.52% | 1.443 | 0.980 |
| Q8 KV | 4 MiB | Raw + Zstd1 | 3.31% | 1.357 | 0.832 |
| Q8 KV | 4 MiB | Scale split + Zstd1 | 4.68% | 1.601 | 0.988 |
| Q8 KV | 4 MiB | Scale split + zigzag + Zstd1 | 4.68% | 1.610 | 0.988 |
| Q8 KV | 4 MiB | Scale split + scalar bitshuffle + Zstd1 | 1.53% | 3.497 | 4.236 |

LZ4 without preprocessing did not reduce either payload category. Q8 scale splitting with LZ4 saved only 0.66% at the larger block size. Zigzag provided no material additional gain. The complete 40-row aggregate and 120 individual measurements are in the adjacent results directory.

## Keep/reject decisions

- Keep exact F32 byte shuffling and exponent separation as measured research candidates. Their best ratios are nearly equal. Byte shuffling is simpler. An implementation would still need bounded capture/restore, cache-budget accounting, format handling, and a full request-latency comparison.
- Leave runtime compression disabled for this deployment. On the current 128 KiB block store, the attractive larger-block ratio does not apply directly.
- Reject Q8 scale splitting, zigzag, and scalar bitshuffle for the current hot cache. The memory gain is small compared with CPU cost and added representation handling.
- Reject integer XOR preprocessing as the preferred F32 choice. It saved less than byte shuffling or exponent separation, without a meaningful decode advantage.

For scale, applying the best sampled F32 ratio to one 87.94 MiB checkpoint would save approximately 12.58 MiB and add approximately 138 ms of encoding and 60 ms of decoding. Three checkpoints would save about 37.7 MiB. These are linear estimates from sampled tensor data, not measured whole-checkpoint or server gains. Shared checkpoints must be counted only once.

## Reproduce

Run when the CPU is quiet, outside inference timing:

```sh
c++ -O3 -march=native -std=c++17 -shared -fPIC \
  tools/server/benchmarks/cache-research/exact-preprocess.cpp \
  -o /tmp/libexact-preprocess.so
python3 tools/server/benchmarks/cache-research/screen-compression.py \
  /path/to/a.bin /path/to/a2.bin /path/to/aux.bin \
  --preprocess-lib /tmp/libexact-preprocess.so \
  --sample-mib 32 --block-kib 128 4096 --repeats 3 \
  --output /tmp/rtx3060-exact-compression.json
```

Omit `--preprocess-lib` to screen only ordinary LZ4 and Zstd1 on raw KV and checkpoint bytes. Use `--inspect` for metadata-only parsing.

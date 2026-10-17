#pragma once

// Host side of the NPU prefill path: bfp16ebs8 activation encoding and the unpermute of the fp32 tile outputs.
//
// bfp16ebs8 ("block floating point, 8 values share an exponent"): 9 bytes per 8 consecutive k of one token: [shared exponent E][8 x int8 mantissa],
// value = q * 2^(E - 133). Encoding is bit-exact with the lab model stage4/bfpmodel.py (enc_rne): round-to-nearest-even mantissa relative to the
// largest exponent of the 8 values, and if any mantissa overflows int8 after rounding the exponent is bumped by one and all 8 are re-rounded
// (this is what the NPU's to_v64bfp16ebs8 does).
//
// Activation tile ("slab") layout consumed by the prefill kernel (pf.cc): one slab = 64 tokens x 256 k = 18432 B =
//   [z 4][kb32 32][mb in pair 2] blocks of 72 B = 8 token rows x 9 B     (token block rb = 2*z + mb, k block kb32 of 8 k)
// A dispatch (one 64-token tile of one split) reads an A BO of   [hdr object 18432 B][nsb passes x nbk slabs]   (the slabs are repeated per
// superblock pass: stride-0 DMA repeats are not available).

#include <cstddef>
#include <cstdint>

#include "ggml-xdna2-layout.h"

namespace xdna2 {

// OpenMP team size for the backend's host loops (physical cores, or GGML_XDNA2_THREADS)
int pf_threads();

constexpr size_t PF_SLAB_BYTES = 18432;       // 64 tokens x 256 k in bfp16ebs8
constexpr int    PF_TILE_TOK   = 64;
constexpr int    PF_MAX_TILES  = 8;           // tiles per wave (512 tokens)

// A BO bytes of one tile: header object + the nbk slabs, stored once (the shim DMA re-reads the slabs for every superblock pass)
inline size_t pf_a_tile_bytes(int64_t nbk) { return (1 + size_t(nbk)) * PF_SLAB_BYTES; }
// Y bytes of one tile of a split: [col 8][nsb][core 4][GSB groups][1024 f32]
inline size_t pf_y_tile_bytes(int64_t nsb) { return size_t(8) * size_t(nsb) * 4 * layout::GSB * 1024 * sizeof(float); }

// one block of 8 floats -> 9 bytes [E][8 x int8]; reference (scalar, 64-bit) and the vectorized version used by the encoder (bit identical)
void pf_enc8_ref(const float * x, uint8_t * out);
void pf_enc8_fast(const float * x, uint8_t * out);

// Encode ONE tile (tokens [tile*64, +64) of n_tok rows of K floats; row t at x + t*row_stride_bytes) into `slabs`: [nbk][PF_SLAB_BYTES]
// (one pass; tokens >= n_tok are zero). Threaded over the k slabs.
void pf_encode_tile(uint8_t * slabs, const float * x, size_t row_stride_bytes, int64_t n_tok, int64_t k, int tile);

// Fill the A BO region of one tile (pf_a_tile_bytes) from its one-pass slabs: header object + the slabs.
// hdr: i32 [nsb, nbk, qtype, G (groups per core)].
void pf_build_a_tile(uint8_t * a_tile, const uint8_t * slabs, int64_t nbk, int64_t nsb, int qtype, int64_t groups);

// Copy the fp32 outputs of one tile of one split into dst (row j at dst + j*dst_row_stride bytes, g.n floats per row).
void pf_unpermute_tile(const float * y_tile, int tile, int64_t n_tok, const layout::Geom & g, int64_t split, float * dst, size_t dst_row_stride);

} // namespace xdna2

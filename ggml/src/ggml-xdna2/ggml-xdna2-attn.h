#pragma once

// Host side of the NPU flash-attention path (kernels/fa.cc + genfa.py): bfp16ebs8 encoding of the Q objects and of the bfp16 shadow of the
// KV cache, and the normalization + scatter of the outputs.  Shape: head dim 256, 16 q heads, 4 kv heads (q head h reads kv head h / 4).
//
// Objects are 4608 B = 64 tiles of 72 B (8 rows x [E][8 x int8], see ggml-xdna2-bfp.h):
//   Q object   (16 queries of one q head)  [dt 32][qt 2] tiles, row r = query qt*8 + r, values d = dt*8 .. +8   (pre-scaled by log2(e) * scale)
//   K block    (16 keys of one kv head)    [dt 32][kt 2] tiles, row r = key kt*8 + r,   values d = dt*8 .. +8
//   V block    (16 keys of one kv head)    [dt 32][kt 2] tiles, row r = d dt*8 + r,     values = keys kt*8 .. +8   (V^T)
// Shadow of one layer: K and V each [kv head 4][cap_rows / 16 blocks][4608 B].
// Output of one core (16 queries of one q head): f32 [qt 2][dt 32][8 q][8 d] unnormalized, then the row sums [qt 2][8 q][8 (equal)].

#include <cstddef>
#include <cstdint>

namespace xdna2 {

constexpr int     FA_D        = 256;
constexpr int     FA_NH       = 16;          // q heads
constexpr int     FA_NKV      = 4;           // kv heads
constexpr int     FA_BLK      = 16;          // keys per block = queries per q tile
constexpr size_t  FA_OBJ      = 4608;        // bytes of a Q object / K block / V block
constexpr int64_t FA_OC       = 2 * 32 * 64 + 128;   // f32 per core output
constexpr int     FA_PAIR_Q   = 32;          // queries of a pair (2 q tiles)
constexpr size_t  FA_PAIR_QB  = size_t(FA_NKV) * 8 * FA_OBJ;           // Q objects of one pair: [g 4][core 8]
constexpr size_t  FA_PAIR_OB  = size_t(8) * 4 * FA_OC * sizeof(float);  // output bytes of one pair: [col 8][core 4][FA_OC]

// One Q object: queries at positions pos0 .. pos0 + 15 of q head `head`; the query at position p is row p - p0 of q when 0 <= p - p0 < n_tok,
// otherwise it is encoded as zeros.  q row t of head h at (char *) q + t * tok_stride + h * head_stride.
void fa_encode_q(uint8_t * obj, const float * q, size_t tok_stride, size_t head_stride, int64_t n_tok, int64_t p0, int64_t pos0, int head,
                 float prescale);

// K and V blocks of block `blk` of kv head `head` from the cache views (row = cell, f16 or bf16): rows >= n_valid are zeros.
void fa_encode_kv_block(uint8_t * kblk, uint8_t * vblk, const void * k, const void * v, bool bf16, size_t row_stride, size_t head_stride,
                        int head, int64_t blk, int64_t n_valid);

// Fingerprint of the cache rows of one block (first and last row of K and V, all heads): detects blocks rewritten since they were encoded.
uint64_t fa_block_fingerprint(const void * k, const void * v, size_t row_stride, size_t row_bytes, int64_t blk);

// Normalize and scatter the outputs of `pairs` pairs (pair r covers positions pos0 + 32 r .. + 31) into dst: query position p -> dst row t = p - p0
// (only 0 <= t < n_tok), head h at (char *) dst + t * tok_stride + h * head_stride.  Threaded.
void fa_scatter_out(const float * out, int64_t pairs, int64_t pos0, int64_t p0, int64_t n_tok, float * dst, size_t tok_stride, size_t head_stride);

} // namespace xdna2

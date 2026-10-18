#pragma once

// Host side of the NPU flash-attention path (kernels/fa.cc + genfa.py): bfp16ebs8 encoding of the Q objects and of the bfp16 shadow of the
// KV cache, and the normalization + scatter of the outputs.  Shapes (FaShape): head dim D 128 or 256, 4 or 8 kv heads, up to 4 q heads per kv
// head (q head h reads kv head h / grp).
//
// Objects are 16 x D bfp16 = D/8 * 2 tiles of 72 B (8 rows x [E][8 x int8], see ggml-xdna2-bfp.h):
//   Q object   (16 queries of one q head)  [dt D/8][qt 2] tiles, row r = query qt*8 + r, values d = dt*8 .. +8   (pre-scaled by log2(e) * scale)
//   K block    (16 keys of one kv head)    [dt D/8][kt 2] tiles, row r = key kt*8 + r,   values d = dt*8 .. +8
//   V block    (16 keys of one kv head)    [dt D/8][kt 2] tiles, row r = d dt*8 + r,     values = keys kt*8 .. +8   (V^T)
// Shadow of one layer: K and V each [kv head][cap_rows / 16 blocks][obj].
// Output of one core (16 queries of one q head): f32 [qt 2][dt D/8][8 q][8 d] unnormalized, then the row sums [qt 2][8 q][8 (equal)].
//
// Core mapping (genfa.py): kv head g owns cpk = 8 / nkv columns; its 4 * cpk cores are (qi = column within the kv head: q tile of the unit,
// hh = row: q head within the group; cores with hh >= grp idle).  A dispatch unit covers 16 * cpk queries.

#include <cstddef>
#include <cstdint>
#include <string>

namespace xdna2 {

constexpr int FA_BLK = 16;          // keys per block = queries per q tile

struct FaShape {
    int d   = 256;                  // head dim
    int nkv = 4;                    // kv heads
    int nh  = 16;                   // q heads

    int     grp() const { return nh / nkv; }
    int     cpk() const { return 8 / nkv; }                              // columns per kv head
    int     nc() const { return 4 * cpk(); }                             // cores per kv head
    int     unit_q() const { return FA_BLK * cpk(); }                    // queries per dispatch unit
    int     ndt() const { return d / 8; }
    size_t  obj() const { return size_t(FA_BLK) * size_t(d) * 9 / 8; }  // bytes of a Q object / K block / V block
    int64_t oc() const { return 2 * int64_t(ndt()) * 64 + 128; }         // f32 per core output
    size_t  unit_qb() const { return size_t(nkv) * size_t(nc()) * obj(); }                   // Q objects of a unit: [g][core]
    size_t  unit_ob() const { return size_t(8) * 4 * size_t(oc()) * sizeof(float); }       // output bytes of a unit: [col 8][core 4][oc]
    std::string variant() const { return "d" + std::to_string(d) + "k" + std::to_string(nkv); }
    bool    supported() const { return (d == 128 || d == 256) && (nkv == 4 || nkv == 8) && nh % nkv == 0 && grp() >= 1 && grp() <= 4; }
};

// One Q object: queries at positions pos0 .. pos0 + 15 of q head `head`; the query at position p is row p - p0 of q when 0 <= p - p0 < n_tok,
// otherwise it is encoded as zeros.  q row t of head h at (char *) q + t * tok_stride + h * head_stride.
void fa_encode_q(const FaShape & s, uint8_t * obj, const float * q, size_t tok_stride, size_t head_stride, int64_t n_tok, int64_t p0, int64_t pos0,
                 int head, float prescale);

// K and V blocks of block `blk` of kv head `head` from the cache views (row = cell, f16 or bf16): rows >= n_valid are zeros.
// rot > 0: row r of the shadow is cell (r + rot) % n_cells (a sliding-window cache whose cells wrapped around, see AttnJob::rot).
void fa_encode_kv_block(const FaShape & s, uint8_t * kblk, uint8_t * vblk, const void * k, const void * v, bool bf16, size_t row_stride,
                        size_t head_stride, int head, int64_t blk, int64_t n_valid, int64_t rot = 0, int64_t n_cells = 0);

// Fingerprint of the cache rows of one block (first and last row of K and V, all heads): detects blocks rewritten since they were encoded.
uint64_t fa_block_fingerprint(const void * k, const void * v, size_t row_stride, size_t row_bytes, int64_t blk);

// Normalize and scatter the outputs of `units` units (unit r covers positions pos0 + unit_q r .. + unit_q - 1) into dst: query position p -> dst
// row t = p - p0 (only 0 <= t < n_tok), head h at (char *) dst + t * tok_stride + h * head_stride.  Threaded.
void fa_scatter_out(const FaShape & s, const float * out, int64_t units, int64_t pos0, int64_t p0, int64_t n_tok, float * dst, size_t tok_stride,
                    size_t head_stride);

} // namespace xdna2

#pragma once

// Device weight layout of the ggml-xdna2 backend for Q4_0 tensors.
//
// This header is dependency free (no ggml, no XRT) so that the layout can be unit tested on its own.
//
// The layout is a pure PERMUTATION of the bytes/nibbles of the GGUF Q4_0 tensor: the logical payload
// has exactly the same size as the GGUF data (18 bytes per block of 32 weights). Only padding is added,
// to reach the 16 rows x 32 cores granularity (N) and the 8-block record granularity (K).
//
// Source of truth: the working NPU kernel (xdna-decode-lab-20261014/stage0: prep_q4.py, gen_q4.py, q4.cc).
//
//   GGUF block_q4_0 (18 B): fp16 d | uint8 qs[16]; element j (0..15) = qs[j] & 15, element j+16 = qs[j] >> 4,
//                           weight value = (element - 8) * d.
//
//   matrix:  N rows (output features) x K columns (K % 32 == 0), KB = K / 32 blocks per row.
//   padded:  N_pad  = roundup(N, 512)   (32 cores x 16-row groups)
//            KB_pad = roundup(KB, 8)    (one record = 8 blocks along K)
//            padded blocks/rows have a zero fp16 scale (and zero nibbles), so they contribute exactly 0.
//
//   cores:   the NPU runs 32 cores (8 columns x 4 rows). Core c owns the contiguous row range
//            [c*RPC, (c+1)*RPC) with RPC = N_pad / 32, and its records are one contiguous byte range
//            [c*core_bytes, (c+1)*core_bytes) of the matrix region. Within a core the records are ordered
//            group-major: for group g (16 rows) -> for record r (8 blocks along K): rec = g*(KB_pad/8) + r.
//
//   record (2304 B = 16 rows x 8 blocks x 18 B), for the rows n = 0..15 of the group and blocks b = 0..7:
//            bytes [0, 2048)    nibbles. Per block b a 256 B tile at b*256, two 128 B halves h (k = h*16 + kk):
//                                  byte  b*256 + h*128 + kk*8 + n/2
//                               low nibble (bits 0..3) = row n even, high nibble (bits 4..7) = row n odd,
//                               value = element ^ 8  (= int4 two's complement of element - 8).
//            bytes [2048, 2304) fp16 scales, 8 blocks x 16 rows: byte offset 2048 + (b*16 + n) * 2 (little endian).
//
//   A tensor with ne[2]*ne[3] > 1 is stored as consecutive independent matrices (one per 2D slice).
//
// Any byte range of the GGUF stream can be written/read independently (needed for chunked set_tensor).

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace xdna2 {
namespace layout {

constexpr int64_t QK            = 32;    // weights per block
constexpr size_t  BLOCK_BYTES   = 18;    // sizeof(block_q4_0)
constexpr int     N_CORES       = 32;
constexpr int     GROUP_ROWS    = 16;
constexpr int     REC_BLOCKS    = 8;
constexpr size_t  REC_BYTES     = 2304;  // 128 blocks * 18
constexpr size_t  REC_NIB_BYTES = 2048;
constexpr size_t  REC_SC_BYTES  = 256;
constexpr int64_t ROW_GRAN      = int64_t(N_CORES) * GROUP_ROWS; // 512

static_assert(REC_BYTES == size_t(GROUP_ROWS) * REC_BLOCKS * BLOCK_BYTES, "record size");
static_assert(REC_NIB_BYTES + REC_SC_BYTES == REC_BYTES, "record split");

constexpr int64_t roundup(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

struct Q4Geom {
    int64_t n = 0, k = 0;      // logical rows / columns
    int64_t kb = 0;            // k / 32
    int64_t n_pad = 0;         // rows padded to 512
    int64_t kb_pad = 0;        // blocks padded to 8
    int64_t rpc = 0;           // rows per core
    int64_t groups_per_core = 0;
    int64_t recs_per_group = 0;
    size_t  core_bytes = 0;    // bytes of one core's contiguous record stream
    size_t  bytes = 0;         // padded bytes of one 2D matrix (N_pad * KB_pad * 18)
    size_t  row_bytes = 0;     // GGUF bytes of one row (kb * 18)
    size_t  gguf_bytes = 0;    // GGUF bytes of one 2D matrix (n * kb * 18)

    // byte offset of the record holding (row, block kb_idx), relative to the matrix start
    size_t rec_offset(int64_t row, int64_t kb_idx) const {
        const int64_t core = row / rpc;
        const int64_t rr   = row % rpc;
        const int64_t g    = rr / GROUP_ROWS;
        const int64_t rec  = g * recs_per_group + kb_idx / REC_BLOCKS;
        return size_t(core) * core_bytes + size_t(rec) * REC_BYTES;
    }
};

inline Q4Geom q4_geom(int64_t n, int64_t k) {
    Q4Geom g;
    g.n  = n;
    g.k  = k;
    g.kb = k / QK;
    g.n_pad = roundup(n, ROW_GRAN);
    g.kb_pad = roundup(g.kb, REC_BLOCKS);
    g.rpc = g.n_pad / N_CORES;
    g.groups_per_core = g.rpc / GROUP_ROWS;
    g.recs_per_group = g.kb_pad / REC_BLOCKS;
    g.core_bytes = size_t(g.groups_per_core) * size_t(g.recs_per_group) * REC_BYTES;
    g.bytes = g.core_bytes * N_CORES;
    g.row_bytes = size_t(g.kb) * BLOCK_BYTES;
    g.gguf_bytes = size_t(n) * g.row_bytes;
    return g;
}

namespace detail {

inline void put_nib(uint8_t * p, int shift, uint8_t v) {
    *p = uint8_t((*p & ~(0xF << shift)) | (v << shift));
}

// Locations of one block inside its record.
struct BlockPos {
    uint8_t * rec;   // record base
    int b;           // block within record (0..7)
    int n;           // row within group (0..15)
};

// byte offset of the nibble pair holding element e (0..31) of the block
inline size_t nib_off(int b, int n, int e) {
    const int h = e >> 4, kk = e & 15;
    return size_t(b) * 256 + size_t(h) * 128 + size_t(kk) * 8 + size_t(n >> 1);
}
inline size_t sc_off(int b, int n) { return REC_NIB_BYTES + (size_t(b) * 16 + size_t(n)) * 2; }

// write bytes [lo, hi) (0 <= lo < hi <= 18) of one GGUF block to the device layout
inline void pack_block_bytes(uint8_t * rec, int b, int n, const uint8_t * blk, int lo, int hi) {
    const int shift = (n & 1) * 4;
    for (int i = lo; i < hi; ++i) {
        if (i < 2) {
            rec[sc_off(b, n) + size_t(i)] = blk[i - lo];
        } else {
            const int j = i - 2;                 // qs index: low nibble = element j, high nibble = element j + 16
            const uint8_t v = blk[i - lo];
            put_nib(rec + nib_off(b, n, j),      shift, uint8_t((v & 15) ^ 8));
            put_nib(rec + nib_off(b, n, j + 16), shift, uint8_t((v >> 4) ^ 8));
        }
    }
}

inline void unpack_block_bytes(const uint8_t * rec, int b, int n, uint8_t * blk, int lo, int hi) {
    const int shift = (n & 1) * 4;
    for (int i = lo; i < hi; ++i) {
        if (i < 2) {
            blk[i - lo] = rec[sc_off(b, n) + size_t(i)];
        } else {
            const int j = i - 2;
            const uint8_t a = uint8_t(((rec[nib_off(b, n, j)]      >> shift) & 15) ^ 8);
            const uint8_t c = uint8_t(((rec[nib_off(b, n, j + 16)] >> shift) & 15) ^ 8);
            blk[i - lo] = uint8_t(a | (c << 4));
        }
    }
}

// full-block fast paths
inline void pack_block(uint8_t * rec, int b, int n, const uint8_t * blk) {
    const int shift = (n & 1) * 4;
    rec[sc_off(b, n)]     = blk[0];
    rec[sc_off(b, n) + 1] = blk[1];
    uint8_t * lo_half = rec + size_t(b) * 256 + (n >> 1);   // elements 0..15
    uint8_t * hi_half = lo_half + 128;                       // elements 16..31
    for (int j = 0; j < 16; ++j) {
        const uint8_t v = blk[2 + j];
        put_nib(lo_half + size_t(j) * 8, shift, uint8_t((v & 15) ^ 8));
        put_nib(hi_half + size_t(j) * 8, shift, uint8_t((v >> 4) ^ 8));
    }
}

inline void unpack_block(const uint8_t * rec, int b, int n, uint8_t * blk) {
    const int shift = (n & 1) * 4;
    blk[0] = rec[sc_off(b, n)];
    blk[1] = rec[sc_off(b, n) + 1];
    const uint8_t * lo_half = rec + size_t(b) * 256 + (n >> 1);
    const uint8_t * hi_half = lo_half + 128;
    for (int j = 0; j < 16; ++j) {
        const uint8_t a = uint8_t(((lo_half[size_t(j) * 8] >> shift) & 15) ^ 8);
        const uint8_t c = uint8_t(((hi_half[size_t(j) * 8] >> shift) & 15) ^ 8);
        blk[2 + j] = uint8_t(a | (c << 4));
    }
}

// Visit the bytes [off, off + size) of the GGUF stream of one 2D matrix, group by group.
// Different 16-row groups touch disjoint device bytes, so groups can run in parallel.
// op(row, kb_idx, lo, hi, byte_ptr_in_user_buffer_for_lo)
template <typename Op>
inline void for_range(const Q4Geom & g, size_t off, size_t size, Op op) {
    if (size == 0) return;
    const size_t end = off + size;
    const int64_t row0 = int64_t(off / g.row_bytes);
    const int64_t row1 = int64_t((end - 1) / g.row_bytes);
    const int64_t grp0 = row0 / GROUP_ROWS, grp1 = row1 / GROUP_ROWS;
    const bool par = (end - off) >= (size_t(1) << 20);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(par)
#endif
    for (int64_t gi = grp0; gi <= grp1; ++gi) {
        const int64_t r_lo = gi * GROUP_ROWS > row0 ? gi * GROUP_ROWS : row0;
        const int64_t r_hi = (gi + 1) * GROUP_ROWS - 1 < row1 ? (gi + 1) * GROUP_ROWS - 1 : row1;
        for (int64_t row = r_lo; row <= r_hi; ++row) {
            const size_t rs = size_t(row) * g.row_bytes;
            const size_t a  = off > rs ? off - rs : 0;                               // first byte inside the row
            const size_t z  = end < rs + g.row_bytes ? end - rs : g.row_bytes;       // one past last byte inside the row
            const int64_t kb_first = int64_t(a / BLOCK_BYTES), kb_last = int64_t((z - 1) / BLOCK_BYTES);
            for (int64_t kb = kb_first; kb <= kb_last; ++kb) {
                const size_t bs = size_t(kb) * BLOCK_BYTES;
                const int lo = a > bs ? int(a - bs) : 0;
                const int hi = z < bs + BLOCK_BYTES ? int(z - bs) : int(BLOCK_BYTES);
                op(row, kb, lo, hi, rs + bs + size_t(lo) - off);
            }
        }
    }
    (void) par;
}

} // namespace detail

// Write bytes [off, off+size) of the GGUF Q4_0 stream of ONE 2D matrix (src points at stream byte `off`)
// into the matrix region `dev` (g.bytes bytes). Untouched bytes (padding) are not modified.
inline void q4_0_pack_range(const Q4Geom & g, uint8_t * dev, const uint8_t * src, size_t off, size_t size) {
    detail::for_range(g, off, size, [&](int64_t row, int64_t kb, int lo, int hi, size_t src_off) {
        uint8_t * rec = dev + g.rec_offset(row, kb);
        const int b = int(kb % REC_BLOCKS), n = int(row % GROUP_ROWS);
        if (lo == 0 && hi == int(BLOCK_BYTES)) {
            detail::pack_block(rec, b, n, src + src_off);
        } else {
            detail::pack_block_bytes(rec, b, n, src + src_off, lo, hi);
        }
    });
}

// Inverse of q4_0_pack_range: bit exact.
inline void q4_0_unpack_range(const Q4Geom & g, const uint8_t * dev, uint8_t * dst, size_t off, size_t size) {
    detail::for_range(g, off, size, [&](int64_t row, int64_t kb, int lo, int hi, size_t dst_off) {
        const uint8_t * rec = dev + g.rec_offset(row, kb);
        const int b = int(kb % REC_BLOCKS), n = int(row % GROUP_ROWS);
        if (lo == 0 && hi == int(BLOCK_BYTES)) {
            detail::unpack_block(rec, b, n, dst + dst_off);
        } else {
            detail::unpack_block_bytes(rec, b, n, dst + dst_off, lo, hi);
        }
    });
}

inline void q4_0_pack(const Q4Geom & g, uint8_t * dev, const uint8_t * src)   { q4_0_pack_range(g, dev, src, 0, g.gguf_bytes); }
inline void q4_0_unpack(const Q4Geom & g, const uint8_t * dev, uint8_t * dst) { q4_0_unpack_range(g, dev, dst, 0, g.gguf_bytes); }

// ---------------------------------------------------------------------------------------------------
// Whole tensor (possibly several 2D slices) <-> byte ranges of the logical GGUF stream.
// slice s occupies GGUF bytes [s*gguf_bytes, (s+1)*gguf_bytes) and device bytes [s*bytes, (s+1)*bytes).
// ---------------------------------------------------------------------------------------------------
inline void q4_0_tensor_pack_range(const Q4Geom & g, uint8_t * dev, const uint8_t * src, size_t off, size_t size) {
    while (size > 0) {
        const size_t s = off / g.gguf_bytes, so = off % g.gguf_bytes;
        const size_t n = size < g.gguf_bytes - so ? size : g.gguf_bytes - so;
        q4_0_pack_range(g, dev + s * g.bytes, src, so, n);
        src += n; off += n; size -= n;
    }
}

inline void q4_0_tensor_unpack_range(const Q4Geom & g, const uint8_t * dev, uint8_t * dst, size_t off, size_t size) {
    while (size > 0) {
        const size_t s = off / g.gguf_bytes, so = off % g.gguf_bytes;
        const size_t n = size < g.gguf_bytes - so ? size : g.gguf_bytes - so;
        q4_0_unpack_range(g, dev + s * g.bytes, dst, so, n);
        dst += n; off += n; size -= n;
    }
}

} // namespace layout
} // namespace xdna2

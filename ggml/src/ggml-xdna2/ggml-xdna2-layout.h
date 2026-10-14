#pragma once

// Device weight layouts of the ggml-xdna2 backend.
//
// This header is dependency free (no ggml, no XRT) so that the layouts can be unit tested on their own.
//
// Every layout is a pure PERMUTATION of the bytes/nibbles/bits of the GGUF tensor: the logical payload has
// exactly the same size as the GGUF data. Only padding is added, to reach the 16 rows x 32 cores granularity (N),
// and, for the 32-element block formats, the 8-block record granularity (K).
//
//   Fmt::Q4_0   record 2304 B = 16 rows x 8 blocks   (stage0 kernel, bit identical to prep_q4.py)
//   Fmt::Q4_K   record 2304 B = 16 rows x 1 super-block   (stage2 kernel, bit identical to pack_qk.py)
//   Fmt::Q6_K   record 3360 B = 16 rows x 1 super-block   (stage2 kernel, bit identical to pack_qk.py)
//   Fmt::Q8_0   record 4352 B = 16 rows x 8 blocks   PROVISIONAL (no NPU kernel yet, see section below)
//   Fmt::Q5_K   record 2816 B = 16 rows x 1 super-block   PROVISIONAL (no NPU kernel yet, see section below)
//
// Matrix: N rows (output features) x K columns, KB = K / elems blocks per row (elems = 32 or 256).
//   N_pad  = roundup(N, 512)            (32 cores x 16-row groups); padded rows/blocks are all zero bytes, which
//                                       is a zero scale (and zero min) so they contribute exactly 0
//   KB_pad = roundup(KB, rec_blocks)
//
// Cores: the NPU runs 32 cores (8 columns x 4 rows). A "chunk" is the unit of one NPU dispatch (<= MAX_DISPATCH_BYTES,
// a larger single dispatch returned wrong data on the lab machine). Chunks are consecutive row ranges, each a
// multiple of 512 rows. Inside a chunk, core c owns the contiguous row range [c*RPC, (c+1)*RPC), RPC = chunk_rows/32,
// and its records are one contiguous byte range [c*core_bytes, (c+1)*core_bytes) of the chunk. Within a core the
// records are ordered group-major: group g (16 rows) -> record r along K: rec = g*recs_per_group + r.
//
// A tensor with ne[2]*ne[3] > 1 is stored as consecutive independent matrices (one per 2D slice).
// Any byte range of the GGUF stream can be written/read independently (needed for chunked set_tensor).

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace xdna2 {
namespace layout {

enum class Fmt : int { Q4_0 = 0, Q8_0, Q4_K, Q5_K, Q6_K, COUNT };

struct FmtInfo {
    const char * name;
    int          elems;        // weights per GGUF block
    size_t       block_bytes;  // sizeof(GGUF block)
    int          rec_blocks;   // blocks along K per record
    size_t       rec_bytes;    // 16 rows * rec_blocks * block_bytes
    bool         provisional;  // layout not yet fixed by an existing NPU kernel
};

inline const FmtInfo & fmt_info(Fmt f) {
    static const FmtInfo t[] = {
        { "Q4_0",  32,  18, 8, 16 *  8 *  18, false },
        { "Q8_0",  32,  34, 8, 16 *  8 *  34, true  },
        { "Q4_K", 256, 144, 1, 16 *  1 * 144, false },
        { "Q5_K", 256, 176, 1, 16 *  1 * 176, true  },
        { "Q6_K", 256, 210, 1, 16 *  1 * 210, false },
    };
    return t[int(f)];
}

constexpr int     QK32       = 32;
constexpr int     N_CORES    = 32;
constexpr int     GROUP_ROWS = 16;
constexpr int64_t ROW_GRAN   = int64_t(N_CORES) * GROUP_ROWS; // 512
constexpr size_t  MAX_DISPATCH_BYTES = 512000000;              // see stage2 RESULTS.md (834 MB dispatch was wrong)
constexpr int     MAX_CHUNKS = 32;
constexpr size_t  MAX_BLOCK_BYTES = 210;

constexpr int64_t roundup(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

struct Chunk {
    int64_t row0 = 0;             // first row of the chunk
    int64_t rows = 0;             // padded rows of the chunk (multiple of 512)
    size_t  off = 0;              // byte offset of the chunk inside the matrix region
    size_t  bytes = 0;
    int64_t rpc = 0;              // rows per core
    int64_t groups_per_core = 0;
    size_t  core_bytes = 0;
};

struct Geom {
    Fmt     fmt = Fmt::Q4_0;
    int64_t n = 0, k = 0;         // logical rows / columns
    int64_t kb = 0;               // blocks per row
    int64_t n_pad = 0;
    int64_t kb_pad = 0;
    int64_t recs_per_group = 0;   // records along K
    size_t  rec_bytes = 0;
    size_t  block_bytes = 0;
    int     rec_blocks = 0;
    size_t  bytes = 0;            // padded bytes of one 2D matrix
    size_t  row_bytes = 0;        // GGUF bytes of one row
    size_t  gguf_bytes = 0;       // GGUF bytes of one 2D matrix
    int     nchunks = 0;
    Chunk   chunks[MAX_CHUNKS];

    const Chunk & chunk_of(int64_t row) const {
        int i = 0;
        while (i + 1 < nchunks && row >= chunks[i + 1].row0) ++i;
        return chunks[i];
    }

    // byte offset (relative to the matrix start) of the first record of the 16-row group containing `row`
    size_t group_base(int64_t row) const {
        const Chunk & c = chunk_of(row);
        const int64_t rr = row - c.row0;
        const int64_t core = rr / c.rpc, g = (rr % c.rpc) / GROUP_ROWS;
        return c.off + size_t(core) * c.core_bytes + size_t(g * recs_per_group) * rec_bytes;
    }
};

inline Geom make_geom(Fmt fmt, int64_t n, int64_t k, size_t max_dispatch_bytes = MAX_DISPATCH_BYTES) {
    const FmtInfo & fi = fmt_info(fmt);
    Geom g;
    g.fmt = fmt;
    g.n = n;
    g.k = k;
    g.kb = k / fi.elems;
    g.n_pad = roundup(n, ROW_GRAN);
    g.kb_pad = roundup(g.kb, fi.rec_blocks);
    g.recs_per_group = g.kb_pad / fi.rec_blocks;
    g.rec_bytes = fi.rec_bytes;
    g.block_bytes = fi.block_bytes;
    g.rec_blocks = fi.rec_blocks;
    g.bytes = size_t(g.n_pad) * size_t(g.kb_pad) * fi.block_bytes;
    g.row_bytes = size_t(g.kb) * fi.block_bytes;
    g.gguf_bytes = size_t(n) * g.row_bytes;

    // split into dispatch chunks of (almost) equal size, each a multiple of 512 rows
    const int64_t units = g.n_pad / ROW_GRAN;
    int64_t nch = int64_t((g.bytes + max_dispatch_bytes - 1) / max_dispatch_bytes);
    if (nch < 1) nch = 1;
    if (nch > units) nch = units;
    if (nch > MAX_CHUNKS) nch = MAX_CHUNKS;
    g.nchunks = int(nch);
    int64_t row0 = 0;
    size_t off = 0;
    for (int i = 0; i < g.nchunks; ++i) {
        Chunk & c = g.chunks[i];
        c.row0 = row0;
        c.rows = (units / nch + (i < units % nch ? 1 : 0)) * ROW_GRAN;
        c.rpc = c.rows / N_CORES;
        c.groups_per_core = c.rpc / GROUP_ROWS;
        c.core_bytes = size_t(c.groups_per_core) * size_t(g.recs_per_group) * fi.rec_bytes;
        c.bytes = c.core_bytes * N_CORES;
        c.off = off;
        row0 += c.rows;
        off += c.bytes;
    }
    return g;
}

namespace detail {

inline void put_nib(uint8_t * p, int shift, uint8_t v) {
    *p = uint8_t((*p & ~(0xF << shift)) | (v << shift));
}
inline uint8_t get_nib(const uint8_t * p, int shift) { return uint8_t((*p >> shift) & 15); }

// ---- 4-bit tile addressing shared by Q4_0 / Q4_K / Q5_K ----------------------------------------------------------
// A tile = 16 consecutive weights (k) x 16 rows (n) = 256 nibbles = 128 B:  byte  tile*128 + k*8 + n/2,
// low nibble = even row, high nibble = odd row.
inline uint8_t * tile_nib(uint8_t * rec, int tile, int kk, int n) { return rec + size_t(tile) * 128 + size_t(kk) * 8 + size_t(n >> 1); }

// ---- Q4_0 (stage0 q4.cc): per record 8 blocks; block b = tiles 2b, 2b+1; nibble = element ^ 8; scales fp16 [b][n] at 2048
inline void q4_0_pack(uint8_t * rec, int b, int n, const uint8_t * blk) {
    const int shift = (n & 1) * 4;
    uint8_t * sc = rec + 2048 + (size_t(b) * 16 + size_t(n)) * 2;
    sc[0] = blk[0];
    sc[1] = blk[1];
    for (int j = 0; j < 16; ++j) {
        const uint8_t v = blk[2 + j];
        put_nib(tile_nib(rec, 2 * b + 0, j, n), shift, uint8_t((v & 15) ^ 8));
        put_nib(tile_nib(rec, 2 * b + 1, j, n), shift, uint8_t((v >> 4) ^ 8));
    }
}
inline void q4_0_unpack(const uint8_t * rec, int b, int n, uint8_t * blk) {
    const int shift = (n & 1) * 4;
    const uint8_t * sc = rec + 2048 + (size_t(b) * 16 + size_t(n)) * 2;
    blk[0] = sc[0];
    blk[1] = sc[1];
    for (int j = 0; j < 16; ++j) {
        const uint8_t a = uint8_t(get_nib(tile_nib(const_cast<uint8_t *>(rec), 2 * b + 0, j, n), shift) ^ 8);
        const uint8_t c = uint8_t(get_nib(tile_nib(const_cast<uint8_t *>(rec), 2 * b + 1, j, n), shift) ^ 8);
        blk[2 + j] = uint8_t(a | (c << 4));
    }
}

// ---- Q4_K (stage2 qk.cc / pack_qk.py): block = d, dmin (fp16), scales[12], qs[128]
//   bytes [0,2048) nibble tiles (tile = w/16, w = weight index 0..255; weight w: chunk c = w/64, r = w%64, l = r%32,
//                  nibble = r < 32 ? qs[c*32+l] & 15 : qs[c*32+l] >> 4, UNSIGNED 0..15)
//   [2048,2240)    the 12 packed 6-bit scale/min bytes as planes [p][n]
//   [2240,2272)    d fp16 [n];  [2272,2304) dmin fp16 [n]
inline void q4k_pack(uint8_t * rec, int n, const uint8_t * blk) {
    const int shift = (n & 1) * 4;
    std::memcpy(rec + 2048 + 192 + 2 * n, blk + 0, 2);
    std::memcpy(rec + 2048 + 224 + 2 * n, blk + 2, 2);
    for (int p = 0; p < 12; ++p) rec[2048 + p * 16 + n] = blk[4 + p];
    const uint8_t * qs = blk + 16;
    for (int c = 0; c < 4; ++c) {
        for (int l = 0; l < 32; ++l) {
            const uint8_t v = qs[c * 32 + l];
            const int wl = c * 64 + l, wh = c * 64 + 32 + l;
            put_nib(tile_nib(rec, wl / 16, wl % 16, n), shift, v & 15);
            put_nib(tile_nib(rec, wh / 16, wh % 16, n), shift, v >> 4);
        }
    }
}
inline void q4k_unpack(const uint8_t * rec, int n, uint8_t * blk) {
    const int shift = (n & 1) * 4;
    std::memcpy(blk + 0, rec + 2048 + 192 + 2 * n, 2);
    std::memcpy(blk + 2, rec + 2048 + 224 + 2 * n, 2);
    for (int p = 0; p < 12; ++p) blk[4 + p] = rec[2048 + p * 16 + n];
    uint8_t * qs = blk + 16;
    for (int c = 0; c < 4; ++c) {
        for (int l = 0; l < 32; ++l) {
            const int wl = c * 64 + l, wh = c * 64 + 32 + l;
            const uint8_t a = get_nib(tile_nib(const_cast<uint8_t *>(rec), wl / 16, wl % 16, n), shift);
            const uint8_t b = get_nib(tile_nib(const_cast<uint8_t *>(rec), wh / 16, wh % 16, n), shift);
            qs[c * 32 + l] = uint8_t(a | (b << 4));
        }
    }
}

// ---- Q6_K (stage2 qk.cc / pack_qk.py): block = ql[128], qh[64], scales[16] int8, d fp16
//   weight w: h = w/128, r = w%128, qq = r/32, l = r%32; 6-bit value = lo4 | hi2 << 4 (then -32 in the kernel)
//     lo4 = (qq < 2 ? ql[h*64 + (qq&1)*32 + l] & 15 : ql[h*64 + (qq&1)*32 + l] >> 4),  hi2 = (qh[h*32 + l] >> 2qq) & 3
//   tile t = w/16, k = w%16, element e = (n/8)*128 + k*8 + n%8 inside the tile
//   [0,2048)     lo4 nibble tiles: byte t*128 + e/2, shift 4*(e&1)
//   [2048,3072)  hi2 bits: byte 2048 + t*64 + e%64, bit 2*(e/64)
//   [3072,3328)  int8 scales [t][n];  [3328,3360) d fp16 [n]
inline void q6k_pos(int w, int n, size_t & lo_byte, int & lo_shift, size_t & hi_byte, int & hi_shift) {
    const int t = w >> 4, k = w & 15;
    const int e = (n >> 3) * 128 + k * 8 + (n & 7);
    lo_byte = size_t(t) * 128 + size_t(e >> 1);
    lo_shift = (e & 1) * 4;
    hi_byte = 2048 + size_t(t) * 64 + size_t(e & 63);
    hi_shift = 2 * (e >> 6);
}
inline void q6k_pack(uint8_t * rec, int n, const uint8_t * blk) {
    const uint8_t * ql = blk, * qh = blk + 128;
    for (int w = 0; w < 256; ++w) {
        const int h = w >> 7, r = w & 127, qq = r >> 5, l = r & 31;
        const uint8_t byte = ql[h * 64 + (qq & 1) * 32 + l];
        const uint8_t lo = qq < 2 ? (byte & 15) : (byte >> 4);
        const uint8_t hi = (qh[h * 32 + l] >> (2 * qq)) & 3;
        size_t lb, hb; int ls, hs;
        q6k_pos(w, n, lb, ls, hb, hs);
        put_nib(rec + lb, ls, lo);
        rec[hb] = uint8_t((rec[hb] & ~(3 << hs)) | (hi << hs));
    }
    for (int s = 0; s < 16; ++s) rec[3072 + s * 16 + n] = blk[192 + s];
    std::memcpy(rec + 3328 + 2 * n, blk + 208, 2);
}
inline void q6k_unpack(const uint8_t * rec, int n, uint8_t * blk) {
    uint8_t * ql = blk, * qh = blk + 128;
    std::memset(blk, 0, 192);
    for (int w = 0; w < 256; ++w) {
        const int h = w >> 7, r = w & 127, qq = r >> 5, l = r & 31;
        size_t lb, hb; int ls, hs;
        q6k_pos(w, n, lb, ls, hb, hs);
        const uint8_t lo = get_nib(rec + lb, ls);
        const uint8_t hi = uint8_t((rec[hb] >> hs) & 3);
        ql[h * 64 + (qq & 1) * 32 + l] |= qq < 2 ? lo : uint8_t(lo << 4);
        qh[h * 32 + l] |= uint8_t(hi << (2 * qq));
    }
    for (int s = 0; s < 16; ++s) blk[192 + s] = rec[3072 + s * 16 + n];
    std::memcpy(blk + 208, rec + 3328 + 2 * n, 2);
}

// ============================ PROVISIONAL layouts (no NPU kernel exists yet) ====================================
// Isolated here so that they are easy to swap once the stage3 kernels define the real ones.
// They keep the same conventions as the proven layouts above (16-row groups, tiles of 16 weights x 16 rows).
//
// Q8_0: record = 8 blocks. int8 tiles: byte b*512 + k*16 + n (k = 0..31); scales fp16 [b][n] at 4096.
inline void q8_0_pack(uint8_t * rec, int b, int n, const uint8_t * blk) {
    std::memcpy(rec + 4096 + (size_t(b) * 16 + size_t(n)) * 2, blk, 2);
    for (int k = 0; k < 32; ++k) rec[size_t(b) * 512 + size_t(k) * 16 + size_t(n)] = blk[2 + k];
}
inline void q8_0_unpack(const uint8_t * rec, int b, int n, uint8_t * blk) {
    std::memcpy(blk, rec + 4096 + (size_t(b) * 16 + size_t(n)) * 2, 2);
    for (int k = 0; k < 32; ++k) blk[2 + k] = rec[size_t(b) * 512 + size_t(k) * 16 + size_t(n)];
}

// Q5_K: block = d, dmin, scales[12], qh[32], qs[128]. Weight w has sub-block j = w/32, l = w%32:
//   low nibble as Q4_K, high bit = (qh[l] >> j) & 1.
//   [0,2048) nibble tiles as Q4_K | [2048,2560) high-bit planes: tile t = w/16, bit index kk*16 + n
//   -> byte 2048 + t*32 + kk*2 + n/8, bit n%8 | [2560,2752) scale planes [p][n] | [2752,2784) d | [2784,2816) dmin
inline void q5k_pack(uint8_t * rec, int n, const uint8_t * blk) {
    const int shift = (n & 1) * 4;
    std::memcpy(rec + 2752 + 2 * n, blk + 0, 2);
    std::memcpy(rec + 2784 + 2 * n, blk + 2, 2);
    for (int p = 0; p < 12; ++p) rec[2560 + p * 16 + n] = blk[4 + p];
    const uint8_t * qh = blk + 16, * qs = blk + 48;
    for (int c = 0; c < 4; ++c) {
        for (int l = 0; l < 32; ++l) {
            const uint8_t v = qs[c * 32 + l];
            const int wl = c * 64 + l, wh = c * 64 + 32 + l;
            put_nib(tile_nib(rec, wl / 16, wl % 16, n), shift, v & 15);
            put_nib(tile_nib(rec, wh / 16, wh % 16, n), shift, v >> 4);
        }
    }
    for (int w = 0; w < 256; ++w) {
        const int j = w >> 5, l = w & 31;
        const uint8_t bit = (qh[l] >> j) & 1;
        uint8_t * p = rec + 2048 + size_t(w >> 4) * 32 + size_t(w & 15) * 2 + size_t(n >> 3);
        *p = uint8_t((*p & ~(1 << (n & 7))) | (bit << (n & 7)));
    }
}
inline void q5k_unpack(const uint8_t * rec, int n, uint8_t * blk) {
    const int shift = (n & 1) * 4;
    std::memcpy(blk + 0, rec + 2752 + 2 * n, 2);
    std::memcpy(blk + 2, rec + 2784 + 2 * n, 2);
    for (int p = 0; p < 12; ++p) blk[4 + p] = rec[2560 + p * 16 + n];
    uint8_t * qh = blk + 16, * qs = blk + 48;
    std::memset(qh, 0, 32);
    for (int c = 0; c < 4; ++c) {
        for (int l = 0; l < 32; ++l) {
            const int wl = c * 64 + l, wh = c * 64 + 32 + l;
            const uint8_t a = get_nib(tile_nib(const_cast<uint8_t *>(rec), wl / 16, wl % 16, n), shift);
            const uint8_t b = get_nib(tile_nib(const_cast<uint8_t *>(rec), wh / 16, wh % 16, n), shift);
            qs[c * 32 + l] = uint8_t(a | (b << 4));
        }
    }
    for (int w = 0; w < 256; ++w) {
        const int j = w >> 5, l = w & 31;
        const uint8_t * p = rec + 2048 + size_t(w >> 4) * 32 + size_t(w & 15) * 2 + size_t(n >> 3);
        qh[l] |= uint8_t(((*p >> (n & 7)) & 1) << j);
    }
}
// ================================================================================================================

inline void pack_block(Fmt f, uint8_t * rec, int b, int n, const uint8_t * blk) {
    switch (f) {
        case Fmt::Q4_0: q4_0_pack(rec, b, n, blk); break;
        case Fmt::Q8_0: q8_0_pack(rec, b, n, blk); break;
        case Fmt::Q4_K: q4k_pack(rec, n, blk); break;
        case Fmt::Q5_K: q5k_pack(rec, n, blk); break;
        case Fmt::Q6_K: q6k_pack(rec, n, blk); break;
        default: break;
    }
}
inline void unpack_block(Fmt f, const uint8_t * rec, int b, int n, uint8_t * blk) {
    switch (f) {
        case Fmt::Q4_0: q4_0_unpack(rec, b, n, blk); break;
        case Fmt::Q8_0: q8_0_unpack(rec, b, n, blk); break;
        case Fmt::Q4_K: q4k_unpack(rec, n, blk); break;
        case Fmt::Q5_K: q5k_unpack(rec, n, blk); break;
        case Fmt::Q6_K: q6k_unpack(rec, n, blk); break;
        default: break;
    }
}

// Visit the bytes [off, off + size) of the GGUF stream of one 2D matrix, 16-row group by group.
// Different groups touch disjoint device bytes, so groups can run in parallel.
// op(group_base, row_in_group, kb_idx, lo, hi, offset_of_byte_lo_relative_to_stream_byte_off)
template <typename Op>
inline void for_range(const Geom & g, size_t off, size_t size, Op op) {
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
        const size_t gbase = g.group_base(gi * GROUP_ROWS);
        for (int64_t row = r_lo; row <= r_hi; ++row) {
            const size_t rs = size_t(row) * g.row_bytes;
            const size_t a  = off > rs ? off - rs : 0;                               // first byte inside the row
            const size_t z  = end < rs + g.row_bytes ? end - rs : g.row_bytes;       // one past last byte inside the row
            const int64_t kb_first = int64_t(a / g.block_bytes), kb_last = int64_t((z - 1) / g.block_bytes);
            for (int64_t kb = kb_first; kb <= kb_last; ++kb) {
                const size_t bs = size_t(kb) * g.block_bytes;
                const int lo = a > bs ? int(a - bs) : 0;
                const int hi = z < bs + g.block_bytes ? int(z - bs) : int(g.block_bytes);
                op(gbase, int(row % GROUP_ROWS), kb, lo, hi, rs + bs + size_t(lo) - off);
            }
        }
    }
    (void) par;
}

} // namespace detail

// Write bytes [off, off+size) of the GGUF stream of ONE 2D matrix (src points at stream byte `off`)
// into the matrix region `dev` (g.bytes bytes). Padding is not modified.
inline void pack_range(const Geom & g, uint8_t * dev, const uint8_t * src, size_t off, size_t size) {
    detail::for_range(g, off, size, [&](size_t gbase, int n, int64_t kb, int lo, int hi, size_t src_off) {
        uint8_t * rec = dev + gbase + size_t(kb / g.rec_blocks) * g.rec_bytes;
        const int b = int(kb % g.rec_blocks);
        if (lo == 0 && hi == int(g.block_bytes)) {
            detail::pack_block(g.fmt, rec, b, n, src + src_off);
        } else {
            // partial block: read-modify-write through the logical block (pack/unpack are exact inverses)
            uint8_t tmp[MAX_BLOCK_BYTES];
            detail::unpack_block(g.fmt, rec, b, n, tmp);
            std::memcpy(tmp + lo, src + src_off, size_t(hi - lo));
            detail::pack_block(g.fmt, rec, b, n, tmp);
        }
    });
}

// Inverse of pack_range: bit exact.
inline void unpack_range(const Geom & g, const uint8_t * dev, uint8_t * dst, size_t off, size_t size) {
    detail::for_range(g, off, size, [&](size_t gbase, int n, int64_t kb, int lo, int hi, size_t dst_off) {
        const uint8_t * rec = dev + gbase + size_t(kb / g.rec_blocks) * g.rec_bytes;
        const int b = int(kb % g.rec_blocks);
        if (lo == 0 && hi == int(g.block_bytes)) {
            detail::unpack_block(g.fmt, rec, b, n, dst + dst_off);
        } else {
            uint8_t tmp[MAX_BLOCK_BYTES];
            detail::unpack_block(g.fmt, rec, b, n, tmp);
            std::memcpy(dst + dst_off, tmp + lo, size_t(hi - lo));
        }
    });
}

inline void pack(const Geom & g, uint8_t * dev, const uint8_t * src)   { pack_range(g, dev, src, 0, g.gguf_bytes); }
inline void unpack(const Geom & g, const uint8_t * dev, uint8_t * dst) { unpack_range(g, dev, dst, 0, g.gguf_bytes); }

// Whole tensor (possibly several 2D slices) <-> byte ranges of the logical GGUF stream.
// slice s occupies GGUF bytes [s*gguf_bytes, (s+1)*gguf_bytes) and device bytes [s*bytes, (s+1)*bytes).
inline void tensor_pack_range(const Geom & g, uint8_t * dev, const uint8_t * src, size_t off, size_t size) {
    while (size > 0) {
        const size_t s = off / g.gguf_bytes, so = off % g.gguf_bytes;
        const size_t n = size < g.gguf_bytes - so ? size : g.gguf_bytes - so;
        pack_range(g, dev + s * g.bytes, src, so, n);
        src += n; off += n; size -= n;
    }
}

inline void tensor_unpack_range(const Geom & g, const uint8_t * dev, uint8_t * dst, size_t off, size_t size) {
    while (size > 0) {
        const size_t s = off / g.gguf_bytes, so = off % g.gguf_bytes;
        const size_t n = size < g.gguf_bytes - so ? size : g.gguf_bytes - so;
        unpack_range(g, dev + s * g.bytes, dst, so, n);
        dst += n; off += n; size -= n;
    }
}

} // namespace layout
} // namespace xdna2

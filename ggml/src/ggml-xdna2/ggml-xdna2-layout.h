#pragma once

// Device weight layouts of the ggml-xdna2 backend: UNIFIED decode kernel ABI (lab stage5: qku.cc / genu.py / packu.py).
//
// This header is dependency free (no ggml, no XRT) so that the layouts can be unit tested on their own.
//
// Every layout is a pure PERMUTATION of the bytes/nibbles/bits of the GGUF tensor (the logical payload has exactly the size of
// the GGUF data); only whole rows are padded (N -> multiple of 512, zero bytes = zero scale) and the stream of every core is padded
// to whole chunks.
//
//   Fmt::Q4_K   record 2304 B = 16 rows x 256 weights  (1 GGUF super-block per row)
//   Fmt::Q5_K   record 2816 B
//   Fmt::Q6_K   record 3360 B
//   Fmt::Q8_0   record 4352 B = 16 rows x 8 GGUF blocks of 32
//   (Q4_0 is not supported: its lab kernel uses the older stage0 ABI.)
//
// Matrix: N rows (output features) x K columns, K % 256 == 0, K <= KMAX, NBK = K/256 records per row.
//   N_pad = roundup(N, 512), G_total = N_pad/512 groups of 16 rows per core.
//   A matrix is cut into SPLITS of at most GMAX groups (GMAX*512 = 65536 rows); every split is an independent kernel dispatch with its
//   own W region, A BO and Y BO. Almost every weight has ONE split.
//
// LAYOUT v2 (shared by the decode kernel and the prefill kernel, ONE weight copy).
//
// W region of one split (Gs groups, rows [split*GMAX*512, ...)):  [8 chains][n_w chunks][4 lanes][CHUNK bytes]
//   core c32 = (col*4 + k) (k = 0..3) owns rows [c32*16*Gs, (c32+1)*16*Gs) of the split; chain = col, lane = k.
//   The 16-row groups of a core are cut into SUPERBLOCKS of GSB groups (the last one has gc = Gs - sb*GSB <= GSB groups). The byte stream of
//   a core is the sequence of SLABS (sb, kb), sb = 0.., kb = 0..NBK-1; a slab holds the gc records (16 rows x 256 k) record(sb*GSB+gl, kb),
//   gl = 0..gc-1, back to back, and is padded with zero bytes to a multiple of CHUNK. => a slab is an integer number of chunks, the slab
//   (sb, kb) is the unit that shares one 256-wide activation slice (prefill: activation slab reused for gc records).
//   Chunk j of the stream of (chain, lane) is at   chain*chain_bytes + (j*4 + lane)*CHUNK    with  chain_bytes = n_w*4*CHUNK.
//   A record may straddle chunk boundaries (consecutive chunks of a lane are 4*CHUNK apart in the device layout): RecPtr resolves that.
//   Bytes of a split region: 8*chain_bytes (a multiple of 32 KB); the region of a matrix is the concatenation of its splits.
//
// The decode output of a dispatch is f32 [col 8][Gs][core 4][16] ->  row = split_row0 + ((col*4 + k)*Gs + g)*16 + i.
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

enum class Fmt : int { Q8_0 = 0, Q4_K, Q5_K, Q6_K, COUNT };

struct FmtInfo {
    const char * name;
    int          elems;        // weights per GGUF block
    size_t       block_bytes;  // sizeof(GGUF block)
    int          rec_blocks;   // GGUF blocks per row per record (record = 256 weights per row)
    size_t       rec_bytes;    // 16 rows * rec_blocks * block_bytes
    int          qt;           // quantization type id of the kernel (header field 2)
};

inline const FmtInfo & fmt_info(Fmt f) {
    static const FmtInfo t[] = {
        { "Q8_0",  32,  34, 8, 16 * 8 *  34, 8 },
        { "Q4_K", 256, 144, 1, 16 * 1 * 144, 4 },
        { "Q5_K", 256, 176, 1, 16 * 1 * 176, 5 },
        { "Q6_K", 256, 210, 1, 16 * 1 * 210, 6 },
    };
    return t[int(f)];
}

constexpr int     QK32       = 32;
constexpr int     N_CORES    = 32;
constexpr int     GROUP_ROWS = 16;
constexpr int64_t ROW_GRAN   = int64_t(N_CORES) * GROUP_ROWS; // 512
constexpr int     N_CHAINS   = 8;               // one chain per column
constexpr int     N_LANES    = 4;               // cores per column (lanes of a chain)
constexpr int     GSB        = 5;               // groups per superblock (A-slab reuse of the prefill kernel; L1 budget)
constexpr size_t  MAX_BLOCK_BYTES = 210;
constexpr size_t  MAX_REC_BYTES   = 4352;
constexpr size_t  CHUNK      = 1024;          // kernel stream chunk (ObjectFifo object) in bytes: ABI constant, see kernels/qku.cc QK_S
constexpr int64_t GMAX       = 128;           // groups per dispatch (core staging buffer ost[GMAX][16])
constexpr int64_t KMAX       = 12288;         // activation buffer of the core (kernels/genu.py --kmax)

constexpr int64_t roundup(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

struct Geom {
    Fmt     fmt = Fmt::Q4_K;
    int64_t n = 0, k = 0;         // logical rows / columns
    int64_t n_pad = 0;
    int64_t groups = 0;           // G_total = n_pad / 512
    int64_t nsplit = 0;           // dispatches per matrix
    int64_t nbk = 0;              // records per row (k / 256)
    int64_t kb = 0;               // GGUF blocks per row
    size_t  rec_bytes = 0;
    size_t  block_bytes = 0;
    int     rec_blocks = 0;
    size_t  split_bytes_full = 0; // bytes of a split with GMAX groups (all splits but the last)
    size_t  bytes = 0;            // bytes of one 2D matrix region (all splits)
    size_t  row_bytes = 0;        // GGUF bytes of one row
    size_t  gguf_bytes = 0;       // GGUF bytes of one 2D matrix

    int64_t sp_groups(int64_t s) const { return s + 1 < nsplit ? GMAX : groups - (nsplit - 1) * GMAX; }
    int64_t sp_row0(int64_t s) const { return s * GMAX * ROW_GRAN; }
    // slab = gc records padded to whole chunks; superblocks hold GSB groups (the last one the rest)
    size_t  slab_bytes(int64_t gc) const { return (size_t(gc) * rec_bytes + CHUNK - 1) / CHUNK * CHUNK; }
    int64_t sp_nsb(int64_t s) const { return (sp_groups(s) + GSB - 1) / GSB; }
    int64_t sb_groups(int64_t s, int64_t sb) const { const int64_t r = sp_groups(s) - sb * GSB; return r < GSB ? r : GSB; }
    size_t  lane_bytes_groups(int64_t gs) const {      // bytes of the lane stream of a core with gs groups
        const int64_t full = gs / GSB, rem = gs % GSB;
        return size_t(nbk) * (size_t(full) * slab_bytes(GSB) + (rem ? slab_bytes(rem) : 0));
    }
    // lane stream of a core with gs groups restricted to the first nbk_use records (k-blocks) of every superblock
    size_t  lane_bytes_groups_k(int64_t gs, int64_t nbk_use) const {
        const int64_t full = gs / GSB, rem = gs % GSB;
        return size_t(nbk_use) * (size_t(full) * slab_bytes(GSB) + (rem ? slab_bytes(rem) : 0));
    }
    size_t  sp_lane_bytes(int64_t s) const { return lane_bytes_groups(sp_groups(s)); }
    size_t  sp_nw(int64_t s) const { return sp_lane_bytes(s) / CHUNK; }   // chunks per lane stream
    size_t  sp_chain_bytes(int64_t s) const { return sp_nw(s) * N_LANES * CHUNK; }
    size_t  sp_bytes(int64_t s) const { return sp_chain_bytes(s) * N_CHAINS; }
    size_t  sp_off(int64_t s) const { return size_t(s) * split_bytes_full; }
    size_t  sp_chain_words(int64_t s) const { return sp_chain_bytes(s) / 4; }   // instruction stream parameter
};

// Activation chunks of one dispatch (64 B header + the quantized vector), see compute.h
inline size_t act_chunks(Fmt fmt, int64_t k) {
    const size_t sv = fmt == Fmt::Q8_0 ? size_t(k) + size_t(k) / 8 : size_t(k) + size_t(k / 256) * 4 + size_t(k) / 8;
    return (64 + sv + CHUNK - 1) / CHUNK;
}

inline bool k_supported(int64_t k) { return k > 0 && k % 256 == 0 && k <= KMAX; }

inline Geom make_geom(Fmt fmt, int64_t n, int64_t k) {
    const FmtInfo & fi = fmt_info(fmt);
    Geom g;
    g.fmt = fmt;
    g.n = n;
    g.k = k;
    g.n_pad = roundup(n, ROW_GRAN);
    g.groups = g.n_pad / ROW_GRAN;
    g.nsplit = (g.groups + GMAX - 1) / GMAX;
    g.nbk = k / 256;
    g.kb = k / fi.elems;
    g.rec_bytes = fi.rec_bytes;
    g.block_bytes = fi.block_bytes;
    g.rec_blocks = fi.rec_blocks;
    g.split_bytes_full = g.lane_bytes_groups(GMAX) / CHUNK * N_LANES * CHUNK * N_CHAINS;
    g.bytes = 0;
    for (int64_t s = 0; s < g.nsplit; ++s) g.bytes += g.sp_bytes(s);
    g.row_bytes = size_t(g.kb) * fi.block_bytes;
    g.gguf_bytes = size_t(n) * g.row_bytes;
    return g;
}

// One 16-row record (REC bytes) of the device layout. A record may straddle chunk boundaries (the next chunk of the same lane is
// N_LANES*CHUNK further), so bytes are addressed through operator(): offset in the record -> pointer. Chunk boundaries are multiples
// of 32 and every multi-byte field of a record is 2-byte aligned, so a field never straddles.
struct RecPtr {
    uint8_t * base = nullptr;   // chain base + lane*CHUNK
    size_t    s0 = 0;           // offset of the record in the lane byte stream
    uint8_t * operator()(size_t off) const {
        const size_t o = s0 + off;
        return base + (o / CHUNK) * (N_LANES * CHUNK) + (o % CHUNK);
    }
    bool straddles(size_t rec_bytes) const { return (s0 % CHUNK) + rec_bytes > CHUNK; }
    void gather(uint8_t * dst, size_t rec_bytes) const {
        for (size_t done = 0; done < rec_bytes;) {
            const size_t room = CHUNK - ((s0 + done) % CHUNK);
            const size_t c = room < rec_bytes - done ? room : rec_bytes - done;
            std::memcpy(dst + done, (*this)(done), c);
            done += c;
        }
    }
};

// Record `r` (index along K) of the 16-row group that contains global row `row`; dev = start of the matrix region.
inline RecPtr locate(const Geom & g, uint8_t * dev, int64_t row, int64_t r) {
    const int64_t s = row / (GMAX * ROW_GRAN);
    const int64_t Gs = g.sp_groups(s);
    const int64_t rr = row - g.sp_row0(s);
    const int64_t rpc = Gs * GROUP_ROWS;
    const int64_t c32 = rr / rpc, grp = (rr % rpc) / GROUP_ROWS;
    const size_t chain = size_t(c32 / N_LANES), lane = size_t(c32 % N_LANES);
    const int64_t sb = grp / GSB, gl = grp % GSB;
    const size_t slab = g.slab_bytes(g.sb_groups(s, sb));
    RecPtr p;
    p.base = dev + g.sp_off(s) + chain * g.sp_chain_bytes(s) + lane * CHUNK;
    p.s0 = size_t(sb) * size_t(g.nbk) * g.slab_bytes(GSB) + size_t(r) * slab + size_t(gl) * g.rec_bytes;
    return p;
}

namespace detail {

inline void put_nib(uint8_t * p, int shift, uint8_t v) {
    *p = uint8_t((*p & ~(0xF << shift)) | (v << shift));
}
inline uint8_t get_nib(const uint8_t * p, int shift) { return uint8_t((*p >> shift) & 15); }

// ---- tile addressing -------------------------------------------------------------------------------------------
// "A" tiles (Q4_K): 16 weights (k) x 16 rows (n) = 128 B:  byte  tile*128 + k*8 + n/2, low nibble = even row.
inline uint8_t * tile_a(const RecPtr & rec, int tile, int kk, int n) { return rec(size_t(tile) * 128 + size_t(kk) * 8 + size_t(n >> 1)); }
// "B" tiles (Q5_K), Q6_K): element e = (n/8)*128 + k*8 + n%8 of the 256-element tile; nibble at byte tile*128 + e/2, shift 4*(e&1)
inline int tile_b_e(int kk, int n) { return (n >> 3) * 128 + kk * 8 + (n & 7); }

// ---- Q4_K (kernel qk3.cc QK_T=4): block = d, dmin (fp16), scales[12], qs[128]
//   [0,2048)  nibble tiles (A addressing; tile = w/16, w = weight 0..255: chunk c = w/64, r = w%64, l = r%32,
//             nibble = r < 32 ? qs[c*32+l] & 15 : qs[c*32+l] >> 4, unsigned 0..15)
//   [2048,2240) the 12 packed 6-bit scale/min bytes as planes [p][n] | [2240,2272) d fp16 [n] | [2272,2304) dmin fp16 [n]
inline void q4k_pack(const RecPtr & rec, int n, const uint8_t * blk) {
    const int shift = (n & 1) * 4;
    std::memcpy(rec(2048 + 192 + 2 * n), blk + 0, 2);
    std::memcpy(rec(2048 + 224 + 2 * n), blk + 2, 2);
    for (int p = 0; p < 12; ++p) (*rec(2048 + p * 16 + n)) = blk[4 + p];
    const uint8_t * qs = blk + 16;
    for (int c = 0; c < 4; ++c) {
        for (int l = 0; l < 32; ++l) {
            const uint8_t v = qs[c * 32 + l];
            const int wl = c * 64 + l, wh = c * 64 + 32 + l;
            put_nib(tile_a(rec, wl / 16, wl % 16, n), shift, v & 15);
            put_nib(tile_a(rec, wh / 16, wh % 16, n), shift, v >> 4);
        }
    }
}
inline void q4k_unpack(const RecPtr & rec, int n, uint8_t * blk) {
    const int shift = (n & 1) * 4;
    std::memcpy(blk + 0, rec(2048 + 192 + 2 * n), 2);
    std::memcpy(blk + 2, rec(2048 + 224 + 2 * n), 2);
    for (int p = 0; p < 12; ++p) blk[4 + p] = (*rec(2048 + p * 16 + n));
    uint8_t * qs = blk + 16;
    for (int c = 0; c < 4; ++c) {
        for (int l = 0; l < 32; ++l) {
            const int wl = c * 64 + l, wh = c * 64 + 32 + l;
            const uint8_t a = get_nib(tile_a(rec, wl / 16, wl % 16, n), shift);
            const uint8_t b = get_nib(tile_a(rec, wh / 16, wh % 16, n), shift);
            qs[c * 32 + l] = uint8_t(a | (b << 4));
        }
    }
}

// ---- Q5_K (QK_T=5): block = d, dmin, scales[12], qh[32], qs[128]
//   weight w: sub-block j = w/32, l = w%32, 5-bit value = low nibble (as Q4_K) | ((qh[l] >> j) & 1) << 4
//   tile t = w/16, k = w%16, e = tile_b_e(k, n)
//   [0,2048)     low nibbles: byte t*128 + e/2, shift 4*(e&1)
//   [2048,2560)  high bits: byte 2048 + (t/2)*64 + e%64, bit (e/64) + 4*(t&1)
//   [2560,2752) scale planes [p][n] | [2752,2784) d | [2784,2816) dmin
inline void q5k_pos(int w, int n, size_t & lo_byte, int & lo_shift, size_t & hi_byte, int & hi_bit) {
    const int t = w >> 4, e = tile_b_e(w & 15, n);
    lo_byte = size_t(t) * 128 + size_t(e >> 1);
    lo_shift = (e & 1) * 4;
    hi_byte = 2048 + size_t(t >> 1) * 64 + size_t(e & 63);
    hi_bit = (e >> 6) + 4 * (t & 1);
}
inline void q5k_pack(const RecPtr & rec, int n, const uint8_t * blk) {
    std::memcpy(rec(2752 + 2 * n), blk + 0, 2);
    std::memcpy(rec(2784 + 2 * n), blk + 2, 2);
    for (int p = 0; p < 12; ++p) (*rec(2560 + p * 16 + n)) = blk[4 + p];
    const uint8_t * qh = blk + 16, * qs = blk + 48;
    for (int w = 0; w < 256; ++w) {
        const int c = w >> 6, r = w & 63, l = r & 31;
        const uint8_t lo = r < 32 ? (qs[c * 32 + l] & 15) : (qs[c * 32 + l] >> 4);
        const uint8_t hi = (qh[w & 31] >> (w >> 5)) & 1;
        size_t lb, hb; int ls, hbit;
        q5k_pos(w, n, lb, ls, hb, hbit);
        put_nib(rec(lb), ls, lo);
        (*rec(hb)) = uint8_t(((*rec(hb)) & ~(1 << hbit)) | (hi << hbit));
    }
}
inline void q5k_unpack(const RecPtr & rec, int n, uint8_t * blk) {
    std::memcpy(blk + 0, rec(2752 + 2 * n), 2);
    std::memcpy(blk + 2, rec(2784 + 2 * n), 2);
    for (int p = 0; p < 12; ++p) blk[4 + p] = (*rec(2560 + p * 16 + n));
    uint8_t * qh = blk + 16, * qs = blk + 48;
    std::memset(qh, 0, 32 + 128);
    for (int w = 0; w < 256; ++w) {
        const int c = w >> 6, r = w & 63, l = r & 31;
        size_t lb, hb; int ls, hbit;
        q5k_pos(w, n, lb, ls, hb, hbit);
        const uint8_t lo = get_nib(rec(lb), ls);
        const uint8_t hi = uint8_t(((*rec(hb)) >> hbit) & 1);
        qs[c * 32 + l] |= r < 32 ? lo : uint8_t(lo << 4);
        qh[w & 31] |= uint8_t(hi << (w >> 5));
    }
}

// ---- Q6_K (QK_T=6): block = ql[128], qh[64], scales[16] int8, d fp16
//   weight w: h = w/128, r = w%128, qq = r/32, l = r%32; 6-bit value = lo4 | hi2 << 4 (then -32 in the kernel)
//     lo4 = (qq < 2 ? ql[h*64 + (qq&1)*32 + l] & 15 : ql[h*64 + (qq&1)*32 + l] >> 4),  hi2 = (qh[h*32 + l] >> 2qq) & 3
//   tile t = w/16, k = w%16, e = tile_b_e(k, n)
//   [0,2048)     lo4 nibble tiles: byte t*128 + e/2, shift 4*(e&1)
//   [2048,3072)  hi2 bits: byte 2048 + t*64 + e%64, bit 2*(e/64)
//   [3072,3328)  int8 scales [t][n];  [3328,3360) d fp16 [n]
inline void q6k_pos(int w, int n, size_t & lo_byte, int & lo_shift, size_t & hi_byte, int & hi_shift) {
    const int t = w >> 4, e = tile_b_e(w & 15, n);
    lo_byte = size_t(t) * 128 + size_t(e >> 1);
    lo_shift = (e & 1) * 4;
    hi_byte = 2048 + size_t(t) * 64 + size_t(e & 63);
    hi_shift = 2 * (e >> 6);
}
inline void q6k_pack(const RecPtr & rec, int n, const uint8_t * blk) {
    const uint8_t * ql = blk, * qh = blk + 128;
    for (int w = 0; w < 256; ++w) {
        const int h = w >> 7, r = w & 127, qq = r >> 5, l = r & 31;
        const uint8_t byte = ql[h * 64 + (qq & 1) * 32 + l];
        const uint8_t lo = qq < 2 ? (byte & 15) : (byte >> 4);
        const uint8_t hi = (qh[h * 32 + l] >> (2 * qq)) & 3;
        size_t lb, hb; int ls, hs;
        q6k_pos(w, n, lb, ls, hb, hs);
        put_nib(rec(lb), ls, lo);
        (*rec(hb)) = uint8_t(((*rec(hb)) & ~(3 << hs)) | (hi << hs));
    }
    for (int s = 0; s < 16; ++s) (*rec(3072 + s * 16 + n)) = blk[192 + s];
    std::memcpy(rec(3328 + 2 * n), blk + 208, 2);
}
inline void q6k_unpack(const RecPtr & rec, int n, uint8_t * blk) {
    uint8_t * ql = blk, * qh = blk + 128;
    std::memset(blk, 0, 192);
    for (int w = 0; w < 256; ++w) {
        const int h = w >> 7, r = w & 127, qq = r >> 5, l = r & 31;
        size_t lb, hb; int ls, hs;
        q6k_pos(w, n, lb, ls, hb, hs);
        const uint8_t lo = get_nib(rec(lb), ls);
        const uint8_t hi = uint8_t(((*rec(hb)) >> hs) & 3);
        ql[h * 64 + (qq & 1) * 32 + l] |= qq < 2 ? lo : uint8_t(lo << 4);
        qh[h * 32 + l] |= uint8_t(hi << (2 * qq));
    }
    for (int s = 0; s < 16; ++s) blk[192 + s] = (*rec(3072 + s * 16 + n));
    std::memcpy(blk + 208, rec(3328 + 2 * n), 2);
}

// ---- Q8_0 (QK_T=8): record = 8 GGUF blocks (fp16 d, int8 qs[32]) per row
//   weight k32 (0..31) of block b: kc = k32/16, k = k32%16, row n: nh = n/8, n8 = n%8
//   [0,4096)    int8: b*512 + kc*256 + nh*128 + k*8 + n8   |   [4096,4352) fp16 scales: 4096 + (b*16 + n)*2
inline size_t q8_0_off(int b, int k32, int n) {
    return size_t(b) * 512 + size_t(k32 >> 4) * 256 + size_t(n >> 3) * 128 + size_t(k32 & 15) * 8 + size_t(n & 7);
}
inline void q8_0_pack(const RecPtr & rec, int b, int n, const uint8_t * blk) {
    std::memcpy(rec(4096 + (size_t(b) * 16 + size_t(n)) * 2), blk, 2);
    for (int k = 0; k < 32; ++k) (*rec(q8_0_off(b, k, n))) = blk[2 + k];
}
inline void q8_0_unpack(const RecPtr & rec, int b, int n, uint8_t * blk) {
    std::memcpy(blk, rec(4096 + (size_t(b) * 16 + size_t(n)) * 2), 2);
    for (int k = 0; k < 32; ++k) blk[2 + k] = (*rec(q8_0_off(b, k, n)));
}

inline void pack_block(Fmt f, const RecPtr & rec, int b, int n, const uint8_t * blk) {
    switch (f) {
        case Fmt::Q8_0: q8_0_pack(rec, b, n, blk); break;
        case Fmt::Q4_K: q4k_pack(rec, n, blk); break;
        case Fmt::Q5_K: q5k_pack(rec, n, blk); break;
        case Fmt::Q6_K: q6k_pack(rec, n, blk); break;
        default: break;
    }
}
inline void unpack_block(Fmt f, const RecPtr & rec, int b, int n, uint8_t * blk) {
    switch (f) {
        case Fmt::Q8_0: q8_0_unpack(rec, b, n, blk); break;
        case Fmt::Q4_K: q4k_unpack(rec, n, blk); break;
        case Fmt::Q5_K: q5k_unpack(rec, n, blk); break;
        case Fmt::Q6_K: q6k_unpack(rec, n, blk); break;
        default: break;
    }
}

// Visit the bytes [off, off + size) of the GGUF stream of one 2D matrix, 16-row group by group.
// Different groups touch disjoint device bytes, so groups can run in parallel.
// op(first_row_of_group, row_in_group, kb_idx, lo, hi, offset_of_byte_lo_relative_to_stream_byte_off)
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
        const int64_t gbase = gi * GROUP_ROWS;
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
    detail::for_range(g, off, size, [&](int64_t grow, int n, int64_t kb, int lo, int hi, size_t src_off) {
        const RecPtr rec = locate(g, dev, grow, kb / g.rec_blocks);
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
    detail::for_range(g, off, size, [&](int64_t grow, int n, int64_t kb, int lo, int hi, size_t dst_off) {
        const RecPtr rec = locate(g, const_cast<uint8_t *>(dev), grow, kb / g.rec_blocks);
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

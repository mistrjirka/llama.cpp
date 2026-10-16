#include "ggml-xdna2-cpu.h"
#include "ggml-xdna2-bfp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace xdna2 {

using namespace layout;

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))

#define XD_TGT __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx512vbmi,fma,avx2,f16c")))

namespace {

// ------------------------------------------------------------------------------------------------------------------------------
// activations
// ------------------------------------------------------------------------------------------------------------------------------
// K-quants: ggml's block_q8_K  { float d; int8 qs[256]; int16 bsums[16] }  (bsums = sums of 16)
struct Q8Kblk {
    float   d;
    int8_t  q[256];
    int16_t bs[16];
};
static_assert(sizeof(Q8Kblk) == 292, "block_q8_K");

XD_TGT inline int32_t ld_i32(const void * p) { int32_t v; std::memcpy(&v, p, 4); return v; }

XD_TGT void quant_q8k(const float * x, Q8Kblk * y, int64_t nb) {
    for (int64_t b = 0; b < nb; ++b, x += 256, ++y) {
        __m512 v[16];
        __m512 amaxv = _mm512_setzero_ps();
        for (int i = 0; i < 16; ++i) {
            v[i] = _mm512_loadu_ps(x + 16 * i);
            amaxv = _mm512_max_ps(amaxv, _mm512_abs_ps(v[i]));
        }
        const float amax = _mm512_reduce_max_ps(amaxv);
        if (amax == 0.0f) {
            y->d = 0.0f;
            std::memset(y->q, 0, sizeof(y->q));
            std::memset(y->bs, 0, sizeof(y->bs));
            continue;
        }
        float mx = amax;      // signed value of the first element with the largest magnitude (ggml: ties -> first)
        {
            const __m512 av = _mm512_set1_ps(amax);
            for (int i = 0; i < 16; ++i) {
                const __mmask16 m = _mm512_cmp_ps_mask(_mm512_abs_ps(v[i]), av, _CMP_EQ_OQ);
                if (m) { mx = x[16 * i + __builtin_ctz(m)]; break; }
            }
        }
        const float iscale = -127.0f / mx;
        const __m512 is = _mm512_set1_ps(iscale);
        const __m512i hi = _mm512_set1_epi32(127);
        for (int i = 0; i < 16; ++i) {
            __m512i q = _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v[i], is)), hi);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(y->q + 16 * i), _mm512_cvtepi32_epi8(q));
            y->bs[i] = int16_t(_mm512_reduce_add_epi32(q));
        }
        y->d = 1.0f / iscale;
    }
}

// Q8_0 weights: blocks of 32, q stored BIASED (q + 128, u8) for vpdpbusd, d = fp16-rounded amax/127 (as ggml's block_q8_0)
XD_TGT void quant_q80(const float * x, uint8_t * q, float * d, int64_t nb) {
    const __m512i bias = _mm512_set1_epi32(128);
    for (int64_t b = 0; b < nb; ++b) {
        const __m512 v0 = _mm512_loadu_ps(x + 32 * b), v1 = _mm512_loadu_ps(x + 32 * b + 16);
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        const float dd = amax / 127.0f;
        const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
        d[b] = _cvtsh_ss(_cvtss_sh(dd, _MM_FROUND_TO_NEAREST_INT));
        const __m512 s = _mm512_set1_ps(id);
        const __m512i q0 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v0, s)), bias);
        const __m512i q1 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v1, s)), bias);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(q + 32 * b), _mm512_cvtepi32_epi8(q0));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(q + 32 * b + 16), _mm512_cvtepi32_epi8(q1));
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
// constant permutation tables (built once)
// ------------------------------------------------------------------------------------------------------------------------------
struct alignas(64) Tabs {
    alignas(64) uint8_t  idx_a[2][64];      // Q4_K: nibble A-tile, quad q&1 of a 64 B half-tile:   ((q&1)*4+i)*8 + n/2
    alignas(64) uint8_t  idx_b[4][64];      // Q5_K/Q6_K low nibbles (two sources: rows 0-7 | rows 8-15):  (n/8)*64 + (q*4+i)*4 + (n%8)/2
    alignas(64) uint8_t  idx_h[2][64];      // high bits: (4*(q&1)+i)*8 + n%8
    alignas(64) uint8_t  idx_8[2][64];      // Q8_0 (two sources: rows 0-7 | rows 8-15):   (n/8)*64 + ((q&1)*4+i)*8 + n%8
    alignas(64) int32_t  sv[16];            // (n&1)*4
    alignas(64) int32_t  sh5[2][2][16];     // Q5_K high bit shift [tile parity][q>>1]: (n/8)*2 + (q>>1) + 4*parity
    alignas(64) int32_t  sh6[2][16];        // Q6_K high bits shift [q>>1]:  2*((n/8)*2 + (q>>1))
    Tabs() {
        for (int h = 0; h < 2; ++h)
            for (int n = 0; n < 16; ++n)
                for (int i = 0; i < 4; ++i) {
                    idx_a[h][n * 4 + i] = uint8_t((h * 4 + i) * 8 + n / 2);
                    idx_h[h][n * 4 + i] = uint8_t((4 * h + i) * 8 + n % 8);
                    idx_8[h][n * 4 + i] = uint8_t((n / 8) * 64 + (h * 4 + i) * 8 + n % 8);
                }
        for (int q = 0; q < 4; ++q)
            for (int n = 0; n < 16; ++n)
                for (int i = 0; i < 4; ++i) idx_b[q][n * 4 + i] = uint8_t((n / 8) * 64 + (q * 4 + i) * 4 + (n % 8) / 2);
        for (int n = 0; n < 16; ++n) {
            sv[n] = (n & 1) * 4;
            for (int pt = 0; pt < 2; ++pt)
                for (int h = 0; h < 2; ++h) sh5[pt][h][n] = (n / 8) * 2 + h + 4 * pt;
            for (int h = 0; h < 2; ++h) sh6[h][n] = 2 * ((n / 8) * 2 + h);
        }
    }
};

const Tabs & tabs() {
    static const Tabs t;
    return t;
}

XD_TGT inline __m512i ldz(const void * p) { return _mm512_loadu_si512(p); }
XD_TGT inline __m512 fp16x16(const void * p) { return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p))); }

// One 16-row record read IN PLACE from the lane stream (chunk j of the lane is j*4*CHUNK further): record offset -> address.
// Q4_K/Q5_K/Q8_0 records start 128 B aligned in the stream, so no 64 B load straddles a chunk; the Q6_K record (3360 B) starts only
// 32 B aligned: a 64 B load at in-chunk offset 992 is assembled from two 32 B halves (Str = true).
template <bool Str>
struct RV {
    const uint8_t * lane;
    size_t          s0;
    XD_TGT inline const uint8_t * p(size_t off) const {
        const size_t o = s0 + off;
        return lane + (o >> 10) * (N_LANES * CHUNK) + (o & (CHUNK - 1));
    }
    XD_TGT inline __m512i ld64(size_t off) const {
        const size_t o = s0 + off;
        const uint8_t * q = lane + (o >> 10) * (N_LANES * CHUNK) + (o & (CHUNK - 1));
        if (Str && (o & (CHUNK - 1)) == CHUNK - 32) {
            const uint8_t * q2 = lane + ((o >> 10) + 1) * (N_LANES * CHUNK);
            return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q))),
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q2)), 1);
        }
        return _mm512_loadu_si512(q);
    }
    XD_TGT inline __m128i ld16(size_t off) const { return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p(off))); }
    XD_TGT inline __m512 f16(size_t off) const { return fp16x16(p(off)); }
};

// scale / min of the 8 sub-blocks of a Q4_K / Q5_K record from the 12 planes [p][n]   (ggml get_scale_min_k4)
template <class R>
XD_TGT inline void kq_scales(const R & rv, size_t base, __m512i * sc, __m512i * mn) {
    __m512i P[12];
    for (int p = 0; p < 12; ++p) P[p] = _mm512_cvtepu8_epi32(rv.ld16(base + 16 * p));
    const __m512i m63 = _mm512_set1_epi32(63), m15 = _mm512_set1_epi32(15);
    for (int j = 0; j < 4; ++j) {
        sc[j] = _mm512_and_si512(P[j], m63);
        mn[j] = _mm512_and_si512(P[j + 4], m63);
    }
    for (int j = 4; j < 8; ++j) {
        sc[j] = _mm512_or_si512(_mm512_and_si512(P[j + 4], m15), _mm512_slli_epi32(_mm512_srli_epi32(P[j - 4], 6), 4));
        mn[j] = _mm512_or_si512(_mm512_srli_epi32(P[j + 4], 4), _mm512_slli_epi32(_mm512_srli_epi32(P[j], 6), 4));
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
// record kernels: contribution of one 16-row x 256-k record (contiguous copy `rec`) to the 16 rows (lane n = row n)
// ------------------------------------------------------------------------------------------------------------------------------
template <class R>
XD_TGT inline __m512 rec_q4(const R & rv, const Q8Kblk * a, const Tabs & T) {
    const __m512i sv = ldz(T.sv), m0F = _mm512_set1_epi32(0x0F0F0F0F);
    const __m512i ia0 = ldz(T.idx_a[0]), ia1 = ldz(T.idx_a[1]);
    __m512i sc[8], mn[8];
    kq_scales(rv, 2048, sc, mn);
    __m512i sumi = _mm512_setzero_si512(), summs = _mm512_setzero_si512();
    for (int j = 0; j < 8; ++j) {
        __m512i acc = _mm512_setzero_si512();
        for (int t = 0; t < 2; ++t) {                       // tile 2j+t = 16 k
            const size_t tp = size_t(2 * j + t) * 128;
            for (int hf = 0; hf < 2; ++hf) {                // 64 B = 8 k
                const __m512i Z = rv.ld64(tp + hf * 64);
                for (int h = 0; h < 2; ++h) {               // 4 k
                    const __m512i p = _mm512_permutexvar_epi8(h ? ia1 : ia0, Z);
                    const __m512i w = _mm512_and_si512(_mm512_srlv_epi32(p, sv), m0F);
                    const int kk = j * 32 + t * 16 + hf * 8 + h * 4;
                    acc = _mm512_dpbusd_epi32(acc, w, _mm512_set1_epi32(ld_i32(a->q + kk)));
                }
            }
        }
        sumi  = _mm512_add_epi32(sumi, _mm512_mullo_epi32(acc, sc[j]));
        summs = _mm512_add_epi32(summs, _mm512_mullo_epi32(mn[j], _mm512_set1_epi32(int(a->bs[2 * j]) + int(a->bs[2 * j + 1]))));
    }
    const __m512 d = rv.f16(2240), dmin = rv.f16(2272);
    const __m512 t = _mm512_sub_ps(_mm512_mul_ps(d, _mm512_cvtepi32_ps(sumi)), _mm512_mul_ps(dmin, _mm512_cvtepi32_ps(summs)));
    return _mm512_mul_ps(t, _mm512_set1_ps(a->d));
}

template <class R>
XD_TGT inline __m512 rec_q5(const R & rv, const Q8Kblk * a, const Tabs & T) {
    const __m512i sv = ldz(T.sv), m0F = _mm512_set1_epi32(0x0F0F0F0F), m01 = _mm512_set1_epi32(0x01010101);
    __m512i sc[8], mn[8];
    kq_scales(rv, 2560, sc, mn);
    __m512i sumi = _mm512_setzero_si512(), summs = _mm512_setzero_si512();
    for (int j = 0; j < 8; ++j) {
        const __m512i H = rv.ld64(2048 + j * 64);
        __m512i acc = _mm512_setzero_si512();
        for (int t = 0; t < 2; ++t) {
            const size_t tp = size_t(2 * j + t) * 128;
            const __m512i Z0 = rv.ld64(tp), Z1 = rv.ld64(tp + 64);
            for (int q = 0; q < 4; ++q) {                   // 4 k
                const __m512i p = _mm512_permutex2var_epi8(Z0, ldz(T.idx_b[q]), Z1);
                const __m512i lo = _mm512_and_si512(_mm512_srlv_epi32(p, sv), m0F);
                const __m512i hp = _mm512_permutexvar_epi8(ldz(T.idx_h[q & 1]), H);
                const __m512i hi = _mm512_and_si512(_mm512_srlv_epi32(hp, ldz(T.sh5[t][q >> 1])), m01);
                const __m512i w = _mm512_or_si512(lo, _mm512_slli_epi32(hi, 4));
                const int kk = j * 32 + t * 16 + q * 4;
                acc = _mm512_dpbusd_epi32(acc, w, _mm512_set1_epi32(ld_i32(a->q + kk)));
            }
        }
        sumi  = _mm512_add_epi32(sumi, _mm512_mullo_epi32(acc, sc[j]));
        summs = _mm512_add_epi32(summs, _mm512_mullo_epi32(mn[j], _mm512_set1_epi32(int(a->bs[2 * j]) + int(a->bs[2 * j + 1]))));
    }
    const __m512 d = rv.f16(2752), dmin = rv.f16(2784);
    const __m512 t = _mm512_sub_ps(_mm512_mul_ps(d, _mm512_cvtepi32_ps(sumi)), _mm512_mul_ps(dmin, _mm512_cvtepi32_ps(summs)));
    return _mm512_mul_ps(t, _mm512_set1_ps(a->d));
}

template <class R>
XD_TGT inline __m512 rec_q6(const R & rv, const Q8Kblk * a, const Tabs & T) {
    const __m512i sv = ldz(T.sv), m0F = _mm512_set1_epi32(0x0F0F0F0F), m03 = _mm512_set1_epi32(0x03030303);
    __m512i sumi = _mm512_setzero_si512();
    for (int t = 0; t < 16; ++t) {                          // sub-block of 16 k
        const size_t tp = size_t(t) * 128;
        const __m512i Z0 = rv.ld64(tp), Z1 = rv.ld64(tp + 64), H = rv.ld64(2048 + t * 64);
        __m512i acc = _mm512_setzero_si512();
        for (int q = 0; q < 4; ++q) {
            const __m512i p = _mm512_permutex2var_epi8(Z0, ldz(T.idx_b[q]), Z1);
            const __m512i lo = _mm512_and_si512(_mm512_srlv_epi32(p, sv), m0F);
            const __m512i hp = _mm512_permutexvar_epi8(ldz(T.idx_h[q & 1]), H);
            const __m512i hi = _mm512_and_si512(_mm512_srlv_epi32(hp, ldz(T.sh6[q >> 1])), m03);
            const __m512i w = _mm512_or_si512(lo, _mm512_slli_epi32(hi, 4));
            acc = _mm512_dpbusd_epi32(acc, w, _mm512_set1_epi32(ld_i32(a->q + t * 16 + q * 4)));
        }
        acc = _mm512_sub_epi32(acc, _mm512_set1_epi32(32 * int(a->bs[t])));
        const __m512i sc = _mm512_cvtepi8_epi32(rv.ld16(3072 + t * 16));
        sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(acc, sc));
    }
    const __m512 d = rv.f16(3328);
    return _mm512_mul_ps(_mm512_mul_ps(d, _mm512_cvtepi32_ps(sumi)), _mm512_set1_ps(a->d));
}

// Q8_0: a = 256 biased u8 (8 blocks of 32), ad = 8 f32 block scales
template <class R>
XD_TGT inline __m512 rec_q8(const R & rv, const uint8_t * a, const float * ad, const Tabs & T) {
    const __m512i ones = _mm512_set1_epi32(0x01010101);
    const __m512i i0 = ldz(T.idx_8[0]), i1 = ldz(T.idx_8[1]);
    __m512 y = _mm512_setzero_ps();
    for (int b = 0; b < 8; ++b) {
        __m512i acc1 = _mm512_setzero_si512(), acc2 = _mm512_setzero_si512();
        for (int kc = 0; kc < 2; ++kc) {
            const size_t bp = size_t(b) * 512 + size_t(kc) * 256;
            for (int hf = 0; hf < 2; ++hf) {
                const __m512i Za = rv.ld64(bp + hf * 64), Zb = rv.ld64(bp + 128 + hf * 64);
                for (int h = 0; h < 2; ++h) {
                    const __m512i w = _mm512_permutex2var_epi8(Za, h ? i1 : i0, Zb);
                    const int kk = b * 32 + kc * 16 + hf * 8 + h * 4;
                    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(ld_i32(a + kk)), w);
                    acc2 = _mm512_dpbusd_epi32(acc2, ones, w);
                }
            }
        }
        const __m512i dot = _mm512_sub_epi32(acc1, _mm512_slli_epi32(acc2, 7));
        const __m512 dw = rv.f16(4096 + b * 32);
        y = _mm512_add_ps(y, _mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(dot), dw), _mm512_set1_ps(ad[b])));
    }
    return y;
}

// ------------------------------------------------------------------------------------------------------------------------------
// streaming a record out of the (chunk interleaved) lane stream
// ------------------------------------------------------------------------------------------------------------------------------
struct ActRef {
    const Q8Kblk *  k8 = nullptr;     // K-quants: K/256 blocks
    const uint8_t * q0 = nullptr;     // Q8_0: K biased bytes
    const float *   d0 = nullptr;     // Q8_0: K/32 scales
};


// One item = (split, chain, first lane, nl lanes, superblock sb): the nl lanes (cores) of a chain are processed slab by slab, so that the
// memory is consumed as consecutive 4*CHUNK pages (linear streams, much friendlier to the hardware prefetcher than one lane at a time:
// 55 -> 85 GB/s), and the lines of the next slab are prefetched in address order, spread over the records of the current slab.
template <Fmt F>
XD_TGT void run_item_t(const Geom & g, const uint8_t * w, const ActRef & act, int64_t s, int64_t chain, int64_t lane0, int64_t nl, int64_t sb,
                       int64_t kb0, int64_t kb1, float * out, bool partial, const Tabs & T) {
    const int64_t Gs = g.sp_groups(s);
    const int64_t gc = g.sb_groups(s, sb);
    const uint8_t * cb = w + g.sp_off(s) + size_t(chain) * g.sp_chain_bytes(s);
    const size_t slab_full = g.slab_bytes(GSB), slab = g.slab_bytes(gc);
    const size_t s0_sb = size_t(sb) * size_t(g.nbk) * slab_full;
    const size_t RB = g.rec_bytes;
    __m512 y[N_LANES][GSB];
    for (int l = 0; l < N_LANES; ++l) for (int i = 0; i < GSB; ++i) y[l][i] = _mm512_setzero_ps();
    const size_t slab_pages = slab / CHUNK;       // pages (4 lanes x CHUNK) per slab
    const size_t pf_lines = slab_pages * size_t(nl) * (CHUNK / 64);
    const size_t per_rec = (pf_lines + size_t(nl * gc) - 1) / size_t(nl * gc);
    for (int64_t kb = kb0; kb < kb1; ++kb) {
        const size_t so = s0_sb + size_t(kb) * slab;
        const bool do_pf = kb + 1 < kb1;
        const uint8_t * pf_base = cb + ((s0_sb + size_t(kb + 1) * slab) / CHUNK) * (N_LANES * CHUNK) + size_t(lane0) * CHUNK;
        size_t pf_done = 0;
        for (int64_t l = 0; l < nl; ++l) {
            for (int64_t gl = 0; gl < gc; ++gl) {
                if (do_pf) {       // the next slab: line pf_done = (page, lane, line in chunk)
                    const size_t e = std::min(pf_lines, pf_done + per_rec);
                    for (; pf_done < e; ++pf_done) {
                        const size_t page = pf_done / (size_t(nl) * (CHUNK / 64)), r = pf_done % (size_t(nl) * (CHUNK / 64));
                        _mm_prefetch(reinterpret_cast<const char *>(pf_base + page * (N_LANES * CHUNK) + (r / (CHUNK / 64)) * CHUNK + (r % (CHUNK / 64)) * 64), _MM_HINT_T0);
                    }
                }
                const RV<F == Fmt::Q6_K> rv{ cb + size_t(lane0 + l) * CHUNK, so + size_t(gl) * RB };
                if (F == Fmt::Q4_K)      y[l][gl] = _mm512_add_ps(y[l][gl], rec_q4(rv, act.k8 + kb, T));
                else if (F == Fmt::Q5_K) y[l][gl] = _mm512_add_ps(y[l][gl], rec_q5(rv, act.k8 + kb, T));
                else if (F == Fmt::Q6_K) y[l][gl] = _mm512_add_ps(y[l][gl], rec_q6(rv, act.k8 + kb, T));
                else                     y[l][gl] = _mm512_add_ps(y[l][gl], rec_q8(rv, act.q0 + kb * 256, act.d0 + kb * 8, T));
            }
        }
    }
    for (int64_t l = 0; l < nl; ++l) {
        if (partial) {          // out = this part's scratch [lane][gc][16]
            for (int64_t i = 0; i < gc; ++i) _mm512_storeu_ps(out + (l * gc + i) * GROUP_ROWS, y[l][i]);
            continue;
        }
        const int64_t row0 = g.sp_row0(s) + (chain * N_LANES + lane0 + l) * GROUP_ROWS * Gs + sb * GSB * GROUP_ROWS;
        for (int64_t i = 0; i < gc; ++i) {
            const int64_t r = row0 + i * GROUP_ROWS;
            if (r >= g.n) break;
            const int64_t cnt = std::min<int64_t>(GROUP_ROWS, g.n - r);
            _mm512_mask_storeu_ps(out + r, __mmask16((1u << cnt) - 1u), y[l][i]);
        }
    }
}

XD_TGT void run_item(const Geom & g, const uint8_t * w, const ActRef & act, int64_t s, int64_t chain, int64_t lane0, int64_t nl, int64_t sb,
                     int64_t kb0, int64_t kb1, float * out, bool partial, const Tabs & T) {
    switch (g.fmt) {
        case Fmt::Q4_K: run_item_t<Fmt::Q4_K>(g, w, act, s, chain, lane0, nl, sb, kb0, kb1, out, partial, T); break;
        case Fmt::Q5_K: run_item_t<Fmt::Q5_K>(g, w, act, s, chain, lane0, nl, sb, kb0, kb1, out, partial, T); break;
        case Fmt::Q6_K: run_item_t<Fmt::Q6_K>(g, w, act, s, chain, lane0, nl, sb, kb0, kb1, out, partial, T); break;
        default:        run_item_t<Fmt::Q8_0>(g, w, act, s, chain, lane0, nl, sb, kb0, kb1, out, partial, T); break;
    }
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::atomic<uint64_t> g_busy_ns{0}, g_wall_ns{0};

// A group = (job, split, chain, sb): the 4 lanes x gc x 16 rows an item range works on. Large groups are cut along K into `nparts` parts (items),
// each part writes its partial sums to the scratch of the group; the last part to finish adds them up into dst.
struct Item { int32_t job, split, chain, lane0, nl, sb, kb0, kb1, part, grp; };
struct Grp { int32_t nparts; size_t scr; };     // scratch offset (floats) of the parts, laid out [part][lane][gc][16]
struct InAct { const float * x; int64_t k; bool q80; size_t off; size_t bytes; };

std::vector<uint8_t> g_act_store;     // activation scratch (grows, never shrinks); used by one cpu_decode_run at a time
std::vector<Item>    g_items;
std::vector<Grp>     g_grps;
std::vector<float>   g_scratch;
std::unique_ptr<std::atomic<int32_t>[]> g_done;
size_t g_done_cap = 0;
std::vector<InAct>   g_ins;

XD_TGT void process_item(const Item & m, const CpuDecodeJob & jb, const ActRef & act, const Tabs & T) {
    const Grp & gr = g_grps[m.grp];
    const Geom & g = jb.geom;
    if (gr.nparts == 1) {
        run_item(g, jb.w, act, m.split, m.chain, m.lane0, m.nl, m.sb, m.kb0, m.kb1, jb.dst, false, T);
    } else {
        const int64_t gc = g.sb_groups(m.split, m.sb);
        const size_t psz = size_t(N_LANES * gc * GROUP_ROWS);
        float * scr = g_scratch.data() + gr.scr;
        run_item(g, jb.w, act, m.split, m.chain, m.lane0, m.nl, m.sb, m.kb0, m.kb1, scr + size_t(m.part) * psz, true, T);
        if (g_done[m.grp].fetch_add(1, std::memory_order_acq_rel) == gr.nparts - 1) {      // last part: reduce
            const int64_t Gs = g.sp_groups(m.split);
            for (int64_t l = 0; l < N_LANES; ++l) {
                const int64_t row0 = g.sp_row0(m.split) + (m.chain * N_LANES + l) * GROUP_ROWS * Gs + m.sb * GSB * GROUP_ROWS;
                for (int64_t i = 0; i < gc; ++i) {
                    const int64_t r = row0 + i * GROUP_ROWS;
                    if (r >= g.n) break;
                    __m512 sum = _mm512_loadu_ps(scr + (l * gc + i) * GROUP_ROWS);
                    for (int32_t p = 1; p < gr.nparts; ++p) sum = _mm512_add_ps(sum, _mm512_loadu_ps(scr + size_t(p) * psz + (l * gc + i) * GROUP_ROWS));
                    const int64_t cnt = std::min<int64_t>(GROUP_ROWS, g.n - r);
                    _mm512_mask_storeu_ps(jb.dst + r, __mmask16((1u << cnt) - 1u), sum);
                }
            }
        }
    }
}

} // namespace

bool cpu_decode_available() {
    static const bool ok = [] {
        if (const char * e = std::getenv("GGML_XDNA2_CPU_ISA")) {
            if (*e && std::strcmp(e, "0") == 0) return false;
        }
        __builtin_cpu_init();
        return bool(__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
                    __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vbmi"));
    }();
    return ok;
}

void cpu_decode_stats(double & busy_thread_s, double & wall_s, int & threads) {
    busy_thread_s = g_busy_ns.load() / 1e9;
    wall_s = g_wall_ns.load() / 1e9;
    threads = pf_threads();
}

void cpu_decode_run(const CpuDecodeJob * jobs, size_t nj) {
    if (nj == 0) return;
    const Tabs & T = tabs();
    // 1. unique inputs (several matrices of a layer read the same activation)
    g_ins.clear();
    std::vector<size_t> in_of(nj);
    size_t total = 0;
    for (size_t j = 0; j < nj; ++j) {
        const bool q80 = jobs[j].geom.fmt == Fmt::Q8_0;
        size_t e = g_ins.size();
        for (size_t q = 0; q < g_ins.size(); ++q) {
            if (g_ins[q].x == jobs[j].x && g_ins[q].k == jobs[j].geom.k && g_ins[q].q80 == q80) { e = q; break; }
        }
        if (e == g_ins.size()) {
            const int64_t k = jobs[j].geom.k;
            const size_t bytes = q80 ? size_t(k) + size_t(k / 32) * 4 : size_t(k / 256) * sizeof(Q8Kblk);
            g_ins.push_back(InAct{ jobs[j].x, k, q80, total, bytes });
            total += (bytes + 63) / 64 * 64;
        }
        in_of[j] = e;
    }
    if (g_act_store.size() < total + 64) g_act_store.resize(total + 64);
    uint8_t * store = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(g_act_store.data()) + 63) & ~uintptr_t(63));

    // 2. work items: (split, chain, superblock) with the 4 lanes of the chain (splitting the lanes of a chain over threads breaks the linear
    //    streams: 50 instead of 80 GB/s). Few big items cannot balance 10 threads, so a group is cut along K into parts when the batch has
    //    fewer than ~4 items per thread.
    g_items.clear();
    g_grps.clear();
    size_t base_total = 0;
    for (size_t j = 0; j < nj; ++j) {
        const Geom & g = jobs[j].geom;
        for (int64_t s = 0; s < g.nsplit; ++s) base_total += size_t(N_CHAINS) * size_t(std::max<int64_t>(0, g.sp_nsb(s) - std::min<int64_t>(jobs[j].sb_from, g.sp_nsb(s))));
    }
    const size_t want = size_t(4 * pf_threads());
    const int64_t parts_wanted = base_total >= want ? 1 : int64_t((want + base_total - 1) / base_total);
    size_t scr_total = 0;
    for (size_t j = 0; j < nj; ++j) {
        const Geom & g = jobs[j].geom;
        const int64_t P = std::max<int64_t>(1, std::min<int64_t>(parts_wanted, g.nbk / 2));
        for (int64_t s = 0; s < g.nsplit; ++s) {
            const int64_t Gs = g.sp_groups(s), nsb = g.sp_nsb(s);
            for (int64_t sb = std::min<int64_t>(jobs[j].sb_from, nsb); sb < nsb; ++sb) {
                for (int64_t chain = 0; chain < N_CHAINS; ++chain) {
                    if (g.sp_row0(s) + (chain * N_LANES) * GROUP_ROWS * Gs + sb * GSB * GROUP_ROWS >= g.n) continue;   // padding only
                    const int64_t gc = g.sb_groups(s, sb);
                    const int32_t grp = int32_t(g_grps.size());
                    g_grps.push_back(Grp{ int32_t(P), scr_total });
                    if (P > 1) scr_total += size_t(P) * size_t(N_LANES * gc * GROUP_ROWS);
                    for (int64_t part = 0; part < P; ++part) {
                        g_items.push_back(Item{ int32_t(j), int32_t(s), int32_t(chain), 0, N_LANES, int32_t(sb),
                                                int32_t(g.nbk * part / P), int32_t(g.nbk * (part + 1) / P), int32_t(part), grp });
                    }
                }
            }
        }
    }
    if (g_scratch.size() < scr_total) g_scratch.resize(scr_total);
    if (g_done_cap < g_grps.size()) { g_done.reset(new std::atomic<int32_t>[g_grps.size()]); g_done_cap = g_grps.size(); }
    for (size_t i = 0; i < g_grps.size(); ++i) g_done[i].store(0, std::memory_order_relaxed);
    const double tw0 = now_s();
    std::atomic<size_t> next{0};
    const size_t n_items = g_items.size();
    const size_t n_in = g_ins.size();
    std::atomic<size_t> next_q{0};

#if defined(_OPENMP)
    #pragma omp parallel num_threads(pf_threads())
#endif
    {
        // quantize the inputs (one task per input, tiny)
        for (size_t q = next_q.fetch_add(1); q < n_in; q = next_q.fetch_add(1)) {
            const InAct & ia = g_ins[q];
            uint8_t * p = store + ia.off;
            if (ia.q80) quant_q80(ia.x, p, reinterpret_cast<float *>(p + ia.k), ia.k / 32);
            else        quant_q8k(ia.x, reinterpret_cast<Q8Kblk *>(p), ia.k / 256);
        }
#if defined(_OPENMP)
        #pragma omp barrier
#endif
        const double tb = now_s();
        for (size_t it = next.fetch_add(1); it < n_items; it = next.fetch_add(1)) {
            const Item & m = g_items[it];
            const CpuDecodeJob & jb = jobs[m.job];
            const InAct & ia = g_ins[in_of[m.job]];
            ActRef act;
            const uint8_t * p = store + ia.off;
            if (ia.q80) { act.q0 = p; act.d0 = reinterpret_cast<const float *>(p + ia.k); }
            else        { act.k8 = reinterpret_cast<const Q8Kblk *>(p); }
            process_item(m, jb, act, T);
        }
        g_busy_ns.fetch_add(uint64_t((now_s() - tb) * 1e9));
    }
    g_wall_ns.fetch_add(uint64_t((now_s() - tw0) * 1e9));
}

#else   // not x86-64 GCC/clang

bool cpu_decode_available() { return false; }
void cpu_decode_stats(double & b, double & w, int & t) { b = w = 0; t = 1; }
void cpu_decode_run(const CpuDecodeJob *, size_t) {}

#endif

} // namespace xdna2

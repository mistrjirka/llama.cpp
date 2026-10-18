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
#include <unordered_map>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#include <x86intrin.h>   // _mm_monitorx/_mm_mwaitx (clang rejects including mwaitxintrin.h directly)
#include <sched.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <thread>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace xdna2 {

using namespace layout;

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))

#define XD_TGT __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx512vbmi,fma,avx2,f16c,mwaitx")))

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

// Q4_0 / Q4_1 (Q41): a = 256 biased u8 (8 blocks of 32), ad = 8 f32 block scales; nibbles in the Q4_K tiles, fp16 d [b][n] at 2048, m at 2304
template <bool Q41, class R>
XD_TGT inline __m512 rec_q40(const R & rv, const uint8_t * a, const float * ad, const Tabs & T) {
    const __m512i sv = ldz(T.sv), m0F = _mm512_set1_epi32(0x0F0F0F0F), flip = _mm512_set1_epi32(int(0x80808080u));
    const __m512i ia0 = ldz(T.idx_a[0]), ia1 = ldz(T.idx_a[1]);
    __m512 y = _mm512_setzero_ps();
    for (int b = 0; b < 8; ++b) {
        __m512i acc = _mm512_setzero_si512();
        for (int t = 0; t < 2; ++t) {
            const size_t tp = size_t(2 * b + t) * 128;
            for (int hf = 0; hf < 2; ++hf) {
                const __m512i Z = rv.ld64(tp + hf * 64);
                for (int h = 0; h < 2; ++h) {
                    const __m512i p = _mm512_permutexvar_epi8(h ? ia1 : ia0, Z);
                    const __m512i w = _mm512_and_si512(_mm512_srlv_epi32(p, sv), m0F);
                    const int kk = b * 32 + t * 16 + hf * 8 + h * 4;
                    acc = _mm512_dpbusd_epi32(acc, w, _mm512_xor_si512(_mm512_set1_epi32(ld_i32(a + kk)), flip));
                }
            }
        }
        int asum = -128 * 32;
        for (int k = 0; k < 32; ++k) asum += a[b * 32 + k];
        const __m512 dw = rv.f16(2048 + b * 32);
        __m512 t;
        if (Q41) t = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), dw, _mm512_mul_ps(rv.f16(2304 + b * 32), _mm512_set1_ps(float(asum))));
        else     t = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc, _mm512_set1_epi32(8 * asum))), dw);
        y = _mm512_fmadd_ps(t, _mm512_set1_ps(ad[b]), y);
    }
    return y;
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
    int             nt = 1;           // tokens; token t's activations are tstride bytes after token 0's
    size_t          tstride = 0;
    XD_TGT ActRef tok(int t) const {
        ActRef r = *this;
        const size_t o = size_t(t) * tstride;
        if (k8) r.k8 = reinterpret_cast<const Q8Kblk *>(reinterpret_cast<const uint8_t *>(k8) + o);
        if (q0) { r.q0 = q0 + o; r.d0 = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(d0) + o); }
        r.nt = 1;
        return r;
    }
};


// One item = (split, chain, first lane, nl lanes, superblock sb): the nl lanes (cores) of a chain are processed slab by slab, so that the
// memory is consumed as consecutive 4*CHUNK pages (linear streams, much friendlier to the hardware prefetcher than one lane at a time:
// 55 -> 85 GB/s), and the lines of the next slab are prefetched in address order, spread over the records of the current slab.
template <Fmt F, class R>
XD_TGT inline __m512 rec_any(const R & rv, const ActRef & act, int64_t kb, const Tabs & T) {
    if (F == Fmt::Q4_K)      return rec_q4(rv, act.k8 + kb, T);
    else if (F == Fmt::Q5_K) return rec_q5(rv, act.k8 + kb, T);
    else if (F == Fmt::Q6_K) return rec_q6(rv, act.k8 + kb, T);
    else if (F == Fmt::Q4_0) return rec_q40<false>(rv, act.q0 + kb * 256, act.d0 + kb * 8, T);
    else if (F == Fmt::Q4_1) return rec_q40<true>(rv, act.q0 + kb * 256, act.d0 + kb * 8, T);
    else                     return rec_q8(rv, act.q0 + kb * 256, act.d0 + kb * 8, T);
}

// out: token t's rows at out + t * ostride (floats); partial: the scratch [t][lane][gc][16] (ostride = N_LANES * gc * 16)
template <Fmt F>
XD_TGT void run_item_t(const Geom & g, const uint8_t * w, const ActRef & act, int64_t s, int64_t chain, int64_t lane0, int64_t nl, int64_t sb,
                       int64_t kb0, int64_t kb1, float * out, bool partial, const Tabs & T, size_t ostride, int64_t n_use) {
    const int nt = act.nt;
    const int64_t Gs = g.sp_groups(s);
    const int64_t gc = g.sb_groups(s, sb);
    const uint8_t * cb = w + g.sp_off(s) + size_t(chain) * g.sp_chain_bytes(s);
    const size_t slab_full = g.slab_bytes(g.gsb), slab = g.slab_bytes(gc);
    const size_t s0_sb = size_t(sb) * size_t(g.nbk) * slab_full;
    const size_t RB = g.rec_bytes;
    __m512 y[N_LANES][GSB_MAX];
    for (int l = 0; l < N_LANES; ++l) for (int i = 0; i < GSB_MAX; ++i) y[l][i] = _mm512_setzero_ps();
    alignas(64) float yt[CPU_MAX_TOK][N_LANES][GSB_MAX][GROUP_ROWS];     // nt > 1: the accumulators of every token (the record stays in L1)
    ActRef at[CPU_MAX_TOK];
    if (nt > 1) {
        std::memset(yt, 0, sizeof(float) * size_t(nt) * N_LANES * GSB_MAX * GROUP_ROWS);
        for (int t = 0; t < nt; ++t) at[t] = act.tok(t);
    }
    const size_t slab_pages = slab / CHUNK;       // pages (4 lanes x CHUNK) per slab
    const size_t pf_lines = slab_pages * size_t(nl) * (CHUNK / 64);
    const size_t per_rec = (pf_lines + size_t(nl * gc) - 1) / size_t(nl * gc);
    static const int PFS = [] { const char * e = std::getenv("XD_PFS"); return e ? std::atoi(e) : 1; }();
    static const int PFH = [] { const char * e = std::getenv("XD_PFH"); return e ? std::atoi(e) : 0; }();
    for (int64_t kb = kb0; kb < kb1; ++kb) {
        const size_t so = s0_sb + size_t(kb) * slab;
        const bool do_pf = kb + PFS < kb1;
        const uint8_t * pf_base = cb + ((s0_sb + size_t(kb + PFS) * slab) / CHUNK) * (N_LANES * CHUNK) + size_t(lane0) * CHUNK;
        size_t pf_done = 0;
        for (int64_t l = 0; l < nl; ++l) {
            for (int64_t gl = 0; gl < gc; ++gl) {
                if (do_pf) {       // the slab PFS ahead: line pf_done = (page, lane, line in chunk)
                    const size_t e = std::min(pf_lines, pf_done + per_rec);
                    for (; pf_done < e; ++pf_done) {
                        const size_t page = pf_done / (size_t(nl) * (CHUNK / 64)), r = pf_done % (size_t(nl) * (CHUNK / 64));
                        const char * a = reinterpret_cast<const char *>(pf_base + page * (N_LANES * CHUNK) + (r / (CHUNK / 64)) * CHUNK + (r % (CHUNK / 64)) * 64);
                        if (PFH == 0) _mm_prefetch(a, _MM_HINT_T0); else if (PFH == 1) _mm_prefetch(a, _MM_HINT_T1); else _mm_prefetch(a, _MM_HINT_T2);
                    }
                }
                const RV<F == Fmt::Q6_K> rv{ cb + size_t(lane0 + l) * CHUNK, so + size_t(gl) * RB };
                if (nt == 1) {
                    y[l][gl] = _mm512_add_ps(y[l][gl], rec_any<F>(rv, act, kb, T));
                } else {
                    for (int t = 0; t < nt; ++t) {
                        float * yp = yt[t][l][gl];
                        _mm512_store_ps(yp, _mm512_add_ps(_mm512_load_ps(yp), rec_any<F>(rv, at[t], kb, T)));
                    }
                }
            }
        }
    }
    for (int t = 0; t < nt; ++t) {
        float * o = out + size_t(t) * ostride;
        for (int64_t l = 0; l < nl; ++l) {
            if (partial) {          // out = this part's scratch [t][lane][gc][16]
                for (int64_t i = 0; i < gc; ++i) _mm512_storeu_ps(o + (l * gc + i) * GROUP_ROWS, nt == 1 ? y[l][i] : _mm512_load_ps(yt[t][l][i]));
                continue;
            }
            const int64_t row0 = g.sp_row0(s) + (chain * N_LANES + lane0 + l) * GROUP_ROWS * Gs + sb * g.gsb * GROUP_ROWS;
            for (int64_t i = 0; i < gc; ++i) {
                const int64_t r = row0 + i * GROUP_ROWS;
                if (r >= n_use) break;
                const int64_t cnt = std::min<int64_t>(GROUP_ROWS, n_use - r);
                _mm512_mask_storeu_ps(o + r, __mmask16((1u << cnt) - 1u), nt == 1 ? y[l][i] : _mm512_load_ps(yt[t][l][i]));
            }
        }
    }
}

XD_TGT void run_item(const Geom & g, const uint8_t * w, const ActRef & act, int64_t s, int64_t chain, int64_t lane0, int64_t nl, int64_t sb,
                     int64_t kb0, int64_t kb1, float * out, bool partial, const Tabs & T, size_t ostride, int64_t n_use) {
    switch (g.fmt) {
        case Fmt::Q4_K: run_item_t<Fmt::Q4_K>(g, w, act, s, chain, lane0, nl, sb, kb0, kb1, out, partial, T, ostride, n_use); break;
        case Fmt::Q5_K: run_item_t<Fmt::Q5_K>(g, w, act, s, chain, lane0, nl, sb, kb0, kb1, out, partial, T, ostride, n_use); break;
        case Fmt::Q6_K: run_item_t<Fmt::Q6_K>(g, w, act, s, chain, lane0, nl, sb, kb0, kb1, out, partial, T, ostride, n_use); break;
        case Fmt::Q4_0: run_item_t<Fmt::Q4_0>(g, w, act, s, chain, lane0, nl, sb, kb0, kb1, out, partial, T, ostride, n_use); break;
        case Fmt::Q4_1: run_item_t<Fmt::Q4_1>(g, w, act, s, chain, lane0, nl, sb, kb0, kb1, out, partial, T, ostride, n_use); break;
        default:        run_item_t<Fmt::Q8_0>(g, w, act, s, chain, lane0, nl, sb, kb0, kb1, out, partial, T, ostride, n_use); break;
    }
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::atomic<uint64_t> g_busy_ns{0}, g_wall_ns{0};

// A group = (job, split, chain, sb): the 4 lanes x gc x 16 rows an item range works on. Large groups are cut along K into `nparts` parts (items),
// each part writes its partial sums to the scratch of the group; the last part to finish adds them up into dst.
struct Item { int32_t job, split, chain, lane0, nl, sb, kb0, kb1, part, grp; };
struct Grp { int32_t nparts; size_t scr; };     // scratch offset (floats) of the parts, laid out [part][lane][gc][16]
struct InAct { const float * x; int64_t k; bool q80; size_t off; size_t bytes; int nt; size_t xs; size_t tstride; };

std::vector<uint8_t> g_act_store;     // activation scratch (grows, never shrinks); used by one cpu_decode_run at a time
const Grp *          g_grps = nullptr;       // groups of the running flush
std::vector<float>   g_scratch;
std::unique_ptr<std::atomic<int32_t>[]> g_done;
size_t g_done_cap = 0;
std::vector<InAct>   g_ins;
std::vector<size_t>  g_in_of;
uint8_t *            g_store = nullptr;

XD_TGT void process_item(const Item & m, const CpuDecodeJob & jb, const ActRef & act, const Tabs & T) {
    const Grp & gr = g_grps[m.grp];
    const Geom & g = jb.geom;
    const int nt = jb.n_tok;
    const int64_t n_use = jb.rows();
    const size_t dstride = nt > 1 ? jb.dst_stride / sizeof(float) : 0;
    if (gr.nparts == 1) {
        run_item(g, jb.w, act, m.split, m.chain, m.lane0, m.nl, m.sb, m.kb0, m.kb1, jb.dst, false, T, dstride, n_use);
    } else {
        const int64_t gc = g.sb_groups(m.split, m.sb);
        const size_t tsz = size_t(N_LANES * gc * GROUP_ROWS), psz = tsz * size_t(nt);
        float * scr = g_scratch.data() + gr.scr;
        run_item(g, jb.w, act, m.split, m.chain, m.lane0, m.nl, m.sb, m.kb0, m.kb1, scr + size_t(m.part) * psz, true, T, tsz, n_use);
        if (g_done[m.grp].fetch_add(1, std::memory_order_acq_rel) == gr.nparts - 1) {      // last part: reduce
            const int64_t Gs = g.sp_groups(m.split);
            for (int t = 0; t < nt; ++t) {
                float * dst = jb.dst + size_t(t) * dstride;
                const float * st = scr + size_t(t) * tsz;
                for (int64_t l = 0; l < N_LANES; ++l) {
                    const int64_t row0 = g.sp_row0(m.split) + (m.chain * N_LANES + l) * GROUP_ROWS * Gs + m.sb * g.gsb * GROUP_ROWS;
                    for (int64_t i = 0; i < gc; ++i) {
                        const int64_t r = row0 + i * GROUP_ROWS;
                        if (r >= n_use) break;
                        __m512 sum = _mm512_loadu_ps(st + (l * gc + i) * GROUP_ROWS);
                        for (int32_t p = 1; p < gr.nparts; ++p) sum = _mm512_add_ps(sum, _mm512_loadu_ps(st + size_t(p) * psz + (l * gc + i) * GROUP_ROWS));
                        const int64_t cnt = std::min<int64_t>(GROUP_ROWS, n_use - r);
                        _mm512_mask_storeu_ps(dst + r, __mmask16((1u << cnt) - 1u), sum);
                    }
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

const void * cpu_decode_act(size_t job) { return g_store + g_ins[g_in_of[job]].off; }

XD_TGT static void prepare_acts(const CpuDecodeJob * jobs, size_t nj) {
    // unique inputs (several matrices of a layer read the same activation)
    g_ins.clear();
    g_in_of.assign(nj, 0);
    size_t total = 0;
    for (size_t j = 0; j < nj; ++j) {
        const bool q80 = blocks32(jobs[j].geom.fmt);
        size_t e = g_ins.size();
        for (size_t q = 0; q < g_ins.size(); ++q) {
            if (g_ins[q].x == jobs[j].x && g_ins[q].k == jobs[j].geom.k && g_ins[q].q80 == q80 && g_ins[q].nt == jobs[j].n_tok &&
                g_ins[q].xs == jobs[j].x_stride) { e = q; break; }
        }
        if (e == g_ins.size()) {
            const int64_t k = jobs[j].geom.k;
            const size_t bytes = q80 ? size_t(k) + size_t(k / 32) * 4 : size_t(k / 256) * sizeof(Q8Kblk);
            const size_t ts = (bytes + 63) / 64 * 64;
            g_ins.push_back(InAct{ jobs[j].x, k, q80, total, bytes, jobs[j].n_tok, jobs[j].x_stride, ts });
            total += ts * size_t(jobs[j].n_tok);
        }
        g_in_of[j] = e;
    }
    if (g_act_store.size() < total + 64) g_act_store.resize(total + 64);
    g_store = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(g_act_store.data()) + 63) & ~uintptr_t(63));
    for (const InAct & ia : g_ins) {
        for (int t = 0; t < ia.nt; ++t) {
            uint8_t * p = g_store + ia.off + size_t(t) * ia.tstride;
            const float * x = reinterpret_cast<const float *>(reinterpret_cast<const char *>(ia.x) + size_t(t) * ia.xs);
            if (ia.q80) quant_q80(x, p, reinterpret_cast<float *>(p + ia.k), ia.k / 32);
            else        quant_q8k(x, reinterpret_cast<Q8Kblk *>(p), ia.k / 256);
        }
    }
}

std::vector<SplitPlan> make_split_plan(const Geom & g, double frac) {
    std::vector<SplitPlan> out(size_t(g.nsplit));
    for (int64_t s = 0; s < g.nsplit; ++s) {
        const int64_t Gs = g.sp_groups(s), nsb = g.sp_nsb(s);
        SplitPlan & p = out[size_t(s)];
        if (frac <= 0.0) continue;
        if (nsb >= 2) {
            int64_t best = 0;
            double bd = frac;                                   // k = 0: distance |0 - frac|
            for (int64_t k = 1; k <= nsb; ++k) {
                const double f = double(std::min<int64_t>(k * g.gsb, Gs)) / double(Gs);
                if (std::fabs(f - frac) < bd) { bd = std::fabs(f - frac); best = k; }
            }
            p.sb_npu = best;
            p.kb_npu = best ? g.nbk : 0;
        } else {
            const int64_t kb = std::min<int64_t>(g.nbk, int64_t(std::llround(frac * double(g.nbk))));
            p.sb_npu = kb ? nsb : 0;
            p.kb_npu = kb;
        }
    }
    return out;
}

void cpu_pack_npu_act(Fmt fmt, int64_t k_full, int64_t k_use, int64_t groups, int64_t n_w, int gsb, const void * act, uint8_t * dst) {
    (void) k_full;
    const size_t nact = act_chunks(fmt, k_use);
    thread_local std::vector<uint8_t> a;
    a.assign(nact * CHUNK, 0);
    const int64_t nbk = k_use / 256;
    const int32_t hdr[7] = { int32_t(groups), int32_t(k_use), int32_t(fmt_info(fmt).qt), int32_t(nact - 1), int32_t(n_w), int32_t(groups * nbk), int32_t(gsb) };
    std::memcpy(a.data(), hdr, sizeof(hdr));
    uint8_t * p = a.data() + 64;
    if (blocks32(fmt)) {          // q[K] signed | f32 d[K/32]   (CPU format: q + 128, then f32 d[K/32])
        const uint8_t * q = static_cast<const uint8_t *>(act);
        for (int64_t i = 0; i < k_use; ++i) p[i] = uint8_t(q[i] ^ 0x80);
        std::memcpy(p + k_use, q + k_full, size_t(k_use / 32) * 4);
    } else {                       // q[K] | f32 d[K/256] | i32 bsum32[K/32]
        const Q8Kblk * b = static_cast<const Q8Kblk *>(act);
        float * d = reinterpret_cast<float *>(p + k_use);
        int32_t * bs = reinterpret_cast<int32_t *>(p + k_use + size_t(nbk) * 4);
        for (int64_t i = 0; i < nbk; ++i) {
            std::memcpy(p + i * 256, b[i].q, 256);
            d[i] = b[i].d;
            for (int j = 0; j < 8; ++j) bs[i * 8 + j] = int32_t(b[i].bs[2 * j]) + int32_t(b[i].bs[2 * j + 1]);
        }
    }
    for (size_t r = 0; r < nact; ++r) {   // every chunk duplicated per lane (stride-0 DMA repeat is not supported)
        for (int l = 0; l < N_LANES; ++l) std::memcpy(dst + (r * N_LANES + l) * CHUNK, a.data() + r * CHUNK, CHUNK);
    }
}

namespace {

// The work of one flush (a set of independent matmuls): items + their groups. Built once per distinct flush (the decode graph repeats every token),
// together with the flush that follows it in the token (learned), so that the threads can already stream the first bytes of the NEXT flush into
// their L2 caches while the CPU ops between two flushes run (the DRAM is idle then).
struct FlushPlan {
    std::vector<Item> items;
    std::vector<Grp>  grps;
    size_t scr_total = 0;
    size_t n_static = 0;       // items [0, n_static) are distributed statically (item k -> thread k % T): their first slabs are what the previous flush prefetched
};
struct FlushRec {
    std::vector<CpuDecodeJob> jobs;     // snapshot (geometry, weights, plan; x/dst are stale)
    FlushPlan plan;
    uint64_t  next = 0;                 // signature of the flush that followed this one the last time
};
std::unordered_map<uint64_t, FlushRec> g_flushes;
uint64_t g_prev_sig = 0;

uint64_t flush_sig(const CpuDecodeJob * jobs, size_t nj) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix(nj);
    for (size_t j = 0; j < nj; ++j) {
        mix(reinterpret_cast<uintptr_t>(jobs[j].w)); mix(uint64_t(jobs[j].geom.n)); mix(uint64_t(jobs[j].geom.k)); mix(uint64_t(jobs[j].geom.fmt));
        mix(uint64_t(jobs[j].n_tok)); mix(uint64_t(jobs[j].rows()));
        if (jobs[j].plan) for (int64_t sp = 0; sp < jobs[j].geom.nsplit; ++sp) { mix(uint64_t(jobs[j].plan[sp].sb_npu)); mix(uint64_t(jobs[j].plan[sp].kb_npu)); }
        else mix(0x9e3779b97f4a7c15ull);
    }
    return h ? h : 1;
}

int env_int(const char * name, int def) { const char * e = std::getenv(name); return e && *e ? std::atoi(e) : def; }

void build_plan(const CpuDecodeJob * jobs, size_t nj, FlushPlan & P) {
    // work items: (split, chain, superblock) with the 4 lanes of the chain (splitting the lanes of a chain over threads breaks the linear
    // streams: 50 instead of 80 GB/s). Few big items cannot balance 10 threads, so a group is cut along K into parts when the batch has
    // fewer than ~4 items per thread.
    auto plan_of = [&](size_t j, int64_t s) { return jobs[j].plan ? jobs[j].plan[s] : SplitPlan{ 0, jobs[j].geom.nbk }; };
    // CPU k-block range of superblock sb (empty: the NPU does all of it)
    auto kb_range = [&](size_t j, int64_t s, int64_t sb, int64_t & k0, int64_t & k1) {
        const Geom & g = jobs[j].geom;
        const SplitPlan p = plan_of(j, s);
        k1 = g.nbk;
        if (!p.npu() || sb >= p.sb_npu) k0 = 0;
        else k0 = p.kb_npu;                     // K split: the NPU did [0, kb_npu) of these rows (== nbk: nothing left)
        return k0 < k1;
    };
    P.items.clear();
    P.grps.clear();
    size_t base_total = 0;
    for (size_t j = 0; j < nj; ++j) {
        const Geom & g = jobs[j].geom;
        for (int64_t s = 0; s < g.nsplit; ++s)
            for (int64_t sb = 0; sb < g.sp_nsb(s); ++sb) {
                int64_t k0, k1;
                if (kb_range(j, s, sb, k0, k1)) base_total += N_CHAINS;
            }
    }
    const int nthr = pf_threads();
    const size_t want = size_t(env_int("XD_WANT", 4) * nthr);
    const int64_t parts_wanted = base_total == 0 ? 1 : base_total >= want ? 1 : int64_t((want + base_total - 1) / base_total);
    // the LAST groups are cut into fine K-parts (about one k-block each): the cores are heterogeneous (5.1 / 3.3 GHz), so with coarse items the
    // slow cores are still busy with a big last item while the others idle (measured: the last thread finishes 17 us after the mean)
    const int tail_fine = env_int("XD_TAILP", 6);                 // max parts of a tail group (0: off)
    const size_t tail_groups = size_t(env_int("XD_TAILG", 2) * nthr);
    size_t n_groups_total = 0;
    for (size_t j = 0; j < nj; ++j) {
        const Geom & g = jobs[j].geom;
        for (int64_t s = 0; s < g.nsplit; ++s) {
            const int64_t Gs = g.sp_groups(s);
            for (int64_t sb = 0; sb < g.sp_nsb(s); ++sb) {
                int64_t k0, k1;
                if (!kb_range(j, s, sb, k0, k1)) continue;
                for (int64_t chain = 0; chain < N_CHAINS; ++chain)
                    if (g.sp_row0(s) + (chain * N_LANES) * GROUP_ROWS * Gs + sb * g.gsb * GROUP_ROWS < jobs[j].rows()) ++n_groups_total;
            }
        }
    }
    size_t scr_total = 0;
    size_t gidx = 0;
    for (size_t j = 0; j < nj; ++j) {
        const Geom & g = jobs[j].geom;
        for (int64_t s = 0; s < g.nsplit; ++s) {
            const int64_t Gs = g.sp_groups(s), nsb = g.sp_nsb(s);
            for (int64_t sb = 0; sb < nsb; ++sb) {
                int64_t k0, k1;
                if (!kb_range(j, s, sb, k0, k1)) continue;
                for (int64_t chain = 0; chain < N_CHAINS; ++chain) {
                    if (g.sp_row0(s) + (chain * N_LANES) * GROUP_ROWS * Gs + sb * g.gsb * GROUP_ROWS >= jobs[j].rows()) continue;   // padding only / beyond n_use
                    int64_t P_ = std::max<int64_t>(1, std::min<int64_t>(parts_wanted, (k1 - k0) / 2));
                    if (tail_fine > 0 && gidx + tail_groups >= n_groups_total) P_ = std::max<int64_t>(P_, std::min<int64_t>(tail_fine, k1 - k0));
                    ++gidx;
                    const int64_t gc = g.sb_groups(s, sb);
                    const int32_t grp = int32_t(P.grps.size());
                    P.grps.push_back(Grp{ int32_t(P_), scr_total });
                    if (P_ > 1) scr_total += size_t(P_) * size_t(N_LANES * gc * GROUP_ROWS) * size_t(jobs[j].n_tok);
                    for (int64_t part = 0; part < P_; ++part) {
                        P.items.push_back(Item{ int32_t(j), int32_t(s), int32_t(chain), 0, N_LANES, int32_t(sb),
                                                int32_t(k0 + (k1 - k0) * part / P_), int32_t(k0 + (k1 - k0) * (part + 1) / P_), int32_t(part), grp });
                    }
                }
            }
        }
    }
    P.scr_total = scr_total;
    // static prologue: ~PFB bytes of items per thread
    const size_t T = size_t(pf_threads());
    const size_t pfb = size_t(env_int("XD_PFB", 384)) * 1024;
    P.n_static = 0;
    if (pfb && !P.items.empty()) {
        const Item & it0 = P.items[0];
        const Geom & g = jobs[it0.job].geom;
        const size_t item_bytes = size_t(it0.nl) * size_t(g.sb_groups(it0.split, it0.sb)) * g.rec_bytes * size_t(std::max(1, it0.kb1 - it0.kb0));
        const size_t m = std::max<size_t>(1, pfb / std::max<size_t>(1, item_bytes));
        P.n_static = std::min(T * m, P.items.size());
    }
}

// Physical cores: the first hardware thread of each core (team thread i runs on cores[i]) and its SMT sibling (helper i), if any.
struct CoreMap { std::vector<int> first, sibling; };
const CoreMap & core_map() {
    static const CoreMap m = [] {
        CoreMap r;
        for (int c = 0; c < 4096; ++c) {
            char path[128];
            std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            FILE * f = std::fopen(path, "r");
            if (!f) break;
            int a = -1, b = -1;
            const int n = std::fscanf(f, "%d", &a);
            if (n == 1 && a == c) {
                int ch = std::fgetc(f);
                if (ch == ',' || ch == '-') { if (std::fscanf(f, "%d", &b) != 1) b = -1; }
                r.first.push_back(c);
                r.sibling.push_back(b);
            }
            std::fclose(f);
        }
        return r;
    }();
    return m;
}

// Pin the OpenMP team to one hardware thread per physical core (so the L2 of a core keeps what its helper loaded). The master is unpinned
// again at the end of every flush (cpu_decode_run): OpenMP creates the threads of a larger team later (the prefill loops use pf_threads(), the CPU
// backend -t) as children of the master, and they would inherit its one-core mask (pp512 at -t 6: 10 instead of 516 t/s).
cpu_set_t g_master_mask;
bool      g_master_pinned = false;

void pin_team_thread(int tid) {
    thread_local int pinned_tid = -1;
    if (env_int("XD_PIN", 1) == 0) return;
    if (tid == 0) {
        if (g_master_pinned) return;
        if (sched_getaffinity(0, sizeof(g_master_mask), &g_master_mask) != 0) return;
        g_master_pinned = true;
    } else {
        if (pinned_tid == tid) return;
        pinned_tid = tid;
    }
    const CoreMap & cm = core_map();
    if (cm.first.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cm.first[size_t(tid) % cm.first.size()], &set);
    sched_setaffinity(0, sizeof(set), &set);
}

// ---- cross-flush weight streaming by SMT helper threads ---------------------------------------------------------------------------------------
// The CPU ops between two flushes leave the DRAM idle. The sibling hardware thread of every team thread loads the first bytes of the static items
// of the NEXT flush (plain loads: software prefetches were measured to be dropped on this CPU) into the L2 it shares with the team thread, so the
// next flush starts with a warm L2. Stopped as soon as the next flush starts (epoch).
std::atomic<uint64_t> g_epoch{0};

struct alignas(64) Helper {
    std::atomic<uint32_t> seq{0};
    std::atomic<uint32_t> sleeping{0};
    const FlushRec *      rec = nullptr;
    int                   tid = 0, nth = 1, cpu = -1;
};
std::vector<Helper *> g_helpers;

long futex_call(std::atomic<uint32_t> * addr, int op, uint32_t val) {
    return syscall(SYS_futex, reinterpret_cast<uint32_t *>(addr), op, val, nullptr, nullptr, 0);
}

XD_TGT void helper_stream(const FlushRec & rec, int tid, int nth, uint64_t ep) {
    const FlushPlan & P = rec.plan;
    const size_t budget = size_t(env_int("XD_PFB", 384)) * 1024;
    size_t n_mine = 0;
    for (size_t k = size_t(tid); k < P.n_static; k += size_t(nth)) ++n_mine;
    if (n_mine == 0) return;
    const size_t per_item = budget / n_mine;
    __m512i acc = _mm512_setzero_si512();
    for (size_t k = size_t(tid); k < P.n_static; k += size_t(nth)) {
        const Item & m = P.items[k];
        const Geom & g = rec.jobs[m.job].geom;
        const uint8_t * w = rec.jobs[m.job].w;
        const int64_t gc = g.sb_groups(m.split, m.sb);
        const uint8_t * cb = w + g.sp_off(m.split) + size_t(m.chain) * g.sp_chain_bytes(m.split);
        const size_t slab_full = g.slab_bytes(g.gsb), slab = g.slab_bytes(gc);
        const size_t s0_sb = size_t(m.sb) * size_t(g.nbk) * slab_full;
        size_t left = per_item;
        for (int64_t kb = m.kb0; kb < m.kb1 && left > 0; ++kb) {
            const uint8_t * base = cb + ((s0_sb + size_t(kb) * slab) / CHUNK) * (N_LANES * CHUNK);
            const size_t bytes = std::min(slab / CHUNK * (N_LANES * CHUNK), left);
            for (size_t o = 0; o < bytes; o += 256) {
                acc = _mm512_xor_si512(acc, _mm512_loadu_si512(base + o));
                acc = _mm512_xor_si512(acc, _mm512_loadu_si512(base + o + 64));
                acc = _mm512_xor_si512(acc, _mm512_loadu_si512(base + o + 128));
                acc = _mm512_xor_si512(acc, _mm512_loadu_si512(base + o + 192));
                if ((o & 4095) == 0 && g_epoch.load(std::memory_order_relaxed) != ep) return;
            }
            left -= bytes;
        }
    }
    static volatile uint32_t sink;
    sink = uint32_t(_mm512_reduce_add_epi32(acc));
}

XD_TGT void helper_main(Helper * h) {
    if (h->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(h->cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
    uint32_t seen = 0;
    for (;;) {
        // spin for a while (the next request usually comes within a few 100 us), then sleep on the futex
        const double t0 = now_s();
        uint32_t q;
        static const int spin_mode = env_int("XD_HSPIN", 0);
        while ((q = h->seq.load(std::memory_order_acquire)) == seen) {
            if (spin_mode == 1) {
                _mm_monitorx(&h->seq, 0, 0);
                if (h->seq.load(std::memory_order_acquire) == seen) _mm_mwaitx(0, 0, 100000);     // wake on write to the line or after ~100k cycles
            } else {
                _mm_pause();
            }
            if (spin_mode == 2 || now_s() - t0 > 2e-3) {
                h->sleeping.store(1, std::memory_order_seq_cst);
                if (h->seq.load(std::memory_order_seq_cst) == seen) futex_call(&h->seq, FUTEX_WAIT_PRIVATE, seen);
                h->sleeping.store(0, std::memory_order_relaxed);
            }
        }
        seen = q;
        static const int noload = env_int("XD_HNOLOAD", 0);
        if (!noload) helper_stream(*h->rec, h->tid, h->nth, g_epoch.load(std::memory_order_acquire));
    }
}

void start_helpers(int nthreads) {
    static bool started = false;
    if (started) return;
    started = true;
    if (env_int("XD_XPF", 0) == 0) return;      // EXPERIMENT, off: SMT helper threads cost more than they gain (stage7 RESULTS)
    const CoreMap & cm = core_map();
    for (int t = 0; t < nthreads && size_t(t) < cm.first.size(); ++t) {
        if (cm.sibling[size_t(t)] < 0) { for (Helper * h : g_helpers) (void) h; g_helpers.clear(); return; }   // no SMT: no helpers
    }
    for (int t = 0; t < nthreads && size_t(t) < cm.first.size(); ++t) {
        Helper * h = new Helper;
        h->tid = t; h->nth = nthreads; h->cpu = cm.sibling[size_t(t)];
        g_helpers.push_back(h);
        std::thread([h] { helper_main(h); }).detach();
    }
}

inline void helper_request(int tid, const FlushRec * rec) {
    if (size_t(tid) >= g_helpers.size()) return;
    Helper * h = g_helpers[size_t(tid)];
    h->rec = rec;
    h->seq.fetch_add(1, std::memory_order_seq_cst);
    if (h->sleeping.load(std::memory_order_seq_cst)) futex_call(&h->seq, FUTEX_WAKE_PRIVATE, 1);
}

} // namespace

static double g_tend[64], g_tbeg[64], g_tail_sum = 0, g_start_skew = 0, g_tail_spread = 0; static long g_tail_n = 0, g_tail_last[64];
static struct TailP { ~TailP() { if (g_tail_n) { fprintf(stderr, "TAIL: %ld flushes, avg (last finish - mean finish) %.1f us, spread (last-first) %.1f us, mean start delay %.1f us; last thread histogram:", g_tail_n, g_tail_sum / g_tail_n * 1e6, g_tail_spread / g_tail_n * 1e6, g_start_skew / g_tail_n * 1e6); for (int t = 0; t < 10; ++t) fprintf(stderr, " %ld", g_tail_last[t]); fprintf(stderr, "\n"); } } } g_tailp;
void cpu_decode_run(const CpuDecodeJob * jobs, size_t nj, const std::function<void()> & master_hook) {
    if (nj == 0) return;
    static const bool tail_dbg = std::getenv("XD_TAIL") != nullptr;
    const Tabs & T = tabs();
    prepare_acts(jobs, nj);

    // the plan of this flush (cached) and the learned successor
    const uint64_t sig = flush_sig(jobs, nj);
    FlushRec & rec = g_flushes[sig];
    if (rec.jobs.empty()) {
        rec.jobs.assign(jobs, jobs + nj);
        build_plan(rec.jobs.data(), nj, rec.plan);
    }
    if (g_prev_sig && g_prev_sig != sig) g_flushes[g_prev_sig].next = sig;       // (g_flushes may rehash: re-take the reference)
    g_prev_sig = sig;
    FlushRec & cur = g_flushes[sig];
    const FlushPlan & P = cur.plan;
    start_helpers(pf_threads());
    g_epoch.fetch_add(1, std::memory_order_release);         // stop the helpers still streaming for this flush's predecessor
    const FlushRec * nxt = (cur.next && !g_helpers.empty()) ? &g_flushes[cur.next] : nullptr;
    if (g_scratch.size() < P.scr_total) g_scratch.resize(P.scr_total);
    if (g_done_cap < P.grps.size()) { g_done.reset(new std::atomic<int32_t>[P.grps.size()]); g_done_cap = P.grps.size(); }
    for (size_t i = 0; i < P.grps.size(); ++i) g_done[i].store(0, std::memory_order_relaxed);
    g_grps = P.grps.data();                // process_item reads the groups through g_grps
    const double tw0 = now_s();
    std::atomic<size_t> next{0};
    const size_t n_items = P.items.size();
    const size_t n_static = P.n_static;

#if defined(_OPENMP)
    #pragma omp parallel num_threads(pf_threads())
#endif
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
#else
        const int tid = 0, nth = 1;
#endif
        pin_team_thread(tid);
        if (tid == 0 && master_hook) master_hook();
        const double tb = now_s();
        auto run = [&](const Item & m) {
            const CpuDecodeJob & jb = jobs[m.job];
            const InAct & ia = g_ins[g_in_of[m.job]];
            ActRef act;
            const uint8_t * p = g_store + ia.off;
            if (ia.q80) { act.q0 = p; act.d0 = reinterpret_cast<const float *>(p + ia.k); }
            else        { act.k8 = reinterpret_cast<const Q8Kblk *>(p); }
            act.nt = ia.nt;
            act.tstride = ia.tstride;
            process_item(m, jb, act, T);
        };
        for (size_t k = size_t(tid); k < n_static; k += size_t(nth)) run(P.items[k]);
        for (size_t it = n_static + next.fetch_add(1); it < n_items; it = n_static + next.fetch_add(1)) run(P.items[it]);
        const double te = now_s();
        g_busy_ns.fetch_add(uint64_t((te - tb) * 1e9));
        if (tail_dbg) { g_tend[tid] = te; g_tbeg[tid] = tb; }
        if (nxt) helper_request(tid, nxt);
    }
    if (g_master_pinned) {
        sched_setaffinity(0, sizeof(g_master_mask), &g_master_mask);
        g_master_pinned = false;
    }
    g_wall_ns.fetch_add(uint64_t((now_s() - tw0) * 1e9));
    if (tail_dbg) {
        double mx = 0, sum = 0, sb = 0; int last = 0; double mn = 1e18;
        for (int t = 0; t < 10; ++t) { if (g_tend[t] > mx) { mx = g_tend[t]; last = t; } sum += g_tend[t]; sb += g_tbeg[t]; mn = std::min(mn, g_tend[t]); }
        g_tail_sum += mx - sum / 10; g_tail_n++; g_tail_last[last]++; g_start_skew += sb / 10 - tw0; g_tail_spread += mx - mn;
    }
}

#else   // not x86-64 GCC/clang

bool cpu_decode_available() { return false; }
void cpu_decode_stats(double & b, double & w, int & t) { b = w = 0; t = 1; }
void cpu_decode_run(const CpuDecodeJob *, size_t, const std::function<void()> &) {}
const void * cpu_decode_act(size_t) { return nullptr; }
std::vector<SplitPlan> make_split_plan(const layout::Geom & g, double) { return std::vector<SplitPlan>(size_t(g.nsplit)); }
void cpu_pack_npu_act(layout::Fmt, int64_t, int64_t, int64_t, int64_t, int, const void *, uint8_t *) {}

#endif

} // namespace xdna2

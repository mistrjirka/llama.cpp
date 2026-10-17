#include "ggml-xdna2-bfp.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace xdna2 {

// Team size of the host prefill loops. The default OpenMP team (all hardware threads) oversubscribes the cores while the
// CPU backend's team (one thread per physical core, llama.cpp's default) is still spin-waiting: measured 2.5x slower
// prefill end to end on the Ryzen AI 9 365. Default: physical cores; GGML_XDNA2_THREADS overrides.
int pf_threads() {
    static const int n = [] {
        if (const char * e = std::getenv("GGML_XDNA2_THREADS")) {
            const int v = std::atoi(e);
            if (v > 0) return v;
        }
        int hw = int(std::thread::hardware_concurrency());
        if (hw <= 0) hw = 1;
        if (FILE * f = std::fopen("/sys/devices/system/cpu/smt/active", "r")) {
            int smt = 0;
            if (std::fscanf(f, "%d", &smt) == 1 && smt == 1 && hw > 1) hw /= 2;
            std::fclose(f);
        }
        return hw;
    }();
    return n;
}

namespace {


// 8 floats -> shared exponent + 8 int8 mantissas, bit exact with bfpmodel.enc_rne
inline void enc8(const float * x, uint8_t * out /* 9 bytes */) {
    uint32_t bits[8], m[8];
    int e[8], sign[8];
    int emax = 0;
    for (int i = 0; i < 8; ++i) {
        std::memcpy(&bits[i], &x[i], 4);
        e[i] = int((bits[i] >> 23) & 0xFF);
        sign[i] = int(bits[i] >> 31);
        m[i] = (bits[i] & 0x7FFFFF) | (e[i] != 0 ? (1u << 23) : 0u);
        emax = std::max(emax, e[i]);
    }
    int64_t qm[8];
    auto round_all = [&](int E) {
        bool over = false;
        for (int i = 0; i < 8; ++i) {
            const int total = std::min(17 + (E - e[i]), 62);
            const uint64_t mm = m[i];
            const uint64_t q0 = mm >> total;
            const uint64_t rem = mm & ((uint64_t(1) << total) - 1);
            const uint64_t half = uint64_t(1) << (total - 1);
            const bool up = rem > half || (rem == half && (q0 & 1));
            qm[i] = int64_t(q0 + (up ? 1 : 0));
            over |= (sign[i] == 0 && qm[i] > 127) || (sign[i] == 1 && qm[i] > 128);
        }
        return over;
    };
    int E = emax;
    if (round_all(E)) {
        E += 1;
        round_all(E);
    }
    out[0] = uint8_t(E);
    for (int i = 0; i < 8; ++i) {
        const int q = sign[i] ? -int(std::min<int64_t>(qm[i], 128)) : int(std::min<int64_t>(qm[i], 127));
        out[1 + i] = uint8_t(int8_t(q));
    }
}

#if defined(__x86_64__)
#define XDNA2_HAVE_AVX512 1
// AVX-512VL version of enc8 (one ymm = one block of 8): RNE is done with the add-(half-1+lsb)-and-shift form, the shift is clamped to 25
// (m < 2^24, so every larger shift rounds to 0 exactly like the 64-bit reference above).
#define XDNA2_TARGET_AVX512 __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq")))
XDNA2_TARGET_AVX512
static inline bool enc8_round(__m256i E, __m256i e, __m256i m, __m256i lim, __m256i & qm) {
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i total = _mm256_min_epi32(_mm256_add_epi32(_mm256_set1_epi32(17), _mm256_sub_epi32(E, e)), _mm256_set1_epi32(25));
    const __m256i half_m1 = _mm256_sub_epi32(_mm256_sllv_epi32(one, _mm256_sub_epi32(total, one)), one);
    const __m256i lsb = _mm256_and_si256(_mm256_srlv_epi32(m, total), one);
    qm = _mm256_srlv_epi32(_mm256_add_epi32(_mm256_add_epi32(m, half_m1), lsb), total);
    return _mm256_cmpgt_epi32_mask(qm, lim) != 0;
}
XDNA2_TARGET_AVX512
static inline void enc8_avx(const float * x, uint8_t * out) {
    const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi32(1);
    const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x));
    const __m256i e = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xFF));
    const __m256i sign = _mm256_srli_epi32(bits, 31);
    __m256i m = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFFFF));
    m = _mm256_mask_or_epi32(m, _mm256_cmpneq_epi32_mask(e, zero), m, _mm256_set1_epi32(1 << 23));
    __m256i mx = _mm256_max_epi32(e, _mm256_permute2x128_si256(e, e, 1));
    mx = _mm256_max_epi32(mx, _mm256_shuffle_epi32(mx, 0x4E));
    mx = _mm256_max_epi32(mx, _mm256_shuffle_epi32(mx, 0xB1));
    const __m256i lim = _mm256_add_epi32(_mm256_set1_epi32(127), sign);
    __m256i qm, E = mx;
    if (enc8_round(E, e, m, lim, qm)) {
        E = _mm256_add_epi32(E, one);
        enc8_round(E, e, m, lim, qm);
    }
    const __m256i qmin = _mm256_min_epi32(qm, lim);
    const __m256i neg = _mm256_sub_epi32(zero, sign);                       // 0 or -1
    const __m256i q = _mm256_add_epi32(_mm256_xor_si256(qmin, neg), sign);  // sign ? -qmin : qmin
    out[0] = uint8_t(_mm256_cvtsi256_si32(E));
    _mm_storel_epi64(reinterpret_cast<__m128i *>(out + 1), _mm256_cvtepi32_epi8(q));
}
XDNA2_TARGET_AVX512
static inline void store_line_nt(float * d, const float * s0, const float * s1) {     // 64 B aligned destination
    const __m512 v = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(s0)), _mm256_loadu_ps(s1), 1);
    _mm512_stream_ps(d, v);
}
static const bool g_have_avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq");
#endif

} // namespace

void pf_enc8_ref(const float * x, uint8_t * out) { enc8(x, out); }
void pf_enc8_fast(const float * x, uint8_t * out) {
#ifdef XDNA2_HAVE_AVX512
    if (g_have_avx512) { enc8_avx(x, out); return; }
#endif
    enc8(x, out);
}

void pf_encode_tile(uint8_t * slabs, const float * x, size_t row_stride_bytes, int64_t n_tok, int64_t k, int tile) {
    const int64_t nbk = k / 256;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(pf_threads())
#endif
    for (int64_t kb = 0; kb < nbk; ++kb) {
        uint8_t * slab = slabs + size_t(kb) * PF_SLAB_BYTES;
        for (int z = 0; z < 4; ++z) {
            for (int kb32 = 0; kb32 < 32; ++kb32) {
                for (int mb = 0; mb < 2; ++mb) {
                    const int rb = 2 * z + mb;
                    uint8_t * blk = slab + (size_t(z * 32 + kb32) * 2 + size_t(mb)) * 72;
                    for (int m8 = 0; m8 < 8; ++m8) {
                        const int64_t tok = int64_t(tile) * PF_TILE_TOK + rb * 8 + m8;
                        if (tok < n_tok) {
                            const float * src = reinterpret_cast<const float *>(reinterpret_cast<const char *>(x) + size_t(tok) * row_stride_bytes) + kb * 256 + kb32 * 8;
                            pf_enc8_fast(src, blk + m8 * 9);
                        } else {
                            std::memset(blk + m8 * 9, 0, 9);
                        }
                    }
                }
            }
        }
    }
}

void pf_build_a_tile(uint8_t * a_tile, const uint8_t * slabs, int64_t nbk, int64_t nsb, int qtype, int64_t groups) {
    const int32_t hdr[4] = { int32_t(nsb), int32_t(nbk), int32_t(qtype), int32_t(groups) };
    std::memcpy(a_tile, hdr, sizeof(hdr));              // the header object (the rest of it is never read)
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(nbk >= 8) num_threads(pf_threads())
#endif
    for (int64_t kb = 0; kb < nbk; ++kb) {
        std::memcpy(a_tile + PF_SLAB_BYTES + size_t(kb) * PF_SLAB_BYTES, slabs + size_t(kb) * PF_SLAB_BYTES, PF_SLAB_BYTES);
    }
}

void pf_unpermute_tile(const float * y, int tile, int64_t n_tok, const layout::Geom & g, int64_t split, float * dst, size_t dst_row_stride) {
    using namespace layout;
    const int64_t Gs = g.sp_groups(split), nsb = g.sp_nsb(split), row_base = g.sp_row0(split);
    const int64_t tok0 = int64_t(tile) * PF_TILE_TOK;
    const int ntok = int(std::min<int64_t>(PF_TILE_TOK, n_tok - tok0));
#if defined(_OPENMP)
    #pragma omp parallel num_threads(pf_threads())
#endif
    {
#if defined(_OPENMP)
    #pragma omp for schedule(static) collapse(2) nowait
#endif
    for (int64_t c = 0; c < 8; ++c) {
        for (int64_t sb = 0; sb < nsb; ++sb) {
            const int64_t gc = g.sb_groups(split, sb);
            for (int64_t k = 0; k < 4; ++k) {
                for (int64_t gl = 0; gl < gc; ++gl) {
                    const int64_t row0 = row_base + ((c * 4 + k) * Gs + sb * GSB + gl) * 16;
                    if (row0 >= g.n) continue;
                    const float * blk = y + (((c * nsb + sb) * 4 + k) * GSB + gl) * 1024;     // [mb 8][nb 2][m8 8][n8 8]
                    const bool full = row0 + 16 <= g.n;
                    for (int mb = 0; mb < 8; ++mb) {
                        for (int m8 = 0; m8 < 8; ++m8) {
                            const int t = mb * 8 + m8;
                            if (t >= ntok) continue;
                            float * d = reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + size_t(tok0 + t) * dst_row_stride) + row0;
                            const float * s0 = blk + (mb * 2 + 0) * 64 + m8 * 8, * s1 = blk + (mb * 2 + 1) * 64 + m8 * 8;
                            if (full) {
#ifdef XDNA2_HAVE_AVX512
                                if (g_have_avx512 && ((uintptr_t) d & 63) == 0) { store_line_nt(d, s0, s1); continue; }
#endif
                                std::memcpy(d, s0, 32);
                                std::memcpy(d + 8, s1, 32);
                            } else {
                                const int64_t cnt = g.n - row0;
                                for (int64_t i = 0; i < cnt; ++i) d[i] = i < 8 ? s0[i] : s1[i - 8];
                            }
                        }
                    }
                }
            }
        }
    }
#ifdef XDNA2_HAVE_AVX512
    _mm_sfence();      // make this thread's non-temporal stores globally visible before the region ends
#endif
    }
}
} // namespace xdna2

#include "ggml-xdna2-bfp.h"

#include <algorithm>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace xdna2 {

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

} // namespace

void pf_encode_acts(uint8_t * slabs, const float * x, size_t row_stride_bytes, int64_t n_tok, int64_t k, int tile0, int ntiles) {
    const int64_t nbk = k / 256;
    const int64_t jobs = int64_t(ntiles) * nbk;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t job = 0; job < jobs; ++job) {
        const int64_t t = job / nbk, kb = job % nbk;
        uint8_t * slab = slabs + size_t(job) * PF_SLAB_BYTES;
        for (int z = 0; z < 4; ++z) {
            for (int kb32 = 0; kb32 < 32; ++kb32) {
                for (int mb = 0; mb < 2; ++mb) {
                    const int rb = 2 * z + mb;
                    uint8_t * blk = slab + (size_t(z * 32 + kb32) * 2 + size_t(mb)) * 72;
                    for (int m8 = 0; m8 < 8; ++m8) {
                        const int64_t tok = int64_t(tile0 + t) * PF_TILE_TOK + rb * 8 + m8;
                        if (tok < n_tok) {
                            const float * src = reinterpret_cast<const float *>(reinterpret_cast<const char *>(x) + size_t(tok) * row_stride_bytes) + kb * 256 + kb32 * 8;
                            enc8(src, blk + m8 * 9);
                        } else {
                            std::memset(blk + m8 * 9, 0, 9);
                        }
                    }
                }
            }
        }
    }
}

void pf_build_a_bo(uint8_t * a_bo, size_t tile_stride, const uint8_t * slabs, int ntiles, int64_t nbk, int64_t nsb, int qtype, int64_t groups) {
    const int32_t hdr[4] = { int32_t(nsb), int32_t(nbk), int32_t(qtype), int32_t(groups) };
    const int64_t jobs = int64_t(ntiles) * nsb;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t job = 0; job < jobs; ++job) {
        const int64_t t = job / nsb, s = job % nsb;
        uint8_t * tile = a_bo + size_t(t) * tile_stride;
        if (s == 0) std::memcpy(tile, hdr, sizeof(hdr));              // the header object (rest of it is never read)
        std::memcpy(tile + PF_SLAB_BYTES + size_t(s) * size_t(nbk) * PF_SLAB_BYTES, slabs + size_t(t) * size_t(nbk) * PF_SLAB_BYTES, size_t(nbk) * PF_SLAB_BYTES);
    }
}

void pf_unpermute(const float * y, int ntiles, int tile0, int64_t n_tok, const layout::Geom & g, int64_t split, float * dst, size_t dst_row_stride) {
    using namespace layout;
    const int64_t Gs = g.sp_groups(split), nsb = g.sp_nsb(split), row_base = g.sp_row0(split);
    const size_t y_tile = pf_y_tile_bytes(nsb) / sizeof(float);
    const int64_t jobs = int64_t(ntiles) * 8 * nsb;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t job = 0; job < jobs; ++job) {
        const int64_t t = job / (8 * nsb), c = (job / nsb) % 8, sb = job % nsb;
        const float * ys = y + size_t(t) * y_tile;
        const int64_t gc = g.sb_groups(split, sb);
        for (int64_t k = 0; k < 4; ++k) {
            for (int64_t gl = 0; gl < gc; ++gl) {
                const int64_t row0 = row_base + ((c * 4 + k) * Gs + sb * GSB + gl) * 16;
                if (row0 >= g.n) continue;
                const int64_t cnt = std::min<int64_t>(16, g.n - row0);
                const float * blk = ys + ((((c * nsb + sb) * 4 + k) * GSB + gl) * 1024);     // [mb 8][nb 2][m8 8][n8 8]
                for (int mb = 0; mb < 8; ++mb) {
                    for (int m8 = 0; m8 < 8; ++m8) {
                        const int64_t tok = int64_t(tile0 + t) * PF_TILE_TOK + mb * 8 + m8;
                        if (tok >= n_tok) continue;
                        float * d = reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + size_t(tok) * dst_row_stride) + row0;
                        for (int nb = 0; nb < 2; ++nb) {
                            const float * s = blk + ((mb * 2 + nb) * 8 + m8) * 8;
                            const int64_t lim = std::min<int64_t>(8, cnt - nb * 8);
                            for (int n8 = 0; n8 < lim; ++n8) d[nb * 8 + n8] = s[n8];
                        }
                    }
                }
            }
        }
    }
}

} // namespace xdna2

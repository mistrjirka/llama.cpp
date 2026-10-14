#include "ggml-xdna2-compute.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// ggml-base (ggml-quants.h): Q8_K block = { float d; int8 qs[256]; int16 bsums[16] } (292 B)
extern "C" void quantize_row_q8_K_ref(const float * x, void * y, int64_t k);

namespace xdna2 {

using namespace layout;

ActKind act_kind(Fmt fmt) {
    return fmt == Fmt::Q8_0 ? ActKind::Q8_0F : ActKind::Q8_K;
}

void quantize_acts(ActQ & out, const Geom & g, const float * src, int64_t n_tok, size_t row_stride_bytes) {
    out.kind  = act_kind(g.fmt);
    out.n_tok = n_tok;
    const int64_t k = g.k;

    if (out.kind == ActKind::Q8_0F) {
        const int64_t kb = k / QK32;
        out.k_pad = k;
        out.nd    = kb;
        out.nbs   = 0;
        out.q.assign(size_t(n_tok) * size_t(k), 0);
        out.d.assign(size_t(n_tok) * size_t(kb), 0.0f);
        out.bsum.clear();
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) if(n_tok * k >= (1 << 16))
#endif
        for (int64_t t = 0; t < n_tok; ++t) {
            const float * x = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src) + size_t(t) * row_stride_bytes);
            int8_t * q = out.q.data() + size_t(t) * size_t(k);
            float  * d = out.d.data() + size_t(t) * size_t(kb);
            for (int64_t b = 0; b < kb; ++b) {
                float amax = 0.0f;
                for (int i = 0; i < QK32; ++i) {
                    amax = std::max(amax, std::fabs(x[b * QK32 + i]));
                }
                const float dd = amax / 127.0f;
                const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
                d[b] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(dd));      // ggml block_q8_0 stores d as fp16
                for (int i = 0; i < QK32; ++i) {
                    q[b * QK32 + i] = int8_t(std::roundf(x[b * QK32 + i] * id));
                }
            }
        }
        return;
    }

    // Q8_K: ggml's own quantizer (block: float d; int8 qs[256]; int16 bsums[16])
    const int64_t nb = k / 256;
    out.k_pad = k;
    out.nd    = nb;
    out.nbs   = k / QK32;
    out.q.assign(size_t(n_tok) * size_t(k), 0);
    out.d.assign(size_t(n_tok) * size_t(nb), 0.0f);
    out.bsum.assign(size_t(n_tok) * size_t(out.nbs), 0);
    const size_t blk_bytes = ggml_row_size(GGML_TYPE_Q8_K, 256);
    GGML_ASSERT(blk_bytes == 292);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n_tok * k >= (1 << 16))
#endif
    for (int64_t t = 0; t < n_tok; ++t) {
        const float * x = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src) + size_t(t) * row_stride_bytes);
        std::vector<uint8_t> tmp(size_t(nb) * blk_bytes);
        quantize_row_q8_K_ref(x, tmp.data(), k);
        int8_t  * q  = out.q.data() + size_t(t) * size_t(k);
        float   * d  = out.d.data() + size_t(t) * size_t(nb);
        int32_t * bs = out.bsum.data() + size_t(t) * size_t(out.nbs);
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t * blk = tmp.data() + size_t(b) * blk_bytes;
            std::memcpy(&d[b], blk, 4);
            std::memcpy(q + b * 256, blk + 4, 256);
            for (int j = 0; j < 8; ++j) {
                int16_t a, c;
                std::memcpy(&a, blk + 260 + 4 * j, 2);
                std::memcpy(&c, blk + 260 + 4 * j + 2, 2);
                bs[b * 8 + j] = int32_t(a) + int32_t(c);
            }
        }
    }
}

size_t act_bo_bytes(const Geom & g, int m) {
    const size_t sv = g.fmt == Fmt::Q8_0 ? size_t(g.k) + size_t(g.k) / 8
                                         : size_t(g.k) + size_t(g.nbk) * 4 + size_t(g.k) / 8;
    const size_t bytes = (64 + size_t(m) * sv + 63) & ~size_t(63);
    const size_t recs = (bytes + g.rec_bytes - 1) / g.rec_bytes;
    return recs * g.rec_bytes * N_LANES;
}

void pack_act_bo(const ActQ & act, const Geom & g, int64_t t0, int m, uint8_t * dst) {
    const size_t total = act_bo_bytes(g, m);
    const size_t recs = total / (g.rec_bytes * N_LANES);
    std::vector<uint8_t> a(recs * g.rec_bytes, 0);
    const int32_t hdr[3] = { int32_t(g.groups), int32_t(g.k), int32_t(m) };
    std::memcpy(a.data(), hdr, sizeof(hdr));
    uint8_t * p = a.data() + 64;
    for (int i = 0; i < m; ++i) {
        const int64_t t = t0 + i;
        std::memcpy(p, act.q_row(t), size_t(act.k_pad));
        p += act.k_pad;
        std::memcpy(p, act.d_row(t), size_t(act.nd) * sizeof(float));
        p += act.nd * sizeof(float);
        if (act.nbs) {
            std::memcpy(p, act.bs_row(t), size_t(act.nbs) * sizeof(int32_t));
            p += act.nbs * sizeof(int32_t);
        }
    }
    for (size_t r = 0; r < recs; ++r) {
        std::memcpy(dst + (r * N_LANES + 0) * g.rec_bytes, a.data() + r * g.rec_bytes, g.rec_bytes);
        std::memcpy(dst + (r * N_LANES + 1) * g.rec_bytes, a.data() + r * g.rec_bytes, g.rec_bytes);
    }
}

float fp16_to_f32_qk_kernel(uint16_t h) {
    const uint32_t em = h & 0x7FFFu;
    uint32_t f = (em << 13) + 0x38000000u;
    if (em == 0) f = 0;
    float v;
    std::memcpy(&v, &f, sizeof(v));
    if (em < 0x400u) {
        v = float(int32_t(em)) * 5.9604644775390625e-08f;   // zero and subnormals
    }
    return (h & 0x8000u) ? -v : v;
}

namespace {

// ---- record decoding, straight from the device layout (the way the kernels read it) -------------------------------
// Decoded record: 256 weights per row (16 rows) as small signed ints, laid out [i][n] so that loops over n vectorize.
struct DecRec {
    int8_t  w[256][16];
    int32_t sc[16][16];     // [sub-block][n]   (Q4_0/Q8_0: unused; Q4_K/Q5_K: 8 used; Q6_K: 16 used)
    int32_t mn[8][16];      // Q4_K/Q5_K
    float   d[16];
    float   dmin[16];
    float   dw[8][16];      // Q8_0: per-block fp16 scale as f32
};

inline int nib_at(const uint8_t * rec, int tile, int kk, int n) {      // Q4_K ("A") tiles
    return (rec[tile * 128 + kk * 8 + (n >> 1)] >> ((n & 1) * 4)) & 15;
}
inline uint16_t u16(const uint8_t * p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

void decode_q8_0(const uint8_t * rec, DecRec & r) {
    for (int b = 0; b < 8; ++b) {
        for (int n = 0; n < 16; ++n) r.dw[b][n] = fp16_to_f32_qk_kernel(u16(rec + 4096 + (b * 16 + n) * 2));
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 16; ++n) r.w[b * 32 + k][n] = int8_t(rec[detail::q8_0_off(b, k, n)]);
    }
}
void decode_scales_k(const uint8_t * scp, DecRec & r) {   // scp: 12 planes [p][n]; ggml get_scale_min_k4
    for (int n = 0; n < 16; ++n) {
        for (int j = 0; j < 8; ++j) {
            auto P = [&](int p) { return int(scp[p * 16 + n]); };
            if (j < 4) {
                r.sc[j][n] = P(j) & 63;
                r.mn[j][n] = P(j + 4) & 63;
            } else {
                r.sc[j][n] = (P(j + 4) & 15) | ((P(j - 4) >> 6) << 4);
                r.mn[j][n] = (P(j + 4) >> 4) | ((P(j) >> 6) << 4);
            }
        }
    }
}
void decode_q4_k(const uint8_t * rec, DecRec & r) {
    for (int i = 0; i < 256; ++i)
        for (int n = 0; n < 16; ++n) r.w[i][n] = int8_t(nib_at(rec, i / 16, i % 16, n));
    decode_scales_k(rec + 2048, r);
    for (int n = 0; n < 16; ++n) {
        r.d[n]    = fp16_to_f32_qk_kernel(u16(rec + 2048 + 192 + 2 * n));
        r.dmin[n] = fp16_to_f32_qk_kernel(u16(rec + 2048 + 224 + 2 * n));
    }
}
void decode_q5_k(const uint8_t * rec, DecRec & r) {
    for (int i = 0; i < 256; ++i)
        for (int n = 0; n < 16; ++n) {
            size_t lb, hb; int ls, hbit;
            detail::q5k_pos(i, n, lb, ls, hb, hbit);
            r.w[i][n] = int8_t(((rec[lb] >> ls) & 15) + 16 * ((rec[hb] >> hbit) & 1));
        }
    decode_scales_k(rec + 2560, r);
    for (int n = 0; n < 16; ++n) {
        r.d[n]    = fp16_to_f32_qk_kernel(u16(rec + 2752 + 2 * n));
        r.dmin[n] = fp16_to_f32_qk_kernel(u16(rec + 2784 + 2 * n));
    }
}
void decode_q6_k(const uint8_t * rec, DecRec & r) {
    for (int i = 0; i < 256; ++i) {
        const int t = i >> 4, k = i & 15;
        for (int n = 0; n < 16; ++n) {
            const int e = detail::tile_b_e(k, n);
            const int lo = (rec[t * 128 + (e >> 1)] >> ((e & 1) * 4)) & 15;
            const int hi = (rec[2048 + t * 64 + (e & 63)] >> (2 * (e >> 6))) & 3;
            r.w[i][n] = int8_t((lo | (hi << 4)) - 32);
        }
    }
    for (int s = 0; s < 16; ++s)
        for (int n = 0; n < 16; ++n) r.sc[s][n] = int8_t(rec[3072 + s * 16 + n]);
    for (int n = 0; n < 16; ++n) r.d[n] = fp16_to_f32_qk_kernel(u16(rec + 3328 + 2 * n));
}

// ---- per-token accumulation of one decoded record into y[16] -----------------------------------------------------
// 32-element blocks (Q4_0, Q8_0): rec holds up to 8 consecutive blocks along K starting at block kb0
void acc_blocks32(const DecRec & r, int nblk, const ActQ & a, int64_t t, int64_t kb0, float * y) {   // Q8_0
    const int8_t * aq = a.q_row(t);
    const float  * ad = a.d_row(t);
    for (int b = 0; b < nblk; ++b) {
        const int8_t * av = aq + (kb0 + b) * QK32;
        int32_t acc[16] = {};
        for (int k = 0; k < 32; ++k) {
            const int32_t x = av[k];
            for (int n = 0; n < 16; ++n) acc[n] += int32_t(r.w[b * 32 + k][n]) * x;
        }
        const float da = ad[kb0 + b];
        for (int n = 0; n < 16; ++n) {
            const float t1 = float(acc[n]) * r.dw[b][n];
            y[n] = y[n] + t1 * da;
        }
    }
}

// 256-element super-blocks with 8 sub-blocks of 32 and a min (Q4_K, Q5_K)
void acc_kmin(const DecRec & r, const ActQ & a, int64_t t, int64_t sb, float * y) {
    const int8_t  * av = a.q_row(t) + sb * 256;
    const int32_t * bs = a.bs_row(t) + sb * 8;
    int32_t sumi[16] = {}, summs[16] = {};
    for (int j = 0; j < 8; ++j) {
        int32_t dot[16] = {};
        for (int k = 0; k < 32; ++k) {
            const int32_t x = av[j * 32 + k];
            for (int n = 0; n < 16; ++n) dot[n] += int32_t(r.w[j * 32 + k][n]) * x;
        }
        for (int n = 0; n < 16; ++n) {
            sumi[n]  += r.sc[j][n] * dot[n];
            summs[n] += r.mn[j][n] * bs[j];
        }
    }
    const float da = a.d_row(t)[sb];
    for (int n = 0; n < 16; ++n) {
        const float t1 = r.d[n] * float(sumi[n]);
        const float t2 = r.dmin[n] * float(summs[n]);
        const float tt = t1 - t2;
        y[n] = y[n] + tt * da;
    }
}

// 256-element super-blocks with 16 sub-blocks of 16 and int8 scales (Q6_K)
void acc_q6k(const DecRec & r, const ActQ & a, int64_t t, int64_t sb, float * y) {
    const int8_t * av = a.q_row(t) + sb * 256;
    int32_t sumi[16] = {};
    for (int s = 0; s < 16; ++s) {
        int32_t cv[16] = {};
        for (int k = 0; k < 16; ++k) {
            const int32_t x = av[s * 16 + k];
            for (int n = 0; n < 16; ++n) cv[n] += int32_t(r.w[s * 16 + k][n]) * x;
        }
        for (int n = 0; n < 16; ++n) sumi[n] += cv[n] * r.sc[s][n];
    }
    const float da = a.d_row(t)[sb];
    for (int n = 0; n < 16; ++n) {
        const float t1 = r.d[n] * float(sumi[n]);
        y[n] = y[n] + t1 * da;
    }
}

class HostReferencePath final : public ComputePath {
public:
    const char * name() const override { return "host-reference"; }
    bool supports(const Geom &) const override { return true; }

    bool mul_mat(const MulMatJob & job) override {
        const Geom & g = job.geom;
        const ActQ & a = *job.act;
        const int64_t n_groups = g.n_pad / GROUP_ROWS;

#if defined(_OPENMP)
        #pragma omp parallel
#endif
        {
            std::vector<float> y(size_t(a.n_tok) * 16);
            DecRec rec_dec;
#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for (int64_t gi = 0; gi < n_groups; ++gi) {
                const int64_t row0 = gi * GROUP_ROWS;
                if (row0 >= g.n) {
                    continue; // padding group
                }
                const uint8_t * grp = job.w + g.group_base(row0);
                std::fill(y.begin(), y.end(), 0.0f);

                for (int64_t r = 0; r < g.nbk; ++r) {
                    const uint8_t * rec = grp + size_t(r) * g.rec_stride;
                    const int64_t kb0 = r * g.rec_blocks;
                    const int nblk = g.rec_blocks;
                    switch (g.fmt) {
                        case Fmt::Q8_0: decode_q8_0(rec, rec_dec); break;
                        case Fmt::Q4_K: decode_q4_k(rec, rec_dec); break;
                        case Fmt::Q5_K: decode_q5_k(rec, rec_dec); break;
                        case Fmt::Q6_K: decode_q6_k(rec, rec_dec); break;
                        default: break;
                    }
                    for (int64_t t = 0; t < a.n_tok; ++t) {
                        float * yt = &y[size_t(t) * 16];
                        switch (g.fmt) {
                            case Fmt::Q8_0: acc_blocks32(rec_dec, nblk, a, t, kb0, yt); break;
                            case Fmt::Q4_K:
                            case Fmt::Q5_K: acc_kmin(rec_dec, a, t, kb0, yt); break;
                            case Fmt::Q6_K: acc_q6k(rec_dec, a, t, kb0, yt); break;
                            default: break;
                        }
                    }
                }

                const int64_t nrows = std::min<int64_t>(GROUP_ROWS, g.n - row0);
                for (int64_t t = 0; t < a.n_tok; ++t) {
                    float * d = reinterpret_cast<float *>(reinterpret_cast<char *>(job.dst) + size_t(t) * job.dst_row_stride);
                    for (int64_t n = 0; n < nrows; ++n) d[row0 + n] = y[size_t(t) * 16 + size_t(n)];
                }
            }
        }
        return true;
    }
};

} // namespace

ComputePath & host_reference_path() {
    static HostReferencePath p;
    return p;
}

} // namespace xdna2

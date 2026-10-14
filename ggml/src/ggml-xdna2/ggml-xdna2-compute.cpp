#include "ggml-xdna2-compute.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace xdna2 {

using namespace layout;

void quantize_acts(Q8Act & out, const float * src, int64_t k, int64_t n_tok, size_t row_stride_bytes) {
    const int64_t kb = k / QK;
    out.n_tok  = n_tok;
    out.kb     = kb;
    out.kb_pad = roundup(kb, REC_BLOCKS);
    out.q.assign(size_t(n_tok) * size_t(out.kb_pad) * QK, 0);
    out.d.assign(size_t(n_tok) * size_t(out.kb_pad), 0.0f);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(n_tok * k >= (1 << 16))
#endif
    for (int64_t t = 0; t < n_tok; ++t) {
        const float * x = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src) + size_t(t) * row_stride_bytes);
        int8_t * q = out.q.data() + size_t(t) * size_t(out.kb_pad) * QK;
        float  * d = out.d.data() + size_t(t) * size_t(out.kb_pad);
        for (int64_t b = 0; b < kb; ++b) {
            float amax = 0.0f;
            for (int i = 0; i < QK; ++i) {
                amax = std::max(amax, std::fabs(x[b * QK + i]));
            }
            const float dd = amax / 127.0f;
            const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
            d[b] = dd;
            for (int i = 0; i < QK; ++i) {
                q[b * QK + i] = int8_t(std::roundf(x[b * QK + i] * id));
            }
        }
    }
}

float fp16_to_f32_kernel(uint16_t h) {
    // mirrors q4.cc: sign / (exp,mantissa) split, rebias by integer add, zero special-cased
    const uint32_t em = h & 0x7FFFu;
    uint32_t f = (em << 13) + 0x38000000u;
    if (em == 0) f = 0;
    float v;
    std::memcpy(&v, &f, sizeof(v));
    return (h & 0x8000u) ? -v : v;
}

namespace {

class HostReferencePath final : public ComputePath {
public:
    const char * name() const override { return "host-reference"; }
    bool supports(const Q4Geom &) const override { return true; }

    bool mul_mat_q4_0(const MulMatJob & job) override {
        const Q4Geom & g = job.geom;
        const Q8Act  & a = *job.act;
        const int64_t n_groups = g.n_pad / GROUP_ROWS;
        const int64_t gpc = g.groups_per_core;

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t gi = 0; gi < n_groups; ++gi) {
            const int64_t row0 = gi * GROUP_ROWS;
            if (row0 >= g.n) {
                continue; // padding group
            }
            const int64_t core = gi / gpc, gc = gi % gpc;
            const uint8_t * grp = job.w + size_t(core) * g.core_bytes + size_t(gc) * size_t(g.recs_per_group) * REC_BYTES;

            for (int64_t t = 0; t < a.n_tok; ++t) {
                const int8_t * aq = a.q_row(t);
                const float  * ad = a.d_row(t);
                float y[GROUP_ROWS] = {};

                for (int64_t kb = 0; kb < g.kb; ++kb) {
                    const uint8_t * rec = grp + size_t(kb / REC_BLOCKS) * REC_BYTES;
                    const int b = int(kb % REC_BLOCKS);
                    const uint8_t * nib = rec + size_t(b) * 256;
                    const int8_t  * av  = aq + kb * QK;

                    int32_t acc[GROUP_ROWS] = {};
                    for (int h = 0; h < 2; ++h) {
                        for (int kk = 0; kk < 16; ++kk) {
                            const int32_t x = av[h * 16 + kk];
                            const uint8_t * p = nib + h * 128 + kk * 8;
                            for (int m = 0; m < 8; ++m) {
                                const int32_t lo = int8_t(uint8_t(p[m] << 4)) >> 4;
                                const int32_t hi = int8_t(p[m]) >> 4;
                                acc[2 * m]     += lo * x;
                                acc[2 * m + 1] += hi * x;
                            }
                        }
                    }

                    const uint8_t * sc = rec + REC_NIB_BYTES + size_t(b) * 32;
                    const float da = ad[kb];
                    for (int n = 0; n < GROUP_ROWS; ++n) {
                        uint16_t h16;
                        std::memcpy(&h16, sc + 2 * n, 2);
                        const float t1 = float(acc[n]) * fp16_to_f32_kernel(h16);
                        y[n] = y[n] + t1 * da;
                    }
                }

                float * d = reinterpret_cast<float *>(reinterpret_cast<char *>(job.dst) + size_t(t) * job.dst_row_stride);
                const int64_t nrows = std::min<int64_t>(GROUP_ROWS, g.n - row0);
                for (int64_t n = 0; n < nrows; ++n) {
                    d[row0 + n] = y[n];
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

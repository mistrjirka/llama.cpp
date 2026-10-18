#include "ggml-xdna2-attn.h"

#include "ggml-xdna2-bfp.h"
#include "ggml.h"

#include <cstring>
#include <initializer_list>

namespace xdna2 {

namespace {

constexpr int D_MAX = 256;

inline size_t tile_off(int a, int b, int r) { return (size_t(a * 2 + b) * 8 + size_t(r)) * 9; }

inline void row_to_f32(const void * src, bool bf16, float * dst, int d) {
    if (bf16) ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t *>(src), dst, d);
    else      ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(src), dst, d);
}

} // namespace

void fa_encode_q(const FaShape & s, uint8_t * obj, const float * q, size_t tok_stride, size_t head_stride, int64_t n_tok, int64_t p0, int64_t pos0,
                 int head, float prescale) {
    const int ndt = s.ndt();
    alignas(64) float x[8];
    for (int i = 0; i < FA_BLK; ++i) {
        const int qt = i / 8, r = i % 8;
        const int64_t t = pos0 + i - p0;
        if (t < 0 || t >= n_tok || head >= s.nh) {
            for (int dt = 0; dt < ndt; ++dt) std::memset(obj + tile_off(dt, qt, r), 0, 9);
            continue;
        }
        const float * row = reinterpret_cast<const float *>(reinterpret_cast<const char *>(q) + size_t(t) * tok_stride + size_t(head) * head_stride);
        for (int dt = 0; dt < ndt; ++dt) {
            for (int c = 0; c < 8; ++c) x[c] = row[dt * 8 + c] * prescale;
            pf_enc8_fast(x, obj + tile_off(dt, qt, r));
        }
    }
}

void fa_encode_kv_block(const FaShape & s, uint8_t * kblk, uint8_t * vblk, const void * k, const void * v, bool bf16, size_t row_stride,
                        size_t head_stride, int head, int64_t blk, int64_t n_valid, int64_t rot, int64_t n_cells) {
    const int d = s.d, ndt = s.ndt();
    alignas(64) float kf[FA_BLK][D_MAX], vf[FA_BLK][D_MAX];
    for (int i = 0; i < FA_BLK; ++i) {
        const int64_t row = blk * FA_BLK + i;
        if (row >= n_valid) {
            std::memset(kf[i], 0, sizeof(float) * size_t(d));
            std::memset(vf[i], 0, sizeof(float) * size_t(d));
            continue;
        }
        const int64_t cell = rot > 0 ? (row + rot) % n_cells : row;
        const size_t off = size_t(cell) * row_stride + size_t(head) * head_stride;
        row_to_f32(static_cast<const char *>(k) + off, bf16, kf[i], d);
        row_to_f32(static_cast<const char *>(v) + off, bf16, vf[i], d);
    }
    alignas(64) float x[8];
    for (int dt = 0; dt < ndt; ++dt) {
        for (int kt = 0; kt < 2; ++kt) {
            for (int r = 0; r < 8; ++r) {
                pf_enc8_fast(&kf[kt * 8 + r][dt * 8], kblk + tile_off(dt, kt, r));
                for (int c = 0; c < 8; ++c) x[c] = vf[kt * 8 + c][dt * 8 + r];
                pf_enc8_fast(x, vblk + tile_off(dt, kt, r));
            }
        }
    }
}

uint64_t fa_block_fingerprint(const void * k, const void * v, size_t row_stride, size_t row_bytes, int64_t blk) {
    uint64_t h = 1469598103934665603ull;
    for (int64_t row : { blk * FA_BLK, blk * FA_BLK + FA_BLK - 1 }) {
        for (const void * base : { k, v }) {
            const uint64_t * p = reinterpret_cast<const uint64_t *>(static_cast<const char *>(base) + size_t(row) * row_stride);
            for (size_t i = 0; i < row_bytes / 8; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        }
    }
    return h;
}

void fa_scatter_out(const FaShape & s, const float * out, int64_t units, int64_t pos0, int64_t p0, int64_t n_tok, float * dst, size_t tok_stride,
                    size_t head_stride) {
    const int ndt = s.ndt(), cpk = s.cpk(), grp = s.grp(), uq = s.unit_q();
    const int64_t oc = s.oc();
#if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static) num_threads(pf_threads())
#endif
    for (int64_t r = 0; r < units; ++r) {
        for (int c = 0; c < 8; ++c) {
            const int g = c / cpk, qi = c % cpk;
            for (int hh = 0; hh < grp; ++hh) {
                const int head = grp * g + hh;
                const float * o = out + ((r * 8 + c) * 4 + hh) * oc;
                for (int qt = 0; qt < 2; ++qt) {
                    for (int row = 0; row < 8; ++row) {
                        const int64_t t = pos0 + uq * r + FA_BLK * qi + 8 * qt + row - p0;
                        if (t < 0 || t >= n_tok) continue;
                        const float l = o[2 * ndt * 64 + qt * 64 + row * 8];
                        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                        float * d = reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + size_t(t) * tok_stride + size_t(head) * head_stride);
                        for (int dt = 0; dt < ndt; ++dt) {
                            const float * src = o + (qt * ndt + dt) * 64 + row * 8;
                            for (int e = 0; e < 8; ++e) d[dt * 8 + e] = src[e] * inv;
                        }
                    }
                }
            }
        }
    }
}

} // namespace xdna2

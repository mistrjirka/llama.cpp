#pragma once

// Compute path abstraction of the ggml-xdna2 backend.
//
//   ActQ             activation rows quantized on the host, exactly what the NPU kernels consume:
//                      Q8_0F  (Q4_0, Q8_0 weights): int8 x32 blocks + f32 scale (d = amax/127), like quantize_row_q8_0
//                      Q8_K   (Q4_K, Q5_K, Q6_K weights): ggml's own Q8_K blocks (int8 x256, f32 d, bsums), plus the
//                             per-32 sums (bsum32) the Q4_K kernel uses
//   ComputePath      "who does the dot products":
//     HostReferencePath  exact emulation of the NPU kernel arithmetic on the CPU, directly on the repacked layout
//                        (ggml-xdna2-layout.h). TEST/DEBUG ONLY: selected only with GGML_XDNA2_REFERENCE=1,
//                        never a production fallback.
//     NpuPath            (ggml-xdna2-npu.cpp) forwards to the NpuExecutor. Not wired yet.
//
// Kernel arithmetic being emulated (per output row, blocks in order, f32 accumulation y, no FMA):
//   Q4_0  (stage0 q4.cc)   idot = sum (nib-8)*aq;                            y += (float(idot) * f16(d)) * da
//   Q4_K  (stage2 qk.cc)   sumi = sum_j sc_j * sum_k nib*aq, summs = sum_j mn_j * bsum32_j (exact int32);
//                          y += (f16(d)*float(sumi) - f16(dmin)*float(summs)) * da
//   Q6_K  (stage2 qk.cc)   sumi = sum_s sc_s * sum_k (w6-32)*aq (exact int);  y += (f16(d)*float(sumi)) * da
//   Q8_0, Q5_K             PROVISIONAL: same arithmetic as Q4_0 / Q4_K on 8-bit / 5-bit weights.
// fp16 -> f32 uses the kernels' integer conversions (see fp16_to_f32_*): q4.cc flushes subnormals wrongly,
// qk.cc handles them.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ggml-xdna2-layout.h"

namespace xdna2 {

class Arena;

enum class ActKind { Q8_0F, Q8_K };

struct ActQ {
    ActKind kind  = ActKind::Q8_0F;
    int64_t n_tok = 0;
    int64_t k_pad = 0;       // int8 values per row (K, padded to 8 blocks for 32-blocks formats; padding is zero)
    int64_t nd    = 0;       // scales per row (k_pad/32 or k_pad/256)
    int64_t nbs   = 0;       // bsum32 per row (Q8_K only: k_pad/32)
    std::vector<int8_t>  q;      // [n_tok][k_pad]
    std::vector<float>   d;      // [n_tok][nd]
    std::vector<int32_t> bsum;   // [n_tok][nbs]

    const int8_t *  q_row(int64_t t)  const { return q.data() + size_t(t) * size_t(k_pad); }
    const float *   d_row(int64_t t)  const { return d.data() + size_t(t) * size_t(nd); }
    const int32_t * bs_row(int64_t t) const { return bsum.data() + size_t(t) * size_t(nbs); }
};

ActKind act_kind(layout::Fmt fmt);

// Quantize n_tok rows of K floats (row t at src + t*row_stride_bytes) for weights of geometry `g`.
void quantize_acts(ActQ & out, const layout::Geom & g, const float * src, int64_t n_tok, size_t row_stride_bytes);

// fp16 -> f32 exactly as done by the NPU kernels with integer ops.
float fp16_to_f32_q4_0_kernel(uint16_t h);   // stage0 q4.cc: normal numbers and zero only
float fp16_to_f32_qk_kernel(uint16_t h);     // stage2 qk.cc: also subnormals

// One product for one 2D weight matrix: dst[j][i] = sum_k W[i][k] * act[j][k]
struct MulMatJob {
    const Arena *        arena = nullptr;   // arena holding the weights (needed by the NPU path for sub-BOs)
    size_t               w_off = 0;         // offset of the matrix region in the arena
    const uint8_t *      w = nullptr;       // host mapping of the matrix region (arena->base() + w_off)
    layout::Geom         geom;
    const ActQ *         act = nullptr;     // act->n_tok rows
    float *              dst = nullptr;     // dst row j at (char *) dst + j * dst_row_stride, geom.n floats each
    size_t               dst_row_stride = 0;
};

class ComputePath {
public:
    virtual ~ComputePath() = default;
    virtual const char * name() const = 0;
    virtual bool supports(const layout::Geom & geom) const = 0;
    // returns false if the job could not be executed
    virtual bool mul_mat(const MulMatJob & job) = 0;
};

ComputePath & host_reference_path();

} // namespace xdna2

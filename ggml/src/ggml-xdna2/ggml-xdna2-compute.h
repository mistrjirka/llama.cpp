#pragma once

// Compute path abstraction of the ggml-xdna2 backend.
//
//   ActQ             activation rows quantized on the host, exactly what the NPU kernels consume:
//                      Q8_0F  (Q8_0 weights): int8 x32 blocks + f32 scale (ggml Q8_0: d = amax/127 rounded to fp16, then widened)
//                      Q8_K   (Q4_K, Q5_K, Q6_K weights): ggml's own Q8_K blocks (int8 x256, f32 d, bsums), plus the
//                             per-32 sums (bsum32) the kernels use
//   ComputePath      "who does the dot products":
//     HostReferencePath  exact emulation of the NPU kernel arithmetic on the CPU, directly on the repacked layout
//                        (ggml-xdna2-layout.h). TEST/DEBUG ONLY: selected only with GGML_XDNA2_REFERENCE=1,
//                        never a production fallback.
//     NpuPath            (ggml-xdna2-npu.cpp) forwards to the NpuExecutor.
//
// Kernel arithmetic being emulated (qk3.cc; per output row, 256-weight records in order, f32 accumulation y, no FMA):
//   Q8_0  y += (float(sum_b q*aq) * f16(d_b)) * da_b                        for the 8 blocks of the record
//   Q4_K  sumi = sum_j sc_j * sum_k nib*aq, summs = sum_j mn_j * bsum32_j (exact int);
//         y += (f16(d)*float(sumi) - f16(dmin)*float(summs)) * da
//   Q5_K  same with 5-bit weights
//   Q6_K  sumi = sum_s sc_s * sum_k (w6-32)*aq (exact int);  y += (f16(d)*float(sumi)) * da
// fp16 -> f32 uses the kernel's integer conversion (fp16_to_f32_qk_kernel; handles subnormals).

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
    int64_t k_pad = 0;       // int8 values per row (= K)
    int64_t nd    = 0;       // scales per row (K/32 or K/256)
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

// fp16 -> f32 exactly as done by the NPU kernels with integer ops (normal numbers, zero and subnormals).
float fp16_to_f32_qk_kernel(uint16_t h);

// Activation BO of the stage3 ABI for M consecutive tokens t0 .. t0+M-1 of `act`:
//   [64 B header: i32 G, K, M][M vectors: Q8_0F: qs K | f32 d K/32;  Q8_K: qs K | f32 d K/256 | i32 bsum K/32]
//   padded to ACT_RECS records and every record duplicated per lane (2x), see act_bo_bytes().
size_t act_bo_bytes(const layout::Geom & g, int m);
void   pack_act_bo(const ActQ & act, const layout::Geom & g, int64_t t0, int m, uint8_t * dst);

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
    // independent jobs (an NPU path may submit them together)
    virtual bool mul_mat_batch(const MulMatJob * jobs, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            if (!mul_mat(jobs[i])) return false;
        }
        return true;
    }
};

ComputePath & host_reference_path();

} // namespace xdna2

#pragma once

// Compute path abstraction of the ggml-xdna2 backend.
//
//   Q8Act            activation rows quantized on the host to Q8_0-style blocks (int8 x 32 + f32 scale),
//                    exactly what the NPU kernel consumes.
//   ComputePath      "who does the Q4_0 x Q8_0 dot products":
//     HostReferencePath  exact emulation of the NPU kernel arithmetic, running on the CPU directly on the
//                        repacked layout (ggml-xdna2-layout.h). Always available. Used for validation.
//     NpuPath            (ggml-xdna2-npu.cpp) forwards to the NpuExecutor. Not wired yet.
//
// Kernel arithmetic (stage0/q4.cc), per output row, blocks kb = 0 .. KB-1 in order:
//     idot = sum_{k<32} (nibble_k - 8) * act_q[kb][k]            (exact int32)
//     y    = y + (float(idot) * fp16_to_f32(w_scale)) * act_d[kb]   (f32, no FMA, sequential over kb)
// fp16_to_f32 is the kernel's integer-op conversion: exact for normal numbers and zero; fp16 subnormals,
// inf and nan are NOT handled like IEEE (see fp16_to_f32_kernel) and are emulated as the kernel does.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ggml-xdna2-layout.h"

namespace xdna2 {

class Arena;

struct Q8Act {
    int64_t n_tok  = 0;
    int64_t kb     = 0;      // real blocks per row
    int64_t kb_pad = 0;      // padded blocks per row (multiple of 8), padding blocks are zero
    std::vector<int8_t> q;   // [n_tok][kb_pad*32]
    std::vector<float>  d;   // [n_tok][kb_pad]

    const int8_t * q_row(int64_t t) const { return q.data() + size_t(t) * size_t(kb_pad) * 32; }
    const float *  d_row(int64_t t) const { return d.data() + size_t(t) * size_t(kb_pad); }
};

// Quantize n_tok rows of K floats (row t at src + t*row_stride_bytes) like ggml quantize_row_q8_0
// (d = amax/127, q = roundf(x/d)) but with an f32 scale, as the NPU kernel expects. kb_pad blocks per row.
void quantize_acts(Q8Act & out, const float * src, int64_t k, int64_t n_tok, size_t row_stride_bytes);

// fp16 -> f32 exactly as done by the NPU kernel with integer ops (normal numbers and zero only).
float fp16_to_f32_kernel(uint16_t h);

// One Q4_0 x Q8_0 product for one 2D weight matrix: dst[j][i] = sum_k W[i][k] * act[j][k]
struct MulMatJob {
    const Arena *        arena = nullptr;   // arena holding the weights (needed by the NPU path for sub-BOs)
    size_t               w_off = 0;         // offset of the matrix region in the arena
    const uint8_t *      w = nullptr;       // host mapping of the matrix region (arena->base() + w_off)
    layout::Q4Geom       geom;
    const Q8Act *        act = nullptr;     // act->n_tok rows, act->kb_pad == geom.kb_pad
    float *              dst = nullptr;     // dst row j at (char *) dst + j * dst_row_stride, geom.n floats each
    size_t               dst_row_stride = 0;
};

class ComputePath {
public:
    virtual ~ComputePath() = default;
    virtual const char * name() const = 0;
    virtual bool supports(const layout::Q4Geom & geom) const = 0;
    // returns false if the job could not be executed (caller falls back to the host reference path)
    virtual bool mul_mat_q4_0(const MulMatJob & job) = 0;
};

ComputePath & host_reference_path();

} // namespace xdna2

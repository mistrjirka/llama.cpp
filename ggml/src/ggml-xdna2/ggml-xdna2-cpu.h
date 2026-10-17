#pragma once

// CPU decode GEMV (M = 1) that reads the NPU weight layout v2 DIRECTLY (ggml-xdna2-layout.h): no second copy of the weights.
//
//   y[i] = sum_k W[i][k] * x[k]     for the rows of one 2D weight matrix, all four formats (Q4_K, Q5_K, Q6_K, Q8_0).
//
// AVX-512 (VNNI + VBMI) implementation, selected at run time; cpu_decode_available() is false on other CPUs (the caller then keeps
// the NPU / host reference path).  Activations are quantized here (Q8_K-like 256 blocks with the 16-sums for the K-quants, biased
// Q8_0 blocks for Q8_0), the products are exact integer dot products per (sub-)block, scaled in f32 like ggml's vec_dot.
//
// Work is organised in items = (split, core c32, superblock sb): the 16-row groups sb*GSB .. of one core of the NPU layout, i.e. the
// rows [row0(split) + c32*16*Gs + sb*GSB*16, +16*gc).  A job may restrict the CPU to the superblocks >= sb_from of every split
// (the NPU then does the prefix [0, sb_from) of every chain: row split between NPU and CPU, see ggml-xdna2-xrt.cpp).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "ggml-xdna2-layout.h"

namespace xdna2 {

// Row/K split of a matrix between the NPU and the CPU (per split of the matrix): the NPU streams a PREFIX of every chain of the layout, i.e. the
// first sb_npu superblocks (all kb) of every core, or, for a matrix with one superblock, the first kb_npu records of every row.  The CPU does
// the complement.  (kb_npu == nbk: the NPU does complete rows; kb_npu < nbk: sb_npu == nsb == 1 and the CPU adds the K-suffix of the same rows.)
struct SplitPlan {
    int64_t sb_npu = 0;
    int64_t kb_npu = 0;
    bool    npu() const { return sb_npu > 0 && kb_npu > 0; }
};

// npu_frac = share of the weight bytes that should go to the NPU (0: all on the CPU, 1: all on the NPU)
std::vector<SplitPlan> make_split_plan(const layout::Geom & g, double npu_frac);

class Arena;

struct CpuDecodeJob {
    layout::Geom        geom;
    const uint8_t *     w = nullptr;      // matrix region (layout v2), g.bytes
    const float *       x = nullptr;      // K floats per token
    float *             dst = nullptr;    // g.n floats per token
    int                 n_tok = 1;        // tokens (<= CPU_MAX_TOK): the weights are read once for all of them
    size_t              x_stride = 0;     // bytes between the tokens of x
    size_t              dst_stride = 0;   // bytes between the tokens of dst
    const SplitPlan *   plan = nullptr;   // geom.nsplit entries; nullptr = the CPU computes the whole matrix
    const Arena *       arena = nullptr;  // for the NPU part
    size_t              w_off = 0;
};

constexpr int CPU_MAX_TOK = 8;

// AVX-512 VNNI/VBMI available (and not disabled with GGML_XDNA2_CPU_ISA=0)
bool cpu_decode_available();

// busy time of the team (sum over threads, inside the item loop) and wall time of the parallel regions (for GGML_XDNA2_STATS)
void cpu_decode_stats(double & busy_thread_s, double & wall_s, int & threads);

// Run n independent jobs on the OpenMP team of pf_threads() threads (blocking): the activations are quantized first (serially, vectorized),
// then `master_hook` (if any) runs on thread 0 -- the NPU part is started there -- while the other threads already work on the CPU part.
// Inside the hook, cpu_decode_act(j) is the quantized activation of job j.
void cpu_decode_run(const CpuDecodeJob * jobs, size_t n, const std::function<void()> & master_hook = nullptr);
const void * cpu_decode_act(size_t job);

// The activation BO of the NPU decode kernel for a (K-prefix of a) quantized activation of the CPU format: logical content
// [64 B header: G, K, qtype, n_act-1, n_w, G*K/256][q | d | bsum32], every chunk duplicated for the 4 lanes (dst: layout::act_chunks(fmt, k_use) * 4 * CHUNK bytes)
void cpu_pack_npu_act(layout::Fmt fmt, int64_t k_full, int64_t k_use, int64_t groups, int64_t n_w, const void * act, uint8_t * dst);

} // namespace xdna2

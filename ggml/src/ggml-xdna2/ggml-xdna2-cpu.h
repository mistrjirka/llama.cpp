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

#include "ggml-xdna2-layout.h"

namespace xdna2 {

struct CpuDecodeJob {
    layout::Geom    geom;
    const uint8_t * w = nullptr;      // matrix region (layout v2), g.bytes
    const float *   x = nullptr;      // K floats
    float *         dst = nullptr;    // g.n floats
    int64_t         sb_from = 0;      // the CPU computes the superblocks >= sb_from of every split (0 = the whole matrix)
};

// AVX-512 VNNI/VBMI available (and not disabled with GGML_XDNA2_CPU_ISA=0)
bool cpu_decode_available();

// busy time of the team (sum over threads, inside the item loop) and wall time of the parallel regions (for GGML_XDNA2_STATS)
void cpu_decode_stats(double & busy_thread_s, double & wall_s, int & threads);

// Run n independent jobs on the OpenMP team of pf_threads() threads (blocking).
void cpu_decode_run(const CpuDecodeJob * jobs, size_t n);

} // namespace xdna2

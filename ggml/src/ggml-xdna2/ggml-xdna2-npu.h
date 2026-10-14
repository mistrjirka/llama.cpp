#pragma once

// NpuExecutor: the (future) real NPU compute path of the ggml-xdna2 backend. Today only a stub exists
// (ggml-xdna2-npu.cpp) that reports "unavailable"; the backend then uses the host reference path.
//
// ---------------------------------------------------------------------------------------------------
// What the real implementation has to do (wiring TODO list)
// ---------------------------------------------------------------------------------------------------
//  1. Device: open xrt::device(0) lazily (never at backend registration), register xclbins.
//  2. Kernel cache: for each distinct padded shape (n_pad, k_pad) load
//         <cache_dir>/q4_n<n_pad>_k<k_pad>.xclbin   and   <cache_dir>/q4_n<n_pad>_k<k_pad>.insts.bin
//     (cache_dir = $GGML_XDNA2_CACHE_DIR, default $XDG_CACHE_HOME/ggml-xdna2 or ~/.cache/ggml-xdna2).
//     has_kernel() is a pure lookup (no compile); a missing shape means "run on host reference / CPU".
//     One xrt::hw_context per xclbin; the number of simultaneously resident contexts is limited by the
//     driver, so contexts must be created lazily and evicted LRU.
//  3. Weight buffers: alloc_weight_arena() returns an XrtBoArena (ggml-xdna2-arena.h): ONE xrt::bo per
//     ggml buffer, memory group = kernel->group_id(<weight arg>) of the kernels that will read it.
//     bind_weights() creates the per-tensor sub-BO xrt::bo(arena.bo(), geom.bytes, w_off) once and caches it.
//     The layout of that region is ggml-xdna2-layout.h, per-core contiguous: core c = bytes
//     [c*core_bytes, (c+1)*core_bytes).
//  4. ACTIVATION IS A SEPARATE SMALL BO. The current stage0 kernel (gen_q4.py/prep_q4.py) embeds ACT_RECS
//     activation records (KB*32 int8 + KB*4 f32 scales, padded to 2304 B records) at the START of every
//     core-chain of the weight buffer (replicated per chain). That cannot work here: the weights live in
//     the big ggml buffer and are shared by many calls. The kernel/transport must be changed so that
//       - the weight shim DMAs stream only the weight records of the sub-BO (offset w_off), and
//       - activations come from a second input BO (n_tok x [KB*32 B int8 | KB*4 B f32], broadcast to all
//         32 cores / replicated by the shim BDs, record padded) and results go to a third BO (N f32).
//     Q8Act (ggml-xdna2-compute.h) already has exactly that content; pack it with pack_act_blob().
//  5. Dispatch: instruction stream per (n_pad,k_pad) as xrt::bo(cacheable); one xrt::run per matmul,
//     or an xrt::runlist for all MUL_MATs of a graph split (batch them in graph_compute) and wait once.
//     n_tok > 1: loop on the host (kernel is a GEMV), or a batched kernel later.
//  6. Coherency: sync_to_device() after writing the activation BO, sync_from_device() (CLFLUSH + MFENCE,
//     ggml-xdna2-sync.h) before reading the result BO. Weight BO is synced once in set_tensor.
//  7. Padding rows: n_pad rows are computed by the kernel, the output BO has n_pad floats; only the first
//     n are copied into dst.
//  8. Fall back to HostReferencePath whenever available()/has_kernel() are false or run() fails.

#include <cstdint>
#include <memory>
#include <string>

#include "ggml-xdna2-arena.h"
#include "ggml-xdna2-compute.h"

namespace xdna2 {

class NpuExecutor {
public:
    virtual ~NpuExecutor() = default;

    // device present and executor usable
    virtual bool available() const = 0;

    // kernel (xclbin + insts) for geom.n_pad x geom.k_pad is in the cache dir
    virtual bool has_kernel(const layout::Q4Geom & geom) = 0;

    // storage for one ggml buffer; nullptr => the backend uses a HostArena
    virtual std::unique_ptr<Arena> alloc_weight_arena(size_t bytes) = 0;

    // register the tensor region (sub-BO) before first use; idempotent
    virtual bool bind_weights(const Arena & arena, size_t w_off, const layout::Q4Geom & geom) = 0;

    // run n jobs (runlist), blocking. Jobs must all have has_kernel() == true.
    virtual bool run(const MulMatJob * jobs, size_t n_jobs) = 0;

    virtual std::string cache_dir() const = 0;
};

// Serialize activations for the kernel: [n_tok][ q int8 (kb_pad*32) | d f32 (kb_pad) ] (stub for TODO 4).
size_t act_blob_bytes_per_token(const layout::Q4Geom & geom);
void   pack_act_blob(const Q8Act & act, uint8_t * dst);

// Global executor (created on first use; honours GGML_XDNA2_CACHE_DIR). Never null.
NpuExecutor & npu_executor();

// ComputePath that forwards to npu_executor().
ComputePath & npu_path();

// true if an XDNA2 NPU device node exists. Does NOT open the device.
bool npu_device_present();

} // namespace xdna2

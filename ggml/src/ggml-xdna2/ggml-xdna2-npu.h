#pragma once

// NpuExecutor: the (future) real NPU compute path of the ggml-xdna2 backend. Today only a stub exists
// (ggml-xdna2-npu.cpp) that reports "unavailable"; the backend then uses the host reference path.
//
// ---------------------------------------------------------------------------------------------------
// What the real implementation has to do (wiring TODO list)
// ---------------------------------------------------------------------------------------------------
//  1. Device: open xrt::device(0) lazily (never at backend registration), register xclbins.
//  2. Kernels: kernel_cache() (ggml-xdna2-kernels.h) already builds and stores them. For each chunk of
//     Geom (one dispatch) and M activation vectors use KernelCache::wait(key) and load
//         <kernel_cache().dir_of(key)>/kernel.xclbin   and   .../insts.bin
//     (cache_dir = $GGML_XDNA2_CACHE_DIR, default $XDG_CACHE_HOME/ggml-xdna2 or ~/.cache/ggml-xdna2).
//     init_tensor requests the builds (request_kernels()), so they overlap with the rest of the model load.
//     One xrt::hw_context per xclbin; the number of simultaneously resident contexts is limited by the
//     driver, so contexts must be created lazily and evicted LRU.
//  3. Weight buffers: alloc_weight_arena() returns an XrtBoArena (ggml-xdna2-arena.h): ONE xrt::bo per
//     ggml buffer, memory group = kernel->group_id(<weight arg>) of the kernels that will read it.
//     bind_weights() creates the per-tensor sub-BO xrt::bo(arena.bo(), geom.bytes, w_off) once and caches it.
//     The layout of that region is ggml-xdna2-layout.h: chunk i (one dispatch) = bytes
//     [chunk.off, chunk.off + chunk.bytes), inside it core c = [c*core_bytes, (c+1)*core_bytes).
//     Tensors with several chunks (output.weight) need one run per chunk, results land at chunk.row0.
//  4. ACTIVATION IS A SEPARATE SMALL BO. The current stage0/stage2 kernels (gen_q4.py, gen_qk.py; the latter is
//     what kernels/ contains) embed ACT_RECS activation records (the Q8 blob, padded to weight-record size) at the START of every
//     core-chain of the weight buffer (replicated per chain). That cannot work here: the weights live in
//     the big ggml buffer and are shared by many calls. The kernel/transport must be changed so that
//       - the weight shim DMAs stream only the weight records of the sub-BO (offset w_off), and
//       - activations come from a second input BO (n_tok x [KB*32 B int8 | KB*4 B f32], broadcast to all
//         32 cores / replicated by the shim BDs, record padded) and results go to a third BO (N f32).
//     ActQ (ggml-xdna2-compute.h) already has exactly that content; pack it with pack_act_blob().
//     The stage3 ABI (separate activation BO, Q8_0/Q5_K kernels) replaces kernels/*.cc|py and build_kernel.sh; the cache
//     key hashes those files, so old binaries are never reused. The provisional Q8_0/Q5_K layouts must then be fixed.
//  5. Dispatch: instruction stream per (n_pad,k_pad) as xrt::bo(cacheable); one xrt::run per matmul,
//     or an xrt::runlist for all MUL_MATs of a graph split (batch them in graph_compute) and wait once.
//     n_tok > 1: loop on the host (kernel is a GEMV), or a batched kernel later.
//  6. Coherency: sync_to_device() after writing the activation BO, sync_from_device() (CLFLUSH + MFENCE,
//     ggml-xdna2-sync.h) before reading the result BO. Weight BO is synced once in set_tensor.
//  7. Padding rows: n_pad rows are computed by the kernel, the output BO has n_pad floats; only the first
//     n are copied into dst.
//  8. NO silent fallback: a weight is only placed in our buffer if placement_ok() (executor available and every chunk
//     kernel cached or buildable); a failing run() aborts. The host reference path exists only for
//     GGML_XDNA2_REFERENCE=1 (tests/debug).

#include <cstdint>
#include <memory>
#include <string>

#include "ggml-xdna2-arena.h"
#include "ggml-xdna2-compute.h"
#include "ggml-xdna2-kernels.h"

namespace xdna2 {

class NpuExecutor {
public:
    virtual ~NpuExecutor() = default;

    // device present and executor usable
    virtual bool available() const = 0;

    // storage for one ggml buffer; nullptr => the backend uses a HostArena
    virtual std::unique_ptr<Arena> alloc_weight_arena(size_t bytes) = 0;

    // register the tensor region (sub-BO) before first use; idempotent
    virtual bool bind_weights(const Arena & arena, size_t w_off, const layout::Geom & geom) = 0;

    // run n jobs (runlist), blocking. Jobs must all have has_kernel() == true.
    virtual bool run(const MulMatJob * jobs, size_t n_jobs) = 0;

    virtual std::string cache_dir() const = 0;
};

// Serialize activations for the kernels:
//   Q8_0F: [n_tok][ q int8 (k_pad) | d f32 (k_pad/32) ]
//   Q8_K : [n_tok][ q int8 (K) | d f32 (K/256) | bsum32 int32 (K/32) ]   (stage2 SV layout)
size_t act_blob_bytes_per_token(const ActQ & act);
void   pack_act_blob(const ActQ & act, uint8_t * dst);

// Global executor (created on first use). Never null.
NpuExecutor & npu_executor();

// ComputePath that forwards to npu_executor().
ComputePath & npu_path();

// Global kernel cache (honours GGML_XDNA2_CACHE_DIR). Never null.
KernelCache & kernel_cache();

// GGML_XDNA2_REFERENCE=1: run the host reference emulation instead of the NPU (tests/debug only).
bool reference_mode();

// Placement policy: may a weight of this geometry live in our buffer (supports_op)?
//   reference mode: always. Otherwise only if the NPU executor is usable and every dispatch chunk has a kernel
//   that is cached or can be built with the installed toolchain.
bool placement_ok(const layout::Geom & geom);

// Start background builds of the missing kernels of a weight (called when the tensor is initialised).
void request_kernels(const layout::Geom & geom);

// true if an XDNA2 NPU device node exists. Does NOT open the device.
bool npu_device_present();

} // namespace xdna2

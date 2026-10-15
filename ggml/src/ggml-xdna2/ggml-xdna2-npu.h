#pragma once

// NpuExecutor: the real NPU compute path of the ggml-xdna2 backend (ggml-xdna2-xrt.cpp, needs XRT) and a stub used when
// XRT is not available or the host reference mode is selected.
//
// Kernel ABI (lab stage5, UNIFIED decode kernel): kernel(opcode=3, instr_bo, instr_words, W_bo, A_bo, Y_bo)
//   W_bo   sub-BO of the weight arena covering one SPLIT of a 2D weight matrix (layout: ggml-xdna2-layout.h; chunk stream, GGUF sized + chunk padding)
//   A_bo   [n_act chunks][2 lanes][4096 B]: 64 B header (i32 G, K, qtype, n_act-1, n_w, G*K/256) + one activation vector (pack_act_bo in compute.h)
//   Y_bo   f32 [col 8][Gs][core 4][16] -> row = ((col*4 + k)*Gs + g)*16 + i (+ split row base)
//   ONE xclbin serves every qtype, every K <= 12288 and every N; the qtype/K/G are read by the cores from the A_bo header and only 3 values of
//   the instruction stream (weight words per chain, act words, groups) are patched by KernelCache::insts_for.
//
// Execution model: ONE hw context for the whole process. Per MUL_MAT tensor a persistent Job (per split: W sub-BO, A BO, Y BO, xrt::run with all
// arguments set once). Per token the activation is quantized on the host, written to A_bo, the run started and waited, and Y_bo unpermuted into dst.
// Independent MUL_MATs of one graph split (any qtype/K/N) are submitted as ONE xrt::runlist.
//
// No silent fallback: a weight is only placed in our buffer if placement_ok() (executor available and the kernel for its
// (qtype, K) cached or buildable); a failing run() aborts. The host reference path exists only for GGML_XDNA2_REFERENCE=1.

#include <cstdint>
#include <memory>
#include <string>

#include "ggml-xdna2-arena.h"
#include "ggml-xdna2-compute.h"
#include "ggml-xdna2-kernels.h"

namespace xdna2 {

// One product on the PREFILL kernel (64-token tiles): activations are the raw f32 rows (encoded to bfp16ebs8 by the executor)
struct PrefillJob {
    const Arena *   arena = nullptr;
    size_t          w_off = 0;
    layout::Geom    geom;
    const float *   x = nullptr;         // n_tok rows of geom.k floats
    size_t          x_stride = 0;        // bytes between rows
    int64_t         n_tok = 0;
    float *         dst = nullptr;       // dst row j at (char *) dst + j * dst_row_stride, geom.n floats each
    size_t          dst_row_stride = 0;
};

struct ExecStats {
    uint64_t runs = 0;            // xrt runs (tokens x matmuls)
    uint64_t batches = 0;         // runlist / single submissions
    double   t_pack_a = 0;        // write + sync activation BOs (s)
    double   t_wait = 0;          // start .. wait (NPU time incl. dispatch) (s)
    double   t_unpermute = 0;     // sync Y + copy to dst (s)
    double   t_setup = 0;         // contexts, jobs, instruction streams (s)
    double   t_kernel_wait = 0;   // blocked on kernel builds (s)
    uint64_t n_switch = 0;        // submissions whose hw context differs from the previous one
    double   t_wait_switch = 0;   // wait time of those submissions (s)
    double   bytes_w = 0;         // weight bytes streamed
    // prefill path
    uint64_t pf_jobs = 0, pf_dispatches = 0, pf_waves = 0;
    double   pf_t_encode = 0;     // f32 -> bfp16ebs8 (host, threaded)
    double   pf_t_abo = 0;        // A BO fill (replicated slabs) + sync
    double   pf_t_npu = 0;        // start .. wait of the dispatches
    double   pf_t_unperm = 0;     // sync Y + unpermute into dst
    double   pf_t_setup = 0;
    double   pf_bytes_w = 0;      // weight bytes streamed by the prefill kernel (all tiles)
    uint64_t pf_switches = 0;     // hw context switches (decode <-> prefill)
};

class NpuExecutor {
public:
    virtual ~NpuExecutor() = default;

    // device present and executor usable
    virtual bool available() const = 0;

    // storage for one ggml buffer; nullptr => the backend uses a HostArena
    virtual std::unique_ptr<Arena> alloc_weight_arena(size_t bytes) = 0;

    // forget everything derived from this arena (sub-BOs, runs); called before the arena is destroyed
    virtual void release_arena(const Arena * arena) = 0;

    // run n independent jobs (each act->n_tok tokens), blocking. false = failure.
    virtual bool run(const MulMatJob * jobs, size_t n_jobs) = 0;

    // prefill kernel available (built or buildable)
    virtual bool prefill_available() { return false; }
    // run n independent prefill jobs, blocking. false = failure.
    virtual bool run_prefill(const PrefillJob * jobs, size_t n_jobs) { (void) jobs; (void) n_jobs; return false; }

    virtual ExecStats stats() const = 0;
    virtual void reset_stats() = 0;
};

// Global executor (created on first use). Never null. Real (XRT) executor when built with XRT, the NPU is present and
// GGML_XDNA2_REFERENCE is not set, otherwise a stub that is never available.
NpuExecutor & npu_executor();

// ComputePath that forwards to npu_executor().
ComputePath & npu_path();

// Global kernel caches (honour GGML_XDNA2_CACHE_DIR). Never null.
KernelCache & kernel_cache();            // decode kernel
KernelCache & prefill_kernel_cache();    // prefill kernel

// GGML_XDNA2_REFERENCE=1: run the host reference emulation instead of the NPU (tests/debug only).
bool reference_mode();

// Prefill kernel policy: GGML_XDNA2_PF_DISABLE=1 turns it off; MUL_MATs with at least prefill_min_tokens() activation rows (GGML_XDNA2_PF_MIN, default 32) use it.
bool prefill_enabled();
int  prefill_min_tokens();

// Placement policy: may a weight of this geometry live in our buffer (supports_op)?
//   reference mode: always. Otherwise only if the NPU executor is usable and the kernel for (qtype, K) is cached or can be
//   built with the installed toolchain.
bool placement_ok(const layout::Geom & geom);

// Start a background build of the missing kernel of a weight (called when the tensor is initialised).
void request_kernels(const layout::Geom & geom);

// true if an XDNA2 NPU device node exists. Does NOT open the device.
bool npu_device_present();

#ifdef GGML_XDNA2_HAVE_XRT
std::unique_ptr<NpuExecutor> make_xrt_executor();   // ggml-xdna2-xrt.cpp
#endif

} // namespace xdna2

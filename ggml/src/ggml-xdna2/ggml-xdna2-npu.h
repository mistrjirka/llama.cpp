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
#include "ggml-xdna2-cpu.h"
#include "ggml-xdna2-kernels.h"
#include "ggml-xdna2-bfp.h"

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
    PfEpi           epi = PfEpi::None;   // fused epilogue (SwiGLU or residual add, see pf_unpermute_tiles), second operand at epi_other
    const float *   epi_other = nullptr;
};

// One FLASH_ATTN_EXT (prefill) on the attention kernel: head dim 256, 16 q heads, 4 kv heads, causal over the cache cells 0 .. p0 + t for query t
// (the mask was checked by the caller), cells = positions.  The executor keeps a bfp16 shadow of the layer's KV cache (identified by kv_id).
struct AttnJob {
    const float * q = nullptr;           // query t of head h at (char *) q + t * q_tok_stride + h * q_head_stride (256 floats)
    size_t        q_tok_stride = 0, q_head_stride = 0;
    int64_t       n_tok = 0;             // queries
    int64_t       p0 = 0;                // cell (position) of query 0
    const void *  k = nullptr;           // cache views: cell c of kv head g at (char *) k + c * kv_row_stride + g * kv_head_stride (256 f16/bf16)
    const void *  v = nullptr;
    bool          bf16 = false;
    size_t        kv_row_stride = 0, kv_head_stride = 0;
    int64_t       kv_cap = 0;            // cells of the cache
    const void *  kv_id = nullptr;       // identity of the layer's cache (shadow key)
    float *       dst = nullptr;         // output of query t, head h at (char *) dst + t * dst_tok_stride + h * dst_head_stride
    size_t        dst_tok_stride = 0, dst_head_stride = 0;
    float         scale = 1.0f;
};

struct ExecStats {
    uint64_t runs = 0;            // xrt runs (tokens x matmuls)
    // decode split
    uint64_t sp_calls = 0, sp_runs = 0;
    double   sp_t_pack = 0, sp_t_start = 0, sp_t_wait = 0, sp_t_merge = 0, sp_bytes_w = 0;
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
    double   pf_t_abuild = 0;     // part of pf_t_abo: host replication of the slabs (without the BO sync)
    double   pf_t_ysync = 0;      // part of pf_t_unperm: BO sync of the result
    double   pf_bytes_w = 0;      // weight bytes streamed by the prefill kernel (all tiles)
    double   pf_bytes_x = 0, pf_bytes_a = 0, pf_bytes_y = 0;   // host bytes: f32 activations encoded, A tiles written, Y tiles unpermuted
    uint64_t pf_switches = 0;     // hw context switches (decode <-> prefill)
    // attention path
    uint64_t fa_calls = 0, fa_dispatches = 0, fa_pairs = 0, fa_blocks_enc = 0, fa_switches = 0;
    double   fa_t_kv = 0;         // shadow KV encode (incl. fingerprint checks)
    double   fa_t_q = 0;          // Q encode + BO sync
    double   fa_t_npu = 0;        // start .. wait
    double   fa_t_out = 0;        // output sync + normalize/scatter
    double   fa_t_setup = 0;
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

    // attention kernel available (built or buildable) and enabled
    virtual bool attention_available() { return false; }
    // run one attention job, blocking. false = failure.
    virtual bool run_attention(const AttnJob & job) { (void) job; return false; }

    // Decode split (GGML_XDNA2_DECODE=split): start the NPU part (CpuDecodeJob::plan) of n independent jobs -- the activations are the quantized ones of
    // cpu_decode_act(j) -- and return without waiting; decode_split_end() waits and merges the NPU rows into dst. The jobs must use distinct weights.
    virtual bool decode_split_begin(const CpuDecodeJob * jobs, size_t n) { (void) jobs; (void) n; return false; }
    virtual bool decode_split_end() { return false; }

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
KernelCache & attention_kernel_cache();  // flash-attention kernel

// GGML_XDNA2_REFERENCE=1: run the host reference emulation instead of the NPU (tests/debug only).
bool reference_mode();

// Prefill kernel policy: GGML_XDNA2_PF_DISABLE=1 turns it off; MUL_MATs with at least prefill_min_tokens() activation rows (GGML_XDNA2_PF_MIN, default 2) use it.
bool prefill_enabled();
int  prefill_min_tokens();

// Attention policy: GGML_XDNA2_FA=0 turns it off; FLASH_ATTN_EXT with at least attention_min_tokens() queries (GGML_XDNA2_FA_MIN_TOK, default 32)
// and attention_min_kv() keys (GGML_XDNA2_FA_MIN_KV, default 1024: below that the two hw context switches per layer cost more than the CPU) run on the NPU.
bool    attention_enabled();
int64_t attention_min_tokens();
int64_t attention_min_kv();

// Decode (M = 1) execution mode, GGML_XDNA2_DECODE: "npu" (unified NPU decode kernel), "cpu" (AVX-512 GEMV straight from the NPU layout,
// ggml-xdna2-cpu.cpp, DEFAULT), "split" (NPU prefix + CPU suffix of every matrix, concurrently).
enum class DecodeMode { Npu, Cpu, Split };
DecodeMode decode_mode();

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

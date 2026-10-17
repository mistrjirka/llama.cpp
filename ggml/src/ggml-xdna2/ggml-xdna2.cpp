// ggml-xdna2: dynamically loadable ggml backend for the AMD XDNA2 NPU.
// Template: ggml-blas. See ggml-xdna2-layout.h (device weight layout), ggml-xdna2-compute.h (compute paths)
// and ggml-xdna2-npu.h (NPU executor interface + wiring TODO list).

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-xdna2.h"
#include "ggml-backend-impl.h"

#include "ggml-xdna2-arena.h"
#include "ggml-xdna2-compute.h"
#include "ggml-xdna2-cpu.h"
#include "ggml-xdna2-layout.h"
#include "ggml-xdna2-npu.h"
#include "ggml-xdna2-attn.h"
#include "ggml-xdna2-bfp.h"

#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

using namespace xdna2;

// copies between the host and our buffers (set_tensor / get_tensor) - part of the per-token breakdown
static std::atomic<uint64_t> g_set_calls{0}, g_get_calls{0}, g_set_bytes{0}, g_get_bytes{0};
static std::atomic<uint64_t> g_set_ns{0}, g_get_ns{0};

static bool xdna2_disabled() {
    const char * e = std::getenv("GGML_XDNA2_DISABLE");
    return e && *e && std::strcmp(e, "0") != 0;
}

//
// helpers
//

// ggml quantization types with a device layout (ggml-xdna2-layout.h)
static bool xdna2_fmt_of(ggml_type type, layout::Fmt & fmt) {
    switch (type) {
        case GGML_TYPE_Q8_0: fmt = layout::Fmt::Q8_0; return true;
        case GGML_TYPE_Q4_K: fmt = layout::Fmt::Q4_K; return true;
        case GGML_TYPE_Q5_K: fmt = layout::Fmt::Q5_K; return true;
        case GGML_TYPE_Q6_K: fmt = layout::Fmt::Q6_K; return true;
        default:             return false;
    }
}

static bool xdna2_is_repacked(const ggml_tensor * t) {
    layout::Fmt f;
    return xdna2_fmt_of(t->type, f);
}

static layout::Geom xdna2_geom(const ggml_tensor * t) {
    layout::Fmt f = layout::Fmt::Q4_K;
    xdna2_fmt_of(t->type, f);
    return layout::make_geom(f, t->ne[1], t->ne[0]);
}

static size_t xdna2_slices(const ggml_tensor * t) {
    return size_t(t->ne[2]) * size_t(t->ne[3]);
}

static ggml_backend_buffer_type_t xdna2_buft();

// weight formats the prefill kernel handles (others run on the decode kernel, token by token)
static bool xdna2_pf_supports(layout::Fmt fmt) {
    (void) fmt;
    return true;   // Q4_K, Q5_K, Q6_K, Q8_0
}

//
// buffer
//

struct xdna2_buffer_context {
    std::unique_ptr<Arena> arena;
};

static xdna2_buffer_context * xdna2_ctx(ggml_backend_buffer_t buffer) {
    return static_cast<xdna2_buffer_context *>(buffer->context);
}

static void ggml_backend_xdna2_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    npu_executor().release_arena(xdna2_ctx(buffer)->arena.get());
    delete xdna2_ctx(buffer);
}

static void * ggml_backend_xdna2_buffer_get_base(ggml_backend_buffer_t buffer) {
    return xdna2_ctx(buffer)->arena->base();
}

static void xdna2_check_tensor(const ggml_tensor * t) {
    if (xdna2_is_repacked(t)) {
        GGML_ASSERT(t->view_src == nullptr && "xdna2: views of repacked tensors are not supported");
        GGML_ASSERT(ggml_is_contiguous(t));
    }
}

// device bytes occupied by the tensor
static size_t xdna2_tensor_dev_bytes(const ggml_tensor * t) {
    return xdna2_is_repacked(t) ? xdna2_slices(t) * xdna2_geom(t).bytes : ggml_nbytes(t);
}

static size_t xdna2_arena_off(ggml_backend_buffer_t buffer, const ggml_tensor * t) {
    return size_t(static_cast<const uint8_t *>(t->data) - xdna2_ctx(buffer)->arena->base());
}

static enum ggml_status ggml_backend_xdna2_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor * tensor) {
    if (xdna2_is_repacked(tensor)) {
        // collect the unique weight shapes: missing NPU kernels are built in the background while the model loads
        request_kernels(xdna2_geom(tensor));
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_xdna2_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    xdna2_check_tensor(tensor);
    if (xdna2_is_repacked(tensor)) {
        // logical memset on the GGUF byte stream
        std::vector<uint8_t> tmp(size, value);
        layout::tensor_pack_range(xdna2_geom(tensor), static_cast<uint8_t *>(tensor->data), tmp.data(), offset, size);
        xdna2_ctx(buffer)->arena->sync_to_device(xdna2_arena_off(buffer, tensor), xdna2_tensor_dev_bytes(tensor));
    } else {
        // plain tensors (activations) are only ever read and written by the host: no cache maintenance needed
        std::memset(static_cast<uint8_t *>(tensor->data) + offset, value, size);
    }
}

static void ggml_backend_xdna2_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const auto t_begin = std::chrono::steady_clock::now();
    struct Acc { std::chrono::steady_clock::time_point t; size_t n; ~Acc() { g_set_calls++; g_set_bytes += n;
        g_set_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count(); } } acc{t_begin, size};
    xdna2_check_tensor(tensor);
    Arena * arena = xdna2_ctx(buffer)->arena.get();
    if (xdna2_is_repacked(tensor)) {
        layout::tensor_pack_range(xdna2_geom(tensor), static_cast<uint8_t *>(tensor->data),
                                       static_cast<const uint8_t *>(data), offset, size);
        arena->sync_to_device(xdna2_arena_off(buffer, tensor), xdna2_tensor_dev_bytes(tensor));
    } else {
        std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
    }
}

static void ggml_backend_xdna2_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const auto t_begin = std::chrono::steady_clock::now();
    struct Acc { std::chrono::steady_clock::time_point t; size_t n; ~Acc() { g_get_calls++; g_get_bytes += n;
        g_get_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count(); } } acc{t_begin, size};
    xdna2_check_tensor(tensor);
    Arena * arena = xdna2_ctx(buffer)->arena.get();
    if (xdna2_is_repacked(tensor)) {
        arena->sync_from_device(xdna2_arena_off(buffer, tensor), xdna2_tensor_dev_bytes(tensor));
        layout::tensor_unpack_range(xdna2_geom(tensor), static_cast<const uint8_t *>(tensor->data),
                                         static_cast<uint8_t *>(data), offset, size);
    } else {
        std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
    }
}

static void ggml_backend_xdna2_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // NOTE: padding of repacked tensors must keep a zero scale, so only clear(0) is meaningful for buffers
    // that hold weights. Compute buffers (F32 only) can be cleared with any value.
    Arena * arena = xdna2_ctx(buffer)->arena.get();
    std::memset(arena->base(), value, arena->size());
    arena->sync_to_device(0, arena->size());
}

static const ggml_backend_buffer_i ggml_backend_xdna2_buffer_i = {
    /* .free_buffer   = */ ggml_backend_xdna2_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_xdna2_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_xdna2_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_xdna2_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_xdna2_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_xdna2_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ ggml_backend_xdna2_buffer_clear,
    /* .reset         = */ nullptr,
};

//
// buffer type
//

static const char * ggml_backend_xdna2_buft_get_name(ggml_backend_buffer_type_t) {
    return GGML_XDNA2_NAME;
}

static ggml_backend_buffer_t ggml_backend_xdna2_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    std::unique_ptr<Arena> arena;
    try {
        arena = npu_executor().alloc_weight_arena(size);
        if (!arena) {
            arena = std::make_unique<HostArena>(size);
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes: %s\n", __func__, size, e.what());
        return nullptr;
    }
    const size_t asize = arena->size();
    auto * ctx = new xdna2_buffer_context{std::move(arena)};
    return ggml_backend_buffer_init(buft, ggml_backend_xdna2_buffer_i, ctx, asize);
}

static size_t ggml_backend_xdna2_buft_get_alignment(ggml_backend_buffer_type_t) {
    return ARENA_ALIGN;
}

static size_t ggml_backend_xdna2_buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    if (xdna2_is_repacked(tensor)) {
        // padded to 512 rows (and whole records along K) per 2D slice (see ggml-xdna2-layout.h)
        return xdna2_slices(tensor) * xdna2_geom(tensor).bytes;
    }
    return ggml_nbytes(tensor);
}

static bool ggml_backend_xdna2_buft_is_host(ggml_backend_buffer_type_t) {
    return false; // repacked layout, and the storage may be a device mapped BO
}

static ggml_backend_buffer_type_t xdna2_buft() {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_xdna2_buft_get_name,
            /* .alloc_buffer     = */ ggml_backend_xdna2_buft_alloc_buffer,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ ggml_backend_xdna2_buft_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ ggml_backend_xdna2_buft_get_alloc_size,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ ggml_backend_xdna2_buft_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna2_reg(), 0),
        /* .context = */ nullptr,
    };
    return &buft;
}

ggml_backend_buffer_type_t ggml_backend_xdna2_buffer_type(void) {
    return xdna2_buft();
}

//
// compute
//

struct ggml_backend_xdna2_context {
    std::deque<ActQ>       acts;          // activation scratch, one per queued job
    size_t                 n_acts = 0;
    std::vector<MulMatJob> pending;       // queued independent jobs
    std::vector<PrefillJob> pf_pending;   // queued independent prefill-kernel jobs
    std::vector<CpuDecodeJob> cpu_pending; // queued independent CPU decode GEMVs (GGML_XDNA2_DECODE=cpu)
    std::vector<std::pair<const char *, const char *>> pending_dst;   // [begin, end) of the pending outputs
    double t_quant = 0, t_graph = 0, t_cpu = 0, b_cpu = 0;      // seconds / bytes
    uint64_t n_cpu_flush = 0, n_cpu_jobs = 0;
    double t_last_end = 0, t_between = 0; // time between two graph computes of this backend (CPU splits + copies)
    uint64_t n_graphs = 0, n_mul_mat = 0, n_glu_fused = 0, n_add_fused = 0;
    double   t_fa = 0;                    // FLASH_ATTN_EXT on the NPU (incl. host encode/scatter)
    uint64_t n_fa = 0;
};

// The host reference path is a test/debug tool (GGML_XDNA2_REFERENCE=1) and never a fallback for the NPU.
static ComputePath & xdna2_select_path() {
    return reference_mode() ? host_reference_path() : npu_path();
}

static double xdna2_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// NPU/CPU split of a weight matrix (GGML_XDNA2_DECODE=split): the NPU share of the bytes is GGML_XDNA2_SPLIT (default 0.5); matrices smaller than
// GGML_XDNA2_SPLIT_MIN_MB (default 0) run on the CPU only. The plan of a matrix never changes.
static const SplitPlan * xdna2_split_plan(const layout::Geom & g) {
    static std::map<std::tuple<int, int64_t, int64_t>, std::vector<SplitPlan>> plans;
    static const double frac = [] { const char * e = std::getenv("GGML_XDNA2_SPLIT"); return e && *e ? std::atof(e) : 0.5; }();
    static const double min_mb = [] { const char * e = std::getenv("GGML_XDNA2_SPLIT_MIN_MB"); return e && *e ? std::atof(e) : 0.0; }();
    const auto key = std::make_tuple(int(g.fmt), g.n, g.k);
    auto it = plans.find(key);
    if (it == plans.end()) {
        it = plans.emplace(key, make_split_plan(g, double(g.bytes) / 1e6 >= min_mb ? frac : 0.0)).first;
    }
    return it->second.data();
}

struct TraceEnt { uint64_t n = 0; double t_total = 0, t_hook_start = 0, t_hook = 0, t_cpu_end = 0, t_end = 0; };
static bool xdna2_trace() { static const bool b = std::getenv("GGML_XDNA2_TRACE") != nullptr; return b; }
static std::map<std::tuple<int, int64_t, int64_t, int64_t>, TraceEnt> & xdna2_trace_tab() {
    static std::map<std::tuple<int, int64_t, int64_t, int64_t>, TraceEnt> t;
    static struct P { ~P() { for (auto & kv : t) { const auto & e = kv.second; fprintf(stderr, "trace fmt %d n %lld k %lld jobs %lld: %llu flushes, total %.1f us (hook starts at %.1f, hook takes %.1f, master cpu done at %.1f, npu wait+merge %.1f)\n",
        std::get<0>(kv.first), (long long) std::get<1>(kv.first), (long long) std::get<2>(kv.first), (long long) std::get<3>(kv.first), (unsigned long long) e.n,
        e.t_total / e.n * 1e6, e.t_hook_start / e.n * 1e6, e.t_hook / e.n * 1e6, e.t_cpu_end / e.n * 1e6, e.t_end / e.n * 1e6); } } } p;
    return t;
}

static void xdna2_flush(ggml_backend_xdna2_context * ctx) {
    if (!ctx->cpu_pending.empty()) {
        const double t0 = xdna2_now();
        auto & jobs = ctx->cpu_pending;
        if (decode_mode() == DecodeMode::Split) {
            // the same weights twice in one batch would share their A/Y buffers: the later job runs on the CPU only
            for (size_t i = 0; i < jobs.size(); ++i)
                for (size_t q = 0; q < i; ++q)
                    if (jobs[q].arena == jobs[i].arena && jobs[q].w_off == jobs[i].w_off) jobs[i].plan = nullptr;
            bool began = true;
            double th0 = 0, th1 = 0;
            cpu_decode_run(jobs.data(), jobs.size(), [&] { th0 = xdna2_now(); began = npu_executor().decode_split_begin(jobs.data(), jobs.size()); th1 = xdna2_now(); });
            const double tc = xdna2_now();
            if (!began || !npu_executor().decode_split_end()) {
                GGML_ABORT("%s: NPU decode split failed", __func__);
            }
            if (xdna2_trace()) {
                const double te = xdna2_now();
                auto & e = xdna2_trace_tab()[std::make_tuple(int(jobs[0].geom.fmt), jobs[0].geom.n, jobs[0].geom.k, (int64_t) jobs.size())];
                e.n++; e.t_total += te - t0; e.t_hook_start += th0 - t0; e.t_hook += th1 - th0; e.t_cpu_end += tc - t0; e.t_end += te - tc;
            }
        } else {
            cpu_decode_run(jobs.data(), jobs.size());
            if (xdna2_trace()) {
                auto & e = xdna2_trace_tab()[std::make_tuple(int(jobs[0].geom.fmt), jobs[0].geom.n, jobs[0].geom.k, (int64_t) jobs.size())];
                e.n++; e.t_total += xdna2_now() - t0;
            }
        }
        ctx->t_cpu += xdna2_now() - t0;
        ctx->n_cpu_flush++; ctx->n_cpu_jobs += ctx->cpu_pending.size();
        for (const auto & j : ctx->cpu_pending) ctx->b_cpu += double(j.geom.bytes);
        ctx->cpu_pending.clear();
        if (ctx->pending.empty() && ctx->pf_pending.empty()) {
            ctx->pending_dst.clear();
        }
    }
    if (!ctx->pf_pending.empty()) {
        if (!npu_executor().run_prefill(ctx->pf_pending.data(), ctx->pf_pending.size())) {
            const layout::Geom & g = ctx->pf_pending[0].geom;
            GGML_ABORT("%s: NPU prefill path failed (first job: %s %lld x %lld weight)", __func__, layout::fmt_info(g.fmt).name, (long long) g.n, (long long) g.k);
        }
        ctx->pf_pending.clear();
        if (ctx->pending.empty()) {
            ctx->pending_dst.clear();
        }
    }
    if (ctx->pending.empty()) {
        return;
    }
    ComputePath & path = xdna2_select_path();
    if (!path.mul_mat_batch(ctx->pending.data(), ctx->pending.size())) {
        const layout::Geom & g = ctx->pending[0].geom;
        GGML_ABORT("%s: %s compute path failed (first job: %s %lld x %lld weight)", __func__, path.name(),
                   layout::fmt_info(g.fmt).name, (long long) g.n, (long long) g.k);
    }
    ctx->pending.clear();
    ctx->pending_dst.clear();
    ctx->n_acts = 0;
}

// Queue the products of a MUL_MAT node. Independent products (the input does not overlap a queued output) are executed together.
static void ggml_backend_xdna2_mul_mat(ggml_backend_xdna2_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(xdna2_is_repacked(src0) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->buffer && src0->buffer->buft == xdna2_buft());
    GGML_ASSERT(ne00 == ne10 && ne0 == ne01 && ne1 == ne11 && ne2 == ne12 && ne3 == ne13);
    GGML_ASSERT(nb10 == sizeof(float) && nb0 == sizeof(float));

    const layout::Geom geom = xdna2_geom(src0);
    const Arena * arena = xdna2_ctx(src0->buffer)->arena.get();
    const size_t base_off = xdna2_arena_off(src0->buffer, src0);

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    // read-after-write: src1 produced by a queued job -> run the queue first
    {
        const char * b = static_cast<const char *>(src1->data);
        const char * e = b + ggml_nbytes(src1);
        for (const auto & d : ctx->pending_dst) {
            if (b < d.second && d.first < e) {
                xdna2_flush(ctx);
                break;
            }
        }
    }

    const bool use_cpu = !reference_mode() && ne11 == 1 && decode_mode() != DecodeMode::Npu && cpu_decode_available();
    const bool use_pf = !reference_mode() && ne11 >= prefill_min_tokens() && xdna2_pf_supports(geom.fmt) && npu_executor().prefill_available();

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const int64_t i02 = i12 / r2, i03 = i13 / r3;

            if (use_cpu) {
                CpuDecodeJob cj;
                cj.geom = geom;
                cj.w    = arena->base() + base_off + size_t(i03 * ne02 + i02) * geom.bytes;
                cj.x    = reinterpret_cast<const float *>(static_cast<const char *>(src1->data) + i12 * nb12 + i13 * nb13);
                cj.dst  = reinterpret_cast<float *>(static_cast<char *>(dst->data) + i12 * nb2 + i13 * nb3);
                cj.arena = arena;
                cj.w_off = base_off + size_t(i03 * ne02 + i02) * geom.bytes;
                cj.plan  = decode_mode() == DecodeMode::Split ? xdna2_split_plan(geom) : nullptr;
                ctx->cpu_pending.push_back(cj);
                continue;
            }

            if (use_pf) {
                PrefillJob pj;
                pj.arena          = arena;
                pj.w_off          = base_off + size_t(i03 * ne02 + i02) * geom.bytes;
                pj.geom           = geom;
                pj.x              = reinterpret_cast<const float *>(static_cast<const char *>(src1->data) + i12 * nb12 + i13 * nb13);
                pj.x_stride       = nb11;
                pj.n_tok          = ne11;
                pj.dst            = reinterpret_cast<float *>(static_cast<char *>(dst->data) + i12 * nb2 + i13 * nb3);
                pj.dst_row_stride = nb1;
                ctx->pf_pending.push_back(pj);
                continue;
            }

            if (ctx->n_acts == ctx->acts.size()) {
                ctx->acts.emplace_back();
            }
            ActQ & act = ctx->acts[ctx->n_acts++];
            const double t0 = xdna2_now();
            quantize_acts(act, geom, reinterpret_cast<const float *>(static_cast<const char *>(src1->data) + i12 * nb12 + i13 * nb13),
                          ne11, nb11);
            ctx->t_quant += xdna2_now() - t0;

            MulMatJob job;
            job.arena          = arena;
            job.w_off          = base_off + size_t(i03 * ne02 + i02) * geom.bytes;
            job.w              = arena->base() + job.w_off;
            job.geom           = geom;
            job.act            = &act;
            job.dst            = reinterpret_cast<float *>(static_cast<char *>(dst->data) + i12 * nb2 + i13 * nb3);
            job.dst_row_stride = nb1;
            ctx->pending.push_back(job);
        }
    }
    const char * db = static_cast<const char *>(dst->data);
    ctx->pending_dst.emplace_back(db, db + ggml_nbytes(dst));
    ctx->n_mul_mat++;
}

//
// backend
//

static const char * ggml_backend_xdna2_get_name(ggml_backend_t) {
    return GGML_XDNA2_NAME;
}

static void xdna2_print_stats(const ggml_backend_xdna2_context * ctx, bool to_stderr = false) {
#define XLOG(...) do { if (to_stderr) { fprintf(stderr, __VA_ARGS__); } else { GGML_LOG_INFO(__VA_ARGS__); } } while (0)
    if (ctx->t_cpu > 0) {
        XLOG("ggml-xdna2: CPU decode GEMV from the NPU layout: %.3f s in %llu flushes (%llu matmuls), %.2f GB -> %.1f GB/s\n", ctx->t_cpu, (unsigned long long) ctx->n_cpu_flush, (unsigned long long) ctx->n_cpu_jobs, ctx->b_cpu / 1e9, ctx->b_cpu / 1e9 / ctx->t_cpu);
        double busy, wall; int thr;
        cpu_decode_stats(busy, wall, thr);
        XLOG("ggml-xdna2: CPU decode team: %d threads, utilisation %.1f %% (busy %.2f thread-s of %.2f s wall)\n", thr, 100.0 * busy / (wall * thr), busy, wall);
    }
    {
        const ExecStats st = npu_executor().stats();
        if (st.sp_calls > 0) {
            XLOG("ggml-xdna2: decode split NPU part: %llu batches, %llu dispatches, pack act %.3f s, start %.3f s, wait (after the CPU part) %.3f s, merge %.3f s, weights streamed %.2f GB\n",
                 (unsigned long long) st.sp_calls, (unsigned long long) st.sp_runs, st.sp_t_pack, st.sp_t_start, st.sp_t_wait, st.sp_t_merge, st.sp_bytes_w / 1e9);
        }
    }
    if (ctx->n_mul_mat > 0) {
        const ExecStats st = npu_executor().stats();
        XLOG("ggml-xdna2: time between backend graph computes (CPU splits + tensor copies) %.3f s\n", ctx->t_between);
        XLOG("ggml-xdna2: %llu graph splits, %llu mul_mat, %llu SwiGLU + %llu add fused; host quantize %.3f s, graph total %.3f s | npu: %llu runs in %llu submissions, "
                      "pack act %.3f s, wait %.3f s (%llu hw context switches: %.3f s), unpermute %.3f s, setup %.3f s, kernel wait %.3f s, weights streamed %.2f GB\n",
                      (unsigned long long) ctx->n_graphs, (unsigned long long) ctx->n_mul_mat, (unsigned long long) ctx->n_glu_fused, (unsigned long long) ctx->n_add_fused, ctx->t_quant, ctx->t_graph,
                      (unsigned long long) st.runs, (unsigned long long) st.batches, st.t_pack_a, st.t_wait, (unsigned long long) st.n_switch, st.t_wait_switch, st.t_unpermute, st.t_setup,
                      st.t_kernel_wait, st.bytes_w / 1e9);
    }
    {
        const ExecStats st = npu_executor().stats();
        if (st.pf_jobs > 0) {
            XLOG("ggml-xdna2: prefill: %llu jobs, %llu waves, %llu tile dispatches; encode %.3f s, A-BO fill %.3f s + sync %.3f s, NPU wait %.3f s, Y sync %.3f s + unpermute %.3f s, setup %.3f s, "
                 "weights streamed %.2f GB (%.1f GB/s while on the NPU), %llu hw context switches; host GB x %.2f a %.2f y %.2f\n",
                 (unsigned long long) st.pf_jobs, (unsigned long long) st.pf_waves, (unsigned long long) st.pf_dispatches, st.pf_t_encode, st.pf_t_abuild, st.pf_t_abo - st.pf_t_abuild, st.pf_t_npu,
                 st.pf_t_ysync, st.pf_t_unperm - st.pf_t_ysync, st.pf_t_setup, st.pf_bytes_w / 1e9, st.pf_t_npu > 0 ? st.pf_bytes_w / 1e9 / st.pf_t_npu : 0.0, (unsigned long long) st.pf_switches, st.pf_bytes_x / 1e9, st.pf_bytes_a / 1e9, st.pf_bytes_y / 1e9);
        }
    }
    {
        const ExecStats st = npu_executor().stats();
        if (st.fa_calls > 0) {
            XLOG("ggml-xdna2: attention: %llu calls (%.3f s), %llu pairs in %llu dispatches, %llu KV blocks encoded; KV encode %.3f s, Q encode %.3f s, "
                 "NPU wait %.3f s, output %.3f s, setup %.3f s, %llu hw context switches\n",
                 (unsigned long long) st.fa_calls, ctx->t_fa, (unsigned long long) st.fa_pairs, (unsigned long long) st.fa_dispatches,
                 (unsigned long long) st.fa_blocks_enc, st.fa_t_kv, st.fa_t_q, st.fa_t_npu, st.fa_t_out, st.fa_t_setup, (unsigned long long) st.fa_switches);
        }
    }
    XLOG("ggml-xdna2: host<->buffer copies since start: set_tensor %llu calls %.1f MB %.3f s, get_tensor %llu calls %.1f MB %.3f s\n",
                  (unsigned long long) g_set_calls.load(), g_set_bytes.load() / 1e6, g_set_ns.load() / 1e9,
                  (unsigned long long) g_get_calls.load(), g_get_bytes.load() / 1e6, g_get_ns.load() / 1e9);
}

#undef XLOG
// the stats of the last backend instance, printed at process exit when GGML_XDNA2_STATS=1 (tools that never free the backend)
static ggml_backend_xdna2_context * g_last_ctx = nullptr;
static struct StatsAtExit {
    StatsAtExit() { (void) &npu_executor(); }   // construct first => destroyed after this object
    ~StatsAtExit() {
        const char * e = std::getenv("GGML_XDNA2_STATS");
        if (g_last_ctx && e && *e && std::strcmp(e, "0") != 0) {
            xdna2_print_stats(g_last_ctx, true);
        }
    }
} g_stats_at_exit;

static void ggml_backend_xdna2_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_xdna2_context *>(backend->context);
    {
        const char * e = std::getenv("GGML_XDNA2_STATS");
        xdna2_print_stats(ctx, e && *e && std::strcmp(e, "0") != 0);
    }
    if (g_last_ctx == ctx) g_last_ctx = nullptr;
    delete ctx;
    delete backend;
}


//
// FLASH_ATTN_EXT (prefill) on the NPU attention kernel
//

// shapes the attention kernel serves (the runtime conditions -- a causal mask -- are checked in xdna2_fa_causal_p0)
static bool xdna2_fa_shape_ok(const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * mask = op->src[3];
    if (op->src[4] != nullptr || mask == nullptr || mask->type != GGML_TYPE_F16) return false;
    if (q->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (k->type != v->type || (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16)) return false;
    if (q->ne[0] != FA_D || k->ne[0] != FA_D || v->ne[0] != FA_D || op->ne[0] != FA_D) return false;
    if (q->ne[2] != FA_NH || k->ne[2] != FA_NKV || v->ne[2] != FA_NKV || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1) return false;
    if (op->ne[1] != FA_NH || op->ne[2] != q->ne[1] || op->ne[3] != 1) return false;
    if (k->ne[1] != v->ne[1] || k->nb[1] != v->nb[1] || k->nb[2] != v->nb[2]) return false;
    if (q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) || op->nb[0] != sizeof(float)) return false;
    if (k->nb[1] % 8 || size_t(FA_NKV) * FA_D * 2 > k->nb[1] || k->nb[2] != size_t(FA_D) * 2) return false;   // cells = rows of 4 contiguous heads
    if (mask->ne[0] != k->ne[1] || mask->ne[1] < q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1) return false;
    float max_bias = 0.0f, softcap = 0.0f;
    std::memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
    std::memcpy(&softcap,  (const float *) op->op_params + 2, sizeof(float));
    return max_bias == 0.0f && softcap == 0.0f;
}

// The NPU kernel computes plain causal attention over the cells 0 .. p0 + t for query t. Returns p0 if the mask is exactly that
// (0 on those cells, -inf on all others), else -1. Checked once per mask tensor and graph (all layers share the mask).
static int64_t xdna2_fa_causal_p0(const ggml_tensor * mask, int64_t n_kv, int64_t n_tok, uint64_t graph_id) {
    static const void * c_data = nullptr;
    static uint64_t c_graph = UINT64_MAX;
    static int64_t c_kv = -1, c_tok = -1, c_p0 = -1;
    if (c_data == mask->data && c_graph == graph_id && c_kv == n_kv && c_tok == n_tok) return c_p0;
    const uint16_t zero = 0x0000, ninf = 0xFC00;
    const uint16_t * row0 = static_cast<const uint16_t *>(mask->data);
    int64_t p0 = 0;
    while (p0 < n_kv && row0[p0] == zero) p0++;
    p0 -= 1;                                    // query 0 sees cells 0 .. p0
    bool ok = p0 >= 0 && p0 + n_tok <= n_kv;
    for (int64_t t = 0; ok && t < n_tok; ++t) {
        const uint16_t * row = reinterpret_cast<const uint16_t *>(static_cast<const char *>(mask->data) + t * mask->nb[1]);
        const int64_t vis = p0 + t;
        for (int64_t c = 0; c <= vis && ok; ++c) ok = row[c] == zero;
        for (int64_t c = vis + 1; c < n_kv && ok; ++c) ok = row[c] == ninf;
    }
    c_data = mask->data; c_graph = graph_id; c_kv = n_kv; c_tok = n_tok; c_p0 = ok ? p0 : -1;
    return c_p0;
}

// Host attention for the shapes the kernel serves, any mask: softmax(scale * q k^T + mask) v. Only for masks the NPU kernel cannot express
// (several sequences in one batch, cells not in position order); slow, so it warns once.
static void xdna2_fa_host(const ggml_tensor * op) {
    static bool warned = false;
    if (!warned) {
        GGML_LOG_WARN("ggml-xdna2: FLASH_ATTN_EXT with a non-causal mask runs on the slow host path (the NPU kernel needs a plain causal mask)\n");
        warned = true;
    }
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * mask = op->src[3];
    const int64_t n_kv = k->ne[1], n_tok = q->ne[1];
    float scale = 1.0f;
    std::memcpy(&scale, (const float *) op->op_params + 0, sizeof(float));
    const bool bf16 = k->type == GGML_TYPE_BF16;
    std::vector<float> kf(size_t(n_kv) * FA_D), vf(size_t(n_kv) * FA_D);
    for (int g = 0; g < FA_NKV; ++g) {
        for (int64_t c = 0; c < n_kv; ++c) {
            const char * kr = static_cast<const char *>(k->data) + c * k->nb[1] + g * k->nb[2];
            const char * vr = static_cast<const char *>(v->data) + c * v->nb[1] + g * v->nb[2];
            if (bf16) { ggml_bf16_to_fp32_row((const ggml_bf16_t *) kr, &kf[c * FA_D], FA_D); ggml_bf16_to_fp32_row((const ggml_bf16_t *) vr, &vf[c * FA_D], FA_D); }
            else      { ggml_fp16_to_fp32_row((const ggml_fp16_t *) kr, &kf[c * FA_D], FA_D); ggml_fp16_to_fp32_row((const ggml_fp16_t *) vr, &vf[c * FA_D], FA_D); }
        }
#if defined(_OPENMP)
        #pragma omp parallel for collapse(2) schedule(dynamic, 8) num_threads(pf_threads())
#endif
        for (int hh = 0; hh < FA_NH / FA_NKV; ++hh) {
            for (int64_t t = 0; t < n_tok; ++t) {
                const int h = g * (FA_NH / FA_NKV) + hh;
                const float * qr = (const float *) ((const char *) q->data + t * q->nb[1] + h * q->nb[2]);
                const ggml_fp16_t * mr = (const ggml_fp16_t *) ((const char *) mask->data + t * mask->nb[1]);
                std::vector<float> sc((size_t) n_kv);
                float mx = -INFINITY;
                for (int64_t c = 0; c < n_kv; ++c) {
                    const float m = ggml_fp16_to_fp32(mr[c]);
                    if (m == -INFINITY) { sc[c] = -INFINITY; continue; }
                    float d = 0.0f;
                    for (int e = 0; e < FA_D; ++e) d += qr[e] * kf[c * FA_D + e];
                    sc[c] = d * scale + m;
                    mx = std::max(mx, sc[c]);
                }
                float acc[FA_D] = {}, sum = 0.0f;
                for (int64_t c = 0; c < n_kv && mx != -INFINITY; ++c) {
                    if (sc[c] == -INFINITY) continue;
                    const float p = std::exp(sc[c] - mx);
                    sum += p;
                    for (int e = 0; e < FA_D; ++e) acc[e] += p * vf[c * FA_D + e];
                }
                float * d = (float *) ((char *) op->data + t * op->nb[2] + h * op->nb[1]);
                for (int e = 0; e < FA_D; ++e) d[e] = sum > 0.0f ? acc[e] / sum : 0.0f;
            }
        }
    }
}

static void ggml_backend_xdna2_flash_attn(ggml_backend_xdna2_context * ctx, ggml_tensor * op) {
    xdna2_flush(ctx);      // queued mul_mats first (graph order)
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * mask = op->src[3];
    const int64_t p0 = xdna2_fa_causal_p0(mask, k->ne[1], q->ne[1], ctx->n_graphs);
    if (p0 < 0) {
        xdna2_fa_host(op);
        return;
    }
    const ggml_tensor * kc = k->view_src ? k->view_src : k;
    AttnJob j;
    j.q = (const float *) q->data;
    j.q_tok_stride = q->nb[1];
    j.q_head_stride = q->nb[2];
    j.n_tok = q->ne[1];
    j.p0 = p0;
    j.k = k->data;
    j.v = v->data;
    j.bf16 = k->type == GGML_TYPE_BF16;
    j.kv_row_stride = k->nb[1];
    j.kv_head_stride = k->nb[2];
    j.kv_cap = int64_t(ggml_nbytes(kc) / k->nb[1]);
    j.kv_id = kc->data;
    j.dst = (float *) op->data;
    j.dst_tok_stride = op->nb[2];
    j.dst_head_stride = op->nb[1];
    std::memcpy(&j.scale, (const float *) op->op_params + 0, sizeof(float));
    const double t0 = xdna2_now();
    if (!npu_executor().run_attention(j)) {
        GGML_ABORT("ggml-xdna2: NPU attention failed");
    }
    ctx->t_fa += xdna2_now() - t0;
    ctx->n_fa++;
}

static bool xdna2_fuse_enabled() {
    static const bool b = [] { const char * e = std::getenv("GGML_XDNA2_FUSE"); return !(e && *e == '0'); }();
    return b;
}

// SWIGLU (split form) of two MUL_MAT outputs. When both products are queued prefill jobs, the later one gets the SwiGLU as the epilogue of its
// unpermute and writes the result straight into this node (its own output is never written); otherwise computed on the host.
static void ggml_backend_xdna2_glu(ggml_backend_xdna2_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * up   = dst->src[1];
    if (xdna2_fuse_enabled() && ggml_is_contiguous(dst)) {
        PrefillJob * jg = nullptr, * ju = nullptr;
        for (auto & pj : ctx->pf_pending) {
            if (pj.dst == gate->data && pj.n_tok == gate->ne[1] && pj.geom.n == gate->ne[0]) jg = &pj;
            if (pj.dst == up->data   && pj.n_tok == up->ne[1]   && pj.geom.n == up->ne[0])   ju = &pj;
        }
        if (jg && ju && jg != ju && jg->epi == PfEpi::None && ju->epi == PfEpi::None && jg->dst_row_stride == size_t(dst->nb[1]) &&
            ju->dst_row_stride == size_t(dst->nb[1])) {
            PrefillJob * later = jg > ju ? jg : ju, * earlier = jg > ju ? ju : jg;     // the units run job by job: the earlier output is complete first
            later->epi       = later == jg ? PfEpi::GluGate : PfEpi::GluUp;
            later->epi_other = earlier->dst;
            later->dst       = static_cast<float *>(dst->data);
            const char * db = static_cast<const char *>(dst->data);
            ctx->pending_dst.emplace_back(db, db + ggml_nbytes(dst));
            ctx->n_glu_fused++;
            return;
        }
    }
    xdna2_flush(ctx);
    const int64_t nr = ggml_nrows(dst), n = dst->ne[0];
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(nr >= 16) num_threads(pf_threads())
#endif
    for (int64_t r = 0; r < nr; ++r) {
        const int64_t i1 = r % dst->ne[1], i2 = (r / dst->ne[1]) % dst->ne[2], i3 = r / (dst->ne[1] * dst->ne[2]);
        swiglu_row(reinterpret_cast<float *>(static_cast<char *>(dst->data) + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]),
                   reinterpret_cast<const float *>(static_cast<const char *>(gate->data) + i1 * gate->nb[1] + i2 * gate->nb[2] + i3 * gate->nb[3]),
                   reinterpret_cast<const float *>(static_cast<const char *>(up->data) + i1 * up->nb[1] + i2 * up->nb[2] + i3 * up->nb[3]), n);
    }
}

// ADD (same shapes) of one of our products and another tensor (the residual): the queued prefill job adds it in its unpermute and writes the sum
// straight into this node; otherwise computed on the host.
static void ggml_backend_xdna2_add(ggml_backend_xdna2_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0], * b = dst->src[1];
    if (xdna2_fuse_enabled() && ggml_is_contiguous_rows(dst)) {
        PrefillJob * ja = nullptr, * jb = nullptr;
        for (auto & pj : ctx->pf_pending) {
            if (pj.dst == a->data && pj.n_tok == a->ne[1] && pj.geom.n == a->ne[0]) ja = &pj;
            if (pj.dst == b->data && pj.n_tok == b->ne[1] && pj.geom.n == b->ne[0]) jb = &pj;
        }
        PrefillJob * j = ja ? ja : jb;
        const ggml_tensor * x = ja ? b : a;          // the other operand: must not be a pending output (it is read during the flush)
        bool x_pending = false;
        {
            const char * xb = static_cast<const char *>(x->data), * xe = xb + ggml_nbytes(x);
            for (const auto & d : ctx->pending_dst) if (xb < d.second && d.first < xe) x_pending = true;
        }
        if (j && !(ja && jb) && !x_pending && j->epi == PfEpi::None && j->dst_row_stride == size_t(dst->nb[1]) && size_t(x->nb[1]) == size_t(dst->nb[1]) &&
            ggml_nrows(dst) == dst->ne[1]) {
            j->epi       = PfEpi::Add;
            j->epi_other = static_cast<const float *>(x->data);
            j->dst       = static_cast<float *>(dst->data);
            const char * db = static_cast<const char *>(dst->data);
            ctx->pending_dst.emplace_back(db, db + ggml_nbytes(dst));
            ctx->n_add_fused++;
            return;
        }
    }
    xdna2_flush(ctx);
    const int64_t nr = ggml_nrows(dst), n = dst->ne[0];
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(nr >= 16) num_threads(pf_threads())
#endif
    for (int64_t r = 0; r < nr; ++r) {
        const int64_t i1 = r % dst->ne[1], i2 = (r / dst->ne[1]) % dst->ne[2], i3 = r / (dst->ne[1] * dst->ne[2]);
        float * d = reinterpret_cast<float *>(static_cast<char *>(dst->data) + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
        const float * pa = reinterpret_cast<const float *>(static_cast<const char *>(a->data) + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3]);
        const float * pb = reinterpret_cast<const float *>(static_cast<const char *>(b->data) + i1 * b->nb[1] + i2 * b->nb[2] + i3 * b->nb[3]);
        for (int64_t i = 0; i < n; ++i) d[i] = pa[i] + pb[i];
    }
}

static enum ggml_status ggml_backend_xdna2_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_xdna2_context *>(backend->context);
    const double t_start = xdna2_now();
    if (ctx->t_last_end > 0) ctx->t_between += t_start - ctx->t_last_end;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_xdna2_mul_mat(ctx, node);
                break;

            case GGML_OP_FLASH_ATTN_EXT:
                ggml_backend_xdna2_flash_attn(ctx, node);
                break;

            case GGML_OP_GLU:
                ggml_backend_xdna2_glu(ctx, node);
                break;

            case GGML_OP_ADD:
                ggml_backend_xdna2_add(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    xdna2_flush(ctx);
    ctx->n_graphs++;
    ctx->t_last_end = xdna2_now();
    ctx->t_graph += ctx->t_last_end - t_start;
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i xdna2_backend_i = {
    /* .get_name                = */ ggml_backend_xdna2_get_name,
    /* .free                    = */ ggml_backend_xdna2_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_xdna2_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_xdna2_guid(void) {
    static ggml_guid guid = { 0x58, 0x44, 0x4e, 0x41, 0x32, 0xa7, 0x3c, 0x91, 0x0e, 0xd5, 0x6b, 0x42, 0xf8, 0x17, 0x9a, 0x2d };
    return &guid;
}

ggml_backend_t ggml_backend_xdna2_init(void) {
    return new ggml_backend {
        /* .guid    = */ ggml_backend_xdna2_guid(),
        /* .iface   = */ xdna2_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna2_reg(), 0),
        /* .context = */ (g_last_ctx = new ggml_backend_xdna2_context),
    };
}

bool ggml_backend_is_xdna2(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_xdna2_guid());
}

//
// device
//

static const char * ggml_backend_xdna2_device_get_name(ggml_backend_dev_t) {
    return GGML_XDNA2_NAME;
}

static const char * ggml_backend_xdna2_device_get_description(ggml_backend_dev_t) {
    return "AMD XDNA2 NPU";
}

//
// host buffer type: plain host memory (CPU readable in place, is_host) that is also an NPU buffer object, for the compute tensors.
// Tensors are 64 B aligned (CPU buffers align to 32 B), and the NPU can address them (DMA straight into the destination tensors).
//

static void ggml_backend_xdna2_host_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    std::memset(static_cast<uint8_t *>(tensor->data) + offset, value, size);
}

static void ggml_backend_xdna2_host_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
}

static void ggml_backend_xdna2_host_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
}

static bool ggml_backend_xdna2_host_buffer_cpy_tensor(ggml_backend_buffer_t, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_xdna2_host_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    Arena * arena = xdna2_ctx(buffer)->arena.get();
    std::memset(arena->base(), value, arena->size());
}

static const ggml_backend_buffer_i ggml_backend_xdna2_host_buffer_i = {
    /* .free_buffer   = */ ggml_backend_xdna2_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_xdna2_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_xdna2_host_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_xdna2_host_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_xdna2_host_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_xdna2_host_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_xdna2_host_buffer_clear,
    /* .reset         = */ nullptr,
};

static const char * ggml_backend_xdna2_host_buft_get_name(ggml_backend_buffer_type_t) {
    return GGML_XDNA2_NAME "_Host";
}

static ggml_backend_buffer_t ggml_backend_xdna2_host_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    std::unique_ptr<Arena> arena;
    try {
        arena = npu_executor().alloc_weight_arena(size);     // a host-only BO (cacheable mapping), like the weights
        if (!arena) {
            arena = std::make_unique<HostArena>(size);
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes: %s\n", __func__, size, e.what());
        return nullptr;
    }
    const size_t asize = arena->size();
    auto * ctx = new xdna2_buffer_context{std::move(arena)};
    return ggml_backend_buffer_init(buft, ggml_backend_xdna2_host_buffer_i, ctx, asize);
}

static size_t ggml_backend_xdna2_host_buft_get_alignment(ggml_backend_buffer_type_t) {
    return 64;     // whole cache lines: the prefill output path writes full lines with non-temporal stores
}

static bool ggml_backend_xdna2_host_buft_is_host(ggml_backend_buffer_type_t) {
    return true;
}

static ggml_backend_buffer_type_t xdna2_host_buft() {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_xdna2_host_buft_get_name,
            /* .alloc_buffer     = */ ggml_backend_xdna2_host_buft_alloc_buffer,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ ggml_backend_xdna2_host_buft_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ ggml_backend_xdna2_host_buft_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna2_reg(), 0),
        /* .context = */ nullptr,
    };
    return &buft;
}

static ggml_backend_buffer_type_t ggml_backend_xdna2_device_get_host_buffer_type(ggml_backend_dev_t) {
    return xdna2_host_buft();
}

static void ggml_backend_xdna2_device_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    // buffers live in host memory (system RAM shared with the NPU): nothing to report
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_xdna2_device_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_xdna2_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna2_device_get_name(dev);
    props->description = ggml_backend_xdna2_device_get_description(dev);
    props->type        = ggml_backend_xdna2_device_get_type(dev);
    ggml_backend_xdna2_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_xdna2_device_init_backend(ggml_backend_dev_t, const char *) {
    return ggml_backend_xdna2_init();
}

static ggml_backend_buffer_type_t ggml_backend_xdna2_device_get_buffer_type(ggml_backend_dev_t) {
    return xdna2_buft();
}

static bool xdna2_reject(int why) {
    if (std::getenv("GGML_XDNA2_DEBUG")) fprintf(stderr, "ggml-xdna2: supports_op rejected (%d)\n", why);
    return false;
}

// the tensor a RESHAPE (chain) of t is a reshape of (same data, same row size), else t
static const ggml_tensor * xdna2_through_reshape(const ggml_tensor * t) {
    while (t->op == GGML_OP_RESHAPE && t->view_src && t->src[0] && t->src[0]->ne[0] == t->ne[0]) t = t->src[0];
    return t;
}

static bool ggml_backend_xdna2_device_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const ggml_tensor * src0 = op->src[0];
            const ggml_tensor * src1 = op->src[1];

            layout::Fmt fmt;
            if (!xdna2_fmt_of(src0->type, fmt) || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                return xdna2_reject(1);
            }
            // the weights must be (or, when not allocated yet, will be) in our buffer type; weights in any
            // other buffer cannot be read directly and a per-run copy is never wanted
            if (src0->buffer && src0->buffer->buft != xdna2_buft()) {
                return xdna2_reject(2);
            }
            if (src0->view_src != nullptr || !ggml_is_contiguous(src0)) {
                return xdna2_reject(3);
            }
            if (!layout::k_supported(src0->ne[0]) || src0->ne[0] != src1->ne[0]) {
                return xdna2_reject(4);
            }
            if (src1->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) {
                return xdna2_reject(5);
            }
            if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                return xdna2_reject(6);
            }
            // placement policy: only weights that can really run on the NPU (a kernel exists or can be built)
            // may live in our buffer; everything else must stay in CPU buffers
            return placement_ok(layout::make_geom(fmt, src0->ne[1], src0->ne[0])) || xdna2_reject(99);
        }

        case GGML_OP_FLASH_ATTN_EXT:
        {
            // prefill only (decode stays on the CPU), and only with enough keys to pay for the two hw context switches of the layer
            if (!attention_enabled() || reference_mode()) return xdna2_reject(20);
            if (!xdna2_fa_shape_ok(op)) return xdna2_reject(21);
            if (op->src[0]->ne[1] < attention_min_tokens() || op->src[1]->ne[1] < attention_min_kv()) return xdna2_reject(22);
            return npu_executor().attention_available() || xdna2_reject(23);
        }

        case GGML_OP_GLU:
        {
            // SwiGLU of two of our products (fused into the unpermute of the later one, see ggml_backend_xdna2_glu)
            const ggml_tensor * a = op->src[0], * b = op->src[1];
            if (!xdna2_fuse_enabled() || ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU || b == nullptr || ggml_get_op_params_i32(op, 1) != 0) return xdna2_reject(30);
            if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, op)) return xdna2_reject(31);
            if (a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) return xdna2_reject(32);
            if (xdna2_through_reshape(a)->op != GGML_OP_MUL_MAT || xdna2_through_reshape(b)->op != GGML_OP_MUL_MAT) return xdna2_reject(33);
            return true;
        }

        case GGML_OP_ADD:
        {
            // residual add onto one of our products (fused into its unpermute, see ggml_backend_xdna2_add)
            const ggml_tensor * a = op->src[0], * b = op->src[1];
            if (!xdna2_fuse_enabled()) return xdna2_reject(40);
            if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, op)) return xdna2_reject(41);
            if (a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) return xdna2_reject(42);
            a = xdna2_through_reshape(a);
            b = xdna2_through_reshape(b);
            if ((a->op == GGML_OP_MUL_MAT) == (b->op == GGML_OP_MUL_MAT)) return xdna2_reject(43);
            const ggml_tensor * mm = a->op == GGML_OP_MUL_MAT ? a : b;
            if (mm->src[0]->buffer && mm->src[0]->buffer->buft != xdna2_buft()) return xdna2_reject(44);
            return true;
        }

        default:
            return xdna2_reject(7);
    }
}

static bool ggml_backend_xdna2_device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    // our own buffers (weights in the device layout) and plain host memory: the activations of a MUL_MAT are read in place from the CPU
    // buffers instead of being copied into one of our buffers first (the scheduler copies split inputs the backend does not support)
    return buft == xdna2_buft() || ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i ggml_backend_xdna2_device_i = {
    /* .get_name             = */ ggml_backend_xdna2_device_get_name,
    /* .get_description      = */ ggml_backend_xdna2_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna2_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna2_device_get_type,
    /* .get_props            = */ ggml_backend_xdna2_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna2_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna2_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_xdna2_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_xdna2_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna2_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//
// registry
//

static const char * ggml_backend_xdna2_reg_get_name(ggml_backend_reg_t) {
    return GGML_XDNA2_NAME;
}

static bool xdna2_available() {
    if (xdna2_disabled()) {
        return false;
    }
    // GGML_XDNA2_REFERENCE=1 (tests): host emulation, no NPU needed; otherwise the NPU executor must be usable
    return reference_mode() || (npu_device_present() && npu_executor().available());
}

static size_t ggml_backend_xdna2_reg_get_device_count(ggml_backend_reg_t) {
    return xdna2_available() ? 1 : 0;
}

static ggml_backend_dev_t ggml_backend_xdna2_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_xdna2_device = {
        /* .iface   = */ ggml_backend_xdna2_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_xdna2_device;
}

static const ggml_backend_reg_i ggml_backend_xdna2_reg_i = {
    /* .get_name         = */ ggml_backend_xdna2_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna2_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna2_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_xdna2_reg(void) {
    static ggml_backend_reg ggml_backend_xdna2_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna2_reg_i,
        /* .context     = */ nullptr,
    };

    return &ggml_backend_xdna2_reg;
}

static int ggml_backend_xdna2_score(void) {
    return xdna2_available() ? 1 : 0;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna2_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_xdna2_score)

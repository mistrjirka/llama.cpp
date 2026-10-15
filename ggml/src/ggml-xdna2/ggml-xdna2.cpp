// ggml-xdna2: dynamically loadable ggml backend for the AMD XDNA2 NPU.
// Template: ggml-blas. See ggml-xdna2-layout.h (device weight layout), ggml-xdna2-compute.h (compute paths)
// and ggml-xdna2-npu.h (NPU executor interface + wiring TODO list).

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-xdna2.h"
#include "ggml-backend-impl.h"

#include "ggml-xdna2-arena.h"
#include "ggml-xdna2-compute.h"
#include "ggml-xdna2-layout.h"
#include "ggml-xdna2-npu.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <cstring>
#include <memory>
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
        std::memset(static_cast<uint8_t *>(tensor->data) + offset, value, size);
        xdna2_ctx(buffer)->arena->sync_to_device(xdna2_arena_off(buffer, tensor) + offset, size);
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
        arena->sync_to_device(xdna2_arena_off(buffer, tensor) + offset, size);
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
        arena->sync_from_device(xdna2_arena_off(buffer, tensor) + offset, size);
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
    std::vector<std::pair<const char *, const char *>> pending_dst;   // [begin, end) of the pending outputs
    double t_quant = 0, t_graph = 0;      // seconds
    uint64_t n_graphs = 0, n_mul_mat = 0;
};

// The host reference path is a test/debug tool (GGML_XDNA2_REFERENCE=1) and never a fallback for the NPU.
static ComputePath & xdna2_select_path() {
    return reference_mode() ? host_reference_path() : npu_path();
}

static double xdna2_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void xdna2_flush(ggml_backend_xdna2_context * ctx) {
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

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const int64_t i02 = i12 / r2, i03 = i13 / r3;

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
    if (ctx->n_mul_mat > 0) {
        const ExecStats st = npu_executor().stats();
        XLOG("ggml-xdna2: %llu graph splits, %llu mul_mat; host quantize %.3f s, graph total %.3f s | npu: %llu runs in %llu submissions, "
                      "pack act %.3f s, wait %.3f s (%llu hw context switches: %.3f s), unpermute %.3f s, setup %.3f s, kernel wait %.3f s, weights streamed %.2f GB\n",
                      (unsigned long long) ctx->n_graphs, (unsigned long long) ctx->n_mul_mat, ctx->t_quant, ctx->t_graph,
                      (unsigned long long) st.runs, (unsigned long long) st.batches, st.t_pack_a, st.t_wait, (unsigned long long) st.n_switch, st.t_wait_switch, st.t_unpermute, st.t_setup,
                      st.t_kernel_wait, st.bytes_w / 1e9);
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
    xdna2_print_stats(ctx);
    if (g_last_ctx == ctx) g_last_ctx = nullptr;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_xdna2_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_xdna2_context *>(backend->context);
    const double t_start = xdna2_now();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_xdna2_mul_mat(ctx, node);
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
    ctx->t_graph += xdna2_now() - t_start;
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
        /* .host_buffer           = */ false,
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

        default:
            return xdna2_reject(7);
    }
}

static bool ggml_backend_xdna2_device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == xdna2_buft();
}

static const ggml_backend_device_i ggml_backend_xdna2_device_i = {
    /* .get_name             = */ ggml_backend_xdna2_device_get_name,
    /* .get_description      = */ ggml_backend_xdna2_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna2_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna2_device_get_type,
    /* .get_props            = */ ggml_backend_xdna2_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna2_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna2_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
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

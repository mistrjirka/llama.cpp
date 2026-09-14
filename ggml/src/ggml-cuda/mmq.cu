#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <memory>
#include "moe-stream.cuh"

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
namespace {
struct moe_stream_options {
    bool enabled = false;
    bool trace = false;
    int group = 16;
    int min_tokens = 64;
    size_t bytes = 64u * 1024u * 1024u;
};

static const moe_stream_options & moe_stream_config() {
    static const moe_stream_options config = [] {
        moe_stream_options c;
        const char * enable = std::getenv("GGML_CUDA_MOE_STREAM");
        c.enabled = enable && std::strcmp(enable, "1") == 0;
        c.trace = std::getenv("GGML_CUDA_MOE_STREAM_TRACE") != nullptr;
        auto integer = [](const char * name, long fallback, long low, long high) {
            const char * value = std::getenv(name);
            if (!value) { return fallback; }
            char * end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end == value || *end || parsed < low || parsed > high) {
                GGML_LOG_WARN("moe-stream: ignoring invalid %s=%s\n", name, value);
                return fallback;
            }
            return parsed;
        };
        c.group = integer("GGML_CUDA_MOE_STREAM_GROUP", 16, 1, 512);
        c.min_tokens = integer("GGML_CUDA_MOE_STREAM_MIN_TOKENS", 64, 2, 16384);
        c.bytes = size_t(integer("GGML_CUDA_MOE_STREAM_MIB", 64, 2, 2048)) * 1024 * 1024;
        return c;
    }();
    return config;
}

constexpr size_t moe_stream_padding = 512;
constexpr int moe_stream_max_experts = 4096;

// One pool per backend context, reused across layers; never shared across models
// or independent contexts. This first version streams individual projections.
struct moe_stream_state {
    int device;
    cudaStream_t copy = nullptr;
    cudaEvent_t metadata = nullptr;
    cudaEvent_t ready[2] = {nullptr, nullptr};
    cudaEvent_t consumed[2] = {nullptr, nullptr};
    char * weights[2] = {nullptr, nullptr};
    char * staging[2] = {nullptr, nullptr};
    int32_t * bounds = nullptr;
    bool used[2] = {false, false};
    size_t slot_bytes;
    uint64_t calls = 0, waves = 0, h2d_bytes = 0, staged_bytes = 0;

    explicit moe_stream_state(int device) : device(device), slot_bytes(moe_stream_config().bytes / 2) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&metadata, cudaEventDisableTiming));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&bounds), (moe_stream_max_experts + 1) * sizeof(int32_t)));
        for (int s = 0; s < 2; ++s) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ready[s], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&consumed[s], cudaEventDisableTiming));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&weights[s]), slot_bytes));
            CUDA_CHECK(cudaMemsetAsync(weights[s], 0, slot_bytes, copy));
        }
        if (moe_stream_config().trace) {
            GGML_LOG_INFO("moe-stream: device=%d pool=%.2f MiB group=%d min_tokens=%d\n", device,
                2.0 * slot_bytes / (1024 * 1024), moe_stream_config().group, moe_stream_config().min_tokens);
        }
    }

    ~moe_stream_state() {
        ggml_cuda_set_device(device);
        // DMA and the last readers must finish before freeing either side.
        CUDA_CHECK(cudaStreamSynchronize(copy));
        for (int s = 0; s < 2; ++s) {
            if (used[s]) { CUDA_CHECK(cudaEventSynchronize(consumed[s])); }
            CUDA_CHECK(cudaFree(weights[s]));
            if (staging[s]) { CUDA_CHECK(cudaFreeHost(staging[s])); }
            CUDA_CHECK(cudaEventDestroy(ready[s]));
            CUDA_CHECK(cudaEventDestroy(consumed[s]));
        }
        CUDA_CHECK(cudaFreeHost(bounds));
        CUDA_CHECK(cudaEventDestroy(metadata));
        CUDA_CHECK(cudaStreamDestroy(copy));
        if (moe_stream_config().trace) {
            GGML_LOG_INFO("moe-stream: calls=%llu waves=%llu H2D=%.3f GiB RAM-staging=%.3f GiB\n",
                (unsigned long long) calls, (unsigned long long) waves,
                h2d_bytes / double(1ull << 30), staged_bytes / double(1ull << 30));
        }
    }
};

static std::mutex moe_stream_mutex;
static std::unordered_map<ggml_backend_cuda_context *, std::unique_ptr<moe_stream_state>> moe_stream_states;

static moe_stream_state & moe_stream_get_state(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(moe_stream_mutex);
    auto & state = moe_stream_states[&ctx];
    if (!state) { state.reset(new moe_stream_state(ctx.device)); }
    return *state;
}
} // namespace

bool ggml_cuda_moe_stream_supported(const ggml_tensor * op, const int cc) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(op); GGML_UNUSED(cc);
    return false;
#else
    const auto & cfg = moe_stream_config();
    if (!cfg.enabled || (cc != GGML_CUDA_CC_VOLTA && cc != GGML_CUDA_CC_TURING) ||
        op->op != GGML_OP_MUL_MAT_ID || op->ne[2] < cfg.min_tokens) { return false; }
    const auto * w = op->src[0];
    const auto * x = op->src[1];
    const auto * ids = op->src[2];
    if (!w || !x || !ids || !w->buffer || w->view_src || !ggml_backend_buffer_is_host(w->buffer) ||
        ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        !ggml_is_contiguous(w) || w->ne[3] != 1 || x->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        w->ne[1] % 128 != 0 || w->ne[2] < 2 || w->ne[2] > moe_stream_max_experts ||
        ids->nb[0] != sizeof(int32_t) || x->nb[2] % x->nb[1] || op->nb[2] % op->nb[1] ||
        w->nb[2] + moe_stream_padding > cfg.bytes / 2) { return false; }
    switch (w->type) {
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ4_NL:
            return true;
        default: return false;
    }
#endif
}

void ggml_cuda_moe_stream_release(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(moe_stream_mutex);
    moe_stream_states.erase(&ctx);
}

static void ggml_cuda_moe_stream_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                    const mmq_args & original, cudaStream_t compute) {
    auto & state = moe_stream_get_state(ctx);
    const int n_experts = src0->ne[2];
    const size_t expert_bytes = src0->nb[2];
    const int group = std::min<int>(moe_stream_config().group,
        (state.slot_bytes - moe_stream_padding) / expert_bytes);
    GGML_ASSERT(group > 0 && original.ids_dst != nullptr);
    CUDA_CHECK(cudaMemcpyAsync(state.bounds, original.expert_bounds, (n_experts + 1) * sizeof(int32_t),
        cudaMemcpyDeviceToHost, compute));
    CUDA_CHECK(cudaEventRecord(state.metadata, compute));
    // Exactly one host metadata wait per projection, not one per expert.
    CUDA_CHECK(cudaEventSynchronize(state.metadata));

    cudaPointerAttributes attributes{};
    const cudaError_t pointer_result = cudaPointerGetAttributes(&attributes, src0->data);
    bool pinned = false;
    if (pointer_result == cudaSuccess) {
        pinned = attributes.type == cudaMemoryTypeHost;
    } else if (pointer_result == cudaErrorInvalidValue) {
        (void) cudaGetLastError();
    } else {
        CUDA_CHECK(pointer_result);
    }

    ++state.calls;
    int wave = 0;
    for (int first = 0; first < n_experts; first += group) {
        const int count = std::min(group, n_experts - first);
        if (state.bounds[first] == state.bounds[first + count]) { continue; }
        const int s = wave++ % 2;
        if (state.used[s]) {
            // GPU readers protect the device slot; DMA completion separately
            // protects the reusable pinned host staging slot.
            CUDA_CHECK(cudaStreamWaitEvent(state.copy, state.consumed[s], 0));
            if (!pinned) { CUDA_CHECK(cudaEventSynchronize(state.ready[s])); }
        }
        if (!pinned && !state.staging[s]) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&state.staging[s]), state.slot_bytes));
        }

        // Consecutive live experts are coalesced. Empty experts keep zero rows
        // in the original bounds map, so no token routing has to be repeated.
        int e = first;
        while (e < first + count) {
            if (state.bounds[e] == state.bounds[e + 1]) { ++e; continue; }
            const int begin = e++;
            while (e < first + count && state.bounds[e] != state.bounds[e + 1]) { ++e; }
            const size_t bytes = size_t(e - begin) * expert_bytes;
            const size_t dst_offset = size_t(begin - first) * expert_bytes;
            const char * source = static_cast<const char *>(src0->data) + size_t(begin) * expert_bytes;
            if (!pinned) {
                std::memcpy(state.staging[s] + dst_offset, source, bytes);
                source = state.staging[s] + dst_offset;
                state.staged_bytes += bytes;
            }
            CUDA_CHECK(cudaMemcpyAsync(state.weights[s] + dst_offset, source, bytes, cudaMemcpyHostToDevice, state.copy));
            state.h2d_bytes += bytes;
        }
        CUDA_CHECK(cudaMemsetAsync(state.weights[s] + size_t(count) * expert_bytes, 0, moe_stream_padding, state.copy));
        CUDA_CHECK(cudaEventRecord(state.ready[s], state.copy));
        CUDA_CHECK(cudaStreamWaitEvent(compute, state.ready[s], 0));

        mmq_args args = original;
        args.x = state.weights[s];
        args.expert_bounds = original.expert_bounds + first;
        args.nchannels_x = args.nchannels_y = count;
        // Keep original logical model geometry/tile selection. Only physical
        // channels and their weight base change; output IDs stay global.
        ggml_cuda_mul_mat_q_switch_type(ctx, args, compute);
        CUDA_CHECK(cudaEventRecord(state.consumed[s], compute));
        state.used[s] = true;
        ++state.waves;
    }
}

#else
bool ggml_cuda_moe_stream_supported(const ggml_tensor * op, int cc) {
    GGML_UNUSED(op); GGML_UNUSED(cc);
    return false;
}
void ggml_cuda_moe_stream_release(ggml_backend_cuda_context & ctx) {
    GGML_UNUSED(ctx);
}
static void ggml_cuda_moe_stream_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                    const mmq_args & args, cudaStream_t stream) {
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(args); GGML_UNUSED(stream);
    GGML_ABORT("MoE host streaming is only implemented for NVIDIA SM70/SM75");
}
#endif

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, std::max<int64_t>(ne_get_rows, 128)) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // ncols_opt selects the MMQ J tile width; the launch grid still covers all routed rows.
    // Ornith/Qwen3.5-MoE has 256 experts with top-8 routing and strongly skewed expert loads,
    // so choosing J from the whole microbatch substantially over-tiles most experts. Keep the
    // override exact to the measured SM70/SM75 expert geometries and N <= 512; unrelated MoE
    // models and larger batches retain the generic selector.
    int64_t ncols_opt = ne12;
    if (ne02 == 256 && n_expert_used == 8 && ne12 > 0 && ne12 <= 512 &&
            (cc == GGML_CUDA_CC_VOLTA || cc == GGML_CUDA_CC_TURING)) {
        if (ne00 == 2048 && ne01 == 512) {
            // Gate/up. AD-Q6_K-Q5_K uses Q5_K; the 21 GiB Turing quant uses Q4_K.
            if (src0->type == GGML_TYPE_Q5_K) {
                ncols_opt = ne12 <= 128 ? 8 : ne12 <= 256 ? 16 : 24;
            } else if (src0->type == GGML_TYPE_Q4_K && cc == GGML_CUDA_CC_TURING && ne12 >= 384) {
                ncols_opt = 24;
            }
        } else if (ne00 == 512 && ne01 == 2048) {
            // Down. AD-Q6_K-Q5_K uses Q6_K; the 21 GiB Turing quant uses Q5_K.
            if (src0->type == GGML_TYPE_Q6_K) {
                if (ne12 < 64) {
                    // Keep TG and unmeasured tiny appends on the generic J8 choice.
                    ncols_opt = 8;
                } else if (ne12 <= 128) {
                    // At N=64 SM75 prefers J16, while SM70 still prefers J8.
                    ncols_opt = ne12 == 64 && cc == GGML_CUDA_CC_VOLTA ? 8 : 16;
                } else if (ne12 <= 256) {
                    ncols_opt = 24;
                } else {
                    ncols_opt = 48;
                }
            } else if (src0->type == GGML_TYPE_Q5_K && cc == GGML_CUDA_CC_TURING && ne12 >= 384) {
                ncols_opt = 64;
            }
        }
    } else if (GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt};

    if (ggml_cuda_moe_stream_supported(dst, cc)) {
        ggml_cuda_moe_stream_mmq(ctx, src0, args, stream);
    } else {
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    }
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (turing_mma_available(cc)) {
        // On physical sm_75, dense large-N quantized matmuls cross over to
        // dequantize->FP16 + cuBLAS. A matched sweep on RTX 2080 Ti puts the best
        // default around N=256; lower thresholds convert too many medium-size ops.
        // Keep routed-MoE and small-N/decode on MMQ. Do not change Ampere+ policy.
        if (cc == GGML_CUDA_CC_TURING && n_experts == 0) {
            int64_t threshold = 256;
            if (const char * env = getenv("GGML_CUDA_TURING_CUBLAS_MIN_BATCH")) {
                threshold = std::max<int64_t>(1, atoll(env));
            }
            if (ne11 >= threshold) {
                return false;
            }
        } else if (n_experts == 0 && (type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K)) {
            // Preserve the pre-existing opt-in crossover experiment on non-Turing
            // tensor-core NVIDIA devices.
            if (const char * env = getenv("GGML_CUDA_TURING_CUBLAS_MIN_BATCH")) {
                const int64_t threshold = std::max<int64_t>(1, atoll(env));
                if (ne11 >= threshold) {
                    return false;
                }
            }
        }
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

    if (cc == GGML_CUDA_CC_VOLTA && n_experts > 0) {
        const char * force = getenv("GGML_CUDA_VOLTA_FORCE_MMQ");
        if (force != nullptr && strcmp(force, "moe") == 0) {
            return true;
        }
    }

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

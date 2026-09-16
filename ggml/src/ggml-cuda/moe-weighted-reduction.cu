#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void lf_accum_f32(const float * state, const int32_t * tokens,
                                   const float * weighted, const int32_t * mapping,
                                   float * dst, int64_t width, int rows, int ranks) {
    const int owner = blockIdx.x;
    const int64_t col = int64_t(blockIdx.y)*blockDim.x + threadIdx.x;
    if (col >= width) {
        return;
    }
    float sum = state[int64_t(tokens[owner])*width+col];
    for (int rank=0; rank<ranks; ++rank) {
        const int row = mapping[rank*rows+owner];
        sum = __fadd_rn(sum, weighted[int64_t(row)*width+col]);
    }
    dst[int64_t(owner)*width+col] = sum;
}

void ggml_cuda_op_lf_accum(ggml_backend_cuda_context & ctx,
                          const ggml_tensor * state, const ggml_tensor * tokens,
                          const ggml_tensor * weighted, const ggml_tensor * mapping,
                          int ranks, ggml_tensor * dst) {
    const int rows = int(dst->ne[1]);
    const int64_t width = dst->ne[0];
    GGML_ASSERT(state->type == GGML_TYPE_F32 && weighted->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(tokens->type == GGML_TYPE_I32 && mapping->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(state) && ggml_is_contiguous(weighted) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(tokens) && ggml_is_contiguous(mapping));
    GGML_ASSERT(tokens->ne[0] == rows && weighted->ne[0] == width && weighted->ne[1] == rows+1);
    GGML_ASSERT(mapping->ne[0] == int64_t(rows)*ranks && ranks > 0 && ranks <= 10);
    const dim3 blocks(rows,(width+255)/256,1);
    lf_accum_f32<<<blocks,256,0,ctx.stream()>>>(
        (const float *)state->data,(const int32_t *)tokens->data,
        (const float *)weighted->data,(const int32_t *)mapping->data,
        (float *)dst->data,width,rows,ranks);
    CUDA_CHECK(cudaGetLastError());
}

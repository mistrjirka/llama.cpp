#include "reduce_rows.cuh"
#include "sumrows.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    }
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    if ((nrows / nsm) < 2) {
        // Increase num threads to 512 for small nrows to better hide the latency
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, src0_d, dst_d, ncols);
    } else {
        // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, src0_d, dst_d, ncols);
    }
}


// Qwen4Exp QSA score reduction. The unfused graph has contiguous scores laid out as
// [n_blocks, 4 heads, n_queries, n_stream], applies ReLU, permutes heads to dim 0,
// materializes that permutation, then SUM_ROWS. One thread computes one output score,
// while each warp performs four coalesced loads over neighboring blocks.
static __global__ void qsa_relu4_sum_f32(
        const float * src, float * dst, int64_t n_blocks, int64_t n_queries, int64_t n_stream) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_blocks * n_queries * n_stream;
    if (i >= n) {
        return;
    }

    const int64_t block = i % n_blocks;
    const int64_t qstream = i / n_blocks;
    const int64_t base = block + qstream * (4 * n_blocks);

    // Match op_relu (fmaxf) and reduce_rows_f32's 32-lane reduction tree for ncols=4:
    // lane 0 ends with (x0 + x2) + (x1 + x3).
    const float x0 = fmaxf(src[base + 0 * n_blocks], 0.0f);
    const float x1 = fmaxf(src[base + 1 * n_blocks], 0.0f);
    const float x2 = fmaxf(src[base + 2 * n_blocks], 0.0f);
    const float x3 = fmaxf(src[base + 3 * n_blocks], 0.0f);
    dst[i] = __fadd_rn(__fadd_rn(x0, x2), __fadd_rn(x1, x3));
}

void ggml_cuda_op_qsa_relu4_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * relu, ggml_tensor * dst) {
    const ggml_tensor * src = relu->src[0];
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->ne[1] == 4);
    GGML_ASSERT(ggml_is_contiguous(src));

    const int64_t n_blocks  = src->ne[0];
    const int64_t n_queries = src->ne[2];
    const int64_t n_stream  = src->ne[3];
    GGML_ASSERT(ggml_nelements(dst) == n_blocks * n_queries * n_stream);

    constexpr int threads = 256;
    const int64_t n = n_blocks * n_queries * n_stream;
    const dim3 blocks((unsigned int) ((n + threads - 1) / threads), 1, 1);
    const dim3 block_dims(threads, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks, block_dims, 0, ctx.stream());
    ggml_cuda_kernel_launch(qsa_relu4_sum_f32, launch_params,
            (const float *) src->data, (float *) dst->data, n_blocks, n_queries, n_stream);
}

#include "common.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream);
void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_qsa_relu4_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * relu, ggml_tensor * dst);

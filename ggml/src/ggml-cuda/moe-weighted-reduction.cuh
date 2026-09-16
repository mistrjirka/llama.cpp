#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

void ggml_cuda_op_lf_accum(ggml_backend_cuda_context & ctx,
                          const ggml_tensor * state, const ggml_tensor * tokens,
                          const ggml_tensor * weighted, const ggml_tensor * mapping,
                          int ranks, ggml_tensor * dst);

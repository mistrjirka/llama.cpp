#include "mmvq.cuh"

// The BF16 PFlash scorer does not use quantized MMVQ. Keep stubs so the CUDA backend links without building MMVQ.
int get_mmvq_mmid_max_batch(ggml_type, int) { return 0; }

bool ggml_cuda_mmvq_mmid_grouped_enabled(ggml_type, int, int64_t, int64_t) { return false; }

void ggml_cuda_mul_mat_vec_q(
    ggml_backend_cuda_context &,
    const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *,
    const ggml_cuda_mm_fusion_args_host *) {
    GGML_ABORT("quantized MMVQ unavailable in BF16 PFlash experiment build");
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context &,
    const ggml_tensor *, const ggml_tensor *, ggml_tensor *, const char *, const float *,
    const char *, float *, const int64_t, const int64_t, const int64_t,
    const int64_t, cudaStream_t) {
    GGML_ABORT("quantized MMVQ unavailable in BF16 PFlash experiment build");
}

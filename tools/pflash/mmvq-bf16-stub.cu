#include "mmvq.cuh"

// Experimental BF16-drafter build: PFlash Qwen3-0.6B BF16 never uses
// quantized MMVQ. Keep the ABI surface so ggml-cuda links without compiling
// the multi-million-line generated MMVQ PTX on sm70.
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

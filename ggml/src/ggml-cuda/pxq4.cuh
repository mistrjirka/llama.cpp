#pragma once
#include "common.cuh"
// Receives the same q8_1 activations as stock MMVQ, but uses real PXQ tensor byte strides.
void ggml_cuda_pxq4_mmvq_launch(const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const block_q8_1 * acts, int64_t ne10_padded,
        const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream);
// Volta-only grouped PXQ4 prefill. False means use the reference fallback.
bool ggml_cuda_pxq4_prefill_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_pxq4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// True only for complete, correctly aligned PXQ4 panels. No device access.
bool ggml_cuda_pxq4_layout_supported(const ggml_tensor * tensor);

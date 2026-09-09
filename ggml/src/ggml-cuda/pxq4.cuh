#pragma once
#include "common.cuh"
// Receives the same q8_1 activations as stock MMVQ, but uses real PXQ tensor byte strides.
void ggml_cuda_pxq4_mmvq_launch(const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const block_q8_1 * acts, int64_t ne10_padded,
        const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream);
void ggml_cuda_pxq4hq_mmvq_launch(const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const block_q8_1 * acts, int64_t ne10_padded,
        const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream);
// Volta-only grouped PXQ4 prefill. False means use the reference fallback.
bool ggml_cuda_pxq4_prefill_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_pxq4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// True only for complete, correctly aligned PXQ4 panels. No device access.
bool ggml_cuda_pxq4_layout_supported(const ggml_tensor * tensor);
// Direct decoded-weight/F32-activation control; no additional Q8 or s8 snapping.
void ggml_cuda_pxq4_mmvf_launch(const ggml_tensor * w, const ggml_tensor * x,
        const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream);

// Shared PXQ-family helpers.
bool ggml_cuda_is_pxq_type(ggml_type type);
bool ggml_cuda_pxq_layout_supported(const ggml_tensor * tensor);
void ggml_cuda_pxq_dequant_f16(ggml_type type, const void * src, half * dst, int64_t nrows, int64_t K, cudaStream_t stream);
void ggml_cuda_pxq_dequant_f32(ggml_type type, const void * src, float * dst, int64_t nrows, int64_t K, cudaStream_t stream);
void ggml_cuda_pxq_mmvq_launch(const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const block_q8_1 * acts, int64_t ne10_padded,
        const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream);
bool ggml_cuda_pxq_prefill_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_pxq_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_pxq_dense_prefill_supported(const ggml_tensor * dst, int cc);
void ggml_cuda_pxq_dense_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_pxq_mmvf_launch(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream);

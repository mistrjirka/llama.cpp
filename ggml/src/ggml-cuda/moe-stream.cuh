#pragma once

struct ggml_tensor;
struct ggml_backend_cuda_context;

// Experimental host-weight streaming for quantized MUL_MAT_ID prefill only.
bool ggml_cuda_moe_stream_supported(const ggml_tensor * op, int cc);
void ggml_cuda_moe_stream_release(ggml_backend_cuda_context & ctx);

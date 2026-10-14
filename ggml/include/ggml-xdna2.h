#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_XDNA2_NAME "XDNA2"

// AMD XDNA2 NPU backend (device type ACCEL).
//
// Weights of supported matrix multiplications (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K x F32) are repacked into the device
// layout when they are uploaded to a buffer of this backend's buffer type, provided an NPU kernel for them exists
// or can be built; everything else stays on the CPU backend.
//
// Environment:
//   GGML_XDNA2_DISABLE=1      do not register the device
//   GGML_XDNA2_CACHE_DIR=dir  NPU kernel cache (xclbin + instruction streams, built on demand; default ~/.cache/ggml-xdna2)
//   GGML_XDNA2_REFERENCE=1    TEST/DEBUG: run the host emulation of the NPU kernels instead of the NPU

GGML_BACKEND_API ggml_backend_t ggml_backend_xdna2_init(void);

GGML_BACKEND_API bool ggml_backend_is_xdna2(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_xdna2_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna2_reg(void);

#ifdef  __cplusplus
}
#endif

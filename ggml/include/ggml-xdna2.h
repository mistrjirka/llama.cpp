#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_XDNA2_NAME "XDNA2"

// AMD XDNA2 NPU backend (device type ACCEL).
//
// Weights of supported matrix multiplications (Q4_0 x F32) are repacked into the device layout when they are
// uploaded to a buffer of this backend's buffer type; everything else stays on the CPU backend.
//
// Environment:
//   GGML_XDNA2_DISABLE=1      do not register the device
//   GGML_XDNA2_CACHE_DIR=dir  directory with the prebuilt xclbin/instruction files (default ~/.cache/ggml-xdna2)

GGML_BACKEND_API ggml_backend_t ggml_backend_xdna2_init(void);

GGML_BACKEND_API bool ggml_backend_is_xdna2(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_xdna2_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna2_reg(void);

#ifdef  __cplusplus
}
#endif

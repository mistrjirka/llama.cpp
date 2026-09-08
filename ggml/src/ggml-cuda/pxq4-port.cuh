// PXQ4 compatibility path for v100-optimized.
// Wire format and frozen tables are compatible with PXA PXQ4 (GGML type id 252).
// PXA is MIT-licensed; format provenance: https://github.com/poisonxa16/pxa
//
// This first-stage path intentionally dequantizes a whole PXQ4 matrix to F16 and lets the
// existing cuBLAS matmul run. Native MMVQ/WMMA paths can replace it without changing the
// on-disk format or loader contract.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

static constexpr int PXQ4_PORT_QK = 32;
static constexpr int PXQ4_PORT_BM = 64;
static constexpr int PXQ4_PORT_HDR = 128;       // 64 fp16 row anchors
static constexpr int PXQ4_PORT_SLAB = 1088;     // 64 scale bytes + 64*16 code bytes
static constexpr int PXQ4_PORT_CODE_OFF = 64;

// Frozen PX16 and SUB16 tables used by the published PXQ4 format. Values are fp16-snapped
// format constants represented exactly as fp32 hex literals.
static __device__ __constant__ float pxq4_port_book[16] = {
    -0x1.f9c0000000000p-1f, -0x1.7880000000000p-1f, -0x1.1e00000000000p-1f, -0x1.adc0000000000p-2f,
    -0x1.3440000000000p-2f, -0x1.8e40000000000p-3f, -0x1.8740000000000p-4f,  0x0.0p+0f,
     0x1.5b00000000000p-4f,  0x1.5ec0000000000p-3f,  0x1.0c40000000000p-2f,  0x1.7140000000000p-2f,
     0x1.e280000000000p-2f,  0x1.3380000000000p-1f,  0x1.8800000000000p-1f,  0x1.0000000000000p+0f,
};
static __device__ __constant__ float pxq4_port_sub[16] = {
    0x1.b7c0000000000p-3f, 0x1.36c0000000000p-2f, 0x1.72c0000000000p-2f, 0x1.a2c0000000000p-2f,
    0x1.ccc0000000000p-2f, 0x1.f300000000000p-2f, 0x1.0bc0000000000p-1f, 0x1.1e00000000000p-1f,
    0x1.3040000000000p-1f, 0x1.4380000000000p-1f, 0x1.5800000000000p-1f, 0x1.6ec0000000000p-1f,
    0x1.8880000000000p-1f, 0x1.a640000000000p-1f, 0x1.cac0000000000p-1f, 0x1.f9c0000000000p-1f,
};

static __global__ void pxq4_port_dequant_f16_kernel(
        const uint8_t * __restrict__ src, half * __restrict__ dst,
        int kslabs, int64_t k, int64_t nrows) {
    const int64_t row = (int64_t) blockIdx.y * PXQ4_PORT_BM + threadIdx.x;
    const int kb = blockIdx.x;
    if (threadIdx.x >= PXQ4_PORT_BM || row >= nrows || kb >= kslabs) return;

    const int64_t panel_id = row / PXQ4_PORT_BM;
    const int r = (int)(row & (PXQ4_PORT_BM - 1));
    const size_t panel_stride = PXQ4_PORT_HDR + (size_t) kslabs * PXQ4_PORT_SLAB;
    const uint8_t * panel = src + (size_t) panel_id * panel_stride;
    const uint8_t * slab = panel + PXQ4_PORT_HDR + (size_t) kb * PXQ4_PORT_SLAB;
    const float anchor = __half2float(((const half *) panel)[r]);
    const uint8_t sc = slab[r];
    const float eff0 = anchor * pxq4_port_sub[sc & 0x0f];
    const float eff1 = anchor * pxq4_port_sub[sc >> 4];
    const uint8_t * q = slab + PXQ4_PORT_CODE_OFF + 16*r;
    half * out = dst + row*k + (int64_t)kb*PXQ4_PORT_QK;
#pragma unroll
    for (int b = 0; b < 16; ++b) {
        const float eff = b < 8 ? eff0 : eff1;
        const uint8_t c = q[b];
        out[2*b + 0] = __float2half_rn(eff * pxq4_port_book[c & 0x0f]);
        out[2*b + 1] = __float2half_rn(eff * pxq4_port_book[c >> 4]);
    }
}

static inline void pxq4_port_dequant_f16(
        const void * src, half * dst, int64_t nrows, int64_t k, cudaStream_t stream) {
    GGML_ASSERT(nrows % PXQ4_PORT_BM == 0);
    GGML_ASSERT(k % PXQ4_PORT_QK == 0);
    const int kslabs = (int)(k / PXQ4_PORT_QK);
    dim3 grid((unsigned)kslabs, (unsigned)(nrows / PXQ4_PORT_BM), 1);
    pxq4_port_dequant_f16_kernel<<<grid, PXQ4_PORT_BM, 0, stream>>>((const uint8_t *)src, dst, kslabs, k, nrows);
    CUDA_CHECK(cudaGetLastError());
}

// Copyright (c) 2026 PXA Network (wire formats / frozen numeric tables).
// SPDX-License-Identifier: MIT
// Shared PXQ slab policies for the v100-optimized port. See LICENSE-PXA.
#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "../../include/ggml-pxq1-tables.h"
#include "../../include/ggml-pxq2-tables.h"
#include "../../include/ggml-pxq3-tables.h"
#include "../../include/ggml-pxq6-tables.h"

static __device__ __constant__ float pxqa_book1[2]  = PXQ1_BOOK_INIT;
static __device__ __constant__ float pxqa_book2[4]  = PXQ2_BOOK_INIT;
static __device__ __constant__ float pxqa_book3[8]  = PXQ3_BOOK_INIT;
static __device__ __constant__ float pxqa_book4[16] = PXQ6_BOOK_INIT;
static __device__ __constant__ float pxqa_book6[32] = PXQ6_LM32_INIT;
static __device__ __constant__ float pxqa_sub16[16] = PXQ6_SUB16_INIT;
static __device__ __constant__ float pxqa_sub8 [16] = PXQ6_SUB8_INIT;

template <ggml_type TYPE_, int TYPE_SIZE_, int SLAB_, int CODE_BYTES_, int BOOK_N_, bool HQ_ = false>
struct pxqa_policy_base {
    static constexpr ggml_type TYPE = TYPE_;
    static constexpr int QK = 32;
    static constexpr int BM = 64;
    static constexpr int HDR = 128;
    static constexpr int TYPE_SIZE = TYPE_SIZE_;
    static constexpr int SLAB = SLAB_;
    static constexpr int CODE_BYTES = CODE_BYTES_;
    static constexpr int CODE_OFF = HQ_ ? 128 : 64;
    static constexpr int BOOK_N = BOOK_N_;
    static constexpr bool HQ = HQ_;

    __device__ static float subscale(const uint8_t * slab, int row, int j) {
        if constexpr (HQ) {
            const uint8_t sb = slab[2*row + (j >= 16)];
            const int nib = (j & 15) >= 8 ? (sb >> 4) : (sb & 0xf);
            return pxqa_sub8[nib];
        } else {
            const uint8_t sb = slab[row];
            return pxqa_sub16[j >= 16 ? (sb >> 4) : (sb & 0xf)];
        }
    }
};

struct pxqa_p1 : pxqa_policy_base<GGML_TYPE_PXQ1,   5,  PXQ1_SLAB_BYTES,  4,  2> {
    __device__ static float book(int c) { return pxqa_book1[c & 1]; }
    __device__ static int code(const uint8_t * q, int j) {
        const uint32_t w = *(const uint32_t *) q;
        return (w >> j) & 1;
    }
};
struct pxqa_p2 : pxqa_policy_base<GGML_TYPE_PXQ2,   9,  PXQ2_SLAB_BYTES,  8,  4> {
    __device__ static float book(int c) { return pxqa_book2[c & 3]; }
    __device__ static int code(const uint8_t * q, int j) {
        const uint32_t * w = (const uint32_t *) q;
        return (w[j >> 4] >> (2*(j & 15))) & 3;
    }
};
struct pxqa_p3 : pxqa_policy_base<GGML_TYPE_PXQ3,  13,  PXQ3_SLAB_BYTES, 12,  8> {
    __device__ static float book(int c) { return pxqa_book3[c & 7]; }
    __device__ static int code(const uint8_t * q, int j) {
        const uint32_t * w = (const uint32_t *) q;
        const int h = j >> 4;
        const int jj = j & 15;
        return ((w[h] >> (2*jj)) & 3) | (((w[2] >> j) & 1) << 2);
    }
};
struct pxqa_p4 : pxqa_policy_base<GGML_TYPE_PXQ4,  17, PXQ6_SLAB_BYTES, 16, 16> {
    __device__ static float book(int c) { return pxqa_book4[c & 15]; }
    __device__ static int code(const uint8_t * q, int j) {
        const uint8_t b = q[j >> 1];
        return (j & 1) ? (b >> 4) : (b & 0xf);
    }
};
struct pxqa_p4hq : pxqa_policy_base<GGML_TYPE_PXQ4HQ, 18, PXQ6HQ_SLAB_BYTES, 16, 16, true> {
    __device__ static float book(int c) { return pxqa_book4[c & 15]; }
    __device__ static int code(const uint8_t * q, int j) {
        const uint8_t b = q[j >> 1];
        return (j & 1) ? (b >> 4) : (b & 0xf);
    }
};
struct pxqa_p6 : pxqa_policy_base<GGML_TYPE_PXQ6, 21, PXQ6R_SLAB_BYTES, 20, 32> {
    __device__ static float book(int c) { return pxqa_book6[c & 31]; }
    __device__ static int code(const uint8_t * q, int j) {
        const uint8_t b = q[j >> 1];
        const uint32_t hi = *(const uint32_t *)(q + 16);
        return ((j & 1) ? (b >> 4) : (b & 0xf)) | (((hi >> j) & 1) << 4);
    }
};

template <class POL>
static __device__ __forceinline__ size_t pxqa_panel_stride(int kslabs) {
    return POL::HDR + (size_t)kslabs*POL::SLAB;
}

template <class POL>
static __device__ __forceinline__ float2 pxqa_dequant_pair(const uint32_t * q, int b, const float * book) {
    if constexpr (POL::TYPE == GGML_TYPE_PXQ1) {
        const uint32_t w = q[0];
        return make_float2(book[(w >> (2*b)) & 1], book[(w >> (2*b + 1)) & 1]);
    } else if constexpr (POL::TYPE == GGML_TYPE_PXQ2) {
        const uint32_t w = q[b >> 3];
        const int sh = 2*((2*b) & 15);
        return make_float2(book[(w >> sh) & 3], book[(w >> (sh + 2)) & 3]);
    } else if constexpr (POL::TYPE == GGML_TYPE_PXQ3) {
        const int h = b >> 3;
        const int j0 = (2*b) & 15;
        const uint32_t lo = q[h];
        const uint32_t hi = q[2] >> (16*h);
        const int c0 = (int)((lo >> (2*j0)) & 3) | (int)(((hi >> j0) & 1) << 2);
        const int c1 = (int)((lo >> (2*j0 + 2)) & 3) | (int)(((hi >> (j0 + 1)) & 1) << 2);
        return make_float2(book[c0], book[c1]);
    } else if constexpr (POL::TYPE == GGML_TYPE_PXQ4 || POL::TYPE == GGML_TYPE_PXQ4HQ) {
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;
        return make_float2(book[byte & 0xf], book[byte >> 4]);
    } else {
        static_assert(POL::TYPE == GGML_TYPE_PXQ6, "unexpected PXQ policy");
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;
        const uint32_t hi = q[4];
        const int c0 = (byte & 0xf) | (int)(((hi >> (2*b)) & 1) << 4);
        const int c1 = (byte >> 4) | (int)(((hi >> (2*b + 1)) & 1) << 4);
        return make_float2(book[c0], book[c1]);
    }
}

template <class POL, typename dst_t>
static __global__ void pxqa_dequant_matrix_kernel(
        const uint8_t * __restrict__ src, dst_t * __restrict__ dst,
        int kslabs, int64_t K, int64_t nrows) {
    // PXQ logical rows are panel-interleaved. Decode one 64x32 slab into shared memory,
    // then transpose the STORE mapping so each warp emits contiguous K values. The old
    // one-thread-per-row direct stores were extremely wasteful on V100 (32 distant sectors
    // for each store instruction), and dominated dense-model prefill.
    __shared__ dst_t tile[64][34]; // +2 avoids pathological half bank strides
    __shared__ float book[32];
    __shared__ float subs[16];

    const int tid = threadIdx.x;
    if (tid < POL::BOOK_N) book[tid] = POL::book(tid);
    if (tid < 16) subs[tid] = POL::HQ ? pxqa_sub8[tid] : pxqa_sub16[tid];
    __syncthreads();

    const int64_t slab_id = blockIdx.x;
    const int64_t panel_id = slab_id / kslabs;
    const int kb = (int)(slab_id % kslabs);
    const int row_in_panel = tid;
    const int64_t row = panel_id*POL::BM + row_in_panel;
    if (row >= nrows) return;
    const uint8_t * panel = src + (size_t)panel_id*pxqa_panel_stride<POL>(kslabs);
    const uint8_t * slab = panel + POL::HDR + (size_t)kb*POL::SLAB;
    const uint8_t * q = slab + POL::CODE_OFF + (size_t)row_in_panel*POL::CODE_BYTES;
    const float anchor = __half2float(((const half *)panel)[row_in_panel]);

    // Keep packed codes and the small set of effective row scales in registers. The original
    // oracle-style loop reloaded bitfields/scales for every one of the 32 outputs; that is
    // mathematically simple but instruction-heavy on Turing. Pair decode preserves the same
    // (anchor * subscale) * book arithmetic while cutting repeated global loads and integer work.
    uint32_t qr[5] = {0, 0, 0, 0, 0};
    if constexpr (POL::TYPE == GGML_TYPE_PXQ1) {
        qr[0] = *(const uint32_t *)q;
    } else if constexpr (POL::TYPE == GGML_TYPE_PXQ2) {
        *(uint2 *)qr = *(const uint2 *)q;
    } else if constexpr (POL::TYPE == GGML_TYPE_PXQ3) {
        const uint32_t * qs = (const uint32_t *)q;
        qr[0] = qs[0]; qr[1] = qs[1]; qr[2] = qs[2];
    } else if constexpr (POL::TYPE == GGML_TYPE_PXQ4 || POL::TYPE == GGML_TYPE_PXQ4HQ) {
        *(uint4 *)qr = *(const uint4 *)q;
    } else {
        static_assert(POL::TYPE == GGML_TYPE_PXQ6, "unexpected PXQ policy");
        const uint32_t * qs = (const uint32_t *)q;
        qr[0] = qs[0]; qr[1] = qs[1]; qr[2] = qs[2]; qr[3] = qs[3]; qr[4] = qs[4];
    }

    float eff[4];
    if constexpr (POL::HQ) {
        const uint8_t sb0 = slab[2*row_in_panel];
        const uint8_t sb1 = slab[2*row_in_panel + 1];
        eff[0] = anchor * subs[sb0 & 0xf];
        eff[1] = anchor * subs[sb0 >> 4];
        eff[2] = anchor * subs[sb1 & 0xf];
        eff[3] = anchor * subs[sb1 >> 4];
    } else {
        const uint8_t sb = slab[row_in_panel];
        eff[0] = anchor * subs[sb & 0xf];
        eff[1] = anchor * subs[sb >> 4];
    }

#pragma unroll
    for (int b = 0; b < 16; ++b) {
        const float e = eff[POL::HQ ? (b >> 2) : (b >> 3)];
        const float2 v = pxqa_dequant_pair<POL>(qr, b, book);
        tile[row_in_panel][2*b]     = ggml_cuda_cast<dst_t>(e * v.x);
        tile[row_in_panel][2*b + 1] = ggml_cuda_cast<dst_t>(e * v.y);
    }
    __syncthreads();

    // Two warps cover all 64 rows. lane == K offset makes every store instruction contiguous.
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int r = warp; r < POL::BM; r += 2) {
        dst[((int64_t)panel_id*POL::BM + r)*K + (int64_t)kb*POL::QK + lane] = tile[r][lane];
    }
}

template <class POL, typename dst_t>
static void pxqa_dequant_matrix(const void * src, dst_t * dst, int64_t nrows, int64_t K, cudaStream_t stream) {
    GGML_ASSERT(nrows > 0 && nrows % POL::BM == 0 && K > 0 && K % POL::QK == 0);
    const int kslabs = (int)(K/POL::QK);
    const int64_t nslabs = (nrows/POL::BM)*(int64_t)kslabs;
    pxqa_dequant_matrix_kernel<POL><<<(unsigned)nslabs, POL::BM, 0, stream>>>((const uint8_t *)src, dst, kslabs, K, nrows);
}

template <typename dst_t>
static void pxqa_dequant_dispatch(ggml_type type, const void * src, dst_t * dst, int64_t nrows, int64_t K, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_PXQ1:   pxqa_dequant_matrix<pxqa_p1  >(src,dst,nrows,K,stream); break;
        case GGML_TYPE_PXQ2:   pxqa_dequant_matrix<pxqa_p2  >(src,dst,nrows,K,stream); break;
        case GGML_TYPE_PXQ3:   pxqa_dequant_matrix<pxqa_p3  >(src,dst,nrows,K,stream); break;
        case GGML_TYPE_PXQ4:   pxqa_dequant_matrix<pxqa_p4  >(src,dst,nrows,K,stream); break;
        case GGML_TYPE_PXQ4HQ: pxqa_dequant_matrix<pxqa_p4hq>(src,dst,nrows,K,stream); break;
        case GGML_TYPE_PXQ6:   pxqa_dequant_matrix<pxqa_p6  >(src,dst,nrows,K,stream); break;
        default: GGML_ABORT("not a PXQ slab type");
    }
    CUDA_CHECK(cudaGetLastError());
}

// Fast Q8_1 / signed-int8 book images. These follow PXA's documented contract:
// q_i = rint(book_i * 127 / absmax(book)), with absmax/127 folded into the effective scale.
static __device__ __constant__ int8_t pxqa_s8_p1[2] = {-127, 127};
static __device__ __constant__ int8_t pxqa_s8_p2[4] = {-127, -34, 34, 126};
static __device__ __constant__ int8_t pxqa_s8_p3[8] = {-127, -77, -42, -13, 13, 41, 76, 127};
static __device__ __constant__ int8_t pxqa_s8_p4[16] = {-125,-93,-71,-53,-38,-25,-12,0,11,22,33,46,60,76,97,127};
static __device__ __constant__ int8_t pxqa_s8_p6[32] = {
    -127,-120,-108,-96,-86,-77,-68,-60,-52,-45,-38,-31,-25,-18,-12,-6,
    0,7,13,20,27,34,41,49,57,65,75,85,96,107,120,127
};

template <class POL> struct pxqa_fast_traits;
template <> struct pxqa_fast_traits<pxqa_p1> { static constexpr float BFOLD = 1.0f/127.0f; __device__ static int q(int c){return pxqa_s8_p1[c];} };
template <> struct pxqa_fast_traits<pxqa_p2> { static constexpr float BFOLD = 0.70556640625f/127.0f; __device__ static int q(int c){return pxqa_s8_p2[c];} };
template <> struct pxqa_fast_traits<pxqa_p3> { static constexpr float BFOLD = 0.90673828125f/127.0f; __device__ static int q(int c){return pxqa_s8_p3[c];} };
template <> struct pxqa_fast_traits<pxqa_p4> { static constexpr float BFOLD = 1.0f/127.0f; __device__ static int q(int c){return pxqa_s8_p4[c];} };
template <> struct pxqa_fast_traits<pxqa_p4hq> { static constexpr float BFOLD = 1.0f/127.0f; __device__ static int q(int c){return pxqa_s8_p4[c];} };
template <> struct pxqa_fast_traits<pxqa_p6> { static constexpr float BFOLD = 1.0f/127.0f; __device__ static int q(int c){return pxqa_s8_p6[c];} };

template <class POL>
static __device__ __forceinline__ int pxqa_pack4_s8(const uint8_t * q, int j0) {
    uint32_t v = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int c = POL::code(q, j0 + b);
        v |= (uint32_t)(uint8_t)pxqa_fast_traits<POL>::q(c) << (8*b);
    }
    return (int)v;
}

template <class POL>
static __device__ __forceinline__ float pxqa_dot32_q8_1(
        const uint8_t * slab, int row, const block_q8_1 * a, float anchor) {
    const uint8_t * q = slab + POL::CODE_OFF + (size_t)row*POL::CODE_BYTES;
    const int * aq = (const int *)a->qs;
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const int qw = pxqa_pack4_s8<POL>(q, 4*w);
        const int d = ggml_cuda_dp4a(qw, aq[w], 0);
        sum += POL::subscale(slab, row, 4*w) * (float)d;
    }
    return anchor * pxqa_fast_traits<POL>::BFOLD * __low2float(a->ds) * sum;
}

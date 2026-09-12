#include "convert.cuh"
#include "dequantize.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
            y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d * (q[l] & 0xF) + dm);
        y[l+16] = ggml_cuda_cast<dst_t>(d * (q[l] >>  4) + dm);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d.x * (q[l] & 0xF) + d.y);
        y[l+16] = ggml_cuda_cast<dst_t>(d.x * (q[l] >>  4) + d.y);
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q2_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q3_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q4_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q5_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q6_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_m(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_nl(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_mxfp4(vx, i, yy + i*QK_K, threadIdx.x);
}

// Volta MXFP4 path: remap four lanes per 32-value block so both the packed input
// bytes and the two 64-bit FP16 output stores are contiguous within each group.
// FP32 arithmetic and final FP16 rounding are unchanged from the stock helper.
static __global__ void dequantize_block_mxfp4_packed(const void * __restrict__ vx,
        half * __restrict__ yy, const int nb) {
    const int ibs = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (ibs >= nb) {
        return;
    }

    const int tid = threadIdx.x % WARP_SIZE;
    const int ib = tid / 4;
    const int il = tid % 4;
    const block_mxfp4 * x = (const block_mxfp4 *) vx + int64_t(ibs) * (QK_K / QK_MXFP4);
    const block_mxfp4 & xb = x[ib];
    half * y = yy + int64_t(ibs) * QK_K + 32 * ib + 4 * il;
    const uint8_t * q4 = xb.qs + 4 * il;
    const float d = ggml_cuda_e8m0_to_fp32(xb.e);

    uint32_t lo0 = 0, lo1 = 0, hi0 = 0, hi1 = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t q = q4[j];
        const uint32_t lo = __half_as_ushort(ggml_cuda_cast<half>(d * kvalues_mxfp4[q & 0xf] * 0.5f));
        const uint32_t hi = __half_as_ushort(ggml_cuda_cast<half>(d * kvalues_mxfp4[q >> 4] * 0.5f));
        if (j < 2) {
            lo0 |= lo << (16 * j);
            hi0 |= hi << (16 * j);
        } else {
            lo1 |= lo << (16 * (j - 2));
            hi1 |= hi << (16 * (j - 2));
        }
    }

    *(uint2 *) (y +  0) = make_uint2(lo0, lo1);
    *(uint2 *) (y + 16) = make_uint2(hi0, hi1);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

// Below this size launch overhead dominates and the grouped/packed paths can be
// marginally slower. All measured model weight matrices are comfortably larger.
static constexpr int VOLTA_KQUANT_MIN_BLOCKS = 512;

// Vectorized loads below rely only on alignment guaranteed by the GGUF block
// layouts themselves, so consecutive blocks remain safe as well.
static_assert(sizeof(block_q3_K) % alignof(uint16_t) == 0 &&
              offsetof(block_q3_K, hmask) % alignof(uint16_t) == 0 &&
              offsetof(block_q3_K, qs) % alignof(uint16_t) == 0);
static_assert(sizeof(block_q4_K) % alignof(uint32_t) == 0 &&
              offsetof(block_q4_K, qs) % alignof(uint32_t) == 0);
static_assert(sizeof(block_q5_K) % alignof(uint16_t) == 0 &&
              offsetof(block_q5_K, qh) % alignof(uint16_t) == 0 &&
              offsetof(block_q5_K, qs) % alignof(uint16_t) == 0);

// Q2_K: group multiple independent stock 64-thread quant blocks into one CTA
// while preserving the exact dequantization helper.
static __global__ void dequantize_block_q2_K_grouped(const void * __restrict__ vx,
        half * __restrict__ yy, const int nb) {
    const int ib = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    if (ib >= nb) {
        return;
    }

    const int tid = threadIdx.x % 64;
    dequantize_q2_K(vx, ib, yy + int64_t(ib) * QK_K, tid);
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    if constexpr (std::is_same_v<dst_t, half>) {
        static const int grouped = [] {
            const char * v = std::getenv("GGML_CUDA_VOLTA_Q2K_GROUPED");
            return v ? std::atoi(v) : 3;
        }();
        if (nb >= VOLTA_KQUANT_MIN_BLOCKS && (grouped == 2 || grouped == 3) &&
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_VOLTA) {
            const int threads = grouped == 2 ? 128 : 256;
            const int blocks_per_cta = threads / 64;
            dequantize_block_q2_K_grouped<<<(nb + blocks_per_cta - 1) / blocks_per_cta, threads, 0, stream>>>(vx, y, nb);
            return;
        }
    }
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

// Volta Q3_K path: keep the stock thread/output mapping, but load each
// thread's four q/hmask bytes as aligned 16-bit pairs and emit its four FP16
// results with one aligned 64-bit store. This targets the unusually poor global
// load/store sector utilization of the stock Q3_K conversion on Volta.
static __global__ void dequantize_block_q3_K_packed(const void * __restrict__ vx,
        half * __restrict__ yy, const int nb) {
    const int ib = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    if (ib >= nb) {
        return;
    }

    const int tid = threadIdx.x % 64;
    const int r = tid / 4;
    const int t = r / 2;
    const int is0 = r % 2;
    const int l0 = 16 * is0 + 4 * (tid % 4);
    const int n = t / 4;
    const int j = t - 4 * n;
    const int is = 8 * n + 2 * j + is0;
    const int shift = 2 * j;
    const uint8_t m = 1u << (4 * n + j);

    const block_q3_K * x = (const block_q3_K *) vx + ib;
    const int8_t us = is <  4 ? (x->scales[is-0] & 0xF) | (((x->scales[is+8] >> 0) & 3) << 4) :
                      is <  8 ? (x->scales[is-0] & 0xF) | (((x->scales[is+4] >> 2) & 3) << 4) :
                      is < 12 ? (x->scales[is-8] >>  4) | (((x->scales[is+0] >> 4) & 3) << 4) :
                                (x->scales[is-8] >>  4) | (((x->scales[is-4] >> 6) & 3) << 4);
    const float d_all = x->d;
    const float dl = d_all * (us - 32);

    // block_q3_K is 110 bytes, so consecutive blocks are only 2-byte aligned.
    // Use aligned 16-bit pairs rather than relying on unaligned uint32_t loads.
    const uint16_t q01 = *(const uint16_t *) (x->qs + 32 * n + l0 + 0);
    const uint16_t q23 = *(const uint16_t *) (x->qs + 32 * n + l0 + 2);
    const uint16_t hm01 = *(const uint16_t *) (x->hmask + l0 + 0);
    const uint16_t hm23 = *(const uint16_t *) (x->hmask + l0 + 2);
    const uint32_t q4 = uint32_t(q01) | (uint32_t(q23) << 16);
    const uint32_t hm4 = uint32_t(hm01) | (uint32_t(hm23) << 16);

    uint32_t out_lo = 0;
    uint32_t out_hi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint8_t q = (q4 >> (8 * p)) & 0xff;
        const uint8_t hm = (hm4 >> (8 * p)) & 0xff;
        const int qv = int((q >> shift) & 3) - ((hm & m) ? 0 : 4);
        const uint32_t h = __half_as_ushort(__float2half_rn(dl * qv));
        if (p < 2) {
            out_lo |= h << (16 * p);
        } else {
            out_hi |= h << (16 * (p - 2));
        }
    }

    half * y = yy + int64_t(ib) * QK_K + 128 * n + 32 * j + l0;
    const uint64_t packed_out = uint64_t(out_lo) | (uint64_t(out_hi) << 32);
    *(uint64_t *) y = packed_out;
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    if constexpr (std::is_same_v<dst_t, half>) {
        static const int packed = [] {
            const char * v = std::getenv("GGML_CUDA_VOLTA_Q3K_PACKED");
            return v ? std::atoi(v) : 3;
        }();
        if (nb >= VOLTA_KQUANT_MIN_BLOCKS && packed >= 1 && packed <= 3 &&
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_VOLTA) {
            const int threads = packed == 1 ? 64 : packed == 2 ? 128 : 256;
            const int blocks_per_cta = threads / 64;
            dequantize_block_q3_K_packed<<<(nb + blocks_per_cta - 1) / blocks_per_cta, threads, 0, stream>>>(vx, y, nb);
            return;
        }
    }
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

// Volta Q4_K packed path retains FP32 dequantization arithmetic and FP16 rounding.
static __global__ void dequantize_block_q4_K_packed(const void * __restrict__ vx,
        half * __restrict__ yy, const int nb) {
    const int ib = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (ib >= nb) {
        return;
    }
    const int tid = threadIdx.x % WARP_SIZE;
    const int il = tid / 8;
    const int ir = tid % 8;
    const block_q4_K * x = (const block_q4_K *) vx + ib;
    half * y = yy + int64_t(ib) * QK_K + 64 * il + 4 * ir;
    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);
    uint8_t sc, m;
    get_scale_min_k4(2 * il, x->scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, x->scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;
    const uint32_t q = *(const uint32_t *) (x->qs + 32 * il + 4 * ir);
    uint32_t lo[2] = {0, 0};
    uint32_t hi[2] = {0, 0};
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t b = (q >> (8 * l)) & 255;
        lo[l / 2] |= uint32_t(__half_as_ushort(__float2half_rn(d1 * (b & 15) - m1))) << (16 * (l % 2));
        hi[l / 2] |= uint32_t(__half_as_ushort(__float2half_rn(d2 * (b >> 4) - m2))) << (16 * (l % 2));
    }
    *(uint2 *) (y +  0) = make_uint2(lo[0], lo[1]);
    *(uint2 *) (y + 32) = make_uint2(hi[0], hi[1]);
}

// Volta Q5_K path: pair adjacent FP16 outputs into aligned 32-bit stores.
// A Q5_K block uses 64 threads; multiple independent blocks share one CTA.
static __global__ void dequantize_block_q5_K_packed(const void * __restrict__ vx,
        half * __restrict__ yy, const int nb) {
    const int ib = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    if (ib >= nb) {
        return;
    }

    const int tid = threadIdx.x % 64;
    const int il = tid / 16;
    const int ir = tid % 16;
    const int is = 2 * il;

    const block_q5_K * x = (const block_q5_K *) vx + ib;
    half * y = yy + int64_t(ib) * QK_K + 64 * il + 2 * ir;

    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x->scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x->scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    const uint16_t ql2 = *(const uint16_t *) (x->qs + 32 * il + 2 * ir);
    const uint16_t qh2 = *(const uint16_t *) (x->qh + 2 * ir);
    const uint8_t ql0 = ql2 & 0xff;
    const uint8_t ql1 = ql2 >> 8;
    const uint8_t qh0 = qh2 & 0xff;
    const uint8_t qh1 = qh2 >> 8;
    const uint8_t hm0 = 1u << (2 * il);
    const uint8_t hm1 = hm0 << 1;

    const uint32_t lo =
        uint32_t(__half_as_ushort(__float2half_rn(d1 * ((ql0 & 0xF) + (qh0 & hm0 ? 16 : 0)) - m1))) |
        (uint32_t(__half_as_ushort(__float2half_rn(d1 * ((ql1 & 0xF) + (qh1 & hm0 ? 16 : 0)) - m1))) << 16);
    const uint32_t hi =
        uint32_t(__half_as_ushort(__float2half_rn(d2 * ((ql0 >> 4) + (qh0 & hm1 ? 16 : 0)) - m2))) |
        (uint32_t(__half_as_ushort(__float2half_rn(d2 * ((ql1 >> 4) + (qh1 & hm1 ? 16 : 0)) - m2))) << 16);

    *(uint32_t *) (y +  0) = lo;
    *(uint32_t *) (y + 32) = hi;
}

// Volta Q6_K path: keep the stock dequantization arithmetic and output
// mapping, but group multiple independent 64-thread quant blocks into one CTA.
static __global__ void dequantize_block_q6_K_grouped(const void * __restrict__ vx,
        half * __restrict__ yy, const int nb) {
    const int ib = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    if (ib >= nb) {
        return;
    }

    const int tid = threadIdx.x % 64;
    dequantize_q6_K(vx, ib, yy + int64_t(ib) * QK_K, tid);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    if constexpr (std::is_same_v<dst_t, half>) {
        static const int packed = [] {
            const char * v = std::getenv("GGML_CUDA_VOLTA_Q4K_PACKED");
            return v ? std::atoi(v) : 2;
        }();
        if (nb >= VOLTA_KQUANT_MIN_BLOCKS && packed >= 1 && packed <= 3 &&
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_VOLTA) {
            const int threads = packed == 1 ? 32 : packed == 2 ? 128 : 256;
            dequantize_block_q4_K_packed<<<(nb + threads / 32 - 1) / (threads / 32), threads, 0, stream>>>(vx, y, nb);
            return;
        }
    }
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    if constexpr (std::is_same_v<dst_t, half>) {
        static const int packed = [] {
            const char * v = std::getenv("GGML_CUDA_VOLTA_Q5K_PACKED");
            return v ? std::atoi(v) : 3;
        }();
        if (nb >= VOLTA_KQUANT_MIN_BLOCKS && packed >= 1 && packed <= 3 &&
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_VOLTA) {
            const int threads = packed == 1 ? 64 : packed == 2 ? 128 : 256;
            const int blocks_per_cta = threads / 64;
            dequantize_block_q5_K_packed<<<(nb + blocks_per_cta - 1) / blocks_per_cta, threads, 0, stream>>>(vx, y, nb);
            return;
        }
    }
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    if constexpr (std::is_same_v<dst_t, half>) {
        static const int grouped = [] {
            const char * v = std::getenv("GGML_CUDA_VOLTA_Q6K_GROUPED");
            return v ? std::atoi(v) : 3;
        }();
        if (nb >= VOLTA_KQUANT_MIN_BLOCKS && (grouped == 2 || grouped == 3) &&
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_VOLTA) {
            const int threads = grouped == 2 ? 128 : 256;
            const int blocks_per_cta = threads / 64;
            dequantize_block_q6_K_grouped<<<(nb + blocks_per_cta - 1) / blocks_per_cta, threads, 0, stream>>>(vx, y, nb);
            return;
        }
    }
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    if constexpr (std::is_same_v<dst_t, half>) {
        static const int packed = [] {
            const char * v = std::getenv("GGML_CUDA_VOLTA_MXFP4_PACKED");
            return v ? std::atoi(v) : 3;
        }();
        if (nb >= 128 && packed >= 1 && packed <= 3 &&
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_VOLTA) {
            const int threads = packed == 1 ? 32 : packed == 2 ? 128 : 256;
            const int blocks_per_cta = threads / WARP_SIZE;
            dequantize_block_mxfp4_packed<<<(nb + blocks_per_cta - 1) / blocks_per_cta, threads, 0, stream>>>(vx, y, nb);
            return;
        }
    }
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}

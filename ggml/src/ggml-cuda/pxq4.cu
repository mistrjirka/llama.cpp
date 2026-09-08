// Copyright (c) 2026 PXA Network. Portions adapted from the MIT-licensed PXA project.
// See LICENSE-PXA and benches/pxq4-v100/README.md for provenance and numeric contract.
#include "pxq4.cuh"
#include "unary.cuh"
#include <cstdlib>
#include <type_traits>

// PXQ4 native Volta decode path. PXQ4 has panel-interleaved storage and two bytes of
// per-row metadata, so its physical byte strides cannot be represented by the stock
// MMVQ block-stride interface. Keep the generic MMVQ machinery untouched and consume
// real tensor byte strides here.
static __device__ __constant__ __align__(16) int8_t pxq4_mmvq_book_s8[16] = {
    -125, -93, -71, -53, -38, -25, -12, 0, 11, 22, 33, 46, 60, 76, 97, 127
};
static __device__ __constant__ float pxq4_mmvq_sub16[16] = {
    0x1.b7c0000000000p-3f, 0x1.36c0000000000p-2f, 0x1.72c0000000000p-2f, 0x1.a2c0000000000p-2f,
    0x1.ccc0000000000p-2f, 0x1.f300000000000p-2f, 0x1.0bc0000000000p-1f, 0x1.1e00000000000p-1f,
    0x1.3040000000000p-1f, 0x1.4380000000000p-1f, 0x1.5800000000000p-1f, 0x1.6ec0000000000p-1f,
    0x1.8880000000000p-1f, 0x1.a640000000000p-1f, 0x1.cac0000000000p-1f, 0x1.f9c0000000000p-1f
};

static __device__ __forceinline__ int2 pxq4_mmvq_table16_seq(uint32_t q4) {
    const uint32_t * values32 = (const uint32_t *) pxq4_mmvq_book_s8;
    const uint32_t mask = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t v1 = __byte_perm(values32[0], values32[1], q4);
    uint32_t v2 = __byte_perm(values32[2], values32[3], q4);
    const uint32_t lo = __byte_perm(v1, v2, mask);
    v1 = __byte_perm(values32[0], values32[1], q4 >> 16);
    v2 = __byte_perm(values32[2], values32[3], q4 >> 16);
    const uint32_t hi = __byte_perm(v1, v2, mask >> 16);
    return make_int2((int)lo, (int)hi);
}

static __device__ __forceinline__ float pxq4_mmvq_slab_dot(
        const uint8_t * slab, int row, const block_q8_1 * a, float anchor) {
    const uint8_t sb = slab[row];
    const uint32_t * q = (const uint32_t *)(slab + 64 + 16*row);
    const int * aq = (const int *) a->qs;
    float sum = 0.0f;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int2 w = pxq4_mmvq_table16_seq(q[m]);
        int v = ggml_cuda_dp4a(w.x, aq[2*m + 0], 0);
        v     = ggml_cuda_dp4a(w.y, aq[2*m + 1], v);
        const int sub = (sb >> (4*(m >> 1))) & 0xf;
        sum += pxq4_mmvq_sub16[sub] * (float) v;
    }
    return anchor * (1.0f/127.0f) * __low2float(a->ds) * sum;
}

#include "pxq4-mmvq.cuh"

template <bool has_ids, bool has_gate, int RPB>
static __global__ void pxq4_mmvq_kernel(
        const uint8_t * weights, const uint8_t * gate_weights, const block_q8_1 * acts, const int32_t * ids, float * dst,
        uint32_t K, uint32_t nrows, uint32_t ncols_dst, uint32_t nchannels_y, uint32_t nchannels_dst,
        size_t w_nb2, size_t w_nb3,
        uint32_t a_s11, uint32_t a_s12, uint32_t a_s13,
        uint32_t d_s1, uint32_t d_s2, uint32_t d_s3, uint32_t ids_stride,
        uint32_t nsamples_x, uint32_t nsamples_dst, uint32_t nchannels_x, int glu_op, float glu_limit) {
    const int row0 = RPB * blockIdx.x;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst = blockIdx.z;
    const uint32_t token = threadIdx.y;
    if (token >= ncols_dst || row0 >= (int)nrows) return;

    uint32_t channel_x, channel_y, sample_x, sample_y;
    if constexpr (has_ids) {
        channel_x = ids[channel_dst + token*ids_stride];
        channel_y = channel_dst % nchannels_y;
        sample_x = 0;
        sample_y = 0;
    } else {
        const uint32_t ratio_c = nchannels_dst / max(1u, nchannels_x);
        channel_x = channel_dst / max(1u, ratio_c);
        channel_y = channel_dst;
        const uint32_t ratio_s = nsamples_dst / max(1u, nsamples_x);
        sample_x = sample_dst / max(1u, ratio_s);
        sample_y = sample_dst;
    }

    const uint8_t * wbase = weights + (size_t)sample_x*w_nb3 + (size_t)channel_x*w_nb2;
    const uint8_t * gbase = has_gate ? gate_weights + (size_t)sample_x*w_nb3 + (size_t)channel_x*w_nb2 : nullptr;
    const block_q8_1 * abase = acts + (size_t)sample_y*a_s13 + (has_ids
        ? (size_t)channel_y*a_s11 + (size_t)token*a_s12
        : (size_t)channel_y*a_s12 + (size_t)token*a_s11);
    const int kslabs = K / 32;
    const size_t panel_stride = 128u + (size_t)kslabs*1088u;

    float acc[RPB] = {};
    float gacc[RPB] = {};
#pragma unroll
    for (int ri = 0; ri < RPB; ++ri) {
        const int row = row0 + ri;
        if (row >= (int)nrows) continue;
        const uint8_t * panel = wbase + (size_t)(row >> 6)*panel_stride;
        const uint8_t * gpanel = has_gate ? gbase + (size_t)(row >> 6)*panel_stride : nullptr;
        const int r = row & 63;
        const float anchor = __half2float(((const half *)panel)[r]);
        const float ganchor = has_gate ? __half2float(((const half *)gpanel)[r]) : 0.0f;
        for (int kb = threadIdx.x; kb < kslabs; kb += 32) {
            const uint8_t * slab = panel + 128 + (size_t)kb*1088;
            acc[ri] += pxq4_mmvq_slab_dot(slab, r, abase + kb, anchor);
            if constexpr (has_gate) {
                const uint8_t * gslab = gpanel + 128 + (size_t)kb*1088;
                gacc[ri] += pxq4_mmvq_slab_dot(gslab, r, abase + kb, ganchor);
            }
        }
        acc[ri] = warp_reduce_sum<32>(acc[ri]);
        if constexpr (has_gate) gacc[ri] = warp_reduce_sum<32>(gacc[ri]);
    }

    if (threadIdx.x < RPB) {
        const int row = row0 + threadIdx.x;
        if (row < (int)nrows) {
            float result = acc[threadIdx.x];
            if constexpr (has_gate) {
                const float gate = gacc[threadIdx.x];
                switch ((ggml_glu_op)glu_op) {
                    case GGML_GLU_OP_SWIGLU: result *= ggml_cuda_op_silu_single(gate); break;
                    case GGML_GLU_OP_GEGLU: result *= ggml_cuda_op_gelu_single(gate); break;
                    case GGML_GLU_OP_SWIGLU_OAI: result = ggml_cuda_op_swiglu_oai_single(gate, result); break;
                    case GGML_GLU_OP_SWIGLU_CLAMP: result = ggml_cuda_op_swiglu_clamp_single(gate, result, glu_limit); break;
                    default: result *= gate; break;
                }
            }
            if constexpr (has_ids) {
                dst[(size_t)channel_dst*d_s1 + (size_t)token*d_s2 + row] = result;
            } else {
                dst[(size_t)sample_dst*d_s3 + (size_t)channel_dst*d_s2 + (size_t)token*d_s1 + row] = result;
            }
        }
    }
}

void ggml_cuda_pxq4_mmvq_launch(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const block_q8_1 * acts, int64_t ne10_padded, const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream) {
    GGML_ASSERT(src0->ne[0] % 32 == 0 && src0->ne[1] % 64 == 0);
    GGML_ASSERT(src0->nb[1] == ggml_row_size(GGML_TYPE_PXQ4, src0->ne[0]));
    const uint32_t K = (uint32_t)src0->ne[0];
    const uint32_t nrows = (uint32_t)src0->ne[1];
    const uint32_t a_s11 = (uint32_t)(ne10_padded / QK8_1);
    const uint32_t a_s12 = (uint32_t)(src1->ne[1] * a_s11);
    const uint32_t a_s13 = (uint32_t)(src1->ne[2] * a_s12);
    const uint32_t d_s1 = (uint32_t)(dst->nb[1] / sizeof(float));
    const uint32_t d_s2 = (uint32_t)(dst->nb[2] / sizeof(float));
    const uint32_t d_s3 = (uint32_t)(dst->nb[3] / sizeof(float));
    const uint32_t ncols_dst = ids ? (uint32_t)dst->ne[2] : (uint32_t)dst->ne[1];
    const uint32_t nchannels_y = ids ? (uint32_t)src1->ne[1] : (uint32_t)src1->ne[2];
    const uint32_t nchannels_dst = ids ? (uint32_t)dst->ne[1] : (uint32_t)dst->ne[2];
    const uint32_t nsamples_dst = (uint32_t)dst->ne[3];
    const uint32_t ids_stride = ids ? (uint32_t)(ids->nb[1] / sizeof(int32_t)) : 0;
    int rpb = 2;
    if (const char * e = getenv("GGML_CUDA_PXQ4_RPB")) rpb = atoi(e);
    if (rpb != 2 && rpb != 4 && rpb != 8) rpb = 2;
    const bool fuse_gate = fusion && fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale;
    const uint8_t * gate_w = fuse_gate ? (const uint8_t *)fusion->gate->data : nullptr;
    const int glu_op = fusion ? (int)fusion->glu_op : (int)GGML_GLU_OP_SWIGLU;
    const float glu_limit = fusion ? fusion->glu_limit : 0.0f;
    if (fusion) GGML_ASSERT(fuse_gate && "PXQ4 native fusion currently supports gate+GLU without bias/scale");
    // Retain the original kernel only as a diagnostic A/B control.
    const char * coalesced = std::getenv("GGML_CUDA_PXQ4_COALESCED");
    if (!coalesced || std::atoi(coalesced) != 0) {
        pxq4_mmvq_args a{
            (const uint8_t *)src0->data, gate_w, acts, ids ? (const int32_t *)ids->data : nullptr, (float *)dst->data,
            K, nrows, ncols_dst, nchannels_y, nchannels_dst, (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], nsamples_dst,
            src0->nb[2],src0->nb[3],a_s11,a_s12,a_s13,d_s1,d_s2,d_s3,ids_stride,glu_op,glu_limit};
        auto run = [&](auto rows_tag, auto vdr_tag) {
            constexpr int R = decltype(rows_tag)::value;
            constexpr int V = decltype(vdr_tag)::value;
            const dim3 gr((nrows+4*R-1)/(4*R),nchannels_dst,ncols_dst*nsamples_dst), bl(32,4);
            if (ids) {
                if (fuse_gate) pxq4_mmvq_coalesced<true,true,R,V><<<gr,bl,0,stream>>>(a);
                else pxq4_mmvq_coalesced<true,false,R,V><<<gr,bl,0,stream>>>(a);
            } else {
                if (fuse_gate) pxq4_mmvq_coalesced<false,true,R,V><<<gr,bl,0,stream>>>(a);
                else pxq4_mmvq_coalesced<false,false,R,V><<<gr,bl,0,stream>>>(a);
            }
        };
        int wr = fuse_gate && ncols_dst == 1 && K >= 1024 ? 2 : 4;
        if (const char * e = std::getenv("GGML_CUDA_PXQ4_WARP_ROWS")) wr = std::atoi(e);
        int vdr = 4;
        if (const char * e = std::getenv("GGML_CUDA_PXQ4_VDR")) vdr = std::atoi(e);
        if (wr == 2) {
            if (vdr == 2) run(std::integral_constant<int,2>{},std::integral_constant<int,2>{});
            else run(std::integral_constant<int,2>{},std::integral_constant<int,4>{});
        } else {
            if (vdr == 2) run(std::integral_constant<int,4>{},std::integral_constant<int,2>{});
            else run(std::integral_constant<int,4>{},std::integral_constant<int,4>{});
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    auto launch = [&](auto rpb_tag) {
        constexpr int R = decltype(rpb_tag)::value;
        const dim3 grid((nrows + R - 1)/R, nchannels_dst, nsamples_dst);
        const dim3 block(32, ncols_dst, 1);
        if (ids) {
            if (fuse_gate) pxq4_mmvq_kernel<true,true,R><<<grid, block, 0, stream>>>(
                (const uint8_t *)src0->data, gate_w, acts, (const int32_t *)ids->data, (float *)dst->data,
                K, nrows, ncols_dst, nchannels_y, nchannels_dst, src0->nb[2], src0->nb[3],
                a_s11, a_s12, a_s13, d_s1, d_s2, d_s3, ids_stride, (uint32_t)src0->ne[3], nsamples_dst, (uint32_t)src0->ne[2], glu_op, glu_limit);
            else pxq4_mmvq_kernel<true,false,R><<<grid, block, 0, stream>>>(
                (const uint8_t *)src0->data, nullptr, acts, (const int32_t *)ids->data, (float *)dst->data,
                K, nrows, ncols_dst, nchannels_y, nchannels_dst, src0->nb[2], src0->nb[3],
                a_s11, a_s12, a_s13, d_s1, d_s2, d_s3, ids_stride, (uint32_t)src0->ne[3], nsamples_dst, (uint32_t)src0->ne[2], glu_op, glu_limit);
        } else {
            if (fuse_gate) pxq4_mmvq_kernel<false,true,R><<<grid, block, 0, stream>>>(
                (const uint8_t *)src0->data, gate_w, acts, nullptr, (float *)dst->data,
                K, nrows, ncols_dst, nchannels_y, nchannels_dst, src0->nb[2], src0->nb[3],
                a_s11, a_s12, a_s13, d_s1, d_s2, d_s3, 0, (uint32_t)src0->ne[3], nsamples_dst, (uint32_t)src0->ne[2], glu_op, glu_limit);
            else pxq4_mmvq_kernel<false,false,R><<<grid, block, 0, stream>>>(
                (const uint8_t *)src0->data, nullptr, acts, nullptr, (float *)dst->data,
                K, nrows, ncols_dst, nchannels_y, nchannels_dst, src0->nb[2], src0->nb[3],
                a_s11, a_s12, a_s13, d_s1, d_s2, d_s3, 0, (uint32_t)src0->ne[3], nsamples_dst, (uint32_t)src0->ne[2], glu_op, glu_limit);
        }
    };
    if (rpb == 2) launch(std::integral_constant<int,2>{});
    else if (rpb == 8) launch(std::integral_constant<int,8>{});
    else launch(std::integral_constant<int,4>{});
    CUDA_CHECK(cudaGetLastError());

}


#include "pxq4-port.cuh"
#include "pxq4-wmma.cuh"

bool ggml_cuda_pxq4_prefill_supported(const ggml_tensor * dst, int cc) {
    const ggml_tensor * w=dst->src[0], *x=dst->src[1], *ids=dst->src[2];
    const char * e=std::getenv("GGML_CUDA_PXQ4_PREFILL");
    return (!e || std::atoi(e)!=0) && cc==GGML_CUDA_CC_VOLTA && dst->op==GGML_OP_MUL_MAT_ID &&
        w->type==GGML_TYPE_PXQ4 && x->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
        w->ne[0]%32==0 && w->ne[1]%64==0 && w->ne[2]<=512 && w->ne[3]==1 &&
        ids && ids->type==GGML_TYPE_I32 && ids->nb[0]==4 && x->ne[2]>8 &&
        x->nb[0]==4 && dst->nb[0]==4 && x->ne[3]==1 && dst->ne[3]==1;
}

void ggml_cuda_pxq4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w=dst->src[0], *x=dst->src[1], *ids=dst->src[2];
    const int E=w->ne[2],T=x->ne[2],U=ids->ne[0];
    GGML_ASSERT(ggml_cuda_pxq4_prefill_supported(dst,ggml_cuda_info().devices[ctx.device].cc));
    const int max_tiles=(T*U+31)/32+E;
    ggml_cuda_pool_alloc<int> map(ctx.pool(),(size_t)E*T);
    ggml_cuda_pool_alloc<int> counts(ctx.pool(),E),ntiles(ctx.pool(),1);
    ggml_cuda_pool_alloc<pxq4_tile> tiles(ctx.pool(),max_tiles);
    cudaStream_t stream=ctx.stream();
    pxq4_build_map<<<E,256,0,stream>>>((const int32_t*)ids->data,ids->nb[1]/4,T,U,map.get(),counts.get());
    int threads=32;while(threads<E)threads*=2;
    pxq4_build_tiles<<<1,threads,0,stream>>>(counts.get(),E,tiles.get(),ntiles.get());
    pxq4_grouped_wmma<<<dim3(w->ne[1]/64,max_tiles),256,0,stream>>>(
        (const uint8_t*)w->data,(const char*)x->data,(float*)dst->data,
        map.get(),tiles.get(),ntiles.get(),w->ne[0],w->ne[1],T,U,x->ne[1],w->nb[2],x->nb[1],x->nb[2],dst->nb[1],dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

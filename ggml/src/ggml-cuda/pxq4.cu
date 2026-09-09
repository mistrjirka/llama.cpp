// Copyright (c) 2026 PXA Network. Portions adapted from the MIT-licensed PXA project.
// See LICENSE-PXA and benches/pxq4-v100/README.md for provenance and numeric contract.
#include "pxq4.cuh"
#include "pxq-all.cuh"
#include "unary.cuh"
#include "pxq-all-mmvq.cuh"
#include "pxq-all-mmvf.cuh"
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


bool ggml_cuda_is_pxq_type(ggml_type type) {
    return type == GGML_TYPE_PXQ1 || type == GGML_TYPE_PXQ2 || type == GGML_TYPE_PXQ3 ||
           type == GGML_TYPE_PXQ4 || type == GGML_TYPE_PXQ4HQ || type == GGML_TYPE_PXQ6;
}

bool ggml_cuda_pxq_layout_supported(const ggml_tensor * w) {
    if (!w || !ggml_cuda_is_pxq_type(w->type) || w->ne[0] <= 0 || w->ne[1] <= 0 ||
            w->ne[0]%32 || w->ne[1]%64 || w->ne[2]<=0 || w->ne[3]<=0 ||
            w->nb[0] != ggml_type_size(w->type)) return false;
    const size_t row = ggml_row_size(w->type, w->ne[0]);
    if (w->nb[1] != row || w->nb[2] != row*w->ne[1] || w->nb[3] != w->nb[2]*w->ne[2]) return false;
    if (w->view_src) {
        const ggml_tensor * root = w->view_src;
        if (root->type != w->type || root->ne[0] != w->ne[0] || !root->nb[2] || !root->nb[3]) return false;
        size_t offset=w->view_offs;
        const size_t sample=offset/root->nb[3]; offset%=root->nb[3];
        const size_t expert=offset/root->nb[2]; offset%=root->nb[2];
        if(sample>=(size_t)root->ne[3] || expert>=(size_t)root->ne[2] || offset%(row*64)) return false;
        const size_t first_row=offset/row;
        if(first_row+w->ne[1]>(size_t)root->ne[1]) return false;
        if((w->ne[2]>1 || w->ne[3]>1) && (first_row!=0 || w->ne[1]!=root->ne[1])) return false;
    }
    return true;
}

void ggml_cuda_pxq_dequant_f16(ggml_type type, const void * src, half * dst, int64_t nrows, int64_t K, cudaStream_t stream) {
    pxqa_dequant_dispatch(type, src, dst, nrows, K, stream);
}
void ggml_cuda_pxq_dequant_f32(ggml_type type, const void * src, float * dst, int64_t nrows, int64_t K, cudaStream_t stream) {
    pxqa_dequant_dispatch(type, src, dst, nrows, K, stream);
}

bool ggml_cuda_pxq4_layout_supported(const ggml_tensor * w) {
    return w && w->type == GGML_TYPE_PXQ4 && ggml_cuda_pxq_layout_supported(w);
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

void ggml_cuda_pxq4hq_mmvq_launch(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const block_q8_1 * acts, int64_t ne10_padded, const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream) {
    GGML_ASSERT(src0->type == GGML_TYPE_PXQ4HQ && ggml_cuda_pxq_layout_supported(src0));
    const ggml_tensor * gate = fusion ? fusion->gate : nullptr;
    const bool fuse_gate = gate && gate->type == GGML_TYPE_PXQ4HQ && !fusion->x_bias && !fusion->gate_bias &&
                           !fusion->x_scale && !fusion->gate_scale;
    if (fusion) GGML_ASSERT(fuse_gate && "PXQ4-HQ fast fusion supports uniform HQ gate+GLU only");
    const uint32_t K=(uint32_t)src0->ne[0], rows=(uint32_t)src0->ne[1];
    const uint32_t a1=(uint32_t)(ne10_padded/QK8_1), a2=(uint32_t)(src1->ne[1]*a1), a3=(uint32_t)(src1->ne[2]*a2);
    const uint32_t d1=(uint32_t)(dst->nb[1]/sizeof(float)), d2=(uint32_t)(dst->nb[2]/sizeof(float)), d3=(uint32_t)(dst->nb[3]/sizeof(float));
    const uint32_t nt=ids?(uint32_t)dst->ne[2]:(uint32_t)dst->ne[1];
    const uint32_t ay=ids?(uint32_t)src1->ne[1]:(uint32_t)src1->ne[2];
    const uint32_t dc=ids?(uint32_t)dst->ne[1]:(uint32_t)dst->ne[2];
    const uint32_t ns=(uint32_t)dst->ne[3], is=ids?(uint32_t)(ids->nb[1]/sizeof(int32_t)):0;
    pxq4_mmvq_args a{(const uint8_t*)src0->data,gate?(const uint8_t*)gate->data:nullptr,acts,
        ids?(const int32_t*)ids->data:nullptr,(float*)dst->data,K,rows,nt,ay,dc,(uint32_t)src0->ne[2],
        (uint32_t)src0->ne[3],ns,src0->nb[2],src0->nb[3],a1,a2,a3,d1,d2,d3,is,
        fusion?(int)fusion->glu_op:(int)GGML_GLU_OP_SWIGLU,fusion?fusion->glu_limit:0.f};
    int wr = fuse_gate && nt == 1 && K >= 1024 ? 2 : 4;
    if (const char * e=std::getenv("GGML_CUDA_PXQ4HQ_WARP_ROWS")) wr=std::atoi(e);
    int vdr=4; if (const char * e=std::getenv("GGML_CUDA_PXQ4HQ_VDR")) vdr=std::atoi(e);
    auto run=[&](auto rt,auto vt){
        constexpr int R=decltype(rt)::value,V=decltype(vt)::value;
        const dim3 gr((rows+4*R-1)/(4*R),dc,nt*ns), bl(32,4);
        if(ids){ if(fuse_gate)pxq4hq_mmvq_coalesced<true,true,R,V><<<gr,bl,0,stream>>>(a); else pxq4hq_mmvq_coalesced<true,false,R,V><<<gr,bl,0,stream>>>(a); }
        else   { if(fuse_gate)pxq4hq_mmvq_coalesced<false,true,R,V><<<gr,bl,0,stream>>>(a); else pxq4hq_mmvq_coalesced<false,false,R,V><<<gr,bl,0,stream>>>(a); }
    };
    if(wr==2){ if(vdr==2)run(std::integral_constant<int,2>{},std::integral_constant<int,2>{}); else run(std::integral_constant<int,2>{},std::integral_constant<int,4>{}); }
    else     { if(vdr==2)run(std::integral_constant<int,4>{},std::integral_constant<int,2>{}); else run(std::integral_constant<int,4>{},std::integral_constant<int,4>{}); }
    CUDA_CHECK(cudaGetLastError());
}

template <class WPOL, bool HAS_IDS>
static void pxqa_mmvq_launch_up_policy(pxqa_mmvq_args a, const ggml_tensor * gate_tensor, cudaStream_t stream) {
    constexpr int RPW = 4;
    constexpr int NW = 4;
    const dim3 grid((a.rows + RPW*NW - 1)/(RPW*NW), a.channels, a.tokens*a.samples);
    const dim3 block(32, NW, 1);
    if (!gate_tensor) {
        pxqa_mmvq_coalesced<WPOL,WPOL,HAS_IDS,false,RPW,NW><<<grid,block,0,stream>>>(a);
        return;
    }
#define PXQA_GATE_CASE(T, P) case T: pxqa_mmvq_coalesced<WPOL,P,HAS_IDS,true,RPW,NW><<<grid,block,0,stream>>>(a); break
    switch (gate_tensor->type) {
        PXQA_GATE_CASE(GGML_TYPE_PXQ1, pxqa_p1);
        PXQA_GATE_CASE(GGML_TYPE_PXQ2, pxqa_p2);
        PXQA_GATE_CASE(GGML_TYPE_PXQ3, pxqa_p3);
        PXQA_GATE_CASE(GGML_TYPE_PXQ4, pxqa_p4);
        PXQA_GATE_CASE(GGML_TYPE_PXQ4HQ, pxqa_p4hq);
        PXQA_GATE_CASE(GGML_TYPE_PXQ6, pxqa_p6);
        default: GGML_ABORT("PXQ gate tensor has unsupported type %s", ggml_type_name(gate_tensor->type));
    }
#undef PXQA_GATE_CASE
}

void ggml_cuda_pxq_mmvq_launch(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const block_q8_1 * acts, int64_t ne10_padded, const ggml_cuda_mm_fusion_args_host * fusion, cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_pxq_layout_supported(src0));
    const ggml_tensor * gate_tensor = fusion ? fusion->gate : nullptr;
    const bool fuse_gate = gate_tensor && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale;
    if (fusion) {
        GGML_ASSERT(fuse_gate && "PXQ native fusion supports gate+GLU without bias/scale");
        GGML_ASSERT(ggml_cuda_pxq_layout_supported(gate_tensor));
        GGML_ASSERT(gate_tensor->ne[0] == src0->ne[0] && gate_tensor->ne[1] == src0->ne[1] &&
                    gate_tensor->ne[2] == src0->ne[2] && gate_tensor->ne[3] == src0->ne[3]);
    }
    const uint32_t a_s11 = (uint32_t)(ne10_padded/QK8_1);
    const uint32_t a_s12 = (uint32_t)(src1->ne[1]*a_s11);
    const uint32_t a_s13 = (uint32_t)(src1->ne[2]*a_s12);
    const uint32_t d_s1 = (uint32_t)(dst->nb[1]/sizeof(float));
    const uint32_t d_s2 = (uint32_t)(dst->nb[2]/sizeof(float));
    const uint32_t d_s3 = (uint32_t)(dst->nb[3]/sizeof(float));
    const uint32_t ncols_dst = ids ? (uint32_t)dst->ne[2] : (uint32_t)dst->ne[1];
    const uint32_t nchannels_y = ids ? (uint32_t)src1->ne[1] : (uint32_t)src1->ne[2];
    const uint32_t nchannels_dst = ids ? (uint32_t)dst->ne[1] : (uint32_t)dst->ne[2];
    const uint32_t nsamples_dst = (uint32_t)dst->ne[3];
    const uint32_t ids_stride = ids ? (uint32_t)(ids->nb[1]/sizeof(int32_t)) : 0;
    pxqa_mmvq_args a {
        (const uint8_t *)src0->data,
        gate_tensor ? (const uint8_t *)gate_tensor->data : nullptr,
        acts,
        ids ? (const int32_t *)ids->data : nullptr,
        (float *)dst->data,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], ncols_dst, nchannels_y, nchannels_dst,
        (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], nsamples_dst,
        src0->nb[2], src0->nb[3], gate_tensor ? gate_tensor->nb[2] : 0, gate_tensor ? gate_tensor->nb[3] : 0,
        a_s11,a_s12,a_s13,d_s1,d_s2,d_s3,ids_stride,
        fusion ? (int)fusion->glu_op : (int)GGML_GLU_OP_SWIGLU,
        fusion ? fusion->glu_limit : 0.0f
    };
#define PXQA_UP_CASE(T, P) case T: if (ids) pxqa_mmvq_launch_up_policy<P,true>(a,gate_tensor,stream); else pxqa_mmvq_launch_up_policy<P,false>(a,gate_tensor,stream); break
    switch (src0->type) {
        PXQA_UP_CASE(GGML_TYPE_PXQ1, pxqa_p1);
        PXQA_UP_CASE(GGML_TYPE_PXQ2, pxqa_p2);
        PXQA_UP_CASE(GGML_TYPE_PXQ3, pxqa_p3);
        PXQA_UP_CASE(GGML_TYPE_PXQ4, pxqa_p4);
        PXQA_UP_CASE(GGML_TYPE_PXQ4HQ, pxqa_p4hq);
        PXQA_UP_CASE(GGML_TYPE_PXQ6, pxqa_p6);
        default: GGML_ABORT("not a PXQ tensor");
    }
#undef PXQA_UP_CASE
    CUDA_CHECK(cudaGetLastError());
}

template <class WPOL, bool HAS_IDS>
static void pxqa_mmvf_launch_up_policy(pxqa_mmvf_args a, const ggml_tensor * gate_tensor, cudaStream_t stream) {
    const dim3 grid((a.rows + 63)/64, a.channels, a.tokens*a.samples);
    if (!gate_tensor) {
        pxqa_mmvf_panel<WPOL,WPOL,HAS_IDS,false><<<grid,256,0,stream>>>(a);
        return;
    }
#define PXQAF_GATE(T,P) case T: pxqa_mmvf_panel<WPOL,P,HAS_IDS,true><<<grid,256,0,stream>>>(a); break
    switch (gate_tensor->type) {
        PXQAF_GATE(GGML_TYPE_PXQ1,pxqa_p1); PXQAF_GATE(GGML_TYPE_PXQ2,pxqa_p2);
        PXQAF_GATE(GGML_TYPE_PXQ3,pxqa_p3); PXQAF_GATE(GGML_TYPE_PXQ4,pxqa_p4);
        PXQAF_GATE(GGML_TYPE_PXQ4HQ,pxqa_p4hq); PXQAF_GATE(GGML_TYPE_PXQ6,pxqa_p6);
        default: GGML_ABORT("bad PXQ gate type");
    }
#undef PXQAF_GATE
}

void ggml_cuda_pxq_mmvf_launch(const ggml_tensor * src0,const ggml_tensor * src1,const ggml_tensor * ids,
        ggml_tensor * dst,const ggml_cuda_mm_fusion_args_host * fusion,cudaStream_t stream){
    GGML_ASSERT(ggml_cuda_pxq_layout_supported(src0)&&src1->type==GGML_TYPE_F32&&dst->type==GGML_TYPE_F32);
    const ggml_tensor *gate=fusion?fusion->gate:nullptr;
    if(fusion){GGML_ASSERT(gate&&!fusion->x_bias&&!fusion->gate_bias&&!fusion->x_scale&&!fusion->gate_scale);GGML_ASSERT(ggml_cuda_pxq_layout_supported(gate));}
    pxqa_mmvf_args a{
        (const uint8_t*)src0->data,gate?(const uint8_t*)gate->data:nullptr,(const float*)src1->data,
        ids?(const int32_t*)ids->data:nullptr,(float*)dst->data,
        (uint32_t)src0->ne[0],(uint32_t)src0->ne[1],ids?(uint32_t)dst->ne[2]:(uint32_t)dst->ne[1],
        ids?(uint32_t)src1->ne[1]:(uint32_t)src1->ne[2],ids?(uint32_t)dst->ne[1]:(uint32_t)dst->ne[2],
        (uint32_t)src0->ne[2],(uint32_t)src0->ne[3],(uint32_t)dst->ne[3],
        src0->nb[2],src0->nb[3],gate?gate->nb[2]:0,gate?gate->nb[3]:0,
        (uint32_t)(src1->nb[1]/sizeof(float)),(uint32_t)(src1->nb[2]/sizeof(float)),(uint32_t)(src1->nb[3]/sizeof(float)),
        (uint32_t)(dst->nb[1]/sizeof(float)),(uint32_t)(dst->nb[2]/sizeof(float)),(uint32_t)(dst->nb[3]/sizeof(float)),
        ids?(uint32_t)(ids->nb[1]/sizeof(int32_t)):0,
        fusion?(int)fusion->glu_op:(int)GGML_GLU_OP_SWIGLU,fusion?fusion->glu_limit:0.f};
#define PXQAF_UP(T,P) case T: if(ids)pxqa_mmvf_launch_up_policy<P,true>(a,gate,stream);else pxqa_mmvf_launch_up_policy<P,false>(a,gate,stream);break
    switch(src0->type){
        PXQAF_UP(GGML_TYPE_PXQ1,pxqa_p1);PXQAF_UP(GGML_TYPE_PXQ2,pxqa_p2);PXQAF_UP(GGML_TYPE_PXQ3,pxqa_p3);
        PXQAF_UP(GGML_TYPE_PXQ4,pxqa_p4);PXQAF_UP(GGML_TYPE_PXQ4HQ,pxqa_p4hq);PXQAF_UP(GGML_TYPE_PXQ6,pxqa_p6);
        default: GGML_ABORT("bad PXQ type");
    }
#undef PXQAF_UP
    CUDA_CHECK(cudaGetLastError());
}



#include "pxq4-port.cuh"
#include "pxq4-mmvf.cuh"
#include "pxq4-wmma.cuh"
#include "pxq-all-wmma.cuh"

bool ggml_cuda_pxq4_prefill_supported(const ggml_tensor * dst, int cc) {
    const ggml_tensor * w=dst->src[0], *x=dst->src[1], *ids=dst->src[2];
    const char * e=std::getenv("GGML_CUDA_PXQ4_PREFILL");
    return (!e || std::atoi(e)!=0) && cc==GGML_CUDA_CC_VOLTA && dst->op==GGML_OP_MUL_MAT_ID &&
        ggml_cuda_pxq4_layout_supported(w) && x->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
        w->ne[0]%32==0 && w->ne[1]%64==0 && w->ne[2]<=512 && w->ne[3]==1 &&
        ids && ids->type==GGML_TYPE_I32 && ids->nb[0]==4 && x->ne[2]>8 &&
        x->nb[0]==4 && dst->nb[0]==4 && x->ne[3]==1 && dst->ne[3]==1;
}

void ggml_cuda_pxq4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w=dst->src[0], *x=dst->src[1], *ids=dst->src[2];
    const int E=w->ne[2],T=x->ne[2],U=ids->ne[0];
    GGML_ASSERT(ggml_cuda_pxq4_prefill_supported(dst,ggml_cuda_info().devices[ctx.device].cc));
    const int max_tiles=(T*U+31)/32+E;
    ggml_cuda_pool_alloc<int> map(ctx.pool(),(size_t)E*T*U);
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


bool ggml_cuda_pxq_prefill_supported(const ggml_tensor * dst, int cc) {
    const ggml_tensor * w=dst->src[0], *x=dst->src[1], *ids=dst->src[2];
    if (!w || !ggml_cuda_is_pxq_type(w->type)) return false;
    if (w->type == GGML_TYPE_PXQ4) return ggml_cuda_pxq4_prefill_supported(dst,cc);
    const char * all=std::getenv("GGML_CUDA_PXQ_NATIVE");
    if (all && std::atoi(all)==0) return false;
    const char * e=std::getenv("GGML_CUDA_PXQ_PREFILL");
    return (!e || std::atoi(e)!=0) && cc==GGML_CUDA_CC_VOLTA && dst->op==GGML_OP_MUL_MAT_ID &&
        ggml_cuda_pxq_layout_supported(w) && x->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
        w->ne[0]%32==0 && w->ne[1]%64==0 && w->ne[2]<=512 && w->ne[3]==1 &&
        ids && ids->type==GGML_TYPE_I32 && ids->nb[0]==4 && x->ne[2]>8 &&
        x->nb[0]==4 && dst->nb[0]==4 && x->ne[3]==1 && dst->ne[3]==1;
}

template <class POL>
static void pxqa_prefill_policy(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w=dst->src[0], *x=dst->src[1], *ids=dst->src[2];
    const int E=w->ne[2],T=x->ne[2],U=ids->ne[0];
    const int max_tiles=(T*U+31)/32+E;
    ggml_cuda_pool_alloc<int> map(ctx.pool(),(size_t)E*T*U);
    ggml_cuda_pool_alloc<int> counts(ctx.pool(),E),ntiles(ctx.pool(),1);
    ggml_cuda_pool_alloc<pxq4_tile> tiles(ctx.pool(),max_tiles);
    cudaStream_t stream=ctx.stream();
    pxq4_build_map<<<E,256,0,stream>>>((const int32_t*)ids->data,ids->nb[1]/4,T,U,map.get(),counts.get());
    int threads=32;while(threads<E)threads*=2;
    pxq4_build_tiles<<<1,threads,0,stream>>>(counts.get(),E,tiles.get(),ntiles.get());
    pxqa_grouped_wmma<POL><<<dim3(w->ne[1]/64,max_tiles),256,0,stream>>>(
        (const uint8_t*)w->data,(const char*)x->data,(float*)dst->data,
        map.get(),tiles.get(),ntiles.get(),w->ne[0],w->ne[1],T,U,x->ne[1],w->nb[2],x->nb[1],x->nb[2],dst->nb[1],dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_pxq_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_type type=dst->src[0]->type;
    if (type==GGML_TYPE_PXQ4) { ggml_cuda_pxq4_prefill(ctx,dst); return; }
    GGML_ASSERT(ggml_cuda_pxq_prefill_supported(dst,ggml_cuda_info().devices[ctx.device].cc));
    switch(type) {
        case GGML_TYPE_PXQ1: pxqa_prefill_policy<pxqa_p1>(ctx,dst); break;
        case GGML_TYPE_PXQ2: pxqa_prefill_policy<pxqa_p2>(ctx,dst); break;
        case GGML_TYPE_PXQ3: pxqa_prefill_policy<pxqa_p3>(ctx,dst); break;
        case GGML_TYPE_PXQ4HQ: pxqa_prefill_policy<pxqa_p4hq>(ctx,dst); break;
        case GGML_TYPE_PXQ6: pxqa_prefill_policy<pxqa_p6>(ctx,dst); break;
        default: GGML_ABORT("bad PXQ prefill type");
    }
}


bool ggml_cuda_pxq_dense_prefill_supported(const ggml_tensor * dst, int cc) {
    const ggml_tensor * w=dst->src[0], *x=dst->src[1];
    if (!w || !ggml_cuda_is_pxq_type(w->type)) return false;
    const char * all=std::getenv("GGML_CUDA_PXQ_NATIVE");
    if (all && std::atoi(all)==0) return false;
    if (w->type==GGML_TYPE_PXQ4) {
        const char * e=std::getenv("GGML_CUDA_PXQ4_NATIVE");
        if (e && std::atoi(e)==0) return false;
    }
    const char * e=std::getenv("GGML_CUDA_PXQ_DENSE_PREFILL");
    // The generic WMMA prototype is a useful oracle/experiment, but on V100 the coalesced
    // exact dequant + cuBLAS path is substantially faster for dense prefill. Keep it opt-in.
    return (e && std::atoi(e)!=0) && cc==GGML_CUDA_CC_VOLTA && dst->op==GGML_OP_MUL_MAT &&
        ggml_cuda_pxq_layout_supported(w) && x->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
        w->ne[0]%32==0 && w->ne[1]%64==0 && w->ne[2]==1 && w->ne[3]==1 &&
        x->ne[1]>8 && x->ne[2]==1 && x->ne[3]==1 && dst->ne[2]==1 && dst->ne[3]==1 &&
        x->nb[0]==sizeof(float) && dst->nb[0]==sizeof(float);
}

template<class POL>
static void pxqa_dense_prefill_policy(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w=dst->src[0], *x=dst->src[1];
    const int K=(int)w->ne[0], M=(int)w->ne[1], N=(int)x->ne[1];
    dim3 grid((unsigned)(M/64),(unsigned)((N+31)/32),1);
    pxqa_dense_wmma<POL><<<grid,256,0,ctx.stream()>>>(
        (const uint8_t *)w->data,(const char *)x->data,(float *)dst->data,
        K,M,N,x->nb[1],dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_pxq_dense_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_pxq_dense_prefill_supported(dst,ggml_cuda_info().devices[ctx.device].cc));
    switch(dst->src[0]->type) {
        case GGML_TYPE_PXQ1: pxqa_dense_prefill_policy<pxqa_p1>(ctx,dst); break;
        case GGML_TYPE_PXQ2: pxqa_dense_prefill_policy<pxqa_p2>(ctx,dst); break;
        case GGML_TYPE_PXQ3: pxqa_dense_prefill_policy<pxqa_p3>(ctx,dst); break;
        case GGML_TYPE_PXQ4: pxqa_dense_prefill_policy<pxqa_p4>(ctx,dst); break;
        case GGML_TYPE_PXQ4HQ: pxqa_dense_prefill_policy<pxqa_p4hq>(ctx,dst); break;
        case GGML_TYPE_PXQ6: pxqa_dense_prefill_policy<pxqa_p6>(ctx,dst); break;
        default: GGML_ABORT("bad PXQ dense prefill type");
    }
}

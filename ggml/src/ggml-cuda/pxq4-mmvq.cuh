// Copyright (c) 2026 PXA Network (frozen tables and nibble lookup).
// SPDX-License-Identifier: MIT
// Format provenance/license: LICENSE-PXA. Thread mapping is specific to this port.
#pragma once

struct pxq4_mmvq_args {
    const uint8_t * weights;
    const uint8_t * gate;
    const block_q8_1 * acts;
    const int32_t * ids;
    float * dst;
    uint32_t K, rows, tokens, achannels, channels, wchannels, wsamples, samples;
    size_t wnb2, wnb3;
    uint32_t as1, as2, as3, ds1, ds2, ds3, ids_stride;
    int glu_op;
    float glu_limit;
};

// Adjacent lanes fetch adjacent output rows of the SAME slab. Two lanes split each
// 16-byte code row when VDR=2. This fills memory sectors instead of walking distant
// slabs with one lane per slab. Four independent warps per CTA avoid the single-warp
// CTA occupancy ceiling; subscales use shared memory, not divergent constant loads.
template<bool HAS_IDS, bool HAS_GATE, int ROWS_PER_WARP, int VDR, int NWARPS = 4>
__launch_bounds__(32*NWARPS, 2)
static __global__ void pxq4_mmvq_coalesced(pxq4_mmvq_args a) {
    static_assert(ROWS_PER_WARP == 1 || ROWS_PER_WARP == 2 || ROWS_PER_WARP == 4, "warp row grouping");
    static_assert(VDR == 2 || VDR == 4, "code words per lane");
    __shared__ float sub[16];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    if (warp == 0 && lane < 16) sub[lane] = pxq4_mmvq_sub16[lane];
    __syncthreads();
    const int r = (int)blockIdx.x * (ROWS_PER_WARP*NWARPS) + warp*ROWS_PER_WARP + lane%ROWS_PER_WARP;
    if (r >= (int)a.rows) return; // panels guarantee full warps have complete row groups
    const unsigned ch = blockIdx.y;
    const unsigned token = blockIdx.z % a.tokens;
    const unsigned sample = blockIdx.z / a.tokens;
    unsigned ex, ac, ws, as;
    if constexpr (HAS_IDS) {
        ex = a.ids[ch + token*a.ids_stride];
        ac = ch % a.achannels;
        ws = as = 0;
    } else {
        ex = ch / (a.channels/a.wchannels);
        ac = ch;
        ws = sample/(a.samples/a.wsamples);
        as = sample;
    }
    const block_q8_1 * x = a.acts + (size_t)as*a.as3 + (HAS_IDS
        ? (size_t)ac*a.as1 + (size_t)token*a.as2
        : (size_t)ac*a.as2 + (size_t)token*a.as1);
    const int slabs = a.K/32;
    const size_t poff = (size_t)ws*a.wnb3 + (size_t)ex*a.wnb2 + (size_t)(r/64)*(128u+(size_t)slabs*1088u);
    const uint8_t * wp = a.weights + poff;
    const uint8_t * gp = HAS_GATE ? a.gate + poff : nullptr;
    const int rr = r%64;
    const float wa = __half2float(((const half *)wp)[rr])*(1.0f/127.0f);
    const float ga = HAS_GATE ? __half2float(((const half *)gp)[rr])*(1.0f/127.0f) : 0.f;
    const int klane = lane/ROWS_PER_WARP;
    const int word = (klane%(4/VDR))*VDR;
    constexpr int STRIDE = (32/ROWS_PER_WARP)/(4/VDR);
    float acc = 0.f, gacc = 0.f;
    for (int kb = klane/(4/VDR); kb < slabs; kb += STRIDE) {
        const uint8_t * wslab = wp+128+(size_t)kb*1088;
        const int sc = wslab[rr];
        uint32_t qw[VDR];
        if constexpr (VDR == 2) *(uint2 *)qw = *(const uint2 *)(wslab+64+16*rr+4*word);
        else                   *(uint4 *)qw = *(const uint4 *)(wslab+64+16*rr);
        const int * qx = (const int *)x[kb].qs;
        const float ad = __low2float(x[kb].ds);
        float sum = 0.f;
#pragma unroll
        for (int m=0; m<VDR; ++m) {
            const int2 w = pxq4_mmvq_table16_seq(qw[m]);
            int d = ggml_cuda_dp4a(w.x,qx[2*(word+m)],0);
            d = ggml_cuda_dp4a(w.y,qx[2*(word+m)+1],d);
            sum += sub[(sc>>(4*((word+m)/2)))&15]*(float)d;
        }
        acc += wa*ad*sum;
        if constexpr (HAS_GATE) {
            const uint8_t * gslab = gp+128+(size_t)kb*1088;
            const int sg = gslab[rr];
            uint32_t qg[VDR];
            if constexpr (VDR == 2) *(uint2 *)qg = *(const uint2 *)(gslab+64+16*rr+4*word);
            else                   *(uint4 *)qg = *(const uint4 *)(gslab+64+16*rr);
            float gs = 0.f;
#pragma unroll
            for (int m=0; m<VDR; ++m) {
                const int2 g = pxq4_mmvq_table16_seq(qg[m]);
                int d = ggml_cuda_dp4a(g.x,qx[2*(word+m)],0);
                d = ggml_cuda_dp4a(g.y,qx[2*(word+m)+1],d);
                gs += sub[(sg>>(4*((word+m)/2)))&15]*(float)d;
            }
            gacc += ga*ad*gs;
        }
    }
#pragma unroll
    for (int off=16; off>=ROWS_PER_WARP; off/=2) {
        acc += __shfl_xor_sync(0xffffffff,acc,off);
        if constexpr (HAS_GATE) gacc += __shfl_xor_sync(0xffffffff,gacc,off);
    }
    if (lane < ROWS_PER_WARP) {
        if constexpr (HAS_GATE) {
            switch ((ggml_glu_op)a.glu_op) {
                case GGML_GLU_OP_SWIGLU: acc *= ggml_cuda_op_silu_single(gacc); break;
                case GGML_GLU_OP_GEGLU: acc *= ggml_cuda_op_gelu_single(gacc); break;
                case GGML_GLU_OP_SWIGLU_OAI: acc = ggml_cuda_op_swiglu_oai_single(gacc,acc); break;
                case GGML_GLU_OP_SWIGLU_CLAMP: acc = ggml_cuda_op_swiglu_clamp_single(gacc,acc,a.glu_limit); break;
                default: acc *= gacc;
            }
        }
        const size_t off = HAS_IDS ? (size_t)ch*a.ds1+(size_t)token*a.ds2+r
            : (size_t)sample*a.ds3+(size_t)ch*a.ds2+(size_t)token*a.ds1+r;
        a.dst[off] = acc;
    }
}

// PXQ4-HQ twin of the validated coalesced PXQ4 MMVQ kernel.
// The book and 4-bit code stream are identical; HQ differs only in its bs8 scale layout:
// two bytes/row (four nibbles for four 8-element groups) and CODE_OFF=128.
template<bool HAS_IDS, bool HAS_GATE, int ROWS_PER_WARP, int VDR, int NWARPS = 4>
__launch_bounds__(32*NWARPS, 2)
static __global__ void pxq4hq_mmvq_coalesced(pxq4_mmvq_args a) {
    static_assert(ROWS_PER_WARP == 1 || ROWS_PER_WARP == 2 || ROWS_PER_WARP == 4, "warp row grouping");
    static_assert(VDR == 2 || VDR == 4, "code words per lane");
    __shared__ float sub[16];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    if (warp == 0 && lane < 16) sub[lane] = pxqa_sub8[lane];
    __syncthreads();

    const int r = (int)blockIdx.x * (ROWS_PER_WARP*NWARPS) + warp*ROWS_PER_WARP + lane%ROWS_PER_WARP;
    if (r >= (int)a.rows) return;
    const unsigned ch = blockIdx.y;
    const unsigned token = blockIdx.z % a.tokens;
    const unsigned sample = blockIdx.z / a.tokens;
    unsigned ex, ac, ws, as;
    if constexpr (HAS_IDS) {
        ex = a.ids[ch + token*a.ids_stride];
        ac = ch % a.achannels;
        ws = as = 0;
    } else {
        ex = ch / (a.channels/a.wchannels);
        ac = ch;
        ws = sample/(a.samples/a.wsamples);
        as = sample;
    }
    const block_q8_1 * x = a.acts + (size_t)as*a.as3 + (HAS_IDS
        ? (size_t)ac*a.as1 + (size_t)token*a.as2
        : (size_t)ac*a.as2 + (size_t)token*a.as1);
    const int slabs = a.K/32;
    const size_t panel_stride = pxqa_p4hq::HDR + (size_t)slabs*pxqa_p4hq::SLAB;
    const size_t poff = (size_t)ws*a.wnb3 + (size_t)ex*a.wnb2 + (size_t)(r/64)*panel_stride;
    const uint8_t * wp = a.weights + poff;
    const uint8_t * gp = HAS_GATE ? a.gate + poff : nullptr;
    const int rr = r%64;
    const float wa = __half2float(((const half *)wp)[rr])*(1.0f/127.0f);
    const float ga = HAS_GATE ? __half2float(((const half *)gp)[rr])*(1.0f/127.0f) : 0.f;
    const int klane = lane/ROWS_PER_WARP;
    const int word = (klane%(4/VDR))*VDR;
    constexpr int STRIDE = (32/ROWS_PER_WARP)/(4/VDR);
    float acc = 0.f, gacc = 0.f;

    for (int kb = klane/(4/VDR); kb < slabs; kb += STRIDE) {
        const uint8_t * wslab = wp + pxqa_p4hq::HDR + (size_t)kb*pxqa_p4hq::SLAB;
        uint32_t qw[VDR];
        if constexpr (VDR == 2) *(uint2 *)qw = *(const uint2 *)(wslab + pxqa_p4hq::CODE_OFF + 16*rr + 4*word);
        else                   *(uint4 *)qw = *(const uint4 *)(wslab + pxqa_p4hq::CODE_OFF + 16*rr);
        const int * qx = (const int *)x[kb].qs;
        const float ad = __low2float(x[kb].ds);
        float sum = 0.f;
#pragma unroll
        for (int m = 0; m < VDR; ++m) {
            const int wi = word + m; // one 32-bit word == eight quantized values
            const int2 w = pxq4_mmvq_table16_seq(qw[m]);
            int d = ggml_cuda_dp4a(w.x, qx[2*wi], 0);
            d = ggml_cuda_dp4a(w.y, qx[2*wi + 1], d);
            const uint8_t sb = wslab[2*rr + (wi >> 1)];
            const int sc = (sb >> (4*(wi & 1))) & 15;
            sum += sub[sc]*(float)d;
        }
        acc += wa*ad*sum;

        if constexpr (HAS_GATE) {
            const uint8_t * gslab = gp + pxqa_p4hq::HDR + (size_t)kb*pxqa_p4hq::SLAB;
            uint32_t qg[VDR];
            if constexpr (VDR == 2) *(uint2 *)qg = *(const uint2 *)(gslab + pxqa_p4hq::CODE_OFF + 16*rr + 4*word);
            else                   *(uint4 *)qg = *(const uint4 *)(gslab + pxqa_p4hq::CODE_OFF + 16*rr);
            float gs = 0.f;
#pragma unroll
            for (int m = 0; m < VDR; ++m) {
                const int wi = word + m;
                const int2 g = pxq4_mmvq_table16_seq(qg[m]);
                int d = ggml_cuda_dp4a(g.x, qx[2*wi], 0);
                d = ggml_cuda_dp4a(g.y, qx[2*wi + 1], d);
                const uint8_t sb = gslab[2*rr + (wi >> 1)];
                const int sc = (sb >> (4*(wi & 1))) & 15;
                gs += sub[sc]*(float)d;
            }
            gacc += ga*ad*gs;
        }
    }
#pragma unroll
    for (int off = 16; off >= ROWS_PER_WARP; off /= 2) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
        if constexpr (HAS_GATE) gacc += __shfl_xor_sync(0xffffffff, gacc, off);
    }
    if (lane < ROWS_PER_WARP) {
        if constexpr (HAS_GATE) {
            switch ((ggml_glu_op)a.glu_op) {
                case GGML_GLU_OP_SWIGLU: acc *= ggml_cuda_op_silu_single(gacc); break;
                case GGML_GLU_OP_GEGLU: acc *= ggml_cuda_op_gelu_single(gacc); break;
                case GGML_GLU_OP_SWIGLU_OAI: acc = ggml_cuda_op_swiglu_oai_single(gacc, acc); break;
                case GGML_GLU_OP_SWIGLU_CLAMP: acc = ggml_cuda_op_swiglu_clamp_single(gacc, acc, a.glu_limit); break;
                default: acc *= gacc; break;
            }
        }
        const size_t off = HAS_IDS ? (size_t)ch*a.ds1 + (size_t)token*a.ds2 + r
            : (size_t)sample*a.ds3 + (size_t)ch*a.ds2 + (size_t)token*a.ds1 + r;
        a.dst[off] = acc;
    }
}

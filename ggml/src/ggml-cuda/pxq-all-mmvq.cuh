#pragma once

struct pxqa_mmvq_args {
    const uint8_t * weights;
    const uint8_t * gate;
    const block_q8_1 * acts;
    const int32_t * ids;
    float * dst;
    uint32_t K, rows, tokens, achannels, channels, wchannels, wsamples, samples;
    size_t wnb2, wnb3, gnb2, gnb3;
    uint32_t as1, as2, as3, ds1, ds2, ds3, ids_stride;
    int glu_op;
    float glu_limit;
};

template<class WPOL, class GPOL, bool HAS_IDS, bool HAS_GATE, int ROWS_PER_WARP = 4, int NWARPS = 4>
__launch_bounds__(32*NWARPS, 2)
static __global__ void pxqa_mmvq_coalesced(pxqa_mmvq_args a) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int r = (int)blockIdx.x*(ROWS_PER_WARP*NWARPS) + warp*ROWS_PER_WARP + lane%ROWS_PER_WARP;
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
    const size_t wpoff = (size_t)ws*a.wnb3 + (size_t)ex*a.wnb2 + (size_t)(r/64)*pxqa_panel_stride<WPOL>(slabs);
    const uint8_t * wp = a.weights + wpoff;
    const uint8_t * gp = nullptr;
    if constexpr (HAS_GATE) {
        const size_t gpoff = (size_t)ws*a.gnb3 + (size_t)ex*a.gnb2 + (size_t)(r/64)*pxqa_panel_stride<GPOL>(slabs);
        gp = a.gate + gpoff;
    }
    const int rr = r & 63;
    const float wa = __half2float(((const half*)wp)[rr]);
    const float ga = HAS_GATE ? __half2float(((const half*)gp)[rr]) : 0.f;
    const int klane = lane/ROWS_PER_WARP;
    constexpr int STRIDE = 32/ROWS_PER_WARP;
    float acc = 0.f, gacc = 0.f;
    for (int kb = klane; kb < slabs; kb += STRIDE) {
        const uint8_t * wsla = wp + WPOL::HDR + (size_t)kb*WPOL::SLAB;
        acc += pxqa_dot32_q8_1<WPOL>(wsla, rr, x+kb, wa);
        if constexpr (HAS_GATE) {
            const uint8_t * gsla = gp + GPOL::HDR + (size_t)kb*GPOL::SLAB;
            gacc += pxqa_dot32_q8_1<GPOL>(gsla, rr, x+kb, ga);
        }
    }
#pragma unroll
    for (int off=16; off>=ROWS_PER_WARP; off/=2) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
        if constexpr (HAS_GATE) gacc += __shfl_xor_sync(0xffffffff, gacc, off);
    }
    if (lane < ROWS_PER_WARP) {
        if constexpr (HAS_GATE) {
            switch ((ggml_glu_op)a.glu_op) {
                case GGML_GLU_OP_SWIGLU: acc *= ggml_cuda_op_silu_single(gacc); break;
                case GGML_GLU_OP_GEGLU: acc *= ggml_cuda_op_gelu_single(gacc); break;
                case GGML_GLU_OP_SWIGLU_OAI: acc = ggml_cuda_op_swiglu_oai_single(gacc,acc); break;
                case GGML_GLU_OP_SWIGLU_CLAMP: acc = ggml_cuda_op_swiglu_clamp_single(gacc,acc,a.glu_limit); break;
                default: acc *= gacc; break;
            }
        }
        const size_t off = HAS_IDS ? (size_t)ch*a.ds1+(size_t)token*a.ds2+r
            : (size_t)sample*a.ds3+(size_t)ch*a.ds2+(size_t)token*a.ds1+r;
        a.dst[off] = acc;
    }
}

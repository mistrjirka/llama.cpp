#pragma once

struct pxqa_mmvf_args {
    const uint8_t * weights;
    const uint8_t * gate;
    const float * acts;
    const int32_t * ids;
    float * dst;
    uint32_t K, rows, tokens, achannels, channels, wchannels, wsamples, samples;
    size_t wnb2, wnb3, gnb2, gnb3;
    uint32_t as1, as2, as3, ds1, ds2, ds3, ids_stride;
    int glu_op;
    float glu_limit;
};

template<class POL>
static __device__ __forceinline__ float pxqa_dot32_f32(
        const uint8_t * slab, int row, const float * x, float anchor) {
    const uint8_t * q = slab + POL::CODE_OFF + (size_t)row*POL::CODE_BYTES;
    float sum=0.f;
#pragma unroll
    for (int j=0;j<32;++j) {
        const float w = anchor*POL::subscale(slab,row,j)*POL::book(POL::code(q,j));
        sum = fmaf(w,x[j],sum);
    }
    return sum;
}

template<class WPOL, class GPOL, bool HAS_IDS, bool HAS_GATE, int ROWS_PER_WARP=4, int NWARPS=4>
__launch_bounds__(32*NWARPS,2)
static __global__ void pxqa_mmvf_coalesced(pxqa_mmvf_args a) {
    const int lane=threadIdx.x, warp=threadIdx.y;
    const int r=(int)blockIdx.x*(ROWS_PER_WARP*NWARPS)+warp*ROWS_PER_WARP+lane%ROWS_PER_WARP;
    if(r>=(int)a.rows)return;
    const unsigned ch=blockIdx.y;
    const unsigned token=blockIdx.z%a.tokens;
    const unsigned sample=blockIdx.z/a.tokens;
    unsigned ex,ac,ws,as;
    if constexpr(HAS_IDS){ex=a.ids[ch+token*a.ids_stride];ac=ch%a.achannels;ws=as=0;}
    else{ex=ch/(a.channels/a.wchannels);ac=ch;ws=sample/(a.samples/a.wsamples);as=sample;}
    const float * x=a.acts+(size_t)as*a.as3+(HAS_IDS
        ?(size_t)ac*a.as1+(size_t)token*a.as2
        :(size_t)ac*a.as2+(size_t)token*a.as1);
    const int slabs=a.K/32;
    const size_t wpoff=(size_t)ws*a.wnb3+(size_t)ex*a.wnb2+(size_t)(r/64)*pxqa_panel_stride<WPOL>(slabs);
    const uint8_t *wp=a.weights+wpoff;
    const uint8_t *gp=nullptr;
    if constexpr(HAS_GATE){
        const size_t gpoff=(size_t)ws*a.gnb3+(size_t)ex*a.gnb2+(size_t)(r/64)*pxqa_panel_stride<GPOL>(slabs);
        gp=a.gate+gpoff;
    }
    const int rr=r&63;
    const float wa=__half2float(((const half*)wp)[rr]);
    const float ga=HAS_GATE?__half2float(((const half*)gp)[rr]):0.f;
    const int klane=lane/ROWS_PER_WARP;
    constexpr int STRIDE=32/ROWS_PER_WARP;
    float acc=0.f,gacc=0.f;
    for(int kb=klane;kb<slabs;kb+=STRIDE){
        const uint8_t *wsla=wp+WPOL::HDR+(size_t)kb*WPOL::SLAB;
        acc+=pxqa_dot32_f32<WPOL>(wsla,rr,x+(size_t)kb*32,wa);
        if constexpr(HAS_GATE){
            const uint8_t *gsla=gp+GPOL::HDR+(size_t)kb*GPOL::SLAB;
            gacc+=pxqa_dot32_f32<GPOL>(gsla,rr,x+(size_t)kb*32,ga);
        }
    }
#pragma unroll
    for(int off=16;off>=ROWS_PER_WARP;off/=2){
        acc+=__shfl_xor_sync(0xffffffff,acc,off);
        if constexpr(HAS_GATE)gacc+=__shfl_xor_sync(0xffffffff,gacc,off);
    }
    if(lane<ROWS_PER_WARP){
        if constexpr(HAS_GATE){
            switch((ggml_glu_op)a.glu_op){
                case GGML_GLU_OP_SWIGLU: acc*=ggml_cuda_op_silu_single(gacc);break;
                case GGML_GLU_OP_GEGLU: acc*=ggml_cuda_op_gelu_single(gacc);break;
                case GGML_GLU_OP_SWIGLU_OAI: acc=ggml_cuda_op_swiglu_oai_single(gacc,acc);break;
                case GGML_GLU_OP_SWIGLU_CLAMP: acc=ggml_cuda_op_swiglu_clamp_single(gacc,acc,a.glu_limit);break;
                default: acc*=gacc;break;
            }
        }
        const size_t off=HAS_IDS?(size_t)ch*a.ds1+(size_t)token*a.ds2+r
            :(size_t)sample*a.ds3+(size_t)ch*a.ds2+(size_t)token*a.ds1+r;
        a.dst[off]=acc;
    }
}

// V100 direct-decode kernel: one 256-thread CTA owns a full 64-row PXQ panel.
// Threads are mapped as 64 rows x 4 K-segments, matching the physical panel layout:
// adjacent lanes read adjacent rows while four lanes independently accumulate each row's K chain.
// Activations are staged in bounded chunks so large dense down projections do not exceed SM70
// shared-memory limits. `acc` remains live across chunks, so chunking does not introduce an
// extra reduction boundary in a lane's arithmetic chain.
template<class POL>
static __device__ __forceinline__ float pxqa_dot32_f32_shared(
        const uint8_t * slab, int row, const float * x, float anchor,
        const float * book, const float * subs) {
    const uint8_t * q = slab + POL::CODE_OFF + (size_t)row*POL::CODE_BYTES;
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        int si;
        if constexpr (POL::HQ) {
            const uint8_t sb = slab[2*row + (j >= 16)];
            si = ((j & 15) >= 8) ? (sb >> 4) : (sb & 0xf);
        } else {
            const uint8_t sb = slab[row];
            si = j >= 16 ? (sb >> 4) : (sb & 0xf);
        }
        sum = fmaf(anchor * subs[si] * book[POL::code(q, j)], x[j], sum);
    }
    return sum;
}

template<class WPOL, class GPOL, bool HAS_IDS, bool HAS_GATE, int CHUNK_SLABS = 128>
__launch_bounds__(256, 2)
static __global__ void pxqa_mmvf_panel(pxqa_mmvf_args a) {
    const int tid  = threadIdx.x;
    const int row  = tid & 63;
    const int kseg = tid >> 6; // 0..3
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

    const float * x = a.acts + (size_t)as*a.as3 + (HAS_IDS
        ? (size_t)ac*a.as1 + (size_t)token*a.as2
        : (size_t)ac*a.as2 + (size_t)token*a.as1);

    const int slabs = a.K/32;
    const size_t wpoff = (size_t)ws*a.wnb3 + (size_t)ex*a.wnb2
        + (size_t)blockIdx.x*pxqa_panel_stride<WPOL>(slabs);
    const uint8_t * wp = a.weights + wpoff;
    const uint8_t * gp = nullptr;
    if constexpr (HAS_GATE) {
        const size_t gpoff = (size_t)ws*a.gnb3 + (size_t)ex*a.gnb2
            + (size_t)blockIdx.x*pxqa_panel_stride<GPOL>(slabs);
        gp = a.gate + gpoff;
    }

    __shared__ float sx[CHUNK_SLABS*32];
    __shared__ float red[HAS_GATE ? 512 : 256];
    __shared__ float book_u[32], sub_u[16], book_g[32], sub_g[16];
    if (tid < WPOL::BOOK_N) book_u[tid] = WPOL::book(tid);
    if (tid < 16) sub_u[tid] = WPOL::HQ ? pxqa_sub8[tid] : pxqa_sub16[tid];
    if constexpr (HAS_GATE) {
        if (tid < GPOL::BOOK_N) book_g[tid] = GPOL::book(tid);
        if (tid < 16) sub_g[tid] = GPOL::HQ ? pxqa_sub8[tid] : pxqa_sub16[tid];
    }
    __syncthreads();

    const float wa = __half2float(((const half*)wp)[row]);
    const float ga = HAS_GATE ? __half2float(((const half*)gp)[row]) : 0.f;
    float acc = 0.f, gacc = 0.f;

    for (int kb0 = 0; kb0 < slabs; kb0 += CHUNK_SLABS) {
        const int nk = min(CHUNK_SLABS, slabs - kb0);
        for (int i = tid; i < nk*32; i += 256) sx[i] = x[(size_t)kb0*32 + i];
        __syncthreads();

        for (int kb = kb0 + kseg; kb < kb0 + nk; kb += 4) {
            const float * xk = sx + (kb-kb0)*32;
            const uint8_t * wsla = wp + WPOL::HDR + (size_t)kb*WPOL::SLAB;
            acc += pxqa_dot32_f32_shared<WPOL>(wsla, row, xk, wa, book_u, sub_u);
            if constexpr (HAS_GATE) {
                const uint8_t * gsla = gp + GPOL::HDR + (size_t)kb*GPOL::SLAB;
                gacc += pxqa_dot32_f32_shared<GPOL>(gsla, row, xk, ga, book_g, sub_g);
            }
        }
        __syncthreads(); // sx may be overwritten by the next chunk
    }

    red[kseg*64 + row] = acc;
    if constexpr (HAS_GATE) red[256 + kseg*64 + row] = gacc;
    __syncthreads();

    if (kseg == 0) {
        float u = red[row] + red[64+row] + red[128+row] + red[192+row];
        if constexpr (HAS_GATE) {
            float g = red[256+row] + red[320+row] + red[384+row] + red[448+row];
            switch ((ggml_glu_op)a.glu_op) {
                case GGML_GLU_OP_SWIGLU: u *= ggml_cuda_op_silu_single(g); break;
                case GGML_GLU_OP_GEGLU: u *= ggml_cuda_op_gelu_single(g); break;
                case GGML_GLU_OP_SWIGLU_OAI: u = ggml_cuda_op_swiglu_oai_single(g,u); break;
                case GGML_GLU_OP_SWIGLU_CLAMP: u = ggml_cuda_op_swiglu_clamp_single(g,u,a.glu_limit); break;
                default: u *= g; break;
            }
        }
        const int r = (int)blockIdx.x*64 + row;
        if (r < (int)a.rows) {
            const size_t off = HAS_IDS ? (size_t)ch*a.ds1 + (size_t)token*a.ds2 + r
                : (size_t)sample*a.ds3 + (size_t)ch*a.ds2 + (size_t)token*a.ds1 + r;
            a.dst[off] = u;
        }
    }
}

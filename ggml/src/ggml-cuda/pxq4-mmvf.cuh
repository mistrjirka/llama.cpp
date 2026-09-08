// PXQ4 direct floating-point decode. Preserve the published codebook and F32
// activations instead of requantizing either operand into the DP4A representation.
// Panel constants are defined in pxq4-port.cuh; see LICENSE-PXA.
#pragma once

struct pxq4_mmvf_args {
    const uint8_t * weights;
    const uint8_t * gate;
    const char * acts;
    const int32_t * ids;
    char * dst;
    uint32_t K, rows, tokens, achannels, channels, wchannels, wsamples, samples;
    size_t wnb2, wnb3, anb1, anb2, anb3, dnb1, dnb2, dnb3, ids_stride;
    int glu_op;
    float glu_limit;
};

template<bool HAS_IDS, bool HAS_GATE, int R = 4>
__launch_bounds__(128, 2)
static __global__ void pxq4_mmvf_kernel(pxq4_mmvf_args a) {
    __shared__ float book[16], sub[16];
    const int lane=threadIdx.x, warp=threadIdx.y;
    if(warp==0 && lane<16) { book[lane]=pxq4_port_book[lane]; sub[lane]=pxq4_port_sub[lane]; }
    __syncthreads();
    const uint32_t row=blockIdx.x*(4*R)+warp*R+lane%R;
    if(row>=a.rows)return;
    const uint32_t ch=blockIdx.y, token=blockIdx.z%a.tokens, sample=blockIdx.z/a.tokens;
    const uint32_t ex=HAS_IDS ? a.ids[ch+token*a.ids_stride] : ch/(a.channels/a.wchannels);
    const uint32_t ac=HAS_IDS ? ch%a.achannels : ch;
    const uint32_t ws=HAS_IDS ? 0 : sample/(a.samples/a.wsamples);
    const char * xp=a.acts+(HAS_IDS ? token*a.anb2+ac*a.anb1 : sample*a.anb3+ac*a.anb2+token*a.anb1);
    const float * x=reinterpret_cast<const float*>(xp);
    const int slabs=a.K/32;
    const size_t off=ws*a.wnb3+ex*a.wnb2+(row/64)*(128u+size_t(slabs)*1088u);
    const uint8_t * wp=a.weights+off;
    const uint8_t * gp=HAS_GATE ? a.gate+off : nullptr;
    const int rr=row%64;
    const float anchor=__half2float(reinterpret_cast<const half*>(wp)[rr]);
    const float ganchor=HAS_GATE ? __half2float(reinterpret_cast<const half*>(gp)[rr]) : 0.f;
    float acc=0.f, gate=0.f;
    for(int kb=lane/R;kb<slabs;kb+=32/R) {
        const uint8_t * slab=wp+128+size_t(kb)*1088;
        const int scale=slab[rr];
        const uint4 q=*reinterpret_cast<const uint4*>(slab+64+16*rr);
        const uint32_t codes[4]={q.x,q.y,q.z,q.w};
        uint32_t gcodes[4]={}; int gs=0;
        if constexpr(HAS_GATE) {
            const uint8_t * g=gp+128+size_t(kb)*1088;
            const uint4 qg=*reinterpret_cast<const uint4*>(g+64+16*rr);
            gcodes[0]=qg.x;gcodes[1]=qg.y;gcodes[2]=qg.z;gcodes[3]=qg.w;gs=g[rr];
        }
#pragma unroll
        for(int m=0;m<4;++m) {
            const float eff=anchor*sub[(scale>>(4*(m/2)))&15];
            const float geff=HAS_GATE ? ganchor*sub[(gs>>(4*(m/2)))&15] : 0.f;
#pragma unroll
            for(int j=0;j<8;++j) {
                const float xv=x[kb*32+m*8+j];
                const float weight=eff*book[(codes[m]>>(4*j))&15];
                acc=fmaf(weight,xv,acc);
                if constexpr(HAS_GATE) {
                    const float gw=geff*book[(gcodes[m]>>(4*j))&15];
                    gate=fmaf(gw,xv,gate);
                }
            }
        }
    }
#pragma unroll
    for(int stride=16;stride>=R;stride/=2) {
        acc+=__shfl_xor_sync(0xffffffff,acc,stride);
        if constexpr(HAS_GATE)gate+=__shfl_xor_sync(0xffffffff,gate,stride);
    }
    if(lane<R) {
        if constexpr(HAS_GATE) {
            switch((ggml_glu_op)a.glu_op) {
                case GGML_GLU_OP_SWIGLU: acc*=ggml_cuda_op_silu_single(gate);break;
                case GGML_GLU_OP_GEGLU: acc*=ggml_cuda_op_gelu_single(gate);break;
                case GGML_GLU_OP_SWIGLU_OAI: acc=ggml_cuda_op_swiglu_oai_single(gate,acc);break;
                case GGML_GLU_OP_SWIGLU_CLAMP: acc=ggml_cuda_op_swiglu_clamp_single(gate,acc,a.glu_limit);break;
                default:acc*=gate;
            }
        }
        char *dp=a.dst+(HAS_IDS ? ch*a.dnb1+token*a.dnb2 : sample*a.dnb3+ch*a.dnb2+token*a.dnb1);
        reinterpret_cast<float*>(dp)[row]=acc;
    }
}

void ggml_cuda_pxq4_mmvf_launch(const ggml_tensor *w,const ggml_tensor *x,
        const ggml_tensor *ids,ggml_tensor *dst,const ggml_cuda_mm_fusion_args_host *fusion,cudaStream_t stream) {
    GGML_ASSERT(w->type==GGML_TYPE_PXQ4 && w->ne[0]>0 && w->ne[0]%32==0 && w->ne[1]>0 && w->ne[1]%64==0);
    GGML_ASSERT(w->nb[1]==ggml_row_size(GGML_TYPE_PXQ4,w->ne[0]) && w->nb[2]>=w->nb[1]*w->ne[1] && w->nb[3]>=w->nb[2]*w->ne[2]);
    GGML_ASSERT(x->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 && x->nb[0]==4 && dst->nb[0]==4);
    const bool fused=fusion && fusion->gate;
    if(fusion)GGML_ASSERT(fused && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
    if(fused)GGML_ASSERT(ggml_are_same_stride(w,fusion->gate));
    const uint32_t tokens=ids ? dst->ne[2] : dst->ne[1];
    const uint32_t channels=ids ? dst->ne[1] : dst->ne[2];
    pxq4_mmvf_args a{(const uint8_t*)w->data,fused?(const uint8_t*)fusion->gate->data:nullptr,
        (const char*)x->data,ids?(const int32_t*)ids->data:nullptr,(char*)dst->data,
        uint32_t(w->ne[0]),uint32_t(w->ne[1]),tokens,uint32_t(ids?x->ne[1]:x->ne[2]),channels,
        uint32_t(w->ne[2]),uint32_t(w->ne[3]),uint32_t(dst->ne[3]),w->nb[2],w->nb[3],
        x->nb[1],x->nb[2],x->nb[3],dst->nb[1],dst->nb[2],dst->nb[3],ids?ids->nb[1]/4:0,
        fusion?int(fusion->glu_op):int(GGML_GLU_OP_SWIGLU),fusion?fusion->glu_limit:0.f};
    const dim3 grid((a.rows+15)/16,channels,tokens*a.samples),block(32,4);
    if(ids) {
        if(fused)pxq4_mmvf_kernel<true,true><<<grid,block,0,stream>>>(a);
        else pxq4_mmvf_kernel<true,false><<<grid,block,0,stream>>>(a);
    } else {
        if(fused)pxq4_mmvf_kernel<false,true><<<grid,block,0,stream>>>(a);
        else pxq4_mmvf_kernel<false,false><<<grid,block,0,stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
}

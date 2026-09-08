// Kernel-level parity and timings. Build with benches/pxq4-v100/build_test.sh.
#include "../../ggml/src/ggml-cuda/pxq4.cu"
#include "../../ggml/src/ggml-cuda/pxq4-mmvq.cuh"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>

static float elapsed(std::function<void()> f) {
    cudaEvent_t t0,t1; CUDA_CHECK(cudaEventCreate(&t0));CUDA_CHECK(cudaEventCreate(&t1));
    for(int i=0;i<3;++i) f();
    CUDA_CHECK(cudaEventRecord(t0));for(int i=0;i<100;++i) f();CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1));
    CUDA_CHECK(cudaEventDestroy(t0));CUDA_CHECK(cudaEventDestroy(t1));return ms*10;
}
template<int R,int V> void coalesced(pxq4_mmvq_args a,bool fused) {
    dim3 grid((a.rows+R*4-1)/(R*4),a.channels,a.tokens),blk(32,4);
    if(fused) pxq4_mmvq_coalesced<true,true,R,V><<<grid,blk>>>(a);
    else      pxq4_mmvq_coalesced<true,false,R,V><<<grid,blk>>>(a);
}
int main() {
    std::mt19937 rng(90208);
    bool failed=false;
    for(auto sh : {std::pair<int,int>{2048,512}, {512,2048}}) for(int nt:{1,2,5,8}) for(bool fused:{false,true}) {
        int K=sh.first,M=sh.second,E=16,U=8,AC=K==512?U:1;
        size_t stride=(2+17*K/32)*M,wb=stride*E;
        std::vector<uint8_t>w(wb),g(wb);
        for(auto * b:{&w,&g}) {
            for(auto &v:*b)v=(uint8_t)rng();
            size_t ps=128+1088*(K/32);
            for(int e=0;e<E;++e)for(int p=0;p<M/64;++p)for(int r=0;r<64;++r) {
                half h=__float2half(.01f+(rng()%1000)*.0001f);
                memcpy(b->data()+e*stride+p*ps+2*r,&h,2);
            }
        }
        std::vector<block_q8_1>x(K/32*AC*nt);
        for(auto& b:x) { b.ds=make_half2(.01f,0.f);for(auto&v:b.qs)v=(int8_t)(rng()%255-127); }
        std::vector<int>ids(U*nt);
        for(int t=0;t<nt;++t) for(int u=0;u<U;++u)ids[t*U+u]=(t*3+u*5)%E;
        uint8_t *dw,*dg;block_q8_1 *dx;int *di;float *out;
        CUDA_CHECK(cudaMalloc(&dw,wb));CUDA_CHECK(cudaMalloc(&dg,wb));CUDA_CHECK(cudaMalloc(&dx,x.size()*sizeof(x[0])));
        CUDA_CHECK(cudaMalloc(&di,ids.size()*4));CUDA_CHECK(cudaMalloc(&out,M*U*nt*4));
        CUDA_CHECK(cudaMemcpy(dw,w.data(),wb,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dg,g.data(),wb,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx,x.data(),x.size()*sizeof(x[0]),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(di,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
        pxq4_mmvq_args a{dw,dg,dx,di,out,(unsigned)K,(unsigned)M,(unsigned)nt,(unsigned)AC,(unsigned)U,(unsigned)E,1,1,
            stride,wb,(unsigned)(K/32),(unsigned)(K/32*AC),(unsigned)(K/32*AC*nt),(unsigned)M,(unsigned)(M*U),(unsigned)(M*U*nt),(unsigned)U,(int)GGML_GLU_OP_SWIGLU,0};
        auto old=[&](){
            dim3 grid(M/2,U),blk(32,nt);
            if(fused)pxq4_mmvq_kernel<true,true,2><<<grid,blk>>>(dw,dg,dx,di,out,K,M,nt,AC,U,stride,wb,K/32,K/32*AC,K/32*AC*nt,M,M*U,M*U*nt,U,1,1,E,(int)GGML_GLU_OP_SWIGLU,0);
            else pxq4_mmvq_kernel<true,false,2><<<grid,blk>>>(dw,nullptr,dx,di,out,K,M,nt,AC,U,stride,wb,K/32,K/32*AC,K/32*AC*nt,M,M*U,M*U*nt,U,1,1,E,(int)GGML_GLU_OP_SWIGLU,0);
        };
        float us=elapsed(old);std::vector<float> ref(M*U*nt),got(ref.size());CUDA_CHECK(cudaMemcpy(ref.data(),out,ref.size()*4,cudaMemcpyDeviceToHost));
        float peak=0;for(auto v:ref)peak=std::max(peak,fabsf(v));
        printf("K=%d M=%d T=%d fused=%d old=%.2fus",K,M,nt,fused,us);
#define TRY(R,V) {auto fn=[&](){coalesced<R,V>(a,fused);};float t=elapsed(fn);CUDA_CHECK(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));float err=0;for(size_t i=0;i<ref.size();++i){if(!std::isfinite(got[i]))failed=true;err=std::max(err,fabsf(got[i]-ref[i]));} printf(" r%dv%d=%.2f(%.2gx)",R,V,t,err/std::max(peak,1e-6f));if(err/std::max(peak,1e-6f)>2e-5)failed=true;}
        TRY(1,2);TRY(2,2);TRY(4,2);TRY(1,4);TRY(2,4);TRY(4,4);
#undef TRY
        printf("\n");fflush(stdout);
        for(auto p:{(void*)dw,(void*)dg,(void*)dx,(void*)di,(void*)out})CUDA_CHECK(cudaFree(p));
    }
    printf("PARITY_%s\n",failed?"FAILED":"PASS");return failed?1:0;
}

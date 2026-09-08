// WMMA parity against direct FP32 accumulation of exactly FP16-snapped operands.
#include "../../../ggml/src/ggml-cuda/pxq4.cu"
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>
#include <algorithm>
static __global__ void reference(const half * w,const float * x,const int * ids,float * d,
        int K,int M,int T,int U,int AC) {
    int row=blockIdx.x*blockDim.x+threadIdx.x,token=blockIdx.y,slot=blockIdx.z;
    if(row>=M)return;
    int ex=ids[token*U+slot];float sum=0.f;
    for(int k=0;k<K;++k)sum=fmaf(__half2float(w[((size_t)ex*M+row)*K+k]),
        __half2float(__float2half_rn(x[((size_t)token*AC+slot%AC)*K+k])),sum);
    d[((size_t)token*U+slot)*M+row]=sum;
}
int main(int argc,char **argv) {
    std::mt19937 rng(18352);bool failed=false;
    for(int K:{32,96})for(int T:{9,33,257})for(int AC:{1,2})for(int E:{1,5,33,256,512}) {
        const int M=64,U=2;size_t stride=(2+17*K/32)*M,wb=stride*E;
        std::vector<unsigned char>w(wb);for(auto &v:w)v=rng()%256;
        for(int e=0;e<E;++e)for(int p=0;p<M/64;++p)for(int r=0;r<64;++r) {
            half h=__float2half(.01f+(rng()%1000)*.0001f);memcpy(w.data()+e*stride+p*(128+1088*K/32)+2*r,&h,2);
        }
        std::vector<float>x(K*T*AC);for(auto &v:x)v=(int(rng()%2001)-1000)*.001f;
        std::vector<int>ids(T*U);for(int t=0;t<T;++t)for(int u=0;u<U;++u)ids[t*U+u]=argc>1?0:(t%13<9?u%E:(t+u)%E);
        unsigned char *dw;float *dx,*d,*dr;half *wf;int *di,*map,*counts,*nt;pxq4_tile*tiles;
        int maxt=(T*U+31)/32+E;
        CUDA_CHECK(cudaMalloc(&dw,wb));CUDA_CHECK(cudaMalloc(&dx,x.size()*4));CUDA_CHECK(cudaMalloc(&di,ids.size()*4));
        CUDA_CHECK(cudaMalloc(&d,M*T*U*4));CUDA_CHECK(cudaMalloc(&dr,M*T*U*4));CUDA_CHECK(cudaMalloc(&wf,(size_t)M*K*E*2));
        CUDA_CHECK(cudaMalloc(&map,E*T*U*4));CUDA_CHECK(cudaMalloc(&counts,E*4));CUDA_CHECK(cudaMalloc(&nt,4));CUDA_CHECK(cudaMalloc(&tiles,maxt*sizeof(pxq4_tile)));
        CUDA_CHECK(cudaMemcpy(dw,w.data(),wb,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(di,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d,0xff,M*T*U*4)); // NaN catches missing scatter writes.
        pxq4_port_dequant_f16(dw,wf,M*E,K,0);
        reference<<<dim3(M/64,T,U),64>>>(wf,dx,di,dr,K,M,T,U,AC);
        pxq4_build_map<<<E,256>>>(di,U,T,U,map,counts);int threads=32;while(threads<E)threads*=2;pxq4_build_tiles<<<1,threads>>>(counts,E,tiles,nt);
        pxq4_grouped_wmma<<<dim3(M/64,maxt),256>>>(dw,(const char*)dx,d,map,tiles,nt,K,M,T,U,AC,stride,K*4,(size_t)K*AC*4,M*4,M*U*4);
        CUDA_CHECK(cudaDeviceSynchronize());std::vector<float>ref(M*T*U),got(ref.size());
        CUDA_CHECK(cudaMemcpy(ref.data(),dr,ref.size()*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(got.data(),d,ref.size()*4,cudaMemcpyDeviceToHost));
        float peak=0,err=0;double ss=0,se=0;for(size_t i=0;i<ref.size();++i){peak=std::max(peak,fabsf(ref[i]));if(!std::isfinite(got[i]))failed=true;err=std::max(err,fabsf(got[i]-ref[i]));ss+=ref[i]*ref[i];se+=(got[i]-ref[i])*(got[i]-ref[i]);}
        if(err/std::max(peak,1e-8f)>3e-5)failed=true;
        printf("K=%d T=%d AC=%d E=%d peak_rel=%.3g rms_rel=%.3g\n",K,T,AC,E,err/peak,sqrt(se/ss));fflush(stdout);
        for(auto p:{(void*)dw,(void*)dx,(void*)di,(void*)d,(void*)dr,(void*)wf,(void*)map,(void*)counts,(void*)nt,(void*)tiles})CUDA_CHECK(cudaFree(p));
    }
    puts(failed?"PREFILL_EDGE_FAIL":"PREFILL_EDGE_PASS");return failed?1:0;
}

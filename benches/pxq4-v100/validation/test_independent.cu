// Independent scalar CPU reference versus the exported production CUDA launcher.
// Uses scalar nibble extraction and double accumulation; no CUDA lookup/dot helpers.
#include "../../../ggml/src/ggml-cuda/pxq4.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <stdexcept>
#include <cstdlib>
static const double book[] = {-125,-93,-71,-53,-38,-25,-12,0,11,22,33,46,60,76,97,127};
static const double sub[] = {0x1.b7cp-3,0x1.36cp-2,0x1.72cp-2,0x1.a2cp-2,0x1.cccp-2,0x1.f3p-2,0x1.0bcp-1,0x1.1ep-1,0x1.304p-1,0x1.438p-1,0x1.58p-1,0x1.6ecp-1,0x1.888p-1,0x1.a64p-1,0x1.cacp-1,0x1.f9cp-1};
static double dot(const unsigned char *w,int r,int K,const block_q8_1 *x) {
    auto p=w+(r/64)*(128+1088*(K/32)); int rr=r%64;
    half ah;memcpy(&ah,p+rr*2,2);double anchor=__half2float(ah)/127.0, s=0;
    for(int k=0;k<K;++k) {
        auto b=p+128+(k/32)*1088;int j=k%32;
        int code=(b[64+rr*16+j/2]>>(4*(j%2)))&15;
        int scale=(b[rr]>>(j<16?0:4))&15;
        s+=anchor*sub[scale]*book[code]*__low2float(x[k/32].ds)*x[k/32].qs[j];
    } return s;
}
static double glu(double g,double u,int op) {
    switch(op) {
        case GGML_GLU_OP_SWIGLU:return u*g/(1+exp(-g));
        case GGML_GLU_OP_GEGLU:return u*.5*g*(1+tanh(.7978845608028654*g*(1+.044715*g*g)));
        case GGML_GLU_OP_SWIGLU_OAI:g=std::min(g,7.);u=std::max(-7.,std::min(u,7.));return g/(1+exp(-1.702*g))*(1+u);
        case GGML_GLU_OP_SWIGLU_CLAMP:g=std::min(g,2.);return std::clamp(u,-2.,2.)*g/(1+exp(-g));
        default:return g*u;
    }
}
template<class T>T *device(const std::vector<T>&v) {T*p;CUDA_CHECK(cudaMalloc(&p,v.size()*sizeof(T)));CUDA_CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(int argc,char **argv) {
    try {
    const bool compact=argc>1; std::mt19937 rng(81209);int cases=0;double worst=0.;
    for(bool routed:{true,false}) for(int K: (compact?std::vector<int>{96}:std::vector<int>{32,96,512,2048}))
    for(int T: (compact?std::vector<int>{1,8}:std::vector<int>{1,2,3,4,5,6,7,8})) for(int op:{-1,(int)GGML_GLU_OP_SWIGLU,(int)GGML_GLU_OP_GEGLU,(int)GGML_GLU_OP_SWIGLU_OAI,(int)GGML_GLU_OP_SWIGLU_CLAMP}) {
        // General dense broadcasts across channels AND samples. Routed activations use
        // both one shared channel and per-expert channels. Strides include padding.
        int M=128,E=routed?5:2,U=routed?3:4,AC=routed?(T%2?1:3):U,WS=routed?1:2,S=routed?1:4;
        size_t wn2=(2+17*K/32)*M+64,wn3=wn2*E+128,wb=wn3*WS;
        std::vector<unsigned char>w(wb),g(wb);for(auto*p:{&w,&g}) {
            for(auto&v:*p)v=rng()%256;
            for(int ss=0;ss<WS;++ss)for(int e=0;e<E;++e)for(int p0=0;p0<M/64;++p0)for(int r=0;r<64;++r){half h=__float2half((rng()%7==0)?0.f:(.05f+.002f*(rng()%100)));memcpy(p->data()+ss*wn3+e*wn2+p0*(128+1088*K/32)+2*r,&h,2);}
        }
        int KP=((K+511)/512)*512,as1=KP/32,as2=(routed?AC:T)*as1,as3=(routed?T:AC)*as2;
        std::vector<block_q8_1>x(as3*S);for(auto&v:x){v.ds=make_half2(.007f,0.f);for(auto&q:v.qs)q=(int(rng()%255)-127);}
        int is=U+5;std::vector<int>ids(is*T,-1);for(int t=0;t<T;++t)for(int u=0;u<U;++u)ids[t*is+u]=(t+u*2)%E;
        int ds1=M+7,ds2=ds1*(routed?U:T)+11,ds3=ds2*(routed?T:U)+13;
        std::vector<float>initial(ds3*S+64,-12345.f),got(initial.size());
        auto dw=device(w),dg=device(g);auto dx=device(x);auto di=device(ids);auto out=device(initial);
        ggml_tensor wt{},xt{},it{},dt{},gt{};wt.type=GGML_TYPE_PXQ4;wt.ne[0]=K;wt.ne[1]=M;wt.ne[2]=E;wt.ne[3]=WS;wt.nb[0]=17;wt.nb[1]=2+17*K/32;wt.nb[2]=wn2;wt.nb[3]=wn3;wt.data=dw;gt=wt;gt.data=dg;
        xt.type=GGML_TYPE_F32;xt.ne[0]=K;xt.ne[1]=routed?AC:T;xt.ne[2]=routed?T:AC;xt.ne[3]=S;
        dt.type=GGML_TYPE_F32;dt.ne[0]=M;dt.ne[1]=routed?U:T;dt.ne[2]=routed?T:U;dt.ne[3]=S;dt.nb[0]=4;dt.nb[1]=ds1*4;dt.nb[2]=ds2*4;dt.nb[3]=ds3*4;dt.data=out;
        it.type=GGML_TYPE_I32;it.ne[0]=U;it.ne[1]=T;it.nb[0]=4;it.nb[1]=is*4;it.data=di;
        ggml_cuda_mm_fusion_args_host fusion{};fusion.gate=&gt;fusion.glu_op=(ggml_glu_op)op;fusion.glu_limit=2.f;
        double peak=0,err=0;std::vector<unsigned char> written(got.size(),0);
        ggml_cuda_pxq4_mmvq_launch(&wt,&xt,routed?&it:nullptr,&dt,dx,KP,op<0?nullptr:&fusion,0);
        CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));
        for(int ss=0;ss<S;++ss)for(int t=0;t<T;++t)for(int u=0;u<U;++u)for(int r=0;r<M;++r) {
            int ex=routed?ids[t*is+u]:u/(U/E),ws=routed?0:ss/(S/WS);
            auto act=x.data()+(routed?t*as2+(u%AC)*as1:ss*as3+u*as2+t*as1);
            auto off=ws*wn3+ex*wn2;
            double ref=dot(w.data()+off,r,K,act);if(op>=0)ref=glu(dot(g.data()+off,r,K,act),ref,op);
            size_t i=routed?u*ds1+t*ds2+r:ss*ds3+u*ds2+t*ds1+r;written[i]=1;
            if(!std::isfinite(got[i]))throw std::runtime_error("non-finite output");
            peak=std::max(peak,fabs(ref));err=std::max(err,fabs(ref-got[i]));
        }
        for(size_t i=0;i<got.size();++i)if(!written[i]&&got[i]!=initial[i])throw std::runtime_error("padding overwritten");
        double rel=err/std::max(peak,1e-8);worst=std::max(worst,rel);if(rel>3e-5){printf("FAIL routed=%d K=%d T=%d op=%d rel=%g\n",routed,K,T,op,rel);return 1;}
        ++cases;for(void*p:{(void*)dw,(void*)dg,(void*)dx,(void*)di,(void*)out})CUDA_CHECK(cudaFree(p));
    }
    printf("INDEPENDENT_CPU_PASS cases=%d max_peak_relative_error=%.9g\n",cases,worst);return 0;
    }catch(const std::exception&e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}

// Regression: native=0 must not re-enter MMVQ via 2D expert views.
// Choose code 8, for which the published value differs from integer 11/127.
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
int main(int argc,char**argv) {
 try {
  const bool approximate=argc>1 && std::strcmp(argv[1],"--expect-approx")==0;
  double peak_error=0;int cases=0;
  for(bool routed:{false,true})for(int T:{1,8}) {
   constexpr int K=96,M=64,E=2,U=2;
   auto*c=ggml_init({1024*1024,nullptr,true});
   auto*w=routed?ggml_new_tensor_3d(c,GGML_TYPE_PXQ4,K,M,E):ggml_new_tensor_2d(c,GGML_TYPE_PXQ4,K,M);
   auto*x=routed?ggml_new_tensor_3d(c,GGML_TYPE_F32,K,1,T):ggml_new_tensor_2d(c,GGML_TYPE_F32,K,T);
   auto*ids=routed?ggml_new_tensor_2d(c,GGML_TYPE_I32,U,T):nullptr;
   auto*y=routed?ggml_mul_mat_id(c,w,x,ids):ggml_mul_mat(c,w,x);
   auto*g=ggml_new_graph(c);ggml_build_forward_expand(g,y);
   auto*backend=ggml_backend_cuda_init(0);if(!backend)throw std::runtime_error("CUDA initialization");
   auto*buf=ggml_backend_alloc_ctx_tensors(c,backend);if(!buf)throw std::runtime_error("allocation");
   std::vector<unsigned char>weights(ggml_nbytes(w),0x88);
   for(int e=0;e<(routed?E:1);++e)for(int r=0;r<M;++r){size_t off=e*w->nb[2]+2*r;weights[off]=0;weights[off+1]=0x3c;}
   std::vector<float>input(ggml_nelements(x),1.f),out(ggml_nelements(y));
   ggml_backend_tensor_set(w,weights.data(),0,weights.size());ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
   if(ids){std::vector<int>v(U*T);for(int i=0;i<U*T;++i)v[i]=i%U;ggml_backend_tensor_set(ids,v.data(),0,v.size()*4);}
   if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("graph execution");
   ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
   const double expected=K*0x1.304p-1*0x1.5bp-4;
   double error=0;for(float v:out){if(!std::isfinite(v))throw std::runtime_error("nonfinite");error=std::max(error,std::abs(v-expected));}
   printf("routed=%d T=%d expected=%.10g actual=%.10g max_error=%.10g\n",routed,T,expected,out[0],error);
   if((!approximate && error>2e-5)||(approximate && error<1e-3))throw std::runtime_error("wrong dispatch/precision");
   peak_error=std::max(peak_error,error);++cases;
   ggml_backend_buffer_free(buf);ggml_backend_free(backend);ggml_free(c);
  }
  printf("REFERENCE_DISPATCH_PASS cases=%d approximate_control=%d max_abs_error=%.10g\n",cases,approximate,peak_error);
  return 0;
 }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}

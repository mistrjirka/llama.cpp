// Compare the fused router with the graph's individual operations on identical
// inputs. The reference exposes probabilities as an extra output solely to
// prevent that test graph from fusing; the candidate retains optimized routing.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>
static void require(bool ok,const char * why) { if(!ok)throw std::runtime_error(why); }
struct Router {
 ggml_context *ctx=nullptr; ggml_cgraph *graph=nullptr; ggml_tensor *in=nullptr,*out=nullptr,*ids=nullptr;
 ggml_gallocr_t alloc=nullptr;
 Router(ggml_backend_t backend,int n,bool separate,bool alias=false) {
  ctx=ggml_init({2*1024*1024,nullptr,true});require(ctx,"context");
  in=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,512,n);ggml_set_input(in);
  auto *x=ggml_scale(ctx,in,1.0f);ggml_set_name(x,separate?"reference_logits":(alias?"alias_logits":"direct_logits"));
  if(!alias)ggml_set_output(x);
  auto *p=ggml_soft_max(ctx,x);if(separate)ggml_set_output(p);
  auto *top=ggml_argsort_top_k(ctx,p,10);
  auto *w=ggml_get_rows(ctx,ggml_reshape_3d(ctx,p,1,512,n),top);
  w=ggml_reshape_2d(ctx,w,10,n);
  auto *den=ggml_clamp(ctx,ggml_sum_rows(ctx,w),6.103515625e-5f,INFINITY);
  out=ggml_cont(ctx,ggml_reshape_3d(ctx,ggml_div(ctx,w,den),1,10,n));ggml_set_output(out);
  ids=ggml_cont(ctx,top);ggml_set_output(ids);
  graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);ggml_build_forward_expand(graph,ids);
  // Real graph allocation exercises direct and intentionally reusable input storage.
  alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));require(ggml_gallocr_alloc_graph(alloc,graph),"graph allocation");
 }
 ~Router(){if(alloc)ggml_gallocr_free(alloc);if(ctx)ggml_free(ctx);}
};
int main(){try{
 ggml_backend_load_all();int cases=0;size_t different=0;double worst=0,oracle_worst=0;
 const bool strict=std::getenv("ROUTER_REQUIRE_EXACT")!=nullptr;
 for(size_t di=0;di<ggml_backend_dev_count();++di){
  auto *dev=ggml_backend_dev_get(di);if(ggml_backend_dev_type(dev)!=GGML_BACKEND_DEVICE_TYPE_GPU)continue;
  auto *b=ggml_backend_dev_init(dev,nullptr);require(b,"backend");
  for(int n:{1,7,8,9,33,128,257,512}){
   Router fused(b,n,false),aliased(b,n,false,true),ref(b,n,true);std::vector<float>x(512*n),a(10*n),z(10*n),alias_values(10*n);std::vector<int>ia(10*n),iz(10*n),alias_ids(10*n),order(512);
   size_t unequal=0;double maximum=0;
   for(int pass=0;pass<4;++pass){
    for(int r=0;r<n;++r)for(int e=0;e<512;++e){
     unsigned v=(unsigned)(r*65537+e*8191+pass*104729+123);v^=v<<13;v^=v>>17;v^=v<<5;
     x[r*512+e]=(float(int(v%2000001)-1000000)/100000.f)*(pass==2?0.01f:1.0f);
    }
    for(auto *g:{&fused,&aliased,&ref}){ggml_backend_tensor_set(g->in,x.data(),0,x.size()*4);require(ggml_backend_graph_compute(b,g->graph)==GGML_STATUS_SUCCESS,"compute");ggml_backend_synchronize(b);}
    ggml_backend_tensor_get(fused.out,a.data(),0,a.size()*4);ggml_backend_tensor_get(ref.out,z.data(),0,z.size()*4);
    ggml_backend_tensor_get(fused.ids,ia.data(),0,ia.size()*4);ggml_backend_tensor_get(ref.ids,iz.data(),0,iz.size()*4);
    ggml_backend_tensor_get(aliased.out,alias_values.data(),0,alias_values.size()*4);
    ggml_backend_tensor_get(aliased.ids,alias_ids.data(),0,alias_ids.size()*4);
    require(alias_values==a && alias_ids==ia,"aliasing changed optimized routing outputs");
    for(size_t i=0;i<a.size();++i){require(ia[i]==iz[i],"expert IDs differ");require(std::isfinite(a[i])&&std::isfinite(z[i]),"nonfinite");double d=std::abs(double(a[i])-z[i]);maximum=std::max(maximum,d);unequal+=d!=0;}
    for(int r=0;r<n;++r){
     double sum=0,m=x[r*512+ia[r*10]];
     for(int k=0;k<10;++k)sum+=std::exp(double(x[r*512+ia[r*10+k]])-m);
     for(int k=0;k<10;++k){double target=std::exp(double(x[r*512+ia[r*10+k]])-m)/sum;double err=std::abs(a[r*10+k]-target);oracle_worst=std::max(oracle_worst,err);require(err<1e-6,"oracle discrepancy");}
    }
   }
   different+=unequal;worst=std::max(worst,maximum);++cases;
   std::printf("ARITHMETIC backend=%s rows=%d unequal=%zu max_abs=%.9g\n",ggml_backend_name(b),n,unequal,maximum);
   if(n==512 && std::getenv("ROUTER_BENCHMARK")){
    for(auto *g:{&ref,&fused}){
     constexpr int repeats=100;auto start=std::chrono::steady_clock::now();
     for(int i=0;i<repeats;++i)require(ggml_backend_graph_compute(b,g->graph)==GGML_STATUS_SUCCESS,"timing");
     ggml_backend_synchronize(b);double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
     std::printf("ROUTER_TIME backend=%s mode=%s rows=%d mean_us=%.3f\n",ggml_backend_name(b),g==&ref?"separate":"fused",n,us);
    }
   }
  }
  ggml_backend_free(b);
 }
 if(!cases){std::puts("SKIP: GPU backend required");return 77;}std::printf("SUMMARY cases=%d unequal=%zu max_abs=%.9g oracle_max=%.9g\n",cases,different,worst,oracle_worst);
 require(!strict||different==0,"fused arithmetic not identical");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"router equivalence: %s\n",e.what());return 1;}}

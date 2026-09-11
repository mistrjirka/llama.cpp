#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct TInfo { ggml_type type; int64_t ne0, ne1; size_t data_off, tensor_off, row_bytes; };
static TInfo info(const std::string & path, const char * name) {
    ggml_context * ctx_data=nullptr;
    gguf_init_params p{}; p.no_alloc=true; p.ctx=&ctx_data;
    gguf_context * g=gguf_init_from_file(path.c_str(), p);
    if (!g) throw std::runtime_error("gguf open");
    int64_t id=gguf_find_tensor(g,name); if(id<0) throw std::runtime_error("tensor missing");
    ggml_tensor * t=ggml_get_tensor(ctx_data,name); if(!t) throw std::runtime_error("tensor meta missing");
    TInfo x{t->type,t->ne[0],t->ne[1],gguf_get_data_offset(g),gguf_get_tensor_offset(g,id),ggml_row_size(t->type,t->ne[0])};
    gguf_free(g); ggml_free(ctx_data); return x;
}
static std::vector<float> row(const std::string & path,const TInfo & t,int64_t r){
    std::ifstream f(path,std::ios::binary); std::vector<unsigned char> q(t.row_bytes); std::vector<float> y(t.ne0);
    f.seekg((std::streamoff)(t.data_off+t.tensor_off+r*t.row_bytes)); f.read((char*)q.data(),q.size()); if(!f) throw std::runtime_error("read");
    auto * tr=ggml_get_type_traits(t.type); if(!tr || !tr->to_float) throw std::runtime_error("no dequant"); tr->to_float(q.data(),y.data(),t.ne0); return y;
}
int main(int argc,char**argv){ if(argc!=4){fprintf(stderr,"usage target draft tensor\n");return 2;} std::string a=argv[1],b=argv[2],n=argv[3]; auto A=info(a,n.c_str()),B=info(b,n.c_str());
 printf("%s: A=%s %ldx%ld row=%zu B=%s %ldx%ld row=%zu\n",n.c_str(),ggml_type_name(A.type),A.ne0,A.ne1,A.row_bytes,ggml_type_name(B.type),B.ne0,B.ne1,B.row_bytes);
 if(A.ne0!=B.ne0||A.ne1!=B.ne1)return 3; std::vector<int64_t> rs={0,1,7,31,127,1023,A.ne1/2,A.ne1-1}; double se=0,sa=0,sb=0; double maxe=0,maxa=0; long long cnt=0;
 for(auto r:rs){if(r<0||r>=A.ne1)continue;auto x=row(a,A,r),y=row(b,B,r);double rse=0,rae=0,rmax=0;for(int64_t i=0;i<A.ne0;i++){double e=x[i]-y[i];rse+=e*e;rae+=std::abs(e);rmax=std::max(rmax,std::abs(e));se+=e*e;sa+=x[i]*x[i];sb+=y[i]*y[i];maxe=std::max(maxe,std::abs(e));maxa=std::max(maxa,std::max(std::abs((double)x[i]),std::abs((double)y[i])));cnt++;}printf(" row %ld rmse=%.8g mae=%.8g max=%.8g\n",r,std::sqrt(rse/A.ne0),rae/A.ne0,rmax);} printf("ALL n=%lld rmse=%.8g rel_rmse(A)=%.6g max_abs=%.8g max_val=%.8g\n",cnt,std::sqrt(se/cnt),std::sqrt(se/std::max(sa,1e-30)),maxe,maxa); }

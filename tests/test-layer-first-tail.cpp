#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(bool ok,const char * what) { if (!ok) throw std::runtime_error(what); }
static void test(ggml_backend_t backend,ggml_type type,int original_rows) {
    // A real K dimension is essential: width 256 hides the scalar/multi-row
    // reduction difference because its inner loop is too short.
    constexpr int width=2560,height=64,experts=4,k=4;
    ggml_init_params init={8*1024*1024,nullptr,true};
    auto * ctx=ggml_init(init);check(ctx,"tensor context");
    auto * weights=ggml_new_tensor_3d(ctx,type,width,height,experts);
    auto * input=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,1,original_rows);
    auto * ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,k,original_rows);
    auto * reference=ggml_mul_mat_id(ctx,weights,input,ids);
    // One real assignment is repeated to preserve its chronological batch's
    // vector-kernel geometry. Only the first output column is consumed.
    auto * one_input=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,1,original_rows);
    auto * one_id=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,1,original_rows);
    auto * candidate=ggml_mul_mat_id(ctx,weights,one_input,one_id);
    candidate->op_params[0]=0x4c464d51;
    candidate->op_params[1]=original_rows;
    auto * ref_graph=ggml_new_graph(ctx);ggml_build_forward_expand(ref_graph,reference);
    auto * one_graph=ggml_new_graph(ctx);ggml_build_forward_expand(one_graph,candidate);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"backend allocation");
    ggml_backend_buffer_clear(buffer,0);
    std::vector<float> w(width*height*experts),x(width*original_rows),expected(height*k*original_rows),actual(height);
    std::vector<int32_t> routes(k*original_rows);
    for(size_t i=0;i<w.size();++i)w[i]=0.25f*std::sin(float(i)*0.013f);
    std::vector<unsigned char> packed(ggml_nbytes(weights));
    ggml_quantize_init(type);
    const auto bytes=ggml_quantize_chunk(type,w.data(),packed.data(),0,height*experts,width,nullptr);
    check(bytes==packed.size(),"quantized weight size");
    ggml_backend_tensor_set(weights,packed.data(),0,bytes);
    double worst=0;
    std::vector<float> repeated(width*original_rows);
    std::vector<int32_t> repeated_id(original_rows);
    for(int pass=0;pass<2;++pass) {
        for(size_t i=0;i<x.size();++i)x[i]=std::cos(float(i+pass*101)*0.029f);
        for(int t=0;t<original_rows;++t)for(int j=0;j<k;++j)routes[t*k+j]=(t+j+pass)%experts;
        ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));
        ggml_backend_tensor_set(ids,routes.data(),0,routes.size()*sizeof(int32_t));
        check(ggml_backend_graph_compute(backend,ref_graph)==GGML_STATUS_SUCCESS,"reference compute");
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(reference,expected.data(),0,expected.size()*sizeof(float));
        for(int t=0;t<original_rows;++t)for(int j=0;j<k;++j) {
            for(int q=0;q<original_rows;++q)std::copy_n(x.data()+t*width,width,repeated.data()+q*width);
            std::fill(repeated_id.begin(),repeated_id.end(),routes[t*k+j]);
            ggml_backend_tensor_set(one_input,repeated.data(),0,repeated.size()*sizeof(float));
            ggml_backend_tensor_set(one_id,repeated_id.data(),0,repeated_id.size()*sizeof(int32_t));
            check(ggml_backend_graph_compute(backend,one_graph)==GGML_STATUS_SUCCESS,"assignment compute");
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(candidate,actual.data(),0,height*sizeof(float));
            for(int c=0;c<height;++c) {
                const double error=std::abs(double(actual[c])-expected[(t*k+j)*height+c]);
                worst=std::max(worst,error);
                check(std::isfinite(actual[c]) && error==0.0,"small assignment arithmetic differs");
            }
        }
    }
    std::printf("PASS backend=%s type=%s original_rows=%d max_abs=%.9g\n",ggml_backend_name(backend),ggml_type_name(type),original_rows,worst);
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
}
int main() { try {
    ggml_backend_load_all();int cases=0;
    for(size_t i=0;i<ggml_backend_dev_count();++i) {
        auto * dev=ggml_backend_dev_get(i);if(ggml_backend_dev_type(dev)!=GGML_BACKEND_DEVICE_TYPE_GPU)continue;
        auto backend=ggml_backend_dev_init(dev,nullptr);check(backend,"GPU init");
        for(auto type:{GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL})for(int rows:{1,2,3,4,5,6,7,8}){test(backend,type,rows);++cases;}
        ggml_backend_free(backend);
    }
    if (cases==0) { std::puts("SKIP: CUDA GPU backend required");return 77; }
    std::printf("PASS %d shape/type/device cases with changing inputs\n",cases);return 0;
} catch(const std::exception & e) { std::fprintf(stderr,"tail test: %s\n",e.what());return 1; }}

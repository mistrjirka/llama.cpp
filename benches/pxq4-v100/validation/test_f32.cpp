#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include <vector>
#include <cmath>
#include <cstdio>
int main(){
 auto ctx=ggml_init({1024*1024,nullptr,true});auto*w=ggml_new_tensor_2d(ctx,GGML_TYPE_PXQ4,96,64);auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,96,9);auto*y=ggml_mul_mat(ctx,w,x);ggml_mul_mat_set_prec(y,GGML_PREC_F32);
 auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);auto backend=ggml_backend_cuda_init(0);if(!backend)return 2;
 auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)return 3;
 std::vector<unsigned char> weights(ggml_nbytes(w),0xff);
 for(int k=0;k<3;++k)for(int r=0;r<64;++r)weights[128+k*1088+r]=0x88;for(int r=0;r<64;++r){weights[r*2]=0;weights[r*2+1]=0x3c;}
 std::vector<float> input(96*9,1),output(64*9);
 ggml_backend_tensor_set(w,weights.data(),0,weights.size());ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
 auto status=ggml_backend_graph_compute(backend,graph);ggml_backend_tensor_get(y,output.data(),0,output.size()*4);
 for(float v:output)if(!std::isfinite(v)||fabs(v-96*0x1.304p-1)>1e-5)return 4;
 printf("F32_FALLBACK_PASS status=%d\n",int(status));ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_free(ctx);return status==GGML_STATUS_SUCCESS?0:5;
}

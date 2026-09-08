#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include <cstdio>
int main(){
 auto c=ggml_init({1024*1024,nullptr,true});auto*w=ggml_new_tensor_2d(c,GGML_TYPE_PXQ4,512,256);auto*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,512,1);
 auto*bad=ggml_view_2d(c,w,512,64,w->nb[1],w->nb[1]);auto*good=ggml_view_2d(c,w,512,64,w->nb[1],w->nb[1]*64);
 auto backend=ggml_backend_cuda_init(0);if(!backend)return 2;
 int bad_accepted=ggml_backend_supports_op(backend,ggml_mul_mat(c,bad,x));int good_accepted=ggml_backend_supports_op(backend,ggml_mul_mat(c,good,x));
 printf("unaligned_panel_view_accepted=%d aligned_panel_view_accepted=%d\n",bad_accepted,good_accepted);
 ggml_backend_free(backend);ggml_free(c);return bad_accepted||!good_accepted;
}

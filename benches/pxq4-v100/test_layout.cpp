#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <initializer_list>
#include <cstdlib>
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"FAILED line %d: %s\n",__LINE__,#x);return 1;} }while(0)
int main() {
    ggml_init_params p{1024*1024,nullptr,true};auto * c=ggml_init(p);CHECK(c);
    for(ggml_type type : {GGML_TYPE_Q4_0,GGML_TYPE_Q8_0,GGML_TYPE_Q5_K,GGML_TYPE_PXQ4}) {
        auto *t=ggml_new_tensor_3d(c,type,2048,512,16);
        size_t row=ggml_type_size(type)*2048/ggml_blck_size(type)+(type==GGML_TYPE_PXQ4?2:0);
        CHECK(t->nb[1]==row);CHECK(t->nb[2]==row*512);
        CHECK(ggml_nbytes(t)==row*512*16);CHECK(ggml_is_contiguous(t));CHECK(ggml_is_contiguously_allocated(t));
        auto *v=ggml_view_2d(c,t,2048,64,row,64*row);
        CHECK(ggml_nbytes(v)==row*64);CHECK(ggml_is_contiguous(v));
    }
    auto *w=ggml_new_tensor_2d(c,GGML_TYPE_PXQ4,2048,64);
    auto *x=ggml_new_tensor_2d(c,GGML_TYPE_F32,2048,1);
    auto *op=ggml_mul_mat(c,w,x);auto cpu=ggml_backend_cpu_init();
    CHECK(!ggml_backend_supports_op(cpu,op));ggml_backend_free(cpu);ggml_free(c);
    puts("LAYOUT_AND_CPU_GUARD_PASS");return 0;
}

#pragma once
// Experimental rare-expert side branch. It computes every selected contribution;
// acceptance remains a full-model numerical comparison, not just finite outputs.
#include "ggml.h"
#include "llama-layer-first-cpu-mmq.h"
#include <cstdlib>
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <stdexcept>

struct llama_lf_cpu_task {size_t expert;std::vector<size_t> slots;size_t output_offset=0;};
struct llama_lf_cpu_result {std::vector<float> values;double ms=0;};
struct llama_lf_cpu_plan {
    ggml_context * ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;
    std::array<ggml_tensor *,3> weights={};ggml_tensor * x=nullptr,*out=nullptr;ggml_cgraph * graph=nullptr;
    llama_lf_cpu_plan(ggml_backend_t b,const std::array<ggml_tensor *,3> & src,size_t rows,float clamp) {
        ctx=ggml_init({2*1024*1024,nullptr,true});if(!ctx)throw std::runtime_error("CPU expert graph context");
        for(size_t j=0;j<3;++j) {
            weights[j]=ggml_new_tensor_2d(ctx,src[j]->type,src[j]->ne[0],src[j]->ne[1]);
            weights[j]->buffer=src[j]->buffer;weights[j]->data=src[j]->data;
        }
        x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,src[0]->ne[0],rows);
        auto * up=ggml_mul_mat(ctx,weights[0],x),*gate=ggml_mul_mat(ctx,weights[1],x);
        auto * act=clamp>1e-6f ? ggml_mul(ctx,ggml_clamp(ctx,ggml_silu(ctx,gate),-INFINITY,clamp),ggml_clamp(ctx,up,-clamp,clamp)) : ggml_swiglu_split(ctx,gate,up);
        out=ggml_mul_mat(ctx,weights[2],act);graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,b);if(!buffer)throw std::runtime_error("CPU expert graph allocation");
    }
    ~llama_lf_cpu_plan(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
};
inline llama_lf_cpu_result llama_lf_cpu_run(const std::array<ggml_tensor *,3> & weights,
        const std::vector<llama_lf_cpu_task> & tasks,const std::vector<float> & input,
        size_t width,size_t topk,size_t count,int workers,float clamp) {
    const auto start=std::chrono::steady_clock::now();
    llama_lf_cpu_result result;result.values.resize(count*width);
    std::atomic<size_t> next{0};std::exception_ptr failure;std::mutex failure_mutex;
    auto worker=[&] {
        ggml_backend_t backend=nullptr;
        try {
            backend=ggml_backend_cpu_init();if(!backend)throw std::runtime_error("CPU worker backend");
            ggml_backend_cpu_set_n_threads(backend,1);
            std::map<size_t,std::unique_ptr<llama_lf_cpu_plan>> plans;
            while(true) {
                const size_t index=next.fetch_add(1);if(index>=tasks.size())break;
                const auto & task=tasks[index];const size_t rows=task.slots.size();
                const char * method=std::getenv("LLAMA_MOE_LAYER_FIRST_CPU_MMQ");
                if(method && method[0]=='1') {
                    for(size_t j=0;j<rows;++j) {
                        const float * v=input.data()+(task.slots[j]/topk)*width;
                        std::vector<float> x(v,v+width);
                        auto mm=[&](size_t p,const std::vector<float>& a) {
                            auto * w=weights[p];
                            return lf_cpu_mmq(w->type,static_cast<const char *>(w->data)+task.expert*w->nb[2],w->ne[0],w->ne[1],a);
                        };
                        auto up=mm(0,x),gate=mm(1,x);
                        for(size_t q=0;q<gate.size();++q) {
                            float g=gate[q]/(1.0f+std::exp(-gate[q]));
                            if(clamp>1e-6f) {g=std::min(g,clamp);up[q]=std::max(-clamp,std::min(up[q],clamp));}
                            gate[q]=g*up[q];
                        }
                        auto y=mm(2,gate);
                        for(size_t q=0;q<width;++q) {
                            if(!std::isfinite(y[q]))throw std::runtime_error("CPU MMQ nonfinite result");
                            result.values[(task.output_offset+j)*width+q]=y[q];
                        }
                    }
                    continue;
                }
                auto & plan=plans[rows];if(!plan)plan=std::make_unique<llama_lf_cpu_plan>(backend,weights,rows,clamp);
                for(size_t j=0;j<3;++j)plan->weights[j]->data=static_cast<char *>(weights[j]->data)+task.expert*weights[j]->nb[2];
                float * x=static_cast<float *>(plan->x->data);
                for(size_t j=0;j<rows;++j)std::copy_n(input.data()+(task.slots[j]/topk)*width,width,x+j*width);
                if(ggml_backend_graph_compute(backend,plan->graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("CPU expert computation");
                auto * y=static_cast<float *>(plan->out->data);
                for(size_t j=0;j<rows*width;++j) {
                    if(!std::isfinite(y[j]))throw std::runtime_error("CPU expert nonfinite result");
                    result.values[task.output_offset*width+j]=y[j];
                }
            }
            plans.clear();ggml_backend_free(backend);backend=nullptr;
        } catch(...) {if(backend)ggml_backend_free(backend);std::lock_guard<std::mutex> lock(failure_mutex);if(!failure)failure=std::current_exception();}
    };
    std::vector<std::thread> threads;
    try {for(int i=0;i<std::min<int>(workers,tasks.size());++i)threads.emplace_back(worker);}
    catch(...) {for(auto & t:threads)t.join();throw;}
    for(auto & t:threads)t.join();
    if(failure)std::rethrow_exception(failure);
    result.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();return result;
}

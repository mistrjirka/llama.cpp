#pragma once
// Opt-in diagnostic only. Freezes a real router input, then checks that the
// fused GPU result agrees with its mathematical softmax/top-k/normalization.
// The synchronization changes timing: this is an arithmetic check, not a race
// detector or a throughput measurement. No outputs are rewritten.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

struct ggml_cuda_topk_audit {
    ggml_backend_cuda_context & ctx;
    const ggml_tensor * logits;
    const ggml_tensor * weights;
    const ggml_tensor * ids;
    const char * prefix;
    int n, e, k;
    float clamp, scale;
    bool enabled;
    std::vector<float> input;
    ggml_cuda_topk_audit(ggml_backend_cuda_context & c, const ggml_tensor * l,
            const ggml_tensor * w, const ggml_tensor * i, bool eligible,
            float clamp_value, float scale_value)
        : ctx(c),logits(l),weights(w),ids(i),prefix(std::getenv("GGML_CUDA_DIAG_ROUTER_AUDIT_PREFIX")),
          n(int(l->ne[1])),e(int(l->ne[0])),k(int(w->ne[1])),clamp(clamp_value),scale(scale_value),
          enabled(prefix && *prefix && eligible && n>1 && n<=512 && e==512) {
        if (!enabled) return;
        input.resize(size_t(n)*e);
        CUDA_CHECK(cudaMemcpyAsync(input.data(),l->data,input.size()*sizeof(float),cudaMemcpyDeviceToHost,ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    }
    void finish() {
        if (!enabled) return;
        static std::atomic<unsigned long long> sequence{0};
        const auto call=sequence.fetch_add(1);
        std::vector<float> actual(size_t(n)*k);
        std::vector<int32_t> selected(size_t(n)*e);
        CUDA_CHECK(cudaMemcpyAsync(actual.data(),weights->data,actual.size()*sizeof(float),cudaMemcpyDeviceToHost,ctx.stream()));
        CUDA_CHECK(cudaMemcpyAsync(selected.data(),ids->data,ggml_nbytes(ids),cudaMemcpyDeviceToHost,ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        size_t invalid=0,bad_weights=0,wrong_rank=0,ambiguous=0;
        double maximum=0;
        std::vector<int> order(e);std::vector<double> p(e);
        for (int r=0;r<n;++r) {
            std::iota(order.begin(),order.end(),0);
            bool finite=true;for(int j=0;j<e;++j)finite &= std::isfinite(input[size_t(r)*e+j]);
            if(!finite){++invalid;continue;}
            std::sort(order.begin(),order.end(),[&](int a,int b){
                const auto x=input[size_t(r)*e+a],y=input[size_t(r)*e+b];return x>y || (x==y && a<b);
            });
            const double m=input[size_t(r)*e+order[0]];double z=0;
            for(int j=0;j<e;++j){p[j]=std::exp(double(input[size_t(r)*e+j])-m);z+=p[j];}
            for(auto & v:p)v/=z;
            double denom=0;std::vector<unsigned char> seen(e,0);bool valid=true;
            for(int j=0;j<k;++j){const int id=selected[size_t(r)*e+j];if(id<0 || id>=e || seen[id]++){++invalid;valid=false;continue;}denom+=p[id];}
            if(!valid)continue;
            denom=std::max(denom,double(clamp));
            for(int j=0;j<k;++j){
                const int id=selected[size_t(r)*e+j];
                const double expected=p[id]/denom*double(scale);
                const double err=std::abs(double(actual[size_t(r)*k+j])-expected);
                if(!std::isfinite(actual[size_t(r)*k+j]) || err>2e-6)++bad_weights;
                maximum=std::max(maximum,err);
                if(id!=order[j]){const double gap=std::abs(double(input[size_t(r)*e+id])-input[size_t(r)*e+order[j]]);if(gap>2e-5)++wrong_rank;else ++ambiguous;}
            }
        }
        std::fprintf(stderr,"TOPK_AUDIT call=%llu tensor=%s rows=%d experts=%d k=%d invalid=%zu bad_weights=%zu wrong_rank=%zu ambiguous=%zu max_weight_error=%.12g\n",
            call,logits->name,n,e,k,invalid,bad_weights,wrong_rank,ambiguous,maximum);
        if(call<4 || invalid || bad_weights || wrong_rank) {
            const std::string path=std::string(prefix)+"-"+std::to_string(call)+"-logits.f32";
            if(auto * f=std::fopen(path.c_str(),"wb")){std::fwrite(input.data(),sizeof(float),input.size(),f);std::fclose(f);}
        }
    }
};

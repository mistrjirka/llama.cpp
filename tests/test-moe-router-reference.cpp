// Isolated router diagnostic: exact, well-separated expert IDs and a double
// precision softmax/normalization reference. No model sampling or cache involved.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static void test(ggml_backend_t backend, int rows, int variant) {
    constexpr int E=512, K=10;
    auto * ctx=ggml_init({2*1024*1024,nullptr,true}); require(ctx,"context allocation");
    auto * input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,E,rows); ggml_set_input(input);
    auto * logits=ggml_scale(ctx,input,1.0f);
    auto * probs=ggml_soft_max(ctx,logits);
    auto * ids=ggml_argsort_top_k(ctx,probs,K);
    auto * weights=ggml_get_rows(ctx,ggml_reshape_3d(ctx,probs,1,E,rows),ids);
    weights=ggml_reshape_2d(ctx,weights,K,rows);
    auto * sum=ggml_clamp(ctx,ggml_sum_rows(ctx,weights),6.103515625e-5f,INFINITY);
    weights=ggml_div(ctx,weights,sum);
    weights=ggml_reshape_3d(ctx,weights,1,K,rows); ggml_set_output(weights);
    auto * packed_ids=ggml_cont(ctx,ids); ggml_set_output(packed_ids);
    auto * graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,weights); ggml_build_forward_expand(graph,packed_ids);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(alloc,graph),"graph allocation");
    std::vector<float> values(E*rows),actual(K*rows);
    std::vector<int32_t> actual_ids(K*rows), order(E);
    double max_error=0; int cases=0;
    for (int pass=0;pass<4;++pass) {
        for (int r=0;r<rows;++r) for(int e=0;e<E;++e) {
            const int permutation=(e*73+r*19+pass*47)%E;
            values[r*E+e]=(permutation-256)*(variant==0 ? 0.03125f : 0.25f);
        }
        ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"router compute");
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(weights,actual.data(),0,actual.size()*sizeof(float));
        ggml_backend_tensor_get(packed_ids,actual_ids.data(),0,actual_ids.size()*sizeof(int32_t));
        for(int r=0;r<rows;++r) {
            std::iota(order.begin(),order.end(),0);
            std::sort(order.begin(),order.end(),[&](int a,int b){
                return values[r*E+a]>values[r*E+b] || (values[r*E+a]==values[r*E+b] && a<b);
            });
            const double max=values[r*E+order[0]];
            double all_sum=0,selected_sum=0;
            for(int e=0;e<E;++e) all_sum+=std::exp(double(values[r*E+e])-max);
            for(int k=0;k<K;++k) selected_sum+=std::exp(double(values[r*E+order[k]])-max)/all_sum;
            const double denominator=std::max(selected_sum,double(6.103515625e-5f));
            for(int k=0;k<K;++k) {
                const double expected=(std::exp(double(values[r*E+order[k]])-max)/all_sum)/denominator;
                const double error=std::abs(double(actual[r*K+k])-expected);
                if(actual_ids[r*K+k]!=order[k] || !std::isfinite(actual[r*K+k]) || error>1e-6) {
                    std::fprintf(stderr,"FAIL backend=%s rows=%d variant=%d pass=%d row=%d slot=%d id=%d expected_id=%d weight=%.9g expected=%.17g error=%.9g\n",
                        ggml_backend_name(backend),rows,variant,pass,r,k,actual_ids[r*K+k],order[k],actual[r*K+k],expected,error);
                    throw std::runtime_error("router disagrees with separated-score oracle");
                }
                max_error=std::max(max_error,error);
            }
            ++cases;
        }
    }
    std::printf("PASS backend=%s rows=%d variant=%d token_cases=%d max_weight_error=%.9g\n",ggml_backend_name(backend),rows,variant,cases,max_error);
    ggml_backend_synchronize(backend);ggml_gallocr_free(alloc);ggml_free(ctx);
}
int main() {try {
    ggml_backend_load_all();int devices=0,graphs=0;
    for(size_t i=0;i<ggml_backend_dev_count();++i) {
        auto * dev=ggml_backend_dev_get(i);
        if(ggml_backend_dev_type(dev)!=GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        auto * backend=ggml_backend_dev_init(dev,nullptr);require(backend,"backend init");
        for(int n:{1,7,8,9,33,128,257,512})for(int v:{0,1}){test(backend,n,v);++graphs;}
        ggml_backend_free(backend);++devices;
    }
    require(devices>0,"no GPU tested");std::printf("PASS %d graphs, four changing-input repeats each, on %d GPUs\n",graphs,devices);return 0;
} catch(const std::exception & e){std::fprintf(stderr,"router reference: %s\n",e.what());return 1;}}

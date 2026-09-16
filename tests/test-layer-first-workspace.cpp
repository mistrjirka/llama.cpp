#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <limits>

static void require(bool ok,const char * why) { if(!ok)throw std::runtime_error(why); }
static void test(ggml_backend_t backend,int live,int padded) {
    constexpr int d=16,n=11,k=3;
    ggml_init_params store_params={ggml_tensor_overhead()*4,nullptr,true};
    auto * store=ggml_init(store_params);require(store,"store init");
    auto * x=ggml_new_tensor_2d(store,GGML_TYPE_F32,d,n);
    auto * slots=ggml_new_tensor_2d(store,GGML_TYPE_F32,d,n*k);
    auto buffer=ggml_backend_alloc_ctx_tensors(store,backend);require(buffer,"store buffer");
    ggml_init_params graph_params={1024*1024,nullptr,true};
    auto * ctx=ggml_init(graph_params);require(ctx,"graph init");
    auto * token_ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,padded);ggml_set_input(token_ids);
    auto * slot_ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I64,live);ggml_set_input(slot_ids);
    auto * gathered=ggml_get_rows(ctx,x,token_ids);
    auto * real=ggml_view_2d(ctx,gathered,d,live,gathered->nb[1],0);
    auto * scatter=ggml_set_rows(ctx,slots,real,slot_ids);ggml_set_output(scatter);
    auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,scatter);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(alloc,graph),"graph allocation");
    std::vector<float> values(d*n),actual(d*n*k),expected(d*n*k),poison(d*n*k,std::numeric_limits<float>::quiet_NaN());
    std::vector<int32_t> ti(padded);std::vector<int64_t> si(live);
    for(int pass=0;pass<3;++pass) {
        for(int i=0;i<d*n;++i)values[i]=float(i+pass*10000);
        expected=poison;
        for(int i=0;i<live;++i){ti[i]=(i*3+pass)%n;si[i]=(i*5+pass)%(n*k);for(int c=0;c<d;++c)expected[si[i]*d+c]=values[ti[i]*d+c];}
        for(int i=live;i<padded;++i)ti[i]=ti[0];
        ggml_backend_tensor_set(x,values.data(),0,values.size()*sizeof(float));
        ggml_backend_tensor_set(slots,poison.data(),0,poison.size()*sizeof(float));
        ggml_backend_tensor_set(token_ids,ti.data(),0,ti.size()*sizeof(int32_t));
        ggml_backend_tensor_set(slot_ids,si.data(),0,si.size()*sizeof(int64_t));
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(slots,actual.data(),0,actual.size()*sizeof(float));
        for(size_t i=0;i<actual.size();++i) {
            require(std::isnan(expected[i]) ? std::isnan(actual[i]) : actual[i]==expected[i],"padding, gather, scatter or graph reuse changed output");
        }
    }
    ggml_backend_synchronize(backend);ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_buffer_free(buffer);ggml_free(store);
}
int main(){try{
    ggml_backend_load_all();int cases=0,devices=0;
    for(size_t i=0;i<ggml_backend_dev_count();++i){auto * dev=ggml_backend_dev_get(i);auto type=ggml_backend_dev_type(dev);if(type!=GGML_BACKEND_DEVICE_TYPE_CPU&&type!=GGML_BACKEND_DEVICE_TYPE_GPU)continue;
        auto backend=ggml_backend_dev_init(dev,nullptr);require(backend,"backend init");
        for(int live:{1,7,31})for(int padded:{128,257}){test(backend,live,padded);cases+=3;}
        std::printf("PASS workspace gather/scatter/reuse on %s\n",ggml_backend_name(backend));ggml_backend_free(backend);++devices;
    }
    require(devices>0,"no backend tested");std::printf("PASS %d cases on %d backends\n",cases,devices);return 0;
}catch(const std::exception &e){std::fprintf(stderr,"workspace test: %s\n",e.what());return 1;}}

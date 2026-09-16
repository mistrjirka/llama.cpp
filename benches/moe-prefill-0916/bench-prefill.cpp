#include "llama.h"
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "llama-model.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <iomanip>
using clock_type=std::chrono::steady_clock;
static void require(bool ok,const char * msg) { if(!ok) throw std::runtime_error(msg); }
static double elapsed(clock_type::time_point start) { return std::chrono::duration<double,std::milli>(clock_type::now()-start).count(); }
static int env_int(const char * key,int def) { auto * v=std::getenv(key); return v?std::stoi(v):def; }
static void save(const std::string & path,const std::vector<float>& x) {
    std::ofstream f(path,std::ios::binary);f.write((const char *)x.data(),x.size()*sizeof(float));require(bool(f),"save failed");
}
struct difference { double maximum=0,rmse=0; int matching_top1=0; bool bitwise=true; };
static difference compare(const std::vector<float>& a,const std::vector<float>& b,size_t vocab) {
    require(a.size()==b.size() && !a.empty(),"vector shapes differ");difference r;double squares=0;
    for(size_t i=0;i<a.size();++i) { require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite scores");double d=double(a[i])-b[i];r.maximum=std::max(r.maximum,std::abs(d));squares+=d*d;if(a[i]!=b[i])r.bitwise=false; }
    r.rmse=std::sqrt(squares/a.size());
    for(size_t off=0;off<a.size();off+=vocab) {
        auto ia=std::max_element(a.begin()+off,a.begin()+off+vocab)-a.begin();
        auto ib=std::max_element(b.begin()+off,b.begin()+off+vocab)-b.begin();
        r.matching_top1+=ia==ib;
    }
    return r;
}
int main(int argc,char **argv) {try {
    require(argc==7,"usage: bench-prefill MODEL N CHUNK PREFIX REPEATS OUTPUT");
    int n=std::stoi(argv[2]);const int chunk=std::stoi(argv[3]),prefix=std::stoi(argv[4]),reps=std::stoi(argv[5]);
    const std::string out=argv[6];const int samples=env_int("BENCH_SAMPLES",8);int expert_rows=env_int("BENCH_EXPERT_ROWS",512);
    require(n>=32&&n<=131072&&prefix>=0&&n+prefix<250000&&chunk>=32&&chunk<=4096&&samples>=1&&samples<=32&&reps>=0&&reps<=64,"invalid benchmark sizes");
    std::vector<int> sequence;
    if(const char * raw=std::getenv("BENCH_SEQUENCE")) {
        std::stringstream ss(raw);std::string value;
        while(std::getline(ss,value,','))sequence.push_back(std::stoi(value));
        require(sequence.size()==size_t(reps+1),"invalid benchmark sequence");
        for(int count:sequence)require(count>=32 && count<=n,"sequence exceeds maximum tokens");
    }
    ggml_backend_load_all();llama_backend_init();

    const char * raw_layers=std::getenv("BENCH_GPU_LAYERS");
    require(raw_layers && *raw_layers,"BENCH_GPU_LAYERS is required");
    std::vector<int> layer_counts;
    std::stringstream layer_stream(raw_layers);std::string value;
    while(std::getline(layer_stream,value,',')) {
        size_t used=0;int count=std::stoi(value,&used);
        require(used==value.size() && count>0 && count<=256,"invalid per-device layer count");
        layer_counts.push_back(count);
    }
    auto * cuda_reg=ggml_backend_reg_by_name("CUDA");require(cuda_reg,"CUDA backend unavailable");
    const size_t n_devices=ggml_backend_reg_dev_count(cuda_reg);
    require(!layer_counts.empty() && layer_counts.size()==n_devices,"layer counts must match visible CUDA devices");
    require(n_devices<=llama_max_devices(),"too many visible devices");
    std::vector<ggml_backend_dev_t> devices;
    std::vector<float> split(llama_max_devices(),0.0f);
    int expected_layers=0;
    for(size_t i=0;i<n_devices;++i) {
        auto * dev=ggml_backend_reg_dev_get(cuda_reg,i);devices.push_back(dev);
        split[i]=float(layer_counts[i]);expected_layers+=layer_counts[i];
        size_t free=0,total=0;ggml_backend_dev_memory(dev,&free,&total);
        std::cerr<<"PORTABLE_DEVICE index="<<i<<" name="<<ggml_backend_dev_name(dev)
                 <<" description="<<ggml_backend_dev_description(dev)<<" layers="<<layer_counts[i]
                 <<" free_bytes="<<free<<" total_bytes="<<total<<"\n";
    }
    split[n_devices-1]+=1.0f;devices.push_back(nullptr);
    // Every canonical expert source is host-backed; the existing executor manages GPU residency.
    std::vector<std::string> patterns;
    for(int l=0;l<expected_layers;++l)patterns.push_back("blk\\."+std::to_string(l)+"\\.ffn_.*_exps");
    std::vector<llama_model_tensor_buft_override> overrides;
    for(auto & p:patterns)overrides.push_back({p.c_str(),ggml_backend_cuda_host_buffer_type()});
    overrides.push_back({"per_layer_token_embd.weight",ggml_backend_cpu_buffer_type()});
    overrides.push_back({nullptr,nullptr});
    auto mp=llama_model_default_params();mp.n_gpu_layers=-1;mp.split_mode=LLAMA_SPLIT_MODE_LAYER;
    mp.devices=devices.data();mp.tensor_split=split.data();mp.tensor_buft_overrides=overrides.data();
    mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_AUTO;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
    require(bool(model),"model load failed");
    require(int(model->layers.size())==expected_layers,"requested placement does not match model layer count");
    int layer=0;
    for(size_t device=0;device<n_devices;++device)for(int i=0;i<layer_counts[device];++i,++layer) {
        require(model->dev_layer(layer)==devices[device],"actual model layer placement differs from requested placement");
        for(auto * w:{model->layers[layer].ffn_up_exps,model->layers[layer].ffn_gate_exps,model->layers[layer].ffn_down_exps})
            require(w && w->buffer && ggml_backend_buffer_is_host(w->buffer),"canonical expert source is not host-backed");
    }
    std::cerr<<"PORTABLE_PLACEMENT_VERIFIED devices="<<n_devices<<" layers="<<expected_layers<<"\n";
    size_t router_bytes = 0;
    for (size_t il=0;il<model->layers.size();++il) {
        auto * w = model->layers[il].ffn_gate_inp;
        if (w) {
            router_bytes += ggml_nbytes(w);
            std::cerr << "ROUTER layer=" << il << " bytes=" << ggml_nbytes(w)
                      << " type=" << ggml_type_name(w->type)
                      << " buffer=" << ggml_backend_buffer_name(w->buffer) << "\n";
        }
    }
    std::cerr << "ROUTER_TOTAL bytes=" << router_bytes << "\n";
    auto *vocab=llama_model_get_vocab(model.get());const int nv=llama_vocab_n_tokens(vocab);
    std::string text;
    if (const char * path=std::getenv("BENCH_TEXT")) {
        std::ifstream f(path);require(bool(f),"fixture open failed");std::ostringstream ss;ss<<f.rdbuf();text=ss.str();require(!text.empty(),"empty fixture");
        const std::string seed=text;while(text.size()<size_t(n+prefix)*18+4096)text+=seed;
    } else {
        for(int i=0;text.size()<size_t(n+prefix)*18+4096;++i)text+="Review task "+std::to_string(i)+": explain why this C++ function keeps its buffer alive.\nint sum(const std::vector<int>& values) { int total = 0; for (int value : values) total += value; return total; }\nCheck empty inputs, lifetime, allocation and cache behavior.\n";
    }
    std::vector<llama_token> tokens(text.size()+32);int nt=llama_tokenize(vocab,text.data(),text.size(),tokens.data(),tokens.size(),true,true);require(nt>n+prefix+4,"fixture too short");tokens.resize(nt);
    if(const char * path=std::getenv("BENCH_TOKENS")) {
        std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"token fixture open failed");
        const auto bytes=f.tellg();require(bytes>0 && size_t(bytes)%sizeof(llama_token)==0,"invalid token fixture");
        tokens.resize(size_t(bytes)/sizeof(llama_token));f.seekg(0);f.read((char*)tokens.data(),bytes);require(bool(f)&&tokens.size()>=size_t(prefix+n),"token fixture too short");
        while(tokens.size()<size_t(prefix+n+4))tokens.push_back(tokens.back());
    }

    const int continuation_count=env_int("BENCH_CONTINUATION",8);
    require(continuation_count>=3 && continuation_count<=32,"invalid continuation count");
    while(tokens.size()<size_t(prefix+n+continuation_count+1))tokens.push_back(tokens.back());
    std::vector<std::vector<llama_token>> suffixes;
    std::vector<std::string> suffix_labels;
    if(const char * raw=std::getenv("BENCH_SUFFIX_SEQUENCE")) {
        std::stringstream paths(raw);std::string path;
        while(std::getline(paths,path,',')) {
            std::ifstream input(path);require(bool(input),"suffix open failed");
            std::ostringstream data;data<<input.rdbuf();const std::string text=data.str();
            require(!text.empty(),"empty suffix");
            std::vector<llama_token> ids(text.size()+32);
            int count=llama_tokenize(vocab,text.data(),text.size(),ids.data(),ids.size(),false,false);
            require(count>n+continuation_count,"suffix too short");ids.resize(count);
            suffix_labels.push_back(std::filesystem::path(path).stem().string());
            suffixes.push_back(std::move(ids));
        }
        require(suffixes.size()==size_t(reps+1),"suffix sequence count");
    }
    auto cp=llama_context_default_params();cp.n_ctx=env_int("BENCH_CONTEXT",std::max(4096,n+prefix+512));cp.n_batch=cp.n_ubatch=chunk;cp.n_seq_max=1;cp.n_outputs_max=cp.n_outputs_max_per_seq=env_int("DIAG_ALL_OUTPUTS",0)?n:32;cp.n_threads=cp.n_threads_batch=24;cp.type_k=cp.type_v=GGML_TYPE_Q8_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.exact_set_top_k=env_int("BENCH_EXACT_SET_TOP_K",1)!=0;
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);require(bool(ctx),"context init failed");
    require(llama_supports_prefill_request(ctx.get()),"request API unavailable");
    std::vector<int> positions;std::vector<int8_t> flags(n,env_int("DIAG_ALL_OUTPUTS",0)?1:0);
    for(int i=0;i<samples;++i) {int p=samples==1?n-1:i*(n-1)/(samples-1);positions.push_back(p);flags[p]=1;}
    std::vector<uint8_t> snapshot;
    if(prefix) {
        auto begin=clock_type::now();
        if(const char * state=std::getenv("BENCH_PREFIX_STATE")) {
            std::vector<llama_token> saved(prefix);size_t count=0;
            require(llama_state_seq_load_file(ctx.get(),state,0,saved.data(),saved.size(),&count)>0,"saved prefix restore failed");
            require(count==size_t(prefix)&&std::equal(saved.begin(),saved.end(),tokens.begin()),"saved prefix token identities differ");
        } else {
            std::vector<int8_t> none(chunk,0);
            for(int off=0;off<prefix;off+=chunk){auto b=llama_batch_get_one(tokens.data()+off,std::min(chunk,prefix-off));b.logits=none.data();require(llama_decode(ctx.get(),b)==0,"prefix construction failed");}
        }
        llama_synchronize(ctx.get());require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==prefix-1,"prefix position wrong");
        const double prefix_ms=elapsed(begin);snapshot.resize(llama_state_seq_get_size(ctx.get(),0));require(llama_state_seq_get_data(ctx.get(),snapshot.data(),snapshot.size(),0)==snapshot.size(),"snapshot failed");
        std::cerr<<"PREFIX_READY tokens="<<prefix<<" construction_ms="<<prefix_ms<<" snapshot_bytes="<<snapshot.size()<<"\n";
    }
    std::vector<std::string> modes={"baseline","original","reuse","resident","fast"};
    if(const char * value=std::getenv("BENCH_MODES")){modes.clear();std::stringstream ss(value);std::string item;while(std::getline(ss,item,','))modes.push_back(item);}
    std::map<std::string,std::vector<float>> reference,reference_next;
    std::ofstream rows(out+".jsonl");require(bool(rows),"summary open failed");rows<<std::setprecision(12);
    std::vector<int> resident_groups;
    if(const char * raw=std::getenv("BENCH_RESIDENT_GROUP_SEQUENCE")) {
        std::stringstream ss(raw);std::string value;
        while(std::getline(ss,value,','))resident_groups.push_back(std::stoi(value));
        require(resident_groups.size()==size_t(reps+1),"resident group sequence count");
    }
    std::vector<int> output_aliases;
    if(const char * raw=std::getenv("BENCH_OUTPUT_ALIAS_SEQUENCE")) {
        std::stringstream ss(raw);std::string value;
        while(std::getline(ss,value,','))output_aliases.push_back(std::stoi(value));
        require(output_aliases.size()==size_t(reps+1),"output alias sequence count");
    }
    std::vector<int> expert_row_sequence;
    if(const char * raw=std::getenv("BENCH_EXPERT_ROWS_SEQUENCE")) {
        std::stringstream ss(raw);std::string value;
        while(std::getline(ss,value,','))expert_row_sequence.push_back(std::stoi(value));
        require(expert_row_sequence.size()==size_t(reps+1),"expert row sequence count");
    }
    std::vector<int> preparation_sequence;
    if(const char * raw=std::getenv("BENCH_PREPARATION_SEQUENCE")) {
        std::stringstream ss(raw);std::string value;
        while(std::getline(ss,value,','))preparation_sequence.push_back(std::stoi(value));
        require(preparation_sequence.size()==size_t(reps+1),"preparation sequence count");
    }
    std::vector<int> attention_sequence;
    if(const char * raw=std::getenv("BENCH_ATTENTION_SEQUENCE")) {
        std::stringstream ss(raw);std::string value;
        while(std::getline(ss,value,','))attention_sequence.push_back(std::stoi(value));
        require(attention_sequence.size()==size_t(reps+1),"attention sequence count");
    }
    std::vector<int> sparse_sequence;
    if(const char * raw=std::getenv("BENCH_SPARSE_SEQUENCE")) {
        std::stringstream ss(raw);std::string value;
        while(std::getline(ss,value,','))sparse_sequence.push_back(std::stoi(value));
        require(sparse_sequence.size()==size_t(reps+1),"sparse sequence count");
    }

    for(int round=0;round<=reps;++round) {
        const std::string suffix_label=suffix_labels.empty()?"original":suffix_labels[round];
        if(!suffixes.empty()) {
            tokens.resize(prefix);tokens.insert(tokens.end(),suffixes[round].begin(),suffixes[round].end());
            const std::string token_path=out+"-"+suffix_label+".tokens.i32";
            std::ofstream f(token_path,std::ios::binary);
            f.write((const char*)suffixes[round].data(),suffixes[round].size()*sizeof(llama_token));
            require(bool(f),"token save failed");
        }
        if(!sparse_sequence.empty()) {
            const int setting=sparse_sequence[round];require(setting==0 || setting==1,"sparse setting");
            llama_set_selected_attn(ctx.get(),setting!=0);
            std::fprintf(stderr,"SPARSE_SETTING round=%d sparse=%d\n",round,setting);
        }

        if(!attention_sequence.empty()) {
            const int setting=attention_sequence[round];require(setting>=0 && setting<=3,"attention setting");
            setenv("LLAMA_MOE_LAYER_FIRST_SHARE_MASK_HOST",(setting&1)?"1":"0",1);
            require(((setting>>1)&1)==int(cp.exact_set_top_k),"attention sequence radix setting differs from context parameter");
            std::fprintf(stderr,"ATTENTION_SETTING round=%d share_host=%d radix=%d\n",round,setting&1,int(cp.exact_set_top_k));
        }

        if(!preparation_sequence.empty()) {
            const int setting=preparation_sequence[round];
            require(setting>=0 && setting<=3,"invalid preparation setting");
            setenv("LLAMA_MOE_LAYER_FIRST_PREPARED_ROUTES",(setting&1)?"1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_RESIDENT_PLANS",(setting&2)?"1":"0",1);
        }
        if(!expert_row_sequence.empty())expert_rows=expert_row_sequence[round];
        if(!output_aliases.empty())setenv("LLAMA_MOE_LAYER_FIRST_OUTPUT_ALIAS",std::to_string(output_aliases[round]).c_str(),1);
        if(!resident_groups.empty())setenv("LLAMA_MOE_LAYER_FIRST_RESIDENT_GROUP",std::to_string(resident_groups[round]).c_str(),1);
        if(!sequence.empty()) {
            n=sequence[round];positions.clear();flags.assign(n,env_int("DIAG_ALL_OUTPUTS",0)?1:0);
            for(int i=0;i<samples;++i){int pos=samples==1?n-1:i*(n-1)/(samples-1);positions.push_back(pos);flags[pos]=1;}
        }
        auto order=modes;if(round%2)std::reverse(order.begin(),order.end());
        for(const auto &mode:order) {
            const std::string sample_key=mode+"-"+suffix_label+"-n"+std::to_string(n);
            const std::map<std::string,int> levels={{"baseline",-1},{"old",0},{"resident",1},{"masks",2},{"banks",3},{"plans",4},{"async",5},{"accum",5},{"shared",5},{"residency",5},{"stagger",5},{"residency-tight",5},{"adaptive",5},{"cpu1",6},{"cpu4",7},{"async8",5},{"async32",5},{"async64",5}};
            require(levels.count(mode),"unknown mode");const int level=levels.at(mode);
            require(!env_int("BENCH_QUALITY",0) || env_int("DIAG_ALL_OUTPUTS",0),"quality requires all output rows");
            if (env_int("DIAG_FRESH_CONTEXT",0)) {
                ctx.reset();
                ctx.reset(llama_init_from_model(model.get(),cp));
                require(bool(ctx),"fresh context init failed");
            }
            llama_synchronize(ctx.get());llama_memory_clear(llama_get_memory(ctx.get()),true);
            if(prefix)require(llama_state_seq_set_data(ctx.get(),snapshot.data(),snapshot.size(),0)==snapshot.size(),"prefix restore failed");
            setenv("LLAMA_MOE_LAYER_FIRST_ACCUMULATE",mode=="adaptive"?"2":(mode=="accum" || mode=="shared" || mode=="residency" || mode=="residency-tight" || mode=="stagger")?"1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_DEFER_SHARED",(mode=="shared" || mode=="residency" || mode=="residency-tight" || mode=="stagger")?"1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_REQUIRE_RESIDENT",(mode=="shared" || mode=="residency" || mode=="residency-tight" || mode=="stagger")?"1":std::to_string(env_int("BENCH_REQUIRE_RESIDENT",0)).c_str(),1);
            setenv("LLAMA_MOE_LAYER_FIRST_REUSE_GRAPH",(mode=="reuse"||mode=="fast"||mode=="lean"||mode=="device")?"1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_RESIDENT_BATCH",(mode=="resident"||mode=="fast")?"1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_COALESCE",(mode=="coalesce"||mode=="fast"||mode=="lean"||mode=="device")?"1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_DEVICE_MIB",mode=="device"?std::to_string(env_int("BENCH_DEVICE_MIB",1024)).c_str():"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_ROWS",std::to_string(expert_rows).c_str(),1);
            setenv("LLAMA_MOE_LAYER_FIRST_REUSE_GRAPH","1",1);
            setenv("LLAMA_MOE_LAYER_FIRST_COALESCE","1",1);
            setenv("LLAMA_MOE_LAYER_FIRST_RESIDENT_BATCH","0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_DEVICE_MIB",std::to_string(env_int("BENCH_DEVICE_MIB",1024)).c_str(),1);
            setenv("LLAMA_MOE_LAYER_FIRST_RESIDENT_DATA",level>=1 ? "1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_MASK_MIB",level>=2 ? std::to_string(env_int("BENCH_MASK_MIB",768)).c_str():"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_REUSE_BANKS",level>=3 ? "1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_RESIDENT_FAST",level>=3 ? "1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_FIXED_PLANS",level>=4 && env_int("BENCH_FIXED_PLANS",1) ? "1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_ASYNC",level>=5 && env_int("BENCH_ASYNC",1) ? "1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_CPU_CUTOFF",level==6 ? "1" : level==7 ? "4" : "0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_CPU_WORKERS","8",1);
            setenv("LLAMA_MOE_LAYER_FIRST_ADAPTIVE_RESIDENCY",(mode=="residency" || mode=="residency-tight" || mode=="stagger")?"1":"0",1);
            setenv("LLAMA_MOE_LAYER_FIRST_STAGGER_RESIDENCY",mode=="stagger"?"1":"0",1);
            if(mode=="residency-tight")setenv("LLAMA_MOE_LAYER_FIRST_DEVICE_MIB","512",1);
            setenv("LLAMA_MOE_LAYER_FIRST_GROUP",std::to_string(env_int("BENCH_GROUP",16)).c_str(),1);
            setenv("LLAMA_MOE_LAYER_FIRST_WEIGHT_MIB",std::to_string(env_int("BENCH_WEIGHT_MIB",128)).c_str(),1);
            if(mode=="async8" || mode=="async32" || mode=="async64") {setenv("LLAMA_MOE_LAYER_FIRST_GROUP",mode.substr(5).c_str(),1);setenv("LLAMA_MOE_LAYER_FIRST_WEIGHT_MIB","512",1);}
            std::vector<float> actual;actual.reserve(size_t(samples)*nv);int calls=0;double ms=0;
            std::vector<double> losses;
            std::vector<int> choices;
            auto capture=[&](int off,int count){
                for(int p:positions)if(p>=off&&p<off+count){auto *v=llama_get_logits_ith(ctx.get(),p-off);require(v,"missing logits");actual.insert(actual.end(),v,v+nv);}
                if(env_int("BENCH_QUALITY",0))for(int p=off;p<off+count;++p) {
                    auto * v=llama_get_logits_ith(ctx.get(),p-off);require(v,"missing quality logits");
                    const float maximum=*std::max_element(v,v+nv);double total=0;
                    for(int j=0;j<nv;++j){require(std::isfinite(v[j]),"nonfinite quality logits");total+=std::exp(double(v[j])-maximum);}
                    losses.push_back(std::log(total)+maximum-v[tokens[prefix+p+1]]);
                    choices.push_back(int(std::max_element(v,v+nv)-v));
                }
            };
            std::cerr<<"BENCH_START mode="<<mode<<" round="<<round<<" tokens="<<n<<" prefix="<<prefix<<"\n";
            bool capture_round=round==env_int("PROFILE_ROUND",1);
            if(const char * raw=std::getenv("PROFILE_ROUNDS")) {
                capture_round=false;std::stringstream ss(raw);std::string value;
                while(std::getline(ss,value,','))if(round==std::stoi(value))capture_round=true;
            }
            const bool profile_capture=env_int("PROFILE_CAPTURE",0) && capture_round && mode==(std::getenv("PROFILE_MODE")?std::getenv("PROFILE_MODE"):"device");
            if(profile_capture)cudaProfilerStart();
            nvtxRangePushA(("BENCH "+mode+" round="+std::to_string(round)).c_str());
            if(mode=="baseline")for(int off=0;off<n;off+=chunk){int count=std::min(chunk,n-off);auto b=llama_batch_get_one(tokens.data()+prefix+off,count);b.logits=flags.data()+off;auto begin=clock_type::now();require(llama_decode(ctx.get(),b)==0,"ordinary decode failed");llama_synchronize(ctx.get());ms+=elapsed(begin);++calls;capture(off,count);}
            else {auto b=llama_batch_get_one(tokens.data()+prefix,n);b.logits=flags.data();auto begin=clock_type::now();require(llama_prefill_request(ctx.get(),b)==0,"request prefill failed");llama_synchronize(ctx.get());ms=elapsed(begin);++calls;capture(0,n);}
            nvtxRangePop();if(profile_capture)cudaProfilerStop();
            require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==prefix+n-1,"suffix position wrong");
            require(actual.size()==size_t(samples)*nv,"sample count wrong");

            std::vector<float> next;
            std::vector<double> continuation_losses;
            const auto continuation_begin=clock_type::now();
            for(int i=0;i<continuation_count;++i) {
                auto b=llama_batch_get_one(tokens.data()+prefix+n+i,1);
                require(llama_decode(ctx.get(),b)==0,"continuation failed");llama_synchronize(ctx.get());
                auto *v=llama_get_logits_ith(ctx.get(),-1);require(v,"missing continuation scores");
                next.insert(next.end(),v,v+nv);
                const float maximum=*std::max_element(v,v+nv);double total=0;
                for(int j=0;j<nv;++j){require(std::isfinite(v[j]),"nonfinite continuation");total+=std::exp(double(v[j])-maximum);}
                continuation_losses.push_back(std::log(total)+maximum-v[tokens[prefix+n+i+1]]);
            }
            const double continuation_ms=elapsed(continuation_begin);
            if(!reference.count(sample_key)){reference[sample_key]=actual;reference_next[sample_key]=next;save(out+"-"+sample_key+".f32",actual);save(out+"-"+sample_key+"-next.f32",next);}
            save(out+"-"+sample_key+"-round"+std::to_string(round)+".f32",actual);
            save(out+"-"+sample_key+"-round"+std::to_string(round)+"-next.f32",next);
            auto repeat=compare(actual,reference.at(sample_key),nv);auto repeat_next=compare(next,reference_next.at(sample_key),nv);
            rows<<"{\"mode\":\""<<mode<<"\",\"round\":"<<round<<",\"warmup\":"<<(round==0?"true":"false")<<",\"tokens\":"<<n<<",\"prefix\":"<<prefix<<",\"chunk\":"<<chunk<<",\"expert_rows\":"<<expert_rows<<",\"samples\":"<<samples<<",\"calls\":"<<calls<<",\"ms\":"<<ms<<",\"tokens_per_second\":"<<(1000.0*n/ms)<<",\"repeat_max\":"<<repeat.maximum<<",\"repeat_next_max\":"<<repeat_next.maximum;
            for(const std::string ref:{"baseline","original"})if(reference.count(ref)){auto d=compare(actual,reference.at(ref),nv);auto e=compare(next,reference_next.at(ref),nv);rows<<",\"vs_"<<ref<<"_max\":"<<d.maximum<<",\"vs_"<<ref<<"_rmse\":"<<d.rmse<<",\"vs_"<<ref<<"_top1\":"<<d.matching_top1<<",\"vs_"<<ref<<"_next_max\":"<<e.maximum;}
            if(env_int("BENCH_QUALITY",0)) {
                require(losses.size()==size_t(n),"quality row count");
                std::ofstream q(out+"-"+sample_key+"-round"+std::to_string(round)+".quality.json");
                q<<std::setprecision(12)<<"{\"losses\":[";double total=0;
                for(size_t j=0;j<losses.size();++j){if(j)q<<",";q<<losses[j];total+=losses[j];}
                q<<"],\"top1\":[";for(size_t j=0;j<choices.size();++j){if(j)q<<",";q<<choices[j];}q<<"],\"continuation_losses\":[";for(size_t j=0;j<continuation_losses.size();++j){if(j)q<<",";q<<continuation_losses[j];}q<<"]}\n";
                rows<<",\"mean_nll\":"<<total/losses.size()<<",\"perplexity\":"<<std::exp(total/losses.size());
            }

            const int sparse_setting=sparse_sequence.empty()?int(cp.selected_attn):sparse_sequence[round];
            rows<<",\"fixture\":\""<<suffix_label<<"\",\"sparse\":"<<sparse_setting;
            rows<<",\"all_outputs\":"<<env_int("DIAG_ALL_OUTPUTS",0)<<",\"continuation_rows\":"<<continuation_count<<",\"continuation_ms\":"<<continuation_ms;
            rows<<",\"output_alias\":"<<env_int("LLAMA_MOE_LAYER_FIRST_OUTPUT_ALIAS",0);
            if(!std::getenv("BENCH_RECORD_NUMERIC_DIFFERENCE"))require(repeat.maximum==0 && repeat_next.maximum==0,"changed checked outputs");
            rows<<",\"prepared_routes\":"<<env_int("LLAMA_MOE_LAYER_FIRST_PREPARED_ROUTES",0);
            rows<<",\"resident_plans\":"<<env_int("LLAMA_MOE_LAYER_FIRST_RESIDENT_PLANS",0);
            rows<<",\"resident_group\":"<<env_int("LLAMA_MOE_LAYER_FIRST_RESIDENT_GROUP",env_int("BENCH_GROUP",16));
            rows<<"}\n";rows.flush();std::cerr<<"BENCH_END mode="<<mode<<" round="<<round<<" ms="<<ms<<" pp="<<1000.0*n/ms<<"\n";
        }
    }
    ctx.reset();model.reset();llama_backend_free();return 0;
} catch(const std::exception&e){std::cerr<<"bench-prefill: "<<e.what()<<"\n";return 1;}}

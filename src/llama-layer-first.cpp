#include "llama-layer-first.h"
#include "llama-layer-first-inputs.h"
#include "llama-layer-first-cpu.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "llama-memory-hybrid-idx.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-impl.h"
#include "llama-layer-first-profile.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace {
void lf_require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}
size_t lf_mul(size_t a, size_t b) {
    lf_require(b == 0 || a <= SIZE_MAX / b, "layer-first size overflow");
    return a*b;
}
size_t lf_env(const char * name, size_t fallback, size_t low, size_t high) {
    const char * value=std::getenv(name);
    if (!value) { return fallback; }
    char * end=nullptr;
    const long long parsed=std::strtoll(value,&end,10);
    lf_require(end!=value && *end=='\0' && parsed>=0 && size_t(parsed)>=low && size_t(parsed)<=high,
               "invalid layer-first environment value");
    return size_t(parsed);
}
// Full-window token backing stays alive; each slice owns its planar positions.
struct lf_slice {
    llama_ubatch batch;
    std::vector<llama_pos> positions;
    lf_slice(const llama_ubatch & full,uint32_t begin,uint32_t count):batch(full) {
        lf_require(begin<=full.n_tokens && count<=full.n_tokens-begin,"slice outside layer window");
        batch.n_tokens=batch.n_seq_tokens=count;
        batch.token+=begin;batch.n_seq_id+=begin;batch.seq_id+=begin;batch.output+=begin;
        positions.resize(size_t(count)*full.n_pos);
        for (uint32_t dim=0;dim<full.n_pos;++dim) {
            std::copy_n(full.pos+size_t(dim)*full.n_tokens+begin,count,positions.data()+size_t(dim)*count);
        }
        batch.pos=positions.data();
    }
    lf_slice(const lf_slice &)=delete;
    lf_slice & operator=(const lf_slice &)=delete;
};
// Declared after any external buffer owner: detach graphs before that owner
// unwinds, including when allocation, a callback or graph computation fails.
struct lf_graph_cleanup {
    ggml_backend_sched_t scheduler;
    llm_graph_result * result;
    bool active;
    ~lf_graph_cleanup() {
        if (active) {
            ggml_backend_sched_synchronize(scheduler);
            ggml_backend_sched_reset(scheduler);
            result->reset();
        }
    }
};
struct lf_activation_store {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    std::array<ggml_tensor *,7> tensor={};
};
struct lf_weight_pool {
    std::array<ggml_backend_buffer_ptr,2> buffer;
    ggml_backend_ptr transfer;
    ggml_backend_t compute=nullptr;
    std::array<ggml_backend_event_t,2> ready={},consumed={};
    std::array<bool,2> used={};
    ~lf_weight_pool() {
        if(transfer)ggml_backend_synchronize(transfer.get());
        if(compute)ggml_backend_synchronize(compute);
        for(auto * e:ready)if(e)ggml_backend_event_free(e);
        for(auto * e:consumed)if(e)ggml_backend_event_free(e);
    }
};
struct lf_plan_scratch {
    ggml_gallocr_t allocator=nullptr;
    ggml_backend_t backend=nullptr;
    ~lf_plan_scratch() {if(backend)ggml_backend_synchronize(backend);if(allocator)ggml_gallocr_free(allocator);}
};
struct lf_expert_plan {
    std::unique_ptr<llm_graph_result> result;
    ggml_gallocr_t allocator=nullptr;
    ggml_cgraph * graph=nullptr;
    ggml_backend_t backend=nullptr;
    size_t last_used=0;
    std::array<ggml_tensor *,3> weight_leaves={};
    std::vector<ggml_tensor *> scratch_tensors;
    ~lf_expert_plan() {if(backend)ggml_backend_synchronize(backend);if(allocator)ggml_gallocr_free(allocator);}
};
struct lf_weights {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    std::array<ggml_tensor *,3> tensor={};
    std::array<bool,3> uploaded={};
};
}

struct llama_lf_residency {
    struct entry {
        ggml_context_ptr context;
        ggml_backend_buffer_ptr buffer;
        std::array<ggml_tensor *,3> tensor={};
        std::array<ggml_tensor *,3> source={};
        std::vector<uint8_t> valid;
        ggml_backend_dev_t device=nullptr;
    };
    std::map<int,entry> layers;
    std::map<std::pair<int,size_t>,entry> groups;
};

bool llama_layer_first_requested() {
    const char * value=std::getenv("LLAMA_MOE_LAYER_FIRST");
    return value && std::strcmp(value,"1")==0;
}

bool llama_context::layer_first_supported() const {
    if (model.arch!=LLM_ARCH_QWEN4EXP || cparams.ctx_type!=LLAMA_CONTEXT_TYPE_DEFAULT ||
        !cparams.causal_attn || cparams.n_seq_max!=1 || cparams.n_rs_seq!=0 ||
        cparams.embeddings || cparams.embeddings_nextn || !sampling.samplers.empty() || !loras->empty()) { return false; }
    for (bool requested:cparams.embeddings_layer_inp) { if (requested) { return false; } }
    for (int il=0;il<model.hparams.n_layer();++il) {
        const auto & l=model.layers[il];
        if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || l.ffn_gate_up_exps ||
            l.ffn_up_exps_s || l.ffn_gate_exps_s || l.ffn_down_exps_s) { return false; }
    }
    return true;
}

llm_graph_result * llama_context::process_ubatch_layer_first(
        const llama_ubatch & full,llama_memory_context_i * memory_context,ggml_status & status) {
    // Full-request expert leases with bounded scratch and optional transfer overlap.
    status=GGML_STATUS_FAILED;
    bool applied=false;
    try {
        lf_require(layer_first_supported(),"unsupported layer-first context");
        lf_require(full.token && !full.embd && full.n_seqs==1 && full.n_seqs_unq==1 && full.equal_seqs(),
                   "layer-first requires one ordinary text sequence");
        for (uint32_t t=0;t<full.n_tokens;++t) {
            lf_require(full.n_seq_id[t]==1 && full.seq_id[t][0]==full.seq_id[0][0],"multiple sequences in layer-first window");
            lf_require(t==0 || full.pos[t]==full.pos[t-1]+1,"nonconsecutive layer-first positions");
        }
        lf_require(full.n_tokens==balloc->get_n_tokens(),"layer-first window was silently split");
        auto * mctx=dynamic_cast<llama_memory_hybrid_idx_context *>(memory_context);
        lf_require(mctx!=nullptr,"layer-first needs Qwen hybrid/index state");
        const size_t n=full.n_tokens,d=model.hparams.n_embd,hc=model.hparams.dsv4_hc_mult;
        lf_profile_scope request_scope("LF.request tokens=%zu",n);
        const size_t request_tokens=n;
        const size_t k=model.hparams.n_expert_used(),ne=model.hparams.n_expert;
        const size_t accum_mode=lf_env("LLAMA_MOE_LAYER_FIRST_ACCUMULATE",0,0,2);
        const size_t width_h=lf_mul(d,hc),token_tile=cparams.n_ubatch;
        const size_t expert_tile=lf_env("LLAMA_MOE_LAYER_FIRST_ROWS",token_tile,1,16384);
        const bool reuse_graph = lf_env("LLAMA_MOE_LAYER_FIRST_REUSE_GRAPH",0,0,1) != 0;
        const bool coalesce = lf_env("LLAMA_MOE_LAYER_FIRST_COALESCE",0,0,1) != 0;
        const bool resident_batch = lf_env("LLAMA_MOE_LAYER_FIRST_RESIDENT_BATCH",0,0,1) != 0;
        const size_t group_limit=lf_env("LLAMA_MOE_LAYER_FIRST_GROUP",16,1,512);
        const size_t resident_group_limit=lf_env("LLAMA_MOE_LAYER_FIRST_RESIDENT_GROUP",group_limit,1,512);
        const size_t device_limit=lf_env("LLAMA_MOE_LAYER_FIRST_DEVICE_MIB",0,0,65536)*(size_t(1)<<20);
        const size_t cpu_cutoff=lf_env("LLAMA_MOE_LAYER_FIRST_CPU_CUTOFF",0,0,16);
        const int cpu_workers=int(lf_env("LLAMA_MOE_LAYER_FIRST_CPU_WORKERS",8,1,24));
        const bool async_weights=lf_env("LLAMA_MOE_LAYER_FIRST_ASYNC",0,0,1)!=0;
        const bool reuse_banks=async_weights || lf_env("LLAMA_MOE_LAYER_FIRST_REUSE_BANKS",0,0,1)!=0;
        const bool fixed_plans=lf_env("LLAMA_MOE_LAYER_FIRST_FIXED_PLANS",0,0,1)!=0;
        const bool resident_fast=lf_env("LLAMA_MOE_LAYER_FIRST_RESIDENT_FAST",0,0,1)!=0;
        const bool resident_shared_scratch=lf_env("LLAMA_MOE_LAYER_FIRST_RESIDENT_SHARED_SCRATCH",1,0,1)!=0;
        const bool async_inputs=lf_env("LLAMA_MOE_LAYER_FIRST_ASYNC_INPUTS",0,0,1)!=0;
        const bool prepared_routes=lf_env("LLAMA_MOE_LAYER_FIRST_PREPARED_ROUTES",0,0,1)!=0;
        const bool resident_plans=lf_env("LLAMA_MOE_LAYER_FIRST_RESIDENT_PLANS",0,0,1)!=0;
        const bool plan_shared_scratch=lf_env("LLAMA_MOE_LAYER_FIRST_PLAN_SHARED_SCRATCH",0,0,1)!=0;
        const bool resident_requested=lf_env("LLAMA_MOE_LAYER_FIRST_RESIDENT_DATA",0,0,1)!=0;
        const size_t original_extra=fixed_plans ? std::max<size_t>(128,(expert_tile+127)/128*128)*d*sizeof(float) : 0;
        const size_t original_bytes=lf_mul(lf_mul(n,d+d*k+(resident_requested ? width_h+d+hc+k : 0)),sizeof(float))+original_extra;
        const bool original_fits=original_bytes<=device_limit && device_limit-original_bytes>=8192;
        const bool accumulate=accum_mode==1 || (accum_mode==2 && !original_fits);
        const bool accum_audit=accumulate && lf_env("LLAMA_MOE_LAYER_FIRST_ACCUM_AUDIT",0,0,1)!=0;
        const bool defer_shared=accumulate && lf_env("LLAMA_MOE_LAYER_FIRST_DEFER_SHARED",0,0,1)!=0;
        const bool require_resident=lf_env("LLAMA_MOE_LAYER_FIRST_REQUIRE_RESIDENT",0,0,1)!=0;
        const bool adaptive_residency=lf_env("LLAMA_MOE_LAYER_FIRST_ADAPTIVE_RESIDENCY",0,0,1)!=0;
        const size_t adaptive_max_layers=lf_env("LLAMA_MOE_LAYER_FIRST_ADAPTIVE_MAX_LAYERS",2,0,48);
        const bool stagger_residency=adaptive_residency && lf_env("LLAMA_MOE_LAYER_FIRST_STAGGER_RESIDENCY",0,0,1)!=0;
        const std::array<size_t,2> base_resident_layers={
            lf_env("LLAMA_MOE_LAYER_FIRST_BASE_LAYERS_0",0,0,48),
            lf_env("LLAMA_MOE_LAYER_FIRST_BASE_LAYERS_1",0,0,48)};
        lf_require(!stagger_residency || cpu_cutoff==0,"staggered residency CPU placement is not configured");
        lf_require(!stagger_residency || (resident_group_limit>=group_limit && resident_group_limit%group_limit==0 && ne%resident_group_limit==0),
                   "resident group size must divide expert count and contain complete rotating groups");
        const size_t shared_width=defer_shared ? 0 : d;
        const size_t output_k=accumulate ? 1 : k,width_slots=lf_mul(d,output_k);
        const size_t extra_rows=(fixed_plans || accumulate) ? std::max<size_t>(128,(expert_tile+127)/128*128) : 0;
        const size_t full_device_bytes=lf_mul(lf_mul(n,d+width_slots+width_h+shared_width+hc+k),sizeof(float))+extra_rows*d*sizeof(float);
        const bool full_device_fits=full_device_bytes<=device_limit && device_limit-full_device_bytes>=8192;
        const bool resident_data=resident_requested && (!accumulate || full_device_fits);
        lf_require(!accumulate || cpu_cutoff==0,"accumulation requires CPU experts disabled");
        const size_t device_bytes=lf_mul(lf_mul(n,d+width_slots+(resident_data ? width_h+shared_width+hc+k : 0)),sizeof(float))+extra_rows*d*sizeof(float);
        const bool use_device=device_limit && device_bytes<=device_limit && device_limit-device_bytes>=8192;
        const bool full_resident=use_device && resident_data;
        lf_require(!(require_resident || defer_shared) || full_resident,
                   "token state requires full GPU residency; increase the device workspace budget and free VRAM from resident experts");
        lf_require(!adaptive_residency || (full_resident && resident_fast),
                   "adaptive expert residency requires resident token state and batched resident experts");
        lf_require(!accum_audit || (use_device && n<=4096),"accumulator audit requires GPU storage and at most 4096 tokens");
        const size_t weight_limit=lf_env("LLAMA_MOE_LAYER_FIRST_WEIGHT_MIB",128,1,8192)*(size_t(1)<<20);
        const size_t host_limit=lf_env("LLAMA_MOE_LAYER_FIRST_HOST_MIB",32768,1,262144)*(size_t(1)<<20);
        const size_t row_floats=(full_resident ? 0 : width_h+shared_width+hc)+(use_device ? 0 : d+width_slots)+k;
        const size_t host_bytes=lf_mul(n,lf_mul(row_floats,sizeof(float))+lf_mul(k,sizeof(int32_t)));
        lf_require(host_bytes<=host_limit,"layer-first host activation budget exceeded; no chunk-first fallback was run");
        lf_require(token_tile>0 && d>0 && hc>0 && k>0 && k<=ne,"invalid layer-first geometry");
        for (int il=0;il<model.hparams.n_layer();++il) {
            const auto & l=model.layers[il];
            for (auto * w:{l.ffn_up_exps,l.ffn_gate_exps,l.ffn_down_exps}) {
                lf_require(w->buffer && ggml_is_contiguous(w) && w->ne[2]==int64_t(ne),"unsupported expert tensor layout");
            }
        }
        std::vector<float> hidden(full_resident ? 0 : lf_mul(n,width_h));
        std::vector<float> x(use_device ? 0 : lf_mul(n,d));
        std::vector<float> shared(full_resident ? 0 : lf_mul(n,shared_width)),inject(full_resident ? 0 : lf_mul(n,hc));
        std::vector<float> route_weight(lf_mul(n,k)),slot_out(use_device ? 0 : lf_mul(n,width_slots));
        auto host_row=[](std::vector<float> & values,size_t off)->float * {
            return values.empty() ? nullptr : values.data()+off;
        };
        std::vector<int32_t> route_id(lf_mul(n,k));
        std::vector<uint8_t> route_needed(prepared_routes ? ne : 0);
        std::vector<size_t> route_counts(prepared_routes ? ne : 0);
        std::vector<std::vector<size_t>> prepared_assigned(prepared_routes ? ne : 0);
        size_t route_scan_slots=0,assignment_scan_slots=0,resident_plan_rebinds=0;
        // Avoid silently switching tiny expert tails to decode/MMV kernels.
        // The logical tile count controls progress; padding only fixes GEMM geometry.
        const size_t physical_tile=std::max<size_t>(128,(expert_tile+127)/128*128);
        std::vector<float> tile_input(lf_mul(physical_tile,d)),tile_output(lf_mul(physical_tile,d));
        std::vector<int32_t> tile_ids(physical_tile),tile_tokens(physical_tile);
        std::vector<int64_t> tile_slots(physical_tile);
        std::vector<int32_t> sum_tokens(physical_tile),sum_map(lf_mul(physical_tile,k));
        std::vector<float> tile_weight(physical_tile);
        std::vector<int32_t> token_owner(accumulate ? n : 0,-1),sum_count(physical_tile);
        size_t upload_bytes=0,upload_count=0,activation_set_bytes=0,activation_get_bytes=0,compute_tiles=0;
        size_t graph_builds=0,graph_reuses=0,resident_layers=0,weight_copy_calls=0;
        size_t promotion_bytes=0,promotion_hits=0,promotion_allocations=0,promotion_evictions=0;
        double promotion_ms=0;
        std::array<double,8> stage_wall_ms={};
        size_t cpu_experts=0,cpu_assignments=0,cpu_saved_bytes=0;double cpu_total_ms=0,cpu_wait_ms=0;
        double build_ms=0,input_ms=0,compute_ms=0,read_ms=0,weight_ms=0,bank_ms=0,gather_ms=0,scatter_ms=0;
        auto milliseconds = [](auto start) {
            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        };
        const auto started=std::chrono::steady_clock::now();
        LLAMA_LOG_INFO("layer-first: window=%zu mixer_chunk=%zu expert_rows=%zu group_limit=%zu host_workspace=%zu weight_budget=%zu\n",
                       n,token_tile,expert_tile,group_limit,host_bytes,weight_limit);
        synchronize();
        std::map<int,size_t> promotion_targets;
        std::map<std::pair<int,size_t>,size_t> promotion_groups;
        if(adaptive_residency || layer_first_residency) {
            ggml_backend_sched_reset(sched.get());
            gf_res_prev->reset();
            if(!layer_first_residency)layer_first_residency=std::make_shared<llama_lf_residency>();
            const size_t spare=adaptive_residency && full_device_bytes+8192<device_limit
                ? device_limit-full_device_bytes-8192 : 0;
            std::map<ggml_backend_dev_t,size_t> ordinal,remaining,counts;
            std::map<int,size_t> layer_bytes;
            std::map<ggml_backend_dev_t,std::vector<int>> host_layers;
            for(int il=0;il<int(model.hparams.n_layer());++il) {
                auto * dev=model.dev_layer(il);
                if(!ordinal.count(dev)) {const size_t next=ordinal.size();ordinal[dev]=next;}
                const auto & l=model.layers[il];
                const std::array<ggml_tensor *,3> source={l.ffn_up_exps,l.ffn_gate_exps,l.ffn_down_exps};
                const size_t di=ordinal.at(dev);
                const size_t base=adaptive_residency && di<base_resident_layers.size() ? base_resident_layers[di] : 0;
                const bool host=std::all_of(source.begin(),source.end(),[](auto * w){return ggml_backend_buffer_is_host(w->buffer);});
                lf_require(!base || host,"base residency budget requires all expert layers on that device to have host sources");
                if(!host)continue;
                size_t bytes=8192;for(auto * w:source)bytes+=ggml_nbytes(w);
                layer_bytes[il]=bytes;
                host_layers[dev].push_back(il);
            }
            for(const auto & device:host_layers) {
                const size_t di=ordinal.at(device.first);
                const size_t base=adaptive_residency && di<base_resident_layers.size() ? base_resident_layers[di] : 0;
                lf_require(base<=device.second.size(),"base residency exceeds device layer count");
                const size_t kv_savings=adaptive_residency ? mctx->get_attn()->layer_rotation_saved_bytes(device.first) : 0;
                remaining[device.first]=spare+kv_savings;
                if(kv_savings)LLAMA_LOG_INFO("layer-first: KV_RESIDENCY_BUDGET device=%zu reclaimed_bytes=%zu\n",di,kv_savings);
                for(size_t i=0;i<base;++i)remaining[device.first]+=layer_bytes.at(device.second[device.second.size()-1-i]);
            }
            for(int il=int(model.hparams.n_layer())-1;il>=0;--il) {
                auto * dev=model.dev_layer(il);
                const auto & l=model.layers[il];
                const std::array<ggml_tensor *,3> source={l.ffn_up_exps,l.ffn_gate_exps,l.ffn_down_exps};
                if(!std::all_of(source.begin(),source.end(),[](auto * w){return ggml_backend_buffer_is_host(w->buffer);}))continue;
                const size_t di=ordinal.at(dev);
                const size_t base=adaptive_residency && di<base_resident_layers.size() ? base_resident_layers[di] : 0;
                size_t bytes=8192;for(auto * w:source)bytes+=ggml_nbytes(w);
                if(!adaptive_residency || counts[dev]>=base+adaptive_max_layers || bytes>remaining[dev])continue;
                promotion_targets[il]=bytes;remaining[dev]-=bytes;++counts[dev];
            }
            if(stagger_residency)for(const auto & device:host_layers) {
                const auto & layers=device.second;const size_t selected=counts[device.first];
                if(!selected || selected==layers.size())continue; // retain the full-layer fast path
                lf_require(ne%group_limit==0,"staggered group size must divide expert count");
                size_t total_weight=0,target_weight=0,selected_weight=0,cumulative=0;
                for(int il:layers) {
                    total_weight+=layer_bytes.at(il)-8192;
                    if(promotion_targets.count(il))target_weight+=promotion_targets.at(il)-8192;
                }
                for(int il:layers) {
                    const size_t bytes=(layer_bytes.at(il)-8192)/ne*resident_group_limit;
                    for(size_t first=0;first<ne;first+=resident_group_limit) {
                        cumulative+=bytes;
                        const long double target=(long double)cumulative*target_weight/total_weight;
                        if(target-selected_weight>=bytes && bytes<=target_weight-selected_weight) {
                            promotion_groups[{il,first}]=bytes+8192;selected_weight+=bytes;
                        }
                    }
                }
                LLAMA_LOG_INFO("layer-first: STAGGER_BUDGET device=%zu weight_budget=%zu resident_weight=%zu unused_weight_budget=%zu group=%zu\n",
                               ordinal.at(device.first),target_weight,selected_weight,target_weight-selected_weight,resident_group_limit);
                for(int il:layers)promotion_targets.erase(il);
            }
            for(auto it=layer_first_residency->layers.begin();it!=layer_first_residency->layers.end();) {
                if(!promotion_targets.count(it->first)) {it=layer_first_residency->layers.erase(it);++promotion_evictions;}
                else ++it;
            }
            for(auto it=layer_first_residency->groups.begin();it!=layer_first_residency->groups.end();) {
                if(!promotion_groups.count(it->first) || it->second.valid.size()!=resident_group_limit) {it=layer_first_residency->groups.erase(it);++promotion_evictions;}
                else ++it;
            }
            LLAMA_LOG_INFO("layer-first: RESIDENCY_PLAN tokens=%zu workspace_bytes=%zu spare_per_device=%zu target_layers=%zu target_groups=%zu evicted_entries=%zu stagger=%d\n",
                           n,full_device_bytes,spare,promotion_targets.size(),promotion_groups.size(),promotion_evictions,int(stagger_residency));
        }
        if(lf_env("LLAMA_MOE_PROFILE",0,0,1)) {
            for(int il=0;il<int(model.hparams.n_layer());++il) {
                const auto & l=model.layers[il];
                const std::array<ggml_tensor *,3> source={l.ffn_up_exps,l.ffn_gate_exps,l.ffn_down_exps};
                for(size_t j=0;j<3;++j) {
                    auto * w=source[j];
                    lf_profile_scope span("LF.model_span layer=%d projection=%zu ptr=%llu bytes=%zu stride=%zu host=%d",
                        il,j,(unsigned long long)(uintptr_t)w->data,ggml_nbytes(w),w->nb[2],int(ggml_backend_buffer_is_host(w->buffer)));
                }
            }
            auto mark_cached=[&](int il,size_t first,const llama_lf_residency::entry & entry) {
                for(size_t j=0;j<3;++j) {
                    auto * w=entry.tensor[j];
                    lf_profile_scope span("LF.cache_span layer=%d projection=%zu first=%zu count=%lld ptr=%llu bytes=%zu",
                        il,j,first,(long long)w->ne[2],(unsigned long long)(uintptr_t)w->data,ggml_nbytes(w));
                }
            };
            if(layer_first_residency) {
                for(const auto & entry:layer_first_residency->layers)mark_cached(entry.first,0,entry.second);
                for(const auto & entry:layer_first_residency->groups)mark_cached(entry.first.first,entry.first.second,entry.second);
            }
        }
        lf_require(mctx->apply(),"could not apply full layer-first cache window");
        applied=true;
        lf_require(mctx->get_recr()->get_n_rs()==1,"layer-first needs one recurrent state row");
        auto * result=gf_res_prev.get();
        auto choose_backend=[&](ggml_backend_dev_t device)->ggml_backend_t {
            for (auto & b:backends) { if (ggml_backend_get_device(b.get())==device) { return b.get(); } }
            throw std::runtime_error("no backend for layer-first stage");
        };
        // Only pure expert graphs can reuse their allocation. Their weight
        // descriptors remain owned by the current lease (or by the model).
        ggml_cgraph * expert_graph=nullptr;
        std::array<ggml_tensor *,3> graph_weights={};
        ggml_backend_t graph_backend=nullptr;
        uint32_t graph_rows=0,graph_reference_tokens=0;
        size_t graph_token_offset=0;
        int graph_kind=-1,graph_layer=-1;
        size_t graph_real_rows=0,device_activation_peak=0,device_copy_bytes=0;
        std::array<ggml_tensor *,7> device_activations={};
        std::map<ggml_backend_t,std::unique_ptr<lf_activation_store>> stores;
        std::map<ggml_backend_t,std::unique_ptr<lf_weight_pool>> weight_pools;
        std::map<ggml_backend_t,std::unique_ptr<lf_plan_scratch>> resident_scratch;
        std::map<std::string,std::unique_ptr<lf_expert_plan>> resident_plan_cache;
        std::map<std::string,std::unique_ptr<lf_expert_plan>> plans;
        size_t plan_clock=0,plan_peak_bytes=0,plan_evictions=0;
        llama_lf_input_cache inputs_cache(lf_env("LLAMA_MOE_LAYER_FIRST_MASK_MIB",0,0,65536)*(size_t(1)<<20));
        ggml_tensor * previous_hidden=nullptr;
        size_t resident_boundary_bytes=0;
        // Detach graph views before request-owned device storage is released on any exit.
        lf_graph_cleanup stores_cleanup{sched.get(),result,true};
        size_t stage_token_offset=0;
        ggml_tensor * accum_values=nullptr;
        std::function<void()> prefetch_next;
        auto stage=[&](int kind,int layer,const llama_ubatch & u,const llama_memory_context_i * mc,
                       const std::array<const void *,5> & data,ggml_backend_t backend,
                       const std::array<ggml_tensor *,3> & weights=std::array<ggml_tensor *,3>{},size_t real_rows=0,uint32_t reference_tokens=0,bool cached_group=false) {
            lf_profile_scope stage_scope("LF.stage kind=%d layer=%d rows=%u",kind,layer,u.n_tokens);
            const auto stage_begin=std::chrono::steady_clock::now();
            lf_profile_scope build_scope("LF.build");
            auto begin=std::chrono::steady_clock::now();
            // Distinct resident addresses must not each acquire a private scratch
            // allocator and evict the reusable rotating plans. Use the scheduler's
            // bounded shared scratch for those groups; retain the same graph math.
            const bool shared_resident_plan=cached_group && resident_plans;
            if(kind==3 && fixed_plans && use_device && !cparams.cb_eval && (!cached_group || !resident_shared_scratch || resident_plans)) {
                std::ostringstream key;
                key<<backend<<":"<<u.n_tokens<<":"<<reference_tokens<<":"<<std::hexfloat<<model.hparams.swiglu_clamp_exp[layer]<<":"<<device_activations[0]->data<<":"<<device_activations[1]->data;
                if(shared_resident_plan)key<<":resident";
                for(auto * w:weights) {
                    if(!shared_resident_plan)key<<":"<<w->data;
                    key<<":"<<int(w->type)<<":"<<w->ne[0]<<":"<<w->ne[1]<<":"<<w->ne[2];
                }
                const std::string plan_key=key.str();
                const bool pooled_resident=shared_resident_plan && plan_shared_scratch;
                auto & plan_cache=pooled_resident ? resident_plan_cache : plans;
                if(plan_cache.size()>=32 && !plan_cache.count(plan_key)) {
                    lf_profile_scope evict("LF.plan_evict entries=%zu resident=%d",plan_cache.size(),int(pooled_resident));
                    auto oldest=std::min_element(plan_cache.begin(),plan_cache.end(),[](const auto & a,const auto & b) {
                        return a.second->last_used<b.second->last_used;
                    });
                    plan_cache.erase(oldest);++plan_evictions;
                }
                auto & plan=plan_cache[plan_key];
                if(!plan) {
                    lf_profile_scope create("LF.plan_create layer=%d rows=%u",layer,u.n_tokens);
                    plan=std::make_unique<lf_expert_plan>();plan->backend=backend;
                    plan->result=std::make_unique<llm_graph_result>(512);
                    auto params=graph_params(plan->result.get(),u,mc,LLM_GRAPH_TYPE_DEFAULT);
                    params.lf_stage=kind;params.lf_layer=layer;params.lf_activations=device_activations;
                    params.lf_valid_rows=real_rows;params.lf_reference_tokens=reference_tokens;params.lf_accumulate=accumulate;params.lf_accum_audit=accum_audit;params.lf_defer_shared=defer_shared;params.n_outputs=0;
                    // Own descriptors independently of the rotating group's descriptors.
                    for(size_t j=0;j<3;++j) {
                        auto * t=ggml_dup_tensor(plan->result->get_ctx(),weights[j]);
                        t->buffer=weights[j]->buffer;t->data=weights[j]->data;
                        params.lf_weights[j]=t;plan->weight_leaves[j]=t;
                    }
                    plan->graph=model.build_graph(params);
                    lf_require(plan->result->inputs.empty(),"expert plan owns stateful model input");
                    if(pooled_resident) {
                        auto remember=[&](ggml_tensor * t) {
                            if(t && !t->buffer && (!t->data || t->view_src) &&
                               std::find(plan->scratch_tensors.begin(),plan->scratch_tensors.end(),t)==plan->scratch_tensors.end())
                                plan->scratch_tensors.push_back(t);
                        };
                        for(int i=0;i<plan->graph->n_leafs;++i)remember(plan->graph->leafs[i]);
                        for(int i=0;i<plan->graph->n_nodes;++i) {
                            auto * t=plan->graph->nodes[i];remember(t);
                            for(auto * src:t->src)remember(src);
                        }
                    } else {
                        plan->allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                        lf_require(ggml_gallocr_alloc_graph(plan->allocator,plan->graph),"fixed expert plan allocation");
                    }
                    ++graph_builds;
                } else {++graph_reuses;}
                plan->last_used=++plan_clock;
                if(shared_resident_plan) {
                    lf_profile_scope binding("LF.plan_bind layer=%d rows=%u",layer,u.n_tokens);
                    bool changed=false;
                    for(size_t j=0;j<3;++j) {
                        auto * t=plan->weight_leaves[j];
                        lf_require(t && t->type==weights[j]->type && ggml_are_same_shape(t,weights[j]) &&
                                   std::memcmp(t->nb,weights[j]->nb,sizeof(t->nb))==0,"resident plan weight layout changed");
                        changed|=t->data!=weights[j]->data || t->buffer!=weights[j]->buffer;
                        t->data=weights[j]->data;t->buffer=weights[j]->buffer;
                    }
                    // Force the backend to inspect changed weight addresses.
                    if(changed) {plan->graph->uid=0;++resident_plan_rebinds;}
                }
                if(pooled_resident) {
                    lf_profile_scope scratch_scope("LF.plan_scratch layer=%d rows=%u",layer,u.n_tokens);
                    auto & scratch=resident_scratch[backend];
                    if(!scratch) {
                        scratch=std::make_unique<lf_plan_scratch>();scratch->backend=backend;
                        scratch->allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                    }
                    // Previous compute has synchronized; only request-owned scratch is rebound.
                    for(auto * t:plan->scratch_tensors) {t->data=nullptr;t->buffer=nullptr;t->extra=nullptr;}
                    lf_require(ggml_gallocr_reserve(scratch->allocator,plan->graph),"resident scratch reservation");
                    lf_require(ggml_gallocr_alloc_graph(scratch->allocator,plan->graph),"resident scratch allocation");
                    plan->graph->uid=0;
                }
                size_t bytes=0;
                for(const auto & entry:plans)bytes+=ggml_gallocr_get_buffer_size(entry.second->allocator,0);
                for(const auto & entry:resident_scratch)bytes+=ggml_gallocr_get_buffer_size(entry.second->allocator,0);
                plan_peak_bytes=std::max(plan_peak_bytes,bytes);
                build_ms+=milliseconds(begin);build_scope.end();begin=std::chrono::steady_clock::now();
                lf_profile_scope fixed_input_scope("LF.input");
                for(size_t j=0;j<plan->result->lf_in.size();++j)if(auto * t=plan->result->lf_in[j]) {
                    lf_require(data[j],"fixed expert input missing");
                    ggml_backend_tensor_set_async(backend,t,data[j],0,ggml_nbytes(t));activation_set_bytes+=ggml_nbytes(t);
                }
                // Finish the small input transfers before queueing the next bulk copy.
                ggml_backend_synchronize(backend);input_ms+=milliseconds(begin);fixed_input_scope.end();
                if(prefetch_next){auto next=std::move(prefetch_next);prefetch_next={};next();}
                begin=std::chrono::steady_clock::now();
                lf_require(ggml_backend_graph_compute(backend,plan->graph)==GGML_STATUS_SUCCESS,"fixed expert compute failed");
                if(accum_audit)accum_values=plan->result->lf_out[1];
                compute_ms+=milliseconds(begin);stage_wall_ms.at(kind)+=milliseconds(stage_begin);return;
            }
            ggml_backend_sched_synchronize(sched.get());
            const bool pure_experts=kind==3 || kind==6;
            const bool hit=reuse_graph && pure_experts && expert_graph &&
                graph_kind==kind && graph_layer==layer && graph_rows==u.n_tokens &&
                (kind!=6 || graph_token_offset==stage_token_offset) &&
                graph_backend==backend && graph_weights==weights && graph_real_rows==real_rows && graph_reference_tokens==reference_tokens;
            ggml_cgraph * graph=expert_graph;
            if (hit) {
                // Inputs below are rewritten on every tile, including routing.
                // No position-dependent model/cache input belongs to these graphs.
                lf_require(result->inputs.empty(),"expert graph unexpectedly owns stateful inputs");
                ++graph_reuses;
            } else {
                expert_graph=nullptr;
                { lf_profile_scope detail("LF.build_reset kind=%d",kind);
                  ggml_backend_sched_reset(sched.get());result->reset(); }
                ggml_backend_sched_set_eval_callback(sched.get(),cparams.cb_eval,cparams.cb_eval_user_data);
                auto params=graph_params(result,u,mc,LLM_GRAPH_TYPE_DEFAULT);
                params.lf_stage=kind;params.lf_layer=layer;params.lf_weights=weights;
                params.lf_activations=device_activations;params.lf_token_offset=stage_token_offset;
                params.lf_valid_rows=real_rows;params.lf_reference_tokens=reference_tokens;params.lf_accumulate=accumulate;params.lf_accum_audit=accum_audit;params.lf_defer_shared=defer_shared;
                params.n_outputs=kind==5 ? u.n_tokens : 0;
                if (kind==1 && layer==int(model.hparams.n_layer())-1) {
                    params.n_outputs=std::count_if(u.output,u.output+u.n_tokens,[](int8_t v){return v!=0;});
                }
                if(kind==1 && inputs_cache.limit) {
                    inputs_cache.backend=backend;inputs_cache.offset=stage_token_offset;
                    result->lf_input_cache=&inputs_cache;
                }
                { lf_profile_scope detail("LF.build_model kind=%d",kind);graph=model.build_graph(params); }
                lf_require(graph!=nullptr,"could not construct layer-first graph");
                for (auto * t:result->lf_in) { if (t) { ggml_backend_sched_set_tensor_backend(sched.get(),t,backend); } }
                { lf_profile_scope detail("LF.build_allocate kind=%d",kind);
                  lf_require(ggml_backend_sched_alloc_graph(sched.get(),graph),"could not allocate bounded stage graph"); }
                { lf_profile_scope detail("LF.build_inputs kind=%d",kind);result->set_inputs(&u); }
                ++graph_builds;
                if (pure_experts) {
                    lf_require(result->inputs.empty(),"expert graph unexpectedly owns stateful inputs");
                    expert_graph=graph;graph_kind=kind;graph_layer=layer;graph_rows=u.n_tokens;
                    graph_token_offset=stage_token_offset;
                    graph_backend=backend;graph_weights=weights;graph_real_rows=real_rows;graph_reference_tokens=reference_tokens;
                }
            }
            build_ms+=milliseconds(begin);build_scope.end();
            lf_profile_scope input_scope("LF.input");
            begin=std::chrono::steady_clock::now();
            const bool batch_inputs=async_inputs && pure_experts && use_device;
            for (size_t j=0;j<result->lf_in.size();++j) {
                if (auto * t=result->lf_in[j]) {
                    lf_require(data[j]!=nullptr,"missing layer-first stage input");
                    if(batch_inputs)ggml_backend_tensor_set_async(backend,t,data[j],0,ggml_nbytes(t));
                    else ggml_backend_tensor_set(t,data[j],0,ggml_nbytes(t));
                    activation_set_bytes+=ggml_nbytes(t);
                }
            }
            if(batch_inputs)ggml_backend_synchronize(backend);
            input_ms+=milliseconds(begin);input_scope.end();
            // Queue next weights only after the current tile's small inputs are
            // ready; otherwise those inputs can wait behind the next large DMA.
            if(pure_experts && prefetch_next) {
                auto next=std::move(prefetch_next);prefetch_next={};next();
            }
            lf_profile_scope compute_scope("LF.compute");
            begin=std::chrono::steady_clock::now();
            const auto rc=graph_compute(graph,u.n_tokens>1);
            lf_require(rc==GGML_STATUS_SUCCESS,"layer-first stage compute failed");
            ggml_backend_sched_synchronize(sched.get());
            compute_ms+=milliseconds(begin);
            if(accum_audit && kind==3)accum_values=result->lf_out[1];
            stage_wall_ms.at(kind)+=milliseconds(stage_begin);
        };
        auto read=[&](int which,void * target,size_t expected) {
            auto * t=result->lf_out[which];
            lf_require(t && ggml_is_contiguous(t) && ggml_nbytes(t)==expected,"layer-first output layout mismatch");
            const auto begin=std::chrono::steady_clock::now();
            lf_profile_scope read_scope("LF.read bytes=%zu",expected);
            ggml_backend_tensor_get(t,target,0,expected);activation_get_bytes+=expected;
            read_ms+=milliseconds(begin);
        };
        auto device_store_output=[&](int which,int target,size_t off,size_t row_width) {
            auto * src=result->lf_out[which];
            lf_require(src && ggml_is_contiguous(src),"resident output must be contiguous");
            auto * dst=ggml_view_4d(result->get_ctx(),device_activations.at(target),
                src->ne[0],src->ne[1],src->ne[2],src->ne[3],src->nb[1],src->nb[2],src->nb[3],off*row_width*sizeof(float));
            lf_require(ggml_backend_view_init(dst)==GGML_STATUS_SUCCESS,"resident output view failed");
            lf_profile_scope copy_scope("LF.store output=%d target=%d bytes=%zu source=%s same_address=%d",
                which,target,ggml_nbytes(src),ggml_get_name(src),int(src->data==dst->data));
            ggml_backend_tensor_copy(src,dst);device_copy_bytes+=ggml_nbytes(src);
        };
        const auto * rotating_attn=mctx->get_attn();
        struct kv_ring_cleanup {
            const llama_kv_cache_context * cache;
            ~kv_ring_cleanup(){if(cache)cache->layer_rotation_end();}
        } kv_cleanup{rotating_attn};
        rotating_attn->layer_rotation_begin();
        for (int il=0;il<model.hparams.n_layer();++il) {
            // Attention/state still consumes the complete chronological request.
            // Only the final layer's token-local work is compacted to requested
            // outputs, exactly where the ordinary model performs that pruning.
            const bool final_layer=il==int(model.hparams.n_layer())-1;
            const size_t n=final_layer ? size_t(n_outputs) : request_tokens;
            struct work_chunk {size_t source_start,source_count,start,count;};
            std::vector<work_chunk> chunks;
            std::vector<uint32_t> origin_sizes(n);
            size_t next_row=0;
            for(size_t off=0;off<request_tokens;off+=token_tile) {
                const size_t cnt=std::min(token_tile,request_tokens-off);
                const size_t used=final_layer ? size_t(std::count_if(full.output+off,full.output+off+cnt,
                    [](int8_t v){return v!=0;})) : cnt;
                chunks.push_back({off,cnt,next_row,used});
                if(used) {std::fill_n(origin_sizes.data()+next_row,used,uint32_t(used));}
                next_row+=used;
            }
            lf_require(next_row==n,"selected-output row count mismatch");
            auto * backend=choose_backend(model.dev_layer(il));
            lf_require(std::strncmp(ggml_backend_name(backend),"CUDA",4)==0,"layer-first currently requires CUDA layer backends");
            lf_profile_scope layer_scope("LF.layer id=%d",il);
            const auto prev_stage=stage_wall_ms;
            const double prev_weight=weight_ms,prev_bank=bank_ms,prev_build=build_ms,prev_in=input_ms,prev_read=read_ms,prev_compute=compute_ms;
            const auto layer_begin=std::chrono::steady_clock::now();
            const size_t bytes_before=upload_bytes,count_before=upload_count;
            ggml_context_ptr activation_context;
            ggml_backend_buffer_ptr activation_buffer;
            lf_graph_cleanup activation_cleanup{sched.get(),result,use_device || resident_fast};
            bool finished_resident=false;
            device_activations={};
            if (use_device && resident_data) {
                auto & store=stores[backend];
                if (!store) {
                    store=std::make_unique<lf_activation_store>();
                    ggml_init_params init={ggml_tensor_overhead()*16,nullptr,true};
                    store->context.reset(ggml_init(init));lf_require(bool(store->context),"resident descriptors failed");
                    const std::array<size_t,7> widths={d,width_slots,width_h,width_h,shared_width,hc,k};
                    for(size_t j=0;j<widths.size();++j) {
                        if(!widths[j]) {continue;}
                        // The input slice is dead when its attention stage finishes.
                        // Reuse it for the residual consumed by the final layer stage.
                        store->tensor[j]=j==3 ? ggml_view_tensor(store->context.get(),store->tensor[2])
                            : ggml_new_tensor_2d(store->context.get(),GGML_TYPE_F32,
                                j==1 ? d : widths[j],j==1 ? request_tokens*output_k+extra_rows : request_tokens);
                        ggml_format_name(store->tensor[j],"lf_request_store_%zu",j);
                    }
                    store->buffer.reset(ggml_backend_alloc_ctx_tensors(store->context.get(),backend));
                    lf_require(bool(store->buffer),"resident activation allocation failed");
                    lf_require(ggml_backend_buffer_get_size(store->buffer.get())<=device_limit,"resident budget exceeded");
                    device_activation_peak=std::max(device_activation_peak,ggml_backend_buffer_get_size(store->buffer.get()));
                }
                device_activations=store->tensor;
                if(previous_hidden && previous_hidden!=device_activations[2]) {
                    lf_profile_scope scope("LF.device_boundary bytes=%zu",ggml_nbytes(previous_hidden));
                    ggml_backend_tensor_copy(previous_hidden,device_activations[2]);
                    resident_boundary_bytes+=ggml_nbytes(previous_hidden);
                }
                previous_hidden=device_activations[2];
            } else if(use_device && accumulate) {
                auto & store=stores[backend];
                if(!store) {
                    store=std::make_unique<lf_activation_store>();
                    store->context.reset(ggml_init({ggml_tensor_overhead()*4,nullptr,true}));
                    lf_require(bool(store->context),"partial resident descriptors");
                    store->tensor[0]=ggml_new_tensor_2d(store->context.get(),GGML_TYPE_F32,d,request_tokens);
                    store->tensor[1]=ggml_new_tensor_2d(store->context.get(),GGML_TYPE_F32,d,request_tokens+extra_rows);
                    store->buffer.reset(ggml_backend_alloc_ctx_tensors(store->context.get(),backend));
                    lf_require(bool(store->buffer),"partial resident allocation");
                    const size_t size=ggml_backend_buffer_get_size(store->buffer.get());
                    lf_require(size<=device_limit,"partial resident budget exceeded");
                    device_activation_peak=std::max(device_activation_peak,size);
                }
                device_activations=store->tensor;
            } else if (use_device && n>0) {
                const size_t descriptors=16+chunks.size();
                ggml_init_params init={lf_mul(ggml_tensor_overhead(),descriptors),nullptr,true};
                activation_context.reset(ggml_init(init));lf_require(bool(activation_context),"device activation descriptors failed");
                device_activations[0]=ggml_new_tensor_2d(activation_context.get(),GGML_TYPE_F32,d,n);
                device_activations[1]=ggml_new_tensor_2d(activation_context.get(),GGML_TYPE_F32,d,lf_mul(n,output_k)+extra_rows);
                ggml_set_name(device_activations[0],"layer_first_device_x");
                ggml_set_name(device_activations[1],"layer_first_device_slots");
                activation_buffer.reset(ggml_backend_alloc_ctx_tensors(activation_context.get(),backend));
                lf_require(bool(activation_buffer),"device activation allocation failed");
                const size_t size=ggml_backend_buffer_get_size(activation_buffer.get());
                lf_require(size<=device_limit,"device activation allocation exceeded budget");
                device_activation_peak=std::max(device_activation_peak,size);
                // Missing scatter rows remain NaN rather than recycling old answers.
                ggml_backend_buffer_clear(activation_buffer.get(),0xff);
            }
            if(accumulate) {
                if(use_device) {
                    auto * dst=device_activations[1];
                    ggml_backend_tensor_memset(dst,0,0,ggml_nbytes(dst));
                } else {std::fill(slot_out.begin(),slot_out.end(),0.0f);}
            }
            if(!model.hparams.is_recr(il))rotating_attn->layer_rotation_enter(il,backend);
            // Entire token window reaches the current layer's MoE input first.
            for (const auto & chunk:chunks) {
                const size_t off=chunk.start,cnt=chunk.count;
                lf_slice part(full,chunk.source_start,chunk.source_count);
                auto view=mctx->layer_slice(part.batch,chunk.source_start,chunk.source_start>0);
                stage_token_offset=chunk.source_start;
                stage(1,il,part.batch,view.get(),{host_row(hidden,chunk.source_start*width_h),nullptr,nullptr,nullptr,nullptr},backend);
                if (!cnt) {continue;}
                if (use_device && resident_data) {
                    device_store_output(0,3,off,width_h);
                    device_store_output(1,0,off,d);
                    device_store_output(2,5,off,hc);
                    continue;
                }
                read(0,host_row(hidden,off*width_h),cnt*width_h*sizeof(float));
                if (use_device) {
                    auto * dst=ggml_view_2d(result->get_ctx(),device_activations[0],d,cnt,
                        device_activations[0]->nb[1],off*d*sizeof(float));
                    lf_require(ggml_backend_view_init(dst)==GGML_STATUS_SUCCESS,"device activation view init failed");
                    ggml_backend_tensor_copy(result->lf_out[1],dst);
                    device_copy_bytes+=cnt*d*sizeof(float);
                } else { read(1,host_row(x,off*d),cnt*d*sizeof(float)); }
                read(2,host_row(inject,off*hc),cnt*hc*sizeof(float));
            }
            if(!model.hparams.is_recr(il))rotating_attn->layer_rotation_leave(il,backend);
            // Original router and shared FFN. Routing persists over all chunks.
            for (const auto & chunk:chunks) {
                const size_t off=chunk.start,cnt=chunk.count;
                if (!cnt) {continue;}
                lf_slice part(full,chunk.source_start,chunk.source_count);
                // Stages 2/4 are token-local: retain the original chunk's selected
                // row count; they do not update positions or sequence state.
                part.batch.n_tokens=part.batch.n_seq_tokens=cnt;
                stage_token_offset=off;
                stage(2,il,part.batch,mctx,{host_row(x,off*d),nullptr,nullptr,nullptr,nullptr},backend);
                read(1,route_id.data()+off*k,cnt*k*sizeof(int32_t));
                if (use_device && resident_data) {
                    if(!defer_shared) {device_store_output(0,4,off,d);}
                    device_store_output(2,6,off,k);
                    if(accumulate)read(2,route_weight.data()+off*k,cnt*k*sizeof(float));
                } else {
                    read(0,host_row(shared,off*d),cnt*d*sizeof(float));
                    read(2,route_weight.data()+off*k,cnt*k*sizeof(float));
                }
            }
            if(prepared_routes) {
                lf_profile_scope route_scope("LF.route_prepare layer=%d slots=%zu",il,n*k);
                std::fill(route_needed.begin(),route_needed.end(),0);
                std::fill(route_counts.begin(),route_counts.end(),0);
                for(size_t slot=0;slot<n*k;++slot) {
                    const int32_t e=route_id[slot];lf_require(e>=0 && size_t(e)<ne,"invalid prepared expert ID");
                    route_needed[e]=1;++route_counts[e];
                }
                route_scan_slots+=n*k;
            }
            const auto & current_layer=model.layers[il];
            std::array<ggml_tensor *,3> resident_source={current_layer.ffn_up_exps,current_layer.ffn_gate_exps,current_layer.ffn_down_exps};
            auto promote=[&](llama_lf_residency::entry & cached,size_t first,size_t count,size_t budget) {
                lf_profile_scope promotion_scope("LF.promotion layer=%d first=%zu count=%zu",il,first,count);
                const auto promotion_start=std::chrono::steady_clock::now();
                if(!cached.buffer) {
                    cached.context.reset(ggml_init({ggml_tensor_overhead()*8,nullptr,true}));
                    lf_require(bool(cached.context),"resident expert descriptors");
                    cached.source=resident_source;cached.device=model.dev_layer(il);cached.valid.assign(count,0);
                    for(size_t j=0;j<3;++j) {
                        auto * w=resident_source[j];
                        cached.tensor[j]=ggml_new_tensor_3d(cached.context.get(),w->type,w->ne[0],w->ne[1],count);
                        ggml_format_name(cached.tensor[j],"lf_resident_%d_%zu_projection_%zu",il,first,j);
                    }
                    cached.buffer.reset(ggml_backend_alloc_ctx_tensors(cached.context.get(),backend));
                    lf_require(bool(cached.buffer),"resident expert allocation");
                    lf_require(ggml_backend_buffer_get_size(cached.buffer.get())<=budget,"resident expert allocation exceeded budget");
                    ggml_backend_buffer_set_usage(cached.buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    ggml_backend_buffer_clear(cached.buffer.get(),0);
                    ++promotion_allocations;
                }
                lf_require(cached.source==resident_source && cached.device==model.dev_layer(il) && cached.valid.size()==count,"resident expert source changed");
                std::vector<uint8_t> local_needed(prepared_routes ? 0 : count,0);
                const uint8_t * needed=prepared_routes ? route_needed.data()+first : local_needed.data();
                if(!prepared_routes) {
                    for(size_t slot=0;slot<n*k;++slot) {
                        const int32_t e=route_id[slot];lf_require(e>=0 && size_t(e)<ne,"invalid promotion expert ID");
                        if(size_t(e)>=first && size_t(e)<first+count)local_needed[size_t(e)-first]=1;
                    }
                    route_scan_slots+=n*k;
                }
                for(size_t e=0;e<count;++e)if(needed[e] && cached.valid[e])promotion_hits+=3;
                for(size_t j=0;j<3;++j) {
                    auto * w=resident_source[j];
                    for(size_t e=0;e<count;) {
                        if(!needed[e] || cached.valid[e]){++e;continue;}
                        const size_t begin=e++;
                        if(coalesce)while(e<count && needed[e] && !cached.valid[e])++e;
                        const size_t bytes=lf_mul(e-begin,w->nb[2]);
                        lf_profile_scope wr("LF.weight layer=%d projection=%zu first=%zu count=%zu bytes=%zu",il,j,first+begin,e-begin,bytes);
                        ggml_backend_tensor_set(cached.tensor[j],static_cast<const char *>(w->data)+(first+begin)*w->nb[2],begin*w->nb[2],bytes);
                        upload_bytes+=bytes;promotion_bytes+=bytes;upload_count+=e-begin;++weight_copy_calls;
                    }
                }
                for(size_t e=0;e<count;++e)if(needed[e])cached.valid[e]=1;
                promotion_ms+=milliseconds(promotion_start);
            };
            if(promotion_targets.count(il)) {
                auto & cached=layer_first_residency->layers[il];
                promote(cached,0,ne,promotion_targets.at(il));resident_source=cached.tensor;
            } else {
                auto it=promotion_groups.lower_bound({il,0});
                for(;it!=promotion_groups.end() && it->first.first==il;++it)
                    promote(layer_first_residency->groups[it->first],it->first.second,resident_group_limit,it->second);
            }
            const bool all_resident=std::all_of(resident_source.begin(),resident_source.end(),[&](auto * w) {
                return !ggml_backend_buffer_is_host(w->buffer) &&
                    ggml_backend_buft_get_device(ggml_backend_buffer_get_type(w->buffer))==model.dev_layer(il);
            });
            if(resident_fast && all_resident && use_device) {
                for(const auto & chunk:chunks) {
                    if(!chunk.count)continue;
                    lf_slice part(full,chunk.source_start,chunk.source_count);
                    part.batch.n_tokens=part.batch.n_seq_tokens=chunk.count;
                    stage_token_offset=chunk.start;
                    stage(6,il,part.batch,mctx,{nullptr,route_id.data()+chunk.start*k,nullptr,accumulate ? route_weight.data()+chunk.start*k : nullptr,nullptr},backend,resident_source,0,chunk.count<=8 ? uint32_t(chunk.count) : 0);
                    device_store_output(0,1,chunk.start,width_slots);++compute_tiles;
                }
                ++resident_layers;
            } else if ((resident_batch || resident_fast) && all_resident && !use_device) {
                // All expert weights already live on this GPU. Their residency
                // spans the entire request, so token tiles need ZERO uploads.
                // Share/quantize each input once for all top-k selections rather
                // than host-gathering the same token into ten assignment rows.
                const size_t physical_tokens=std::max<size_t>(128,token_tile);
                const size_t scratch_bytes=lf_mul(physical_tokens,lf_mul(d+width_slots,sizeof(float))+lf_mul(k,sizeof(int32_t)));
                lf_require(host_bytes<=host_limit && scratch_bytes<=host_limit-host_bytes,
                           "resident expert scratch exceeds host budget");
                const bool local_slots=resident_fast && lf_mul(physical_tokens,width_slots*sizeof(float))+8192<=device_limit;
                std::vector<float> rx(lf_mul(physical_tokens,d)),ro(local_slots ? 0 : lf_mul(physical_tokens,width_slots));
                std::vector<int32_t> rid(lf_mul(physical_tokens,k));
                std::vector<float> rw(accumulate ? physical_tokens*k : 0);
                if(local_slots) {
                    activation_context.reset(ggml_init({ggml_tensor_overhead()*4,nullptr,true}));
                    lf_require(bool(activation_context),"resident chunk descriptors");
                    device_activations[1]=ggml_new_tensor_2d(activation_context.get(),GGML_TYPE_F32,d,physical_tokens*output_k);
                    activation_buffer.reset(ggml_backend_alloc_ctx_tensors(activation_context.get(),backend));
                    lf_require(bool(activation_buffer),"resident chunk slot buffer");
                    lf_require(ggml_backend_buffer_get_size(activation_buffer.get())<=device_limit,"resident chunk slot budget");
                    device_activation_peak=std::max(device_activation_peak,ggml_backend_buffer_get_size(activation_buffer.get()));
                }
                for (const auto & chunk:chunks) {
                    const size_t off=chunk.start,cnt=chunk.count;
                    if (!cnt) {continue;}
                    const size_t compute_n=cnt<=8 ? cnt : std::max<size_t>(128,cnt);
                    std::copy_n(host_row(x,off*d),cnt*d,rx.data());
                    std::copy_n(route_id.data()+off*k,cnt*k,rid.data());
                    if(accumulate)std::copy_n(route_weight.data()+off*k,cnt*k,rw.data());
                    for (size_t i=0;i<cnt*k;++i) { lf_require(rid[i]>=0 && size_t(rid[i])<ne,"invalid resident expert ID"); }
                    for (size_t i=cnt;i<compute_n;++i) {
                        std::copy_n(rx.data(),d,rx.data()+i*d);
                        std::copy_n(rid.data(),k,rid.data()+i*k);
                        if(accumulate)std::copy_n(rw.data(),k,rw.data()+i*k);
                    }
                    llama_ubatch u=full;u.n_tokens=u.n_seq_tokens=compute_n;
                    stage(6,il,u,mctx,{rx.data(),rid.data(),nullptr,accumulate ? rw.data() : nullptr,nullptr},backend,resident_source,0,cnt<=8 ? uint32_t(cnt) : 0);
                    if(local_slots) {
                        device_store_output(0,1,0,width_slots);
                        u.n_tokens=u.n_seq_tokens=cnt;stage_token_offset=0;
                        stage(4,il,u,mctx,{host_row(hidden,off*width_h),nullptr,
                            route_weight.data()+off*k,host_row(shared,off*d),host_row(inject,off*hc)},backend);
                        read(0,host_row(hidden,off*width_h),cnt*width_h*sizeof(float));
                    } else {
                        read(0,ro.data(),compute_n*width_slots*sizeof(float));
                        std::copy_n(ro.data(),cnt*width_slots,host_row(slot_out,off*width_slots));
                    }
                    ++compute_tiles;
                }
                finished_resident=local_slots;
                ++resident_layers;
            } else {
            std::vector<float> audit_order(accum_audit ? n*d : 0,0.0f);
            std::vector<float> audit_slots(accum_audit ? n*k*d : 0);
            std::vector<float> audit_values(accum_audit ? physical_tile*d : 0);
            std::vector<std::vector<size_t>> local_assigned(prepared_routes ? 0 : ne);
            auto & assigned=prepared_routes ? prepared_assigned : local_assigned;
            {
                lf_profile_scope assignment_scope("LF.assign layer=%d slots=%zu",il,n*k);
                if(prepared_routes)for(size_t e=0;e<ne;++e) {
                    assigned[e].clear();assigned[e].reserve(route_counts[e]);
                }
                for(size_t slot=0;slot<n*k;++slot) {
                    const int32_t e=route_id[slot];lf_require(e>=0 && size_t(e)<ne,"router produced invalid expert ID");
                    assigned[e].push_back(slot);
                }
                assignment_scan_slots+=n*k;
            }
            const auto & layer=model.layers[il];
            const std::array<ggml_tensor *,3> source={layer.ffn_up_exps,layer.ffn_gate_exps,layer.ffn_down_exps};
            size_t per_expert=0;
            for (auto * w:source) { if (ggml_backend_buffer_is_host(w->buffer)) { per_expert+=w->nb[2]; } }
            if(const char * file=std::getenv("LLAMA_MOE_ROUTE_DUMP")) {
                FILE * fp=std::fopen(file,"a");lf_require(fp,"route audit open failed");
                for(size_t e=0;e<ne;++e) {
                    std::fprintf(fp,"%d,%zu,%zu,%zu,%zu,%zu,%zu,%d,%d,%d\n",il,e,assigned[e].size(),per_expert,
                        source[0]->nb[2],source[1]->nb[2],source[2]->nb[2],int(source[0]->type),int(source[1]->type),int(source[2]->type));
                }
                std::fclose(fp);
            }
            if(const char * dir=std::getenv("LLAMA_MOE_CAPTURE_DIR")) {
                if(il==0 || il==2 || il==4 || il==17) {
                    std::vector<float> snapshot(n*d);
                    if(use_device)ggml_backend_tensor_get(device_activations[0],snapshot.data(),0,snapshot.size()*sizeof(float));
                    else snapshot.assign(x.begin(),x.begin()+n*d);
                    char name[4096];std::snprintf(name,sizeof(name),"%s/layer-%d-x.f32",dir,il);
                    FILE * f=std::fopen(name,"wb");lf_require(f,"capture input open failed");
                    const size_t nw=std::fwrite(snapshot.data(),sizeof(float),snapshot.size(),f);std::fclose(f);lf_require(nw==snapshot.size(),"capture input write failed");
                    std::snprintf(name,sizeof(name),"%s/layer-%d-ids.i32",dir,il);f=std::fopen(name,"wb");lf_require(f,"capture route open failed");
                    const size_t ni=std::fwrite(route_id.data(),sizeof(int32_t),n*k,f);std::fclose(f);lf_require(ni==n*k,"capture route write failed");
                }
            }
            std::vector<llama_lf_cpu_task> cpu_tasks;
            std::vector<int64_t> cpu_slots;
            std::vector<float> cpu_input;
            std::future<llama_lf_cpu_result> cpu_future;
            size_t layer_cpu_saved=0;
            if(cpu_cutoff && per_expert && std::all_of(source.begin(),source.end(),[](auto * w){return ggml_backend_buffer_is_host(w->buffer);})) {
                std::vector<size_t> candidates;
                size_t required_bytes=0;
                for(size_t e=0;e<ne;++e) {
                    if(!assigned[e].empty())required_bytes+=per_expert;
                    if(!assigned[e].empty() && assigned[e].size()<=cpu_cutoff)candidates.push_back(e);
                }
                std::stable_sort(candidates.begin(),candidates.end(),[&](size_t a,size_t b){return assigned[a].size()<assigned[b].size();});
                for(size_t e:candidates) {
                    const size_t proposed_rows=cpu_slots.size()+assigned[e].size();
                    const size_t remaining=required_bytes-layer_cpu_saved-per_expert;
                    // Conservative admission estimate, not a speed guarantee: one
                    // millisecond per assignment per worker plus a 2 ms allowance.
                    const double estimated_ms=2.0+double(proposed_rows)/cpu_workers;
                    const double allowed_ms=0.7*double(remaining)/13.0e6;
                    if(estimated_ms>allowed_ms)continue;
                    llama_lf_cpu_task task{e,assigned[e],cpu_slots.size()};
                    for(size_t slot:task.slots)cpu_slots.push_back(int64_t(slot));
                    cpu_tasks.push_back(std::move(task));assigned[e].clear();layer_cpu_saved+=per_expert;
                }
                if(!cpu_tasks.empty()) {
                    lf_profile_scope range("LF.cpu_input layer=%d rows=%zu",il,n);
                    cpu_input.resize(n*d);
                    if(use_device) {
                        ggml_backend_tensor_get(device_activations[0],cpu_input.data(),0,cpu_input.size()*sizeof(float));
                        activation_get_bytes+=cpu_input.size()*sizeof(float);
                    } else std::copy_n(x.data(),cpu_input.size(),cpu_input.data());
                    cpu_experts+=cpu_tasks.size();cpu_assignments+=cpu_slots.size();cpu_saved_bytes+=layer_cpu_saved;
                    cpu_future=std::async(std::launch::async,[&,limit=model.hparams.swiglu_clamp_exp[il],layer_index=il] {
                        lf_profile_scope cpu_range("LF.cpu_compute layer=%d tasks=%zu rows=%zu",layer_index,cpu_tasks.size(),cpu_slots.size());
                        return llama_lf_cpu_run(source,cpu_tasks,cpu_input,d,k,cpu_slots.size(),cpu_workers,limit);
                    });
                }
            }
            const bool pipeline=async_weights && per_expert>0;
            const size_t bank_capacity=weight_limit/(pipeline ? 2 : 1);
            const size_t reserve_padding=3*4096;
            lf_require(bank_capacity>reserve_padding && (per_expert==0 || per_expert<=bank_capacity-reserve_padding),
                       "one expert does not fit the complete ring budget");
            const size_t group=std::min({group_limit,ne,per_expert ? (bank_capacity-reserve_padding)/per_expert : group_limit});
            std::vector<uint8_t> written(n*k,0),uploaded(ne*3,0);
            struct group_work {
                size_t first,count;std::vector<size_t> slots;
                llama_lf_residency::entry * cached=nullptr;
                size_t ring=0;bool prepared=false;
            };
            std::vector<group_work> groups;
            size_t rotating_index=0;
            lf_require(!stagger_residency || group==group_limit,"staggered groups must fit a ring bank");
            for(size_t first=0;first<ne;) {
                const bool cached=promotion_groups.count({il,first})!=0;
                const size_t count=cached ? resident_group_limit : std::min(group,ne-first);
                group_work item{first,count,{}};
                if(prepared_routes) {
                    size_t size=0;for(size_t e=first;e<first+count;++e)size+=assigned[e].size();
                    item.slots.reserve(size);
                }
                for(size_t e=first;e<first+count;++e)item.slots.insert(item.slots.end(),assigned[e].begin(),assigned[e].end());
                if(!item.slots.empty()) {
                    if(cached)item.cached=&layer_first_residency->groups.at({il,first});
                    else item.ring=pipeline ? rotating_index++%2 : 0;
                    groups.push_back(std::move(item));
                }
                first+=count;
            }
            lf_weight_pool * pool=nullptr;
            if(reuse_banks && per_expert) {
                auto & entry=weight_pools[backend];
                if(!entry) {
                    entry=std::make_unique<lf_weight_pool>();entry->compute=backend;
                    const int banks=pipeline ? 2 : 1;
                    for(int j=0;j<banks;++j) {
                        entry->buffer[j].reset(ggml_backend_alloc_buffer(backend,bank_capacity));
                        lf_require(bool(entry->buffer[j]),"persistent weight bank allocation");
                        ggml_backend_buffer_set_usage(entry->buffer[j].get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                        ggml_backend_buffer_clear(entry->buffer[j].get(),0);
                    }
                    if(pipeline) {
                        auto * dev=ggml_backend_get_device(backend);
                        entry->transfer.reset(ggml_backend_dev_init(dev,nullptr));
                        lf_require(bool(entry->transfer),"copy backend creation");
                        for(int j=0;j<2;++j) {
                            entry->ready[j]=ggml_backend_event_new(dev);entry->consumed[j]=ggml_backend_event_new(dev);
                            lf_require(entry->ready[j] && entry->consumed[j],"copy completion events");
                        }
                    }
                }
                pool=entry.get();
            }
            std::array<std::unique_ptr<lf_weights>,2> banks;
            lf_graph_cleanup banks_cleanup{sched.get(),result,true};
            auto prepare_group=[&](size_t index) {
                auto & item=groups.at(index);const size_t first=item.first,count=item.count;
                lf_require(!item.cached && !item.prepared,"invalid rotating group preparation");
                const size_t bi=item.ring;
                lf_profile_scope range("LF.prepare_group layer=%d group=%zu bank=%zu",il,index,bi);
                auto preparation=std::chrono::steady_clock::now();
                if(pipeline && pool->used[bi])ggml_backend_event_synchronize(pool->consumed[bi]);
                // Clear unused expert rows too: MMQ padding may read their scales.
                if(reuse_banks && per_expert)
                    ggml_backend_buffer_clear(pool->buffer[bi].get(),0);
                auto owned=std::make_unique<lf_weights>();auto & bank=*owned;
                ggml_init_params init={ggml_tensor_overhead()*16,nullptr,true};
                bank.context.reset(ggml_init(init));lf_require(bool(bank.context),"weight descriptor allocation failed");
                for(size_t j=0;j<3;++j) {
                    auto * w=source[j];bank.uploaded[j]=ggml_backend_buffer_is_host(w->buffer);
                    if(bank.uploaded[j])bank.tensor[j]=ggml_new_tensor_3d(bank.context.get(),w->type,w->ne[0],w->ne[1],count);
                    else bank.tensor[j]=ggml_view_3d(bank.context.get(),w,w->ne[0],w->ne[1],count,w->nb[1],w->nb[2],first*w->nb[2]);
                    ggml_format_name(bank.tensor[j],"lf_blk_%d_projection_%zu",il,j);
                }
                if(reuse_banks) {
                    if(per_expert) {
                        auto alloc=ggml_tallocr_new(pool->buffer[bi].get());
                        for(size_t j=0;j<3;++j) {
                            if(bank.uploaded[j])lf_require(ggml_tallocr_alloc(&alloc,bank.tensor[j])==GGML_STATUS_SUCCESS,"weight bank capacity");
                            else lf_require(ggml_backend_view_init(bank.tensor[j])==GGML_STATUS_SUCCESS,"resident view");
                        }
                    } else for(auto * t:bank.tensor)lf_require(ggml_backend_view_init(t)==GGML_STATUS_SUCCESS,"resident view");
                } else {
                    bank.buffer.reset(ggml_backend_alloc_ctx_tensors(bank.context.get(),backend));
                    lf_require(bank.buffer || !per_expert,"weight bank allocation");
                    if(bank.buffer) {
                        lf_require(ggml_backend_buffer_get_size(bank.buffer.get())<=weight_limit,"weight budget exceeded");
                        ggml_backend_buffer_set_usage(bank.buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                        ggml_backend_buffer_clear(bank.buffer.get(),0);
                    }
                }
                bank_ms+=milliseconds(preparation);preparation=std::chrono::steady_clock::now();
                for(size_t j=0;j<3;++j) {
                    if(!bank.uploaded[j])continue;
                    auto * w=source[j];
                    for(size_t e=first;e<first+count;) {
                        if(assigned[e].empty()){++e;continue;}
                        const size_t begin=e++;
                        if(coalesce)while(e<first+count && !assigned[e].empty())++e;
                        for(size_t id=begin;id<e;++id)lf_require(uploaded[id*3+j]++==0,"RELOAD: duplicate expert upload");
                        const size_t bytes=lf_mul(e-begin,w->nb[2]);
                        lf_profile_scope wr("LF.weight layer=%d projection=%zu first=%zu count=%zu bytes=%zu",il,j,begin,e-begin,bytes);
                        const void * data=static_cast<const char *>(w->data)+begin*w->nb[2];
                        if(pipeline)ggml_backend_tensor_set_async(pool->transfer.get(),bank.tensor[j],data,(begin-first)*w->nb[2],bytes);
                        else ggml_backend_tensor_set(bank.tensor[j],data,(begin-first)*w->nb[2],bytes);
                        upload_bytes+=bytes;upload_count+=e-begin;++weight_copy_calls;
                    }
                }
                if(pipeline)ggml_backend_event_record(pool->ready[bi],pool->transfer.get());
                weight_ms+=milliseconds(preparation);banks[bi]=std::move(owned);item.prepared=true;
            };
            for(size_t gi=0;gi<groups.size();++gi)if(!groups[gi].cached){prepare_group(gi);break;}
            // Each group owns its bank through ALL assigned tiles. The next bank
            // is filled asynchronously only after its previous last-reader event.
            for(size_t gi=0;gi<groups.size();++gi) {
                auto & item=groups[gi];const size_t bi=item.ring;
                if(!item.cached && !item.prepared)prepare_group(gi);
                lf_weights resident_lease;
                if(item.cached)resident_lease.tensor=item.cached->tensor;
                auto & bank=item.cached ? resident_lease : *banks[bi];
                const size_t first=item.first,count=item.count;auto & slots=item.slots;
                lf_profile_scope group_scope("LF.group layer=%d first=%zu count=%zu assignments=%zu resident=%d bank=%zu",il,first,count,slots.size(),int(item.cached!=nullptr),bi);
                if(pipeline) {
                    if(!item.cached)ggml_backend_event_wait(backend,pool->ready[bi]);
                    for(size_t next=gi+1;next<groups.size();++next)if(!groups[next].cached) {
                        if(!groups[next].prepared)prefetch_next=[&,next]{prepare_group(next);};
                        break; // never overwrite an already-prefetched bank by skipping farther ahead
                    }
                }
                // CUDA's quantized vector path applies to at most eight original
                // token rows. Route those assignments separately, retaining this
                // same weight lease. A per-projection hint below preserves whether
                // its quantization/device uses vector or matrix arithmetic.
                auto origin_tokens = [&](size_t slot) -> uint32_t {
                    return origin_sizes.at(slot/k);
                };
                const size_t bulk_end=size_t(std::stable_partition(slots.begin(),slots.end(),[&](size_t slot) {
                    return origin_tokens(slot)>8;
                })-slots.begin());
                for (size_t off=0;off<slots.size();) {
                    const bool scalar=off>=bulk_end;
                    const uint32_t reference_tokens=scalar ? origin_tokens(slots[off]) : 0;
                    const auto gather_begin=std::chrono::steady_clock::now();
                    const size_t cnt=scalar ? 1 : std::min(expert_tile,bulk_end-off);
                    for (size_t j=0;j<cnt;++j) {
                        const size_t slot=slots[off+j],token=slot/k;
                        if (!use_device) { std::copy_n(host_row(x,token*d),d,tile_input.data()+j*d); }
                        tile_tokens[j]=int32_t(token);tile_slots[j]=int64_t(slot);
                        tile_ids[j]=route_id[slot]-int32_t(first);
                    }
                    // Preserve the multi-token MMVQ reduction, not just the MMVQ/MMQ
                    // family. Extra columns repeat this real row; only cnt is scattered.
                    const size_t compute_n=scalar ? reference_tokens : std::max<size_t>(128,fixed_plans ? (cnt+127)/128*128 : cnt);
                    for (size_t j=cnt;j<compute_n;++j) {
                        if (!use_device) { std::copy_n(tile_input.data(),d,tile_input.data()+j*d); }
                        tile_tokens[j]=tile_tokens[0];tile_ids[j]=tile_ids[0];
                        tile_slots[j]=int64_t((resident_data ? request_tokens : n)*k+j);
                    }
                    if(accumulate && use_device) {
                        std::fill_n(sum_map.data(),compute_n*k,int32_t(compute_n));
                        std::fill_n(tile_weight.data(),compute_n,0.0f);
                        std::fill_n(sum_count.data(),compute_n,0);
                        size_t unique=0;
                        for(size_t j=0;j<cnt;++j) {
                            const size_t slot=slots[off+j],token=slot/k;
                            int32_t & owner=token_owner[token];
                            if(owner<0) {owner=int32_t(unique);sum_tokens[unique++]=int32_t(token);}
                            const size_t rank=sum_count[owner]++;
                            lf_require(rank<k,"too many expert contributions per token");
                            sum_map[rank*compute_n+owner]=int32_t(j);
                            tile_weight[j]=route_weight[slot];
                        }
                        for(size_t j=0;j<unique;++j)token_owner[sum_tokens[j]]=-1;
                        for(size_t j=unique;j<compute_n;++j)sum_tokens[j]=int32_t(n+j);
                    }
                    gather_ms+=milliseconds(gather_begin);
                    // These are assignment rows, not new sequence positions.
                    llama_ubatch u=full;u.n_tokens=u.n_seq_tokens=compute_n;
                    const void * expert_input=use_device ? static_cast<const void *>(tile_tokens.data()) : static_cast<const void *>(tile_input.data());
                    stage(3,il,u,mctx,{expert_input,tile_ids.data(),
                        accumulate ? static_cast<const void *>(sum_tokens.data()) : static_cast<const void *>(tile_slots.data()),
                        accumulate ? tile_weight.data() : nullptr,accumulate ? sum_map.data() : nullptr},
                        backend,bank.tensor,use_device ? ((fixed_plans || accumulate) ? compute_n : cnt) : 0,reference_tokens,item.cached!=nullptr);
                    if (!use_device) { read(0,tile_output.data(),compute_n*d*sizeof(float)); }
                    if(accum_audit && use_device) {
                        lf_require(accum_values && ggml_is_contiguous(accum_values),"missing accumulator audit values");
                        ggml_backend_tensor_get(accum_values,audit_values.data(),0,compute_n*d*sizeof(float));
                        for(size_t j=0;j<cnt;++j) {
                            const size_t slot=slots[off+j],token=slot/k;
                            std::copy_n(audit_values.data()+j*d,d,audit_slots.data()+slot*d);
                            for(size_t col=0;col<d;++col) {
                                volatile float product=audit_values[j*d+col]*route_weight[slot];
                                audit_order[token*d+col]+=product;
                            }
                        }
                    }
                    const auto scatter_begin=std::chrono::steady_clock::now();
                    for (size_t j=0;j<cnt;++j) {
                        const size_t slot=slots[off+j];lf_require(written[slot]++==0,"duplicate output slot");
                        if (!use_device) {
                            if(accumulate) {
                                float * sum=slot_out.data()+(slot/k)*d;
                                const float weight=route_weight[slot];
                                for(size_t col=0;col<d;++col) {
                                    volatile float product=tile_output[j*d+col]*weight;
                                    sum[col]+=product;
                                }
                            } else {std::copy_n(tile_output.data()+j*d,d,host_row(slot_out,slot*d));}
                        }
                    }
                    scatter_ms+=milliseconds(scatter_begin);
                    ++compute_tiles;
                    off+=cnt;
                }
                if(pipeline) {
                    lf_require(!prefetch_next,"group never dispatched its prefetch");
                    if(!item.cached) {ggml_backend_event_record(pool->consumed[bi],backend);pool->used[bi]=true;}
                }
                // All readers completed before bank release; no graph keeps dangling descriptors.
                expert_graph=nullptr; // invalidate before any descriptor can be freed/reused
                ggml_backend_sched_reset(sched.get());result->reset();
            }
            if(cpu_future.valid()) {
                const auto wait_begin=std::chrono::steady_clock::now();
                auto computed=cpu_future.get();const double wait=milliseconds(wait_begin);
                cpu_total_ms+=computed.ms;cpu_wait_ms+=wait;
                for(auto slot:cpu_slots)lf_require(written.at(size_t(slot))++==0,"CPU/GPU duplicate contribution");
                if(use_device) {
                    llama_ubatch u=full;u.n_tokens=u.n_seq_tokens=cpu_slots.size();
                    stage(7,il,u,mctx,{computed.values.data(),cpu_slots.data(),nullptr,nullptr,nullptr},backend);
                } else for(size_t j=0;j<cpu_slots.size();++j)std::copy_n(computed.values.data()+j*d,d,slot_out.data()+size_t(cpu_slots[j])*d);
                LLAMA_LOG_INFO("layer-first: CPU_LAYER layer=%d experts=%zu assignments=%zu saved_bytes=%zu cpu_ms=%.3f join_wait_ms=%.3f\n",il,cpu_tasks.size(),cpu_slots.size(),layer_cpu_saved,computed.ms,wait);
            }
            lf_require(std::all_of(written.begin(),written.end(),[](uint8_t v){return v==1;}),"unwritten expert contribution");
            if(accum_audit && use_device) {
                std::vector<float> actual(n*d);
                ggml_backend_tensor_get(device_activations[1],actual.data(),0,n*d*sizeof(float));
                double order_max=0,reference_max=0,squares=0;
                for(size_t token=0;token<n;++token)for(size_t col=0;col<d;++col) {
                    float ref=0.0f;
                    for(size_t j=0;j<k;++j) {
                        volatile float product=audit_slots[(token*k+j)*d+col]*route_weight[token*k+j];
                        ref+=product;
                    }
                    lf_require(std::isfinite(actual[token*d+col]),"nonfinite accumulator audit");
                    order_max=std::max(order_max,std::abs(double(actual[token*d+col])-audit_order[token*d+col]));
                    const double delta=double(actual[token*d+col])-ref;
                    reference_max=std::max(reference_max,std::abs(delta));squares+=delta*delta;
                }
                LLAMA_LOG_INFO("layer-first: ACCUM_AUDIT layer=%d order_max=%.9g reference_max=%.9g reference_rmse=%.9g\n",
                    il,order_max,reference_max,std::sqrt(squares/(n*d)));
                lf_require(order_max<1e-6,"accumulator differs from same-order CPU sum");
            }
            }
            if(!finished_resident) for (const auto & chunk:chunks) {
                const size_t off=chunk.start,cnt=chunk.count;
                if (!cnt) {continue;}
                lf_slice part(full,chunk.source_start,chunk.source_count);
                part.batch.n_tokens=part.batch.n_seq_tokens=cnt;
                stage_token_offset=off;
                stage(4,il,part.batch,mctx,{host_row(hidden,off*width_h),host_row(slot_out,off*width_slots),
                    route_weight.data()+off*k,host_row(shared,off*d),host_row(inject,off*hc)},backend);
                if(use_device && resident_data) {device_store_output(0,2,off,width_h);}
                else {read(0,host_row(hidden,off*width_h),cnt*width_h*sizeof(float));}
            }
            if (use_device || finished_resident) {
                // Detach all graph views before freeing this layer's activation lease.
                ggml_backend_sched_synchronize(sched.get());expert_graph=nullptr;
                ggml_backend_sched_reset(sched.get());result->reset();device_activations={};
            }
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-layer_begin).count();
            LLAMA_LOG_INFO("layer-first: LAYER_PROFILE layer=%d mix_ms=%.3f route_shared_ms=%.3f expert_ms=%.3f finish_ms=%.3f weight_ms=%.3f bank_ms=%.3f build_ms=%.3f input_ms=%.3f read_ms=%.3f compute_ms=%.3f\n",
                il,stage_wall_ms[1]-prev_stage[1],stage_wall_ms[2]-prev_stage[2],stage_wall_ms[3]-prev_stage[3]+stage_wall_ms[6]-prev_stage[6],
                stage_wall_ms[4]-prev_stage[4],weight_ms-prev_weight,bank_ms-prev_bank,build_ms-prev_build,input_ms-prev_in,read_ms-prev_read,compute_ms-prev_compute);
            LLAMA_LOG_INFO("layer-first: layer=%d tokens=%zu weight_h2d_bytes=%zu uploads=%zu reloads=0 ms=%.3f\n",
                il,request_tokens,upload_bytes-bytes_before,upload_count-count_before,ms);
        }
        // The output head is token-local and permanently resident. Preserve the
        // ordinary chronological chunk's requested-row geometry, rather than
        // changing its quantized matrix dispatch by merging every output row.
        // This does not revisit any expert group or change weight residency.
        const size_t n_vocab = model.vocab.n_tokens();
        std::vector<float> selected(full_resident ? 0 : lf_mul(std::min(n,token_tile),width_h));
        size_t row=0;
        for (size_t off=0;off<n;off+=token_tile) {
            const size_t count=std::min(token_tile,n-off);
            size_t selected_rows=0;
            for (size_t t=off;t<off+count;++t) {
                if (full.output[t]) {
                    if(!full_resident) {
                        std::copy_n(hidden.data()+(row+selected_rows)*width_h,width_h,
                                    selected.data()+selected_rows*width_h);
                    }
                    ++selected_rows;
                }
            }
            if (!selected_rows) { continue; }
            llama_ubatch u=full;u.n_tokens=u.n_seq_tokens=selected_rows;
            if(use_device && resident_data) {
                device_activations=stores.at(choose_backend(model.dev_output()))->tensor;
                stage_token_offset=row;
            }
            stage(5,-1,u,mctx,{selected.data(),nullptr,nullptr,nullptr,nullptr},choose_backend(model.dev_output()));
            lf_require(result->t_logits && logits.data &&
                lf_mul(row+selected_rows,n_vocab)<=logits.size,"layer-first output storage mismatch");
            ggml_backend_tensor_get(result->t_logits,logits.data+row*n_vocab,0,
                                    lf_mul(lf_mul(selected_rows,n_vocab),sizeof(float)));
            row+=selected_rows;
        }
        lf_require(row==size_t(n_outputs),"layer-first output count mismatch");
        // Outputs have been collected above before each head graph is reused.
        // The enclosing decode() must not copy the last chunk over the full result.
        result->t_logits=nullptr;
        if (!row) { ggml_backend_sched_reset(sched.get());result->reset(); }
        rotating_attn->layer_rotation_end();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        LLAMA_LOG_INFO("layer-first: COMPLETE window=%zu expert_rows=%zu weight_h2d_bytes=%zu uploads=%zu reloads=0 expert_tiles=%zu manual_activation_set_bytes=%zu manual_activation_get_bytes=%zu ms=%.3f\n",
            n,expert_tile,upload_bytes,upload_count,compute_tiles,activation_set_bytes,activation_get_bytes,ms);
        if(adaptive_residency || layer_first_residency) {
            size_t bytes=0;
            if(layer_first_residency)for(const auto & item:layer_first_residency->layers)bytes+=ggml_backend_buffer_get_size(item.second.buffer.get());
            if(layer_first_residency)for(const auto & item:layer_first_residency->groups)bytes+=ggml_backend_buffer_get_size(item.second.buffer.get());
            LLAMA_LOG_INFO("layer-first: ADAPTIVE_RESIDENCY enabled=%d layers=%zu groups=%zu allocated_bytes=%zu uploaded_bytes=%zu hit_projections=%zu allocations=%zu evictions=%zu promotion_ms=%.3f\n",
                           int(adaptive_residency),layer_first_residency ? layer_first_residency->layers.size() : 0,layer_first_residency ? layer_first_residency->groups.size() : 0,bytes,promotion_bytes,promotion_hits,promotion_allocations,promotion_evictions,promotion_ms);
        }
        size_t assignment_capacity=0;
        for(const auto & a:prepared_assigned)assignment_capacity+=a.capacity()*sizeof(size_t);
        LLAMA_LOG_INFO("layer-first: PREPARATION routes=%d resident_plans=%d route_scan_slots=%zu assignment_scan_slots=%zu assignment_capacity=%zu resident_plan_rebinds=%zu\n",
                      int(prepared_routes),int(resident_plans),route_scan_slots,assignment_scan_slots,assignment_capacity,resident_plan_rebinds);
        size_t resident_scratch_bytes=0;
        for(const auto & entry:resident_scratch)resident_scratch_bytes+=ggml_gallocr_get_buffer_size(entry.second->allocator,0);
        LLAMA_LOG_INFO("layer-first: RESIDENT_PLAN_CACHE shared_scratch=%d limit=32 entries=%zu scratch_bytes=%zu\n",
                      int(plan_shared_scratch),resident_plan_cache.size(),resident_scratch_bytes);
        LLAMA_LOG_INFO("layer-first: PLAN_CACHE limit=32 entries=%zu evictions=%zu peak_bytes=%zu\n",plans.size(),plan_evictions,plan_peak_bytes);
        LLAMA_LOG_INFO("layer-first: PROFILE window=%zu graph_builds=%zu graph_reuses=%zu resident_layers=%zu build_ms=%.3f input_ms=%.3f compute_ms=%.3f read_ms=%.3f\n",
            n,graph_builds,graph_reuses,resident_layers,build_ms,input_ms,compute_ms,read_ms);
        LLAMA_LOG_INFO("layer-first: TRANSFERS window=%zu weight_calls=%zu weight_ms=%.3f bank_ms=%.3f gather_ms=%.3f scatter_ms=%.3f\n",
            n,weight_copy_calls,weight_ms,bank_ms,gather_ms,scatter_ms);
        LLAMA_LOG_INFO("layer-first: DEVICE_WORKSPACE window=%zu enabled=%d peak_bytes=%zu budget=%zu input_copy_bytes=%zu\n",
            n,int(use_device),device_activation_peak,device_limit,device_copy_bytes);
        LLAMA_LOG_INFO("layer-first: ACCUMULATE enabled=%d full_device=%d bytes_per_token=%zu host_workspace=%zu\n",
            int(accumulate),int(use_device&&resident_data),(d+width_slots+(resident_data ? width_h+shared_width+hc+k : 0))*sizeof(float),host_bytes);
        LLAMA_LOG_INFO("layer-first: STATE_RESIDENCY required=%d full_device=%d defer_shared=%d state_host_bytes=%zu expert_input_host_bytes=%zu sum_host_bytes=%zu shared_host_bytes=%zu state_bytes=%zu\n",
                      int(require_resident),int(full_resident),int(defer_shared),hidden.size()*sizeof(float),
                      x.size()*sizeof(float),slot_out.size()*sizeof(float),shared.size()*sizeof(float),n*width_h*sizeof(float));
        LLAMA_LOG_INFO("layer-first: CPU_SUMMARY cutoff=%zu workers=%d experts=%zu assignments=%zu saved_bytes=%zu cpu_ms=%.3f join_wait_ms=%.3f\n",cpu_cutoff,cpu_workers,cpu_experts,cpu_assignments,cpu_saved_bytes,cpu_total_ms,cpu_wait_ms);
        LLAMA_LOG_INFO("layer-first: ASYNC enabled=%d pools=%zu total_weight_budget=%zu\n",int(async_weights),weight_pools.size(),weight_limit);
        LLAMA_LOG_INFO("layer-first: INPUT_CACHE hits=%zu misses=%zu host_reuses=%zu\n",inputs_cache.hits,inputs_cache.misses,inputs_cache.host_reuses);
        LLAMA_LOG_INFO("layer-first: RESIDENT_DATA enabled=%d boundary_bytes=%zu stores=%zu\n",int(use_device&&resident_data),resident_boundary_bytes,stores.size());
        status=GGML_STATUS_SUCCESS;return result;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("layer-first: %s\n",e.what());
        ggml_backend_sched_synchronize(sched.get());ggml_backend_sched_reset(sched.get());gf_res_prev->reset();
        if (applied) {
            memory->seq_rm(full.seq_id[0][0],-1,-1);
            LLAMA_LOG_ERROR("layer-first: invalidated affected sequence after partial execution\n");
        }
        status=GGML_STATUS_FAILED;return nullptr;
    }
}

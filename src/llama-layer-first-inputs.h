#pragma once
// Request-local, immutable attention-geometry inputs. Never cache token-dependent scores.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <map>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

struct llama_lf_cached_input {
    ggml_context * ctx=nullptr;
    ggml_backend_buffer_t host_buffer=nullptr,device_buffer=nullptr;
    ggml_tensor * host=nullptr,*device=nullptr;
    ggml_backend_t backend=nullptr;
    bool ready=false,host_filled=false;
    std::shared_ptr<llama_lf_cached_input> host_source;
    bool host_ready() const {return host_source ? host_source->host_ready() : host_filled;}
    void mark_host_ready() {if(host_source)host_source->mark_host_ready();else host_filled=true;}
    llama_lf_cached_input(ggml_backend_t b,const ggml_tensor * shape,std::shared_ptr<llama_lf_cached_input> source={}):backend(b),host_source(std::move(source)) {
        ctx=ggml_init({ggml_tensor_overhead()*4,nullptr,true});
        if(!ctx)throw std::runtime_error("mask cache descriptors");
        host=host_source ? host_source->host : ggml_dup_tensor(ctx,shape);device=ggml_dup_tensor(ctx,shape);
        // These geometry inputs are uploaded once and awaited before consumption.
        // Per-request page pinning costs more than it saves for these copies.
        // Keep a diagnostic pinned variant for the paired performance control.
        const char * pin=std::getenv("LLAMA_MOE_LAYER_FIRST_PINNED_MASKS");
        auto * htype=pin && pin[0]=='1' ? ggml_backend_dev_host_buffer_type(ggml_backend_get_device(b)) : ggml_backend_cpu_buffer_type();
        if(!htype)htype=ggml_backend_cpu_buffer_type();
        if(!host_source)host_buffer=ggml_backend_buft_alloc_buffer(htype,ggml_backend_buft_get_alloc_size(htype,host)+4096);
        device_buffer=ggml_backend_alloc_buffer(b,ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(b),device)+4096);
        if((!host_source && !host_buffer) || !device_buffer) {
            if(host_buffer)ggml_backend_buffer_free(host_buffer);
            if(device_buffer)ggml_backend_buffer_free(device_buffer);
            ggml_free(ctx);throw std::runtime_error("mask cache allocation");
        }
        auto da=ggml_tallocr_new(device_buffer);
        ggml_status host_status=GGML_STATUS_SUCCESS;
        if(host_buffer) {auto ha=ggml_tallocr_new(host_buffer);host_status=ggml_tallocr_alloc(&ha,host);}
        if(host_status!=GGML_STATUS_SUCCESS || ggml_tallocr_alloc(&da,device)!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("mask cache tensor allocation");
        ggml_set_name(device,"lf_cached_attention_geometry");
    }
    void upload() {
        if(ready)return;
        mark_host_ready();
        ggml_backend_tensor_set_async(backend,device,host->data,0,ggml_nbytes(host));
        ggml_backend_synchronize(backend);ready=true;
    }
    ~llama_lf_cached_input() {
        ggml_backend_synchronize(backend);
        ggml_backend_buffer_free(device_buffer);ggml_backend_buffer_free(host_buffer);ggml_free(ctx);
    }
};
struct llama_lf_input_cache {
    ggml_backend_t backend=nullptr;
    size_t offset=0,limit=0,hits=0,misses=0,host_reuses=0;
    bool share_host=false;
    std::map<ggml_backend_t,size_t> bytes;
    std::map<std::string,std::shared_ptr<llama_lf_cached_input>> values;
    std::map<std::string,std::weak_ptr<llama_lf_cached_input>> host_values;
    explicit llama_lf_input_cache(size_t cap):limit(cap) {
        const char * e=std::getenv("LLAMA_MOE_LAYER_FIRST_SHARE_MASK_HOST");share_host=e && e[0]=='1';
    }
    llama_lf_cached_input * bind(const std::string & label,const ggml_tensor * shape) {
        if(!shape || !backend || !limit)return nullptr;
        std::ostringstream key;key<<backend<<':'<<offset<<':'<<label<<':'<<int(shape->type);
        for(int i=0;i<4;++i)key<<':'<<shape->ne[i]<<':'<<shape->nb[i];
        auto it=values.find(key.str());if(it!=values.end()){++hits;return it->second.get();}
        const size_t allocation=ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(backend),shape)+4096;
        if(allocation>limit || bytes[backend]>limit-allocation)return nullptr;
        std::ostringstream host_key;host_key<<offset<<':'<<label<<':'<<int(shape->type);
        for(int i=0;i<4;++i)host_key<<':'<<shape->ne[i]<<':'<<shape->nb[i];
        auto source=share_host ? host_values[host_key.str()].lock() : nullptr;
        auto p=std::make_shared<llama_lf_cached_input>(backend,shape,source);auto * result=p.get();
        if(source)++host_reuses;else if(share_host)host_values[host_key.str()]=p;
        bytes[backend]+=allocation;++misses;values.emplace(key.str(),std::move(p));return result;
    }
};

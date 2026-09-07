// LD_PRELOAD profiling only: forwards every call and leaves model computation unchanged.
#include "llama.h"
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <string>
#include <unistd.h>
struct common_speculative;
struct common_sampler;
namespace {
using Clock=std::chrono::steady_clock;
struct Counter { long long calls=0, ns=0, tokens=0; };
struct Stats { std::map<std::string,Counter> data; std::mutex mutex; };
Stats & stats() { static auto * s=new Stats; return *s; }
thread_local const char * phase="target";
bool enabled() { const char * p=getenv("MTP_PROFILE_GATE"); return p && access(p,F_OK)==0; }
template<class T> T symbol(const char * name) { auto p=dlsym(RTLD_NEXT,name); if(!p) { fprintf(stderr,"profile symbol missing: %s\n",name); abort(); } return reinterpret_cast<T>(p); }
struct Scope {
 bool on; std::string name; const char * prev; Clock::time_point start; int tokens;
 Scope(std::string n,const char * p=nullptr,int t=0):on(enabled()),name(std::move(n)),prev(phase),tokens(t) {
  if(on) { start=Clock::now(); nvtxRangePushA(name.c_str()); }
  if(p)phase=p;
 }
 ~Scope() { phase=prev; if(on) {auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count(); nvtxRangePop(); std::lock_guard<std::mutex> l(stats().mutex); auto & c=stats().data[name]; c.calls++; c.ns+=ns; c.tokens+=tokens;} }
};
__attribute__((destructor)) void report() { for(const auto & p:stats().data)fprintf(stderr,"MTP_PROFILE %s calls=%lld ms=%.6f tokens=%lld\n",p.first.c_str(),p.second.calls,p.second.ns/1e6,p.second.tokens); }
}
void common_speculative_draft(common_speculative * p) {
 static auto f=symbol<void(*)(common_speculative*)>("_Z24common_speculative_draftP18common_speculative");
 Scope s("MTP/draft","draft"); f(p);
}
bool common_speculative_process(common_speculative * p,const llama_batch & b) {
 static auto f=symbol<bool(*)(common_speculative*,const llama_batch&)>("_Z26common_speculative_processP18common_speculativeRK11llama_batch");
 Scope s("MTP/catchup","catchup",b.n_tokens); return f(p,b);
}
bool common_speculative_flush_deferred(common_speculative * p) {
 static auto f=symbol<bool(*)(common_speculative*)>("_Z33common_speculative_flush_deferredP18common_speculative");
 Scope s("MTP/flush","flush"); return f(p);
}
bool common_speculative_flush_deferred_before_last(common_speculative * p) {
 static auto f=symbol<bool(*)(common_speculative*)>("_Z45common_speculative_flush_deferred_before_lastP18common_speculative");
 Scope s("MTP/flush_before_last","flush"); return f(p);
}
void common_speculative_accept(common_speculative * p,llama_seq_id seq,uint16_t n) {
 static auto f=symbol<void(*)(common_speculative*,llama_seq_id,uint16_t)>("_Z25common_speculative_acceptP18common_speculativeit");
 Scope s("MTP/accept","accept",n); f(p,seq,n);
}
llama_token common_sampler_sample(common_sampler * p,llama_context * ctx,int idx,bool grammar_first) {
 static auto f=symbol<llama_token(*)(common_sampler*,llama_context*,int,bool)>("_Z21common_sampler_sampleP14common_samplerP13llama_contextib");
 Scope s(std::string(phase)+"/sample"); return f(p,ctx,idx,grammar_first);
}
int32_t llama_decode(llama_context * ctx,llama_batch b) {
 static auto f=symbol<int32_t(*)(llama_context*,llama_batch)>("llama_decode");
 Scope s(std::string(phase)+"/decode",nullptr,b.n_tokens); return f(ctx,b);
}
float * llama_get_embeddings_nextn_ith(llama_context * ctx,int32_t i) {
 static auto f=symbol<float*(*)(llama_context*,int32_t)>("_Z30llama_get_embeddings_nextn_ithP13llama_contexti");
 Scope s(std::string(phase)+"/hidden"); return f(ctx,i);
}

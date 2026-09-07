// Profiling-only wrappers; no changes to scheduling, data, or sampling.
#include "llama.h"
#include <nvtx3/nvToolsExt.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
struct common_speculative; struct common_sampler;
static thread_local const char *component="target";
struct scope {
 const char *old;
 scope(const char*n,const char*k=nullptr):old(component){if(k)component=k;nvtxRangePushA(n);}
 ~scope(){nvtxRangePop();component=old;}
};
template<typename F> static F real(const char*n){void*p=dlsym(RTLD_NEXT,n);if(!p){std::fprintf(stderr,"missing hook %s\n",n);std::abort();}return reinterpret_cast<F>(p);}
void common_speculative_draft(common_speculative*s){static auto f=real<void(*)(common_speculative*)>("_Z24common_speculative_draftP18common_speculative");scope t("MTP/draft","draft");f(s);}
bool common_speculative_process(common_speculative*s,const llama_batch&b){static auto f=real<bool(*)(common_speculative*,const llama_batch&)>("_Z26common_speculative_processP18common_speculativeRK11llama_batch");scope t("MTP/refresh","refresh");return f(s,b);}
bool common_speculative_flush_deferred(common_speculative*s){static auto f=real<bool(*)(common_speculative*)>("_Z33common_speculative_flush_deferredP18common_speculative");scope t("MTP/flush","refresh");return f(s);}
bool common_speculative_flush_deferred_before_last(common_speculative*s){static auto f=real<bool(*)(common_speculative*)>("_Z45common_speculative_flush_deferred_before_lastP18common_speculative");scope t("MTP/flush_before_last","refresh");return f(s);}
llama_token common_sampler_sample(common_sampler*s,llama_context*c,int i,bool g){static auto f=real<llama_token(*)(common_sampler*,llama_context*,int,bool)>("_Z21common_sampler_sampleP14common_samplerP13llama_contextib");char n[96];std::snprintf(n,sizeof(n),"sample/%s/row=%d",component,i);scope t(n);return f(s,c,i,g);}
extern "C" int32_t llama_decode(llama_context*c,llama_batch b){static auto f=real<int32_t(*)(llama_context*,llama_batch)>("llama_decode");char n[96];std::snprintf(n,sizeof(n),"decode/%s/n=%d",component,b.n_tokens);scope t(n);return f(c,b);}
extern "C" int32_t llama_decode_mtp_kv(llama_context*c,llama_batch b){static auto f=real<int32_t(*)(llama_context*,llama_batch)>("llama_decode_mtp_kv");char n[96];std::snprintf(n,sizeof(n),"decode/refresh/n=%d",b.n_tokens);scope t(n,"refresh");return f(c,b);}
extern "C" bool llama_memory_seq_rm(llama_memory_t m,llama_seq_id s,llama_pos a,llama_pos b){static auto f=real<bool(*)(llama_memory_t,llama_seq_id,llama_pos,llama_pos)>("llama_memory_seq_rm");char n[160];std::snprintf(n,sizeof(n),"KV/remove/%s/seq=%d/p0=%d/p1=%d",component,s,a,b);scope t(n);return f(m,s,a,b);}
class server_tokens {public:std::vector<llama_token> get_text_tokens() const;};
std::vector<llama_token> server_tokens::get_text_tokens() const {static auto f=real<std::vector<llama_token>(*)(const server_tokens*)>("_ZNK13server_tokens15get_text_tokensEv");scope t("CPU/copy_prompt");return f(this);}

// Validation-only interposition. Never use this library for timing measurements.
#include "llama.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <atomic>
struct rows_info { int rows=0; bool masked=true; };
static std::unordered_map<llama_context *,rows_info> rows;
static std::atomic<unsigned long> outputs{0}, hidden{0};
template<class F> static F sym(const char *name) {
    void *p=dlsym(RTLD_NEXT,name);
    if (!p) {std::fprintf(stderr,"missing validation symbol %s\n",name); std::abort();}
    return reinterpret_cast<F>(p);
}
static void require(bool ok,const char *why) {if(!ok) {std::fprintf(stderr,"VIEW MISMATCH %s\n",why);std::abort();}}
extern "C" int32_t llama_decode(llama_context *c,llama_batch b) {
    static auto f=sym<int32_t(*)(llama_context*,llama_batch)>("llama_decode");
    int32_t rc=f(c,b); if(!rc) rows[c].rows=b.n_tokens; return rc;
}
// This internal API has C++ linkage.
void llama_set_embeddings_nextn(llama_context *c,bool enabled,bool masked) {
    static auto f=sym<void(*)(llama_context*,bool,bool)>("_Z26llama_set_embeddings_nextnP13llama_contextbb");
    f(c,enabled,masked);rows[c].masked=masked;
}
float *llama_get_embeddings_nextn(llama_context *c) {
    static auto f=sym<float*(*)(llama_context*)>("_Z26llama_get_embeddings_nextnP13llama_context");
    static auto one=sym<float*(*)(llama_context*,int32_t)>("_Z30llama_get_embeddings_nextn_ithP13llama_contexti");
    float *p=f(c);
    if(!rows[c].masked && p) {
        const int width=llama_model_n_embd_out(llama_get_model(c));
        for(int i=0;i<rows[c].rows;++i) {require(one(c,i)==p+(size_t)i*width,"dense hidden row");++hidden;}
    }
    return p;
}
extern "C" bool llama_get_sampling_output_ith(llama_context *c,int32_t i,llama_sampling_output *o) {
    static auto f=sym<bool(*)(llama_context*,int32_t,llama_sampling_output*)>("llama_get_sampling_output_ith");
    const bool ok=f(c,i,o);if(!ok) return false;
    require(o->token==llama_get_sampled_token_ith(c,i),"sampled token");
    require(o->probs==llama_get_sampled_probs_ith(c,i),"probability pointer");
    require(o->logits==llama_get_sampled_logits_ith(c,i),"sampled logits pointer");
    require(o->candidates==llama_get_sampled_candidates_ith(c,i),"candidate pointer");
    if(o->probs) require(o->n_probs==llama_get_sampled_probs_count_ith(c,i),"probability count");
    else if(o->logits) require(o->n_logits==llama_get_sampled_logits_count_ith(c,i),"logits count");
    else require(o->raw_logits==llama_get_logits_ith(c,i),"raw logits pointer");
    ++outputs;return true;
}
__attribute__((destructor)) static void report() {
    std::fprintf(stderr,"VIEW_CHECK outputs=%lu hidden_rows=%lu\n",outputs.load(),hidden.load());
}

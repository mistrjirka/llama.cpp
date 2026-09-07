#include "llama.h"
#include <nvtx3/nvToolsExt.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
struct common_speculative;
static thread_local const char * component = "target";
struct scope {
    const char * previous;
    scope(const char * name, const char * kind = nullptr) : previous(component) {
        if (kind) component = kind;
        nvtxRangePushA(name);
    }
    ~scope() { nvtxRangePop(); component = previous; }
};
static void * lookup(const char *name) {
    void *p=dlsym(RTLD_NEXT,name);
    if (!p) { std::fprintf(stderr,"profile hook: missing %s\n",name); std::abort(); }
    return p;
}
void common_speculative_draft(common_speculative *s) {
    static auto real=reinterpret_cast<void (*)(common_speculative *)>(lookup("_Z24common_speculative_draftP18common_speculative"));
    scope range("MTP/drafting", "draft"); real(s);
}
bool common_speculative_process(common_speculative *s, const llama_batch &b) {
    static auto real=reinterpret_cast<bool (*)(common_speculative *,const llama_batch &)>(lookup("_Z26common_speculative_processP18common_speculativeRK11llama_batch"));
    scope range("MTP/process", "process"); return real(s,b);
}
bool common_speculative_flush_deferred(common_speculative *s) {
    static auto real=reinterpret_cast<bool (*)(common_speculative *)>(lookup("_Z33common_speculative_flush_deferredP18common_speculative"));
    scope range("MTP/flush", "process"); return real(s);
}
bool common_speculative_flush_deferred_before_last(common_speculative *s) {
    static auto real=reinterpret_cast<bool (*)(common_speculative *)>(lookup("_Z45common_speculative_flush_deferred_before_lastP18common_speculative"));
    scope range("MTP/flush-before-last", "process"); return real(s);
}
extern "C" int32_t llama_decode(llama_context *ctx, llama_batch batch) {
    static auto real=reinterpret_cast<int32_t (*)(llama_context *,llama_batch)>(lookup("llama_decode"));
    char label[128];std::snprintf(label,sizeof(label),"decode/%s/n=%d",component,batch.n_tokens);
    scope range(label);return real(ctx,batch);
}

extern "C" int32_t llama_decode_mtp_kv(llama_context *ctx, llama_batch batch) {
    static auto real=reinterpret_cast<int32_t (*)(llama_context *,llama_batch)>(lookup("llama_decode_mtp_kv"));
    char label[128];std::snprintf(label,sizeof(label),"decode/%s/kv-only/n=%d",component,batch.n_tokens);
    scope range(label);return real(ctx,batch);
}

// Validation-only interposer: both paths run from the same restored draft state.
// Compare sequence K/V bytes; full-decode hidden/logit outputs are intentionally unused.
#include "llama.h"
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>
#include <vector>
static unsigned long calls=0, checks=0, total_bytes=0;
static std::set<std::pair<int,int>> shapes;
template<typename F> F symbol(const char *n) {
    void *p=dlsym(RTLD_NEXT,n);
    if (!p) {std::fprintf(stderr,"KV_CHECK missing symbol %s\n",n);std::abort();}
    return reinterpret_cast<F>(p);
}
static void require(bool ok,const char *message) {
    if (!ok) {std::fprintf(stderr,"KV_CHECK FAIL %s\n",message);std::abort();}
}
static std::vector<uint8_t> state(llama_context *c) {
    std::vector<uint8_t>b(llama_state_get_size(c));
    auto n=llama_state_get_data(c,b.data(),b.size());require(n>0,"save context");b.resize(n);return b;
}
static std::vector<uint8_t> seq_state(llama_context*c,llama_seq_id seq) {
    std::vector<uint8_t>b(llama_state_seq_get_size(c,seq));
    auto n=llama_state_seq_get_data(c,b.data(),b.size(),seq);require(n>0,"save sequence");b.resize(n);return b;
}
extern "C" int32_t llama_decode_mtp_kv(llama_context *c,llama_batch b) {
    static auto fast=symbol<int32_t(*)(llama_context*,llama_batch)>("llama_decode_mtp_kv");
    static auto full=symbol<int32_t(*)(llama_context*,llama_batch)>("llama_decode");
    ++calls;
    std::set<llama_seq_id> seqs;
    for(int i=0;i<b.n_tokens;++i)for(int j=0;j<b.n_seq_id[i];++j)seqs.insert(b.seq_id[i][j]);
    const auto shape=std::make_pair(b.n_tokens,(int)seqs.size());
    if(checks>=12 || !shapes.insert(shape).second)return fast(c,b);
    auto before=state(c);
    if(checks==0) {
        require(fast(nullptr,b)==-1,"null context rejected");
        llama_batch bad=b;bad.n_tokens=0;require(fast(c,bad)==-1,"empty batch rejected");
        std::vector<int8_t>flags(b.n_tokens,0);flags[0]=1;bad=b;bad.logits=flags.data();
        require(fast(c,bad)==-1,"output request rejected");
        bad=b;bad.logits=nullptr;require(fast(c,bad)==-1,"implicit output rejected");
        require(state(c)==before,"invalid requests changed state");
    }
    require(full(c,b)==0,"reference decode");
    llama_synchronize(c);
    std::map<llama_seq_id,std::vector<uint8_t>> expected;
    for(auto seq:seqs)expected[seq]=seq_state(c,seq);
    require(llama_state_set_data(c,before.data(),before.size())==before.size(),"restore baseline state");
    const int rc=fast(c,b);require(rc==0,"cache-only decode");llama_synchronize(c);
    for(auto seq:seqs) {
        auto actual=seq_state(c,seq);auto &ref=expected[seq];
        if(actual!=ref){
            size_t diff=0,first=0;for(size_t i=0;i<std::min(actual.size(),ref.size());++i)if(actual[i]!=ref[i]){if(!diff)first=i;++diff;}
            std::fprintf(stderr,"KV_CHECK mismatch seq=%d bytes=%zu/%zu first=%zu different=%zu n=%d\n",seq,actual.size(),ref.size(),first,diff,b.n_tokens);
            std::abort();
        }
        total_bytes+=actual.size();
    }
    ++checks;
    std::fprintf(stderr,"KV_CHECK PASS tokens=%d seqs=%zu checked=%lu bytes=%lu\n",b.n_tokens,seqs.size(),checks,total_bytes);
    return rc;
}
__attribute__((destructor)) static void report(){std::fprintf(stderr,"KV_CHECK TOTAL calls=%lu checks=%lu bytes=%lu\n",calls,checks,total_bytes);}

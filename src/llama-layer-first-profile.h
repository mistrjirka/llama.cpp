#pragma once
// Optional profiling hook. The default executable has no NVTX/CUDA dependency.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
extern "C" void lf_profile_push(const char *) __attribute__((weak));
extern "C" void lf_profile_pop() __attribute__((weak));
struct lf_profile_scope {
    bool active=false;
    lf_profile_scope(const char * fmt, ...) {
        const char * e=std::getenv("LLAMA_MOE_PROFILE");
        active=e && e[0]=='1' && lf_profile_push && lf_profile_pop;
        if(active) {char label[512];va_list ap;va_start(ap,fmt);std::vsnprintf(label,sizeof(label),fmt,ap);va_end(ap);lf_profile_push(label);}
    }
    void end() {if(active){lf_profile_pop();active=false;}}
    ~lf_profile_scope(){end();}
    lf_profile_scope(const lf_profile_scope&)=delete;
};

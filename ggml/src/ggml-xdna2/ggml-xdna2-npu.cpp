#include "ggml-xdna2-npu.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <cstring>

#include <sys/stat.h>

#ifndef GGML_XDNA2_KERNEL_DIR
#define GGML_XDNA2_KERNEL_DIR ""
#endif
#ifndef GGML_XDNA2_MLIR_AIE_ENV
#define GGML_XDNA2_MLIR_AIE_ENV ""
#endif

namespace xdna2 {

namespace {

std::string default_cache_dir() {
    if (const char * d = std::getenv("GGML_XDNA2_CACHE_DIR")) {
        if (*d) return d;
    }
    if (const char * x = std::getenv("XDG_CACHE_HOME")) {
        if (*x) return std::string(x) + "/ggml-xdna2";
    }
    if (const char * h = std::getenv("HOME")) {
        if (*h) return std::string(h) + "/.cache/ggml-xdna2";
    }
    return "ggml-xdna2-cache";
}

// Never available: used with the host reference mode, without XRT, or without an NPU.
class StubExecutor final : public NpuExecutor {
public:
    bool available() const override { return false; }
    std::unique_ptr<Arena> alloc_weight_arena(size_t) override { return nullptr; }
    void release_arena(const Arena *) override {}
    bool run(const MulMatJob *, size_t) override { return false; }
    ExecStats stats() const override { return ExecStats(); }
    void reset_stats() override {}
};

class NpuPath final : public ComputePath {
public:
    const char * name() const override { return "npu"; }
    bool supports(const layout::Geom & geom) const override { return placement_ok(geom); }
    bool mul_mat(const MulMatJob & job) override { return npu_executor().run(&job, 1); }
    bool mul_mat_batch(const MulMatJob * jobs, size_t n) override { return npu_executor().run(jobs, n); }
};

void kernel_log(int level, const std::string & msg) {
    switch (level) {
        case 0:  GGML_LOG_INFO ("ggml-xdna2: kernels: %s\n", msg.c_str()); break;
        case 1:  GGML_LOG_WARN ("ggml-xdna2: kernels: %s\n", msg.c_str()); break;
        default: GGML_LOG_ERROR("ggml-xdna2: kernels: %s\n", msg.c_str()); break;
    }
}

} // namespace

NpuExecutor & npu_executor() {
    static std::unique_ptr<NpuExecutor> e = []() -> std::unique_ptr<NpuExecutor> {
#ifdef GGML_XDNA2_HAVE_XRT
        if (!reference_mode() && npu_device_present()) {
            return make_xrt_executor();
        }
#endif
        return std::make_unique<StubExecutor>();
    }();
    return *e;
}

ComputePath & npu_path() {
    static NpuPath p;
    return p;
}

KernelCache & kernel_cache() {
    static KernelCache c([] {
        KernelCache::Config cfg;
        cfg.cache_dir      = default_cache_dir();
        cfg.kernel_src_dir = GGML_XDNA2_KERNEL_DIR;
        cfg.toolchain_env  = GGML_XDNA2_MLIR_AIE_ENV;
        cfg.max_jobs       = 4;
        cfg.log            = kernel_log;
        return cfg;
    }());
    return c;
}

KernelCache & prefill_kernel_cache() {
    static KernelCache c([] {
        KernelCache::Config cfg;
        cfg.cache_dir      = default_cache_dir();
        cfg.kernel_src_dir = GGML_XDNA2_KERNEL_DIR;
        cfg.toolchain_env  = GGML_XDNA2_MLIR_AIE_ENV;
        cfg.product        = Product::Prefill;
        cfg.max_jobs       = 4;
        cfg.log            = kernel_log;
        return cfg;
    }());
    return c;
}

KernelCache & attention_kernel_cache(const FaShape & shape) {
    static std::mutex mu;
    static std::map<std::string, std::unique_ptr<KernelCache>> caches;
    std::lock_guard<std::mutex> lk(mu);
    auto & c = caches[shape.variant()];
    if (!c) {
        KernelCache::Config cfg;
        cfg.cache_dir      = default_cache_dir();
        cfg.kernel_src_dir = GGML_XDNA2_KERNEL_DIR;
        cfg.toolchain_env  = GGML_XDNA2_MLIR_AIE_ENV;
        cfg.product        = Product::Attention;
        cfg.max_jobs       = 4;
        cfg.log            = kernel_log;
        cfg.variant        = shape.variant();
        cfg.build_args     = { std::to_string(shape.d), std::to_string(shape.nkv) };
        c = std::make_unique<KernelCache>(cfg);
    }
    return *c;
}

bool reference_mode() {
    const char * e = std::getenv("GGML_XDNA2_REFERENCE");
    return e && *e && std::strcmp(e, "0") != 0;
}

bool prefill_enabled() {
    const char * e = std::getenv("GGML_XDNA2_PF_DISABLE");
    return !(e && *e && std::strcmp(e, "0") != 0);
}

int prefill_min_tokens() {
    if (const char * e = std::getenv("GGML_XDNA2_PF_MIN")) {
        if (*e) return std::max(1, std::atoi(e));
    }
    return 2;      // measured (Qwen3.5-4B): the prefill kernel beats n per-token decode dispatches from 2 rows on (pp2 18.3 vs 13.2, pp16 122.5 vs 15.0 t/s)
}

bool attention_enabled() {
    const char * e = std::getenv("GGML_XDNA2_FA");
    return !(e && *e && std::strcmp(e, "0") == 0);
}

static int64_t env_i64(const char * name, int64_t def) {
    const char * e = std::getenv(name);
    return e && *e ? std::max<int64_t>(1, std::atoll(e)) : def;
}

int64_t attention_min_tokens() { return env_i64("GGML_XDNA2_FA_MIN_TOK", 32); }
int64_t attention_min_kv() { return env_i64("GGML_XDNA2_FA_MIN_KV", 1024); }

DecodeMode decode_mode() {
    static const DecodeMode m = [] {
        const char * e = std::getenv("GGML_XDNA2_DECODE");
        if (e && !std::strcmp(e, "cpu")) return DecodeMode::Cpu;
        if (e && !std::strcmp(e, "split")) return DecodeMode::Split;
        if (e && !std::strcmp(e, "npu")) return DecodeMode::Npu;
        return DecodeMode::Cpu;      // fastest end to end (stage7): AVX-512 GEMV from the NPU layout; falls back to the NPU kernel if the CPU lacks VNNI/VBMI
    }();
    return m;
}

bool placement_ok(const layout::Geom & geom) {
    if (reference_mode()) {
        return true;
    }
    if (!npu_executor().available()) {
        return false;
    }
    return KernelCache::supports(geom) && kernel_cache().available();
}

void request_kernels(const layout::Geom & geom) {
    if (reference_mode() || !npu_executor().available()) {
        return;
    }
    if (KernelCache::supports(geom)) {
        kernel_cache().request();
        if (prefill_enabled()) prefill_kernel_cache().request();
    }
}

bool npu_device_present() {
    struct stat st;
    return stat("/dev/accel/accel0", &st) == 0;
}

} // namespace xdna2

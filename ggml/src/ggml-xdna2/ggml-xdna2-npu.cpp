#include "ggml-xdna2-npu.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <cstdlib>
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

// Stub: never available. The real implementation (see the TODO list in ggml-xdna2-npu.h) replaces this class.
class StubExecutor final : public NpuExecutor {
public:
    StubExecutor() : cache_dir_(default_cache_dir()) {}
    bool available() const override { return false; }
    std::unique_ptr<Arena> alloc_weight_arena(size_t) override { return nullptr; }
    bool bind_weights(const Arena &, size_t, const layout::Geom &) override { return false; }
    bool run(const MulMatJob *, size_t) override { return false; }
    std::string cache_dir() const override { return cache_dir_; }
private:
    std::string cache_dir_;
};

class NpuPath final : public ComputePath {
public:
    const char * name() const override { return "npu"; }
    bool supports(const layout::Geom & geom) const override { return placement_ok(geom); }
    bool mul_mat(const MulMatJob & job) override {
        NpuExecutor & e = npu_executor();
        return e.bind_weights(*job.arena, job.w_off, job.geom) && e.run(&job, 1);
    }
};

void kernel_log(int level, const std::string & msg) {
    switch (level) {
        case 0:  GGML_LOG_INFO ("ggml-xdna2: kernels: %s\n", msg.c_str()); break;
        case 1:  GGML_LOG_WARN ("ggml-xdna2: kernels: %s\n", msg.c_str()); break;
        default: GGML_LOG_ERROR("ggml-xdna2: kernels: %s\n", msg.c_str()); break;
    }
}

} // namespace

size_t act_blob_bytes_per_token(const ActQ & act) {
    return size_t(act.k_pad) + size_t(act.nd) * sizeof(float) + size_t(act.nbs) * sizeof(int32_t);
}

void pack_act_blob(const ActQ & act, uint8_t * dst) {
    for (int64_t t = 0; t < act.n_tok; ++t) {
        std::memcpy(dst, act.q_row(t), size_t(act.k_pad));
        dst += act.k_pad;
        std::memcpy(dst, act.d_row(t), size_t(act.nd) * sizeof(float));
        dst += act.nd * sizeof(float);
        if (act.nbs) {
            std::memcpy(dst, act.bs_row(t), size_t(act.nbs) * sizeof(int32_t));
            dst += act.nbs * sizeof(int32_t);
        }
    }
}

NpuExecutor & npu_executor() {
    static StubExecutor e;
    return e;
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

bool reference_mode() {
    const char * e = std::getenv("GGML_XDNA2_REFERENCE");
    return e && *e && std::strcmp(e, "0") != 0;
}

bool placement_ok(const layout::Geom & geom) {
    if (reference_mode()) {
        return true;
    }
    if (!npu_executor().available()) {
        return false;
    }
    for (int i = 0; i < geom.nchunks; ++i) {
        const KernelKey key{ geom.fmt, geom.chunks[i].rows, geom.k, 1 };
        if (!kernel_cache().available(key)) {
            return false;
        }
    }
    return true;
}

void request_kernels(const layout::Geom & geom) {
    if (reference_mode() || !npu_executor().available()) {
        return;
    }
    for (int i = 0; i < geom.nchunks; ++i) {
        kernel_cache().request(KernelKey{ geom.fmt, geom.chunks[i].rows, geom.k, 1 });
    }
}

bool npu_device_present() {
    struct stat st;
    return stat("/dev/accel/accel0", &st) == 0;
}

} // namespace xdna2

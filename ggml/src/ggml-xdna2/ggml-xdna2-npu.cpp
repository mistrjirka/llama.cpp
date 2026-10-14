#include "ggml-xdna2-npu.h"

#include <cstdlib>
#include <cstring>

#include <sys/stat.h>

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
    bool has_kernel(const layout::Q4Geom &) override { return false; }
    std::unique_ptr<Arena> alloc_weight_arena(size_t) override { return nullptr; }
    bool bind_weights(const Arena &, size_t, const layout::Q4Geom &) override { return false; }
    bool run(const MulMatJob *, size_t) override { return false; }
    std::string cache_dir() const override { return cache_dir_; }
private:
    std::string cache_dir_;
};

class NpuPath final : public ComputePath {
public:
    const char * name() const override { return "npu"; }
    bool supports(const layout::Q4Geom & geom) const override {
        NpuExecutor & e = npu_executor();
        return e.available() && e.has_kernel(geom);
    }
    bool mul_mat_q4_0(const MulMatJob & job) override {
        NpuExecutor & e = npu_executor();
        return e.bind_weights(*job.arena, job.w_off, job.geom) && e.run(&job, 1);
    }
};

} // namespace

size_t act_blob_bytes_per_token(const layout::Q4Geom & geom) {
    return size_t(geom.kb_pad) * (layout::QK + sizeof(float));
}

void pack_act_blob(const Q8Act & act, uint8_t * dst) {
    const size_t qb = size_t(act.kb_pad) * layout::QK, db = size_t(act.kb_pad) * sizeof(float);
    for (int64_t t = 0; t < act.n_tok; ++t) {
        std::memcpy(dst, act.q_row(t), qb);
        std::memcpy(dst + qb, act.d_row(t), db);
        dst += qb + db;
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

bool npu_device_present() {
    struct stat st;
    return stat("/dev/accel/accel0", &st) == 0;
}

} // namespace xdna2

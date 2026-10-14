// Real NPU executor of ggml-xdna2 (XRT): see ggml-xdna2-npu.h for the execution model.

#include "ggml-xdna2-npu.h"

#include "ggml-impl.h"

#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <set>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include "ggml-xdna2-sync.h"

namespace xdna2 {

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr size_t kMaxContexts = 8;     // resident xclbins (hw contexts); least recently used are evicted

struct Job;

// One loaded xclbin (hw context + kernel) for a KernelKey and the per-N instruction BOs derived from it.
struct Ctx {
    KernelKey                         key;
    xrt::xclbin                       xb;
    xrt::hw_context                   hw;
    xrt::kernel                       kern;
    std::map<int64_t, xrt::bo>        instr;        // n_pad -> instruction BO
    std::map<int64_t, uint32_t>       instr_words;
    std::list<Job *>                  jobs;         // jobs with a run bound to this context
    std::map<std::vector<Job *>, xrt::runlist> runlists;
    uint64_t                          last_use = 0;
};

struct Job {
    layout::Geom  geom;
    const Arena * arena = nullptr;
    size_t        w_off = 0;
    xrt::bo       w, a, y;
    size_t        a_bytes = 0;
    Ctx *         ctx = nullptr;
    xrt::run      run;
};

class XrtExecutor final : public NpuExecutor {
public:
    XrtExecutor() {
        try {
            dev_ = xrt::device(0);
            ok_ = true;
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("ggml-xdna2: cannot open the NPU device: %s\n", e.what());
        }
    }

    ~XrtExecutor() override {
        std::lock_guard<std::mutex> lk(mu_);
        jobs_.clear();
        ctxs_.clear();
    }

    bool available() const override { return ok_; }

    std::unique_ptr<Arena> alloc_weight_arena(size_t bytes) override {
        return std::make_unique<XrtBoArena>(dev_, bytes);
    }

    void release_arena(const Arena * arena) override {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = jobs_.begin(); it != jobs_.end();) {
            if (it->second->arena == arena) {
                unbind(*it->second);
                it = jobs_.erase(it);
            } else {
                ++it;
            }
        }
    }

    ExecStats stats() const override { return st_; }
    void reset_stats() override { st_ = ExecStats(); }

    bool run(const MulMatJob * jobs, size_t n_jobs) override {
        std::lock_guard<std::mutex> lk(mu_);
        try {
            return run_locked(jobs, n_jobs);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("ggml-xdna2: NPU run failed: %s\n", e.what());
            return false;
        }
    }

private:
    using JobKey = std::tuple<const Arena *, size_t, int, int64_t, int64_t>;

    void unbind(Job & j) {
        if (j.ctx) {
            for (auto it = j.ctx->runlists.begin(); it != j.ctx->runlists.end();) {
                bool uses = false;
                for (Job * x : it->first) uses |= x == &j;
                it = uses ? j.ctx->runlists.erase(it) : std::next(it);
            }
            j.ctx->jobs.remove(&j);
            j.run = xrt::run();
            j.ctx = nullptr;
        }
    }

    Ctx & get_ctx(const KernelKey & key) {
        auto it = ctxs_.find(key);
        if (it != ctxs_.end()) {
            it->second->last_use = ++tick_;
            return *it->second;
        }
        const double t0 = now_s();
        KernelCache & kc = kernel_cache();
        kc.request(key);
        const double tw = now_s();
        if (!kc.wait(key)) {
            throw std::runtime_error("NPU kernel " + key.name() + " is not available (build failed, see the kernel cache)");
        }
        st_.t_kernel_wait += now_s() - tw;
        while (ctxs_.size() >= kMaxContexts) {
            evict_lru();
        }
        auto c = std::make_unique<Ctx>();
        c->key = key;
        for (int attempt = 0;; ++attempt) {
            try {
                c->xb = xrt::xclbin(kc.xclbin_path(key));
                dev_.register_xclbin(c->xb);
                c->hw = xrt::hw_context(dev_, c->xb.get_uuid());
                break;
            } catch (const std::exception & e) {
                if (attempt > 0 || ctxs_.empty()) throw;
                GGML_LOG_WARN("ggml-xdna2: hw context creation failed (%s), evicting an idle context\n", e.what());
                evict_lru();
            }
        }
        std::string kn;
        for (auto & k : c->xb.get_kernels()) {
            if (k.get_name().rfind("MLIR_AIE", 0) == 0) { kn = k.get_name(); break; }
        }
        if (kn.empty()) throw std::runtime_error("no MLIR_AIE kernel in " + kc.xclbin_path(key));
        c->kern = xrt::kernel(c->hw, kn);
        c->last_use = ++tick_;
        Ctx & ref = *c;
        ctxs_[key] = std::move(c);
        st_.t_setup += now_s() - t0;
        return ref;
    }

    void evict_lru() {
        auto victim = ctxs_.begin();
        for (auto it = ctxs_.begin(); it != ctxs_.end(); ++it) {
            if (it->second->last_use < victim->second->last_use) victim = it;
        }
        GGML_LOG_DEBUG("ggml-xdna2: evicting NPU context %s\n", victim->first.name().c_str());
        if (last_ctx_ == victim->second.get()) last_ctx_ = nullptr;
        std::list<Job *> js = victim->second->jobs;
        for (Job * j : js) unbind(*j);
        ctxs_.erase(victim);
    }

    const xrt::bo & get_instr(Ctx & c, int64_t n_pad) {
        auto it = c.instr.find(n_pad);
        if (it != c.instr.end()) return it->second;
        std::vector<uint32_t> w;
        if (!kernel_cache().insts_for(c.key, n_pad, w)) {
            throw std::runtime_error("cannot derive the instruction stream of " + c.key.name() + " for N=" + std::to_string(n_pad));
        }
        xrt::bo bi(dev_, w.size() * 4, xrt::bo::flags::cacheable, c.kern.group_id(1));
        std::memcpy(bi.map<char *>(), w.data(), w.size() * 4);
        bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        c.instr_words[n_pad] = uint32_t(w.size());
        return c.instr.emplace(n_pad, std::move(bi)).first->second;
    }

    Job & get_job(const MulMatJob & mj, Ctx & c) {
        const layout::Geom & g = mj.geom;
        const JobKey key{ mj.arena, mj.w_off, int(g.fmt), g.n, g.k };
        auto it = jobs_.find(key);
        Job * j;
        if (it == jobs_.end()) {
            const double t0 = now_s();
            auto nj = std::make_unique<Job>();
            nj->geom = g;
            nj->arena = mj.arena;
            nj->w_off = mj.w_off;
            const XrtBoArena * xa = dynamic_cast<const XrtBoArena *>(mj.arena);
            if (!xa) throw std::runtime_error("weights are not in an NPU buffer");
            nj->w = xrt::bo(xa->bo(), g.bytes, mj.w_off);
            nj->a_bytes = act_bo_bytes(g, 1);
            nj->a = xrt::ext::bo{dev_, nj->a_bytes};
            nj->y = xrt::ext::bo{dev_, size_t(g.n_pad) * sizeof(float)};
            std::memset(nj->a.map<char *>(), 0, nj->a_bytes);
            nj->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            std::memset(nj->y.map<char *>(), 0, size_t(g.n_pad) * sizeof(float));
            nj->y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            j = nj.get();
            jobs_[key] = std::move(nj);
            st_.t_setup += now_s() - t0;
        } else {
            j = it->second.get();
        }
        if (!j->ctx) {
            const double t0 = now_s();
            const xrt::bo & bi = get_instr(c, g.n_pad);
            xrt::run r(c.kern);
            r.set_arg(0, 3);
            r.set_arg(1, bi);
            r.set_arg(2, c.instr_words[g.n_pad]);
            r.set_arg(3, j->w);
            r.set_arg(4, j->a);
            r.set_arg(5, j->y);
            j->run = std::move(r);
            j->ctx = &c;
            c.jobs.push_back(j);
            st_.t_setup += now_s() - t0;
        }
        return *j;
    }

    bool run_locked(const MulMatJob * jobs, size_t n_jobs) {
        // group by kernel (hw context): contexts are time shared, runlists need one context
        std::map<KernelKey, std::vector<size_t>> groups;
        for (size_t i = 0; i < n_jobs; ++i) {
            groups[KernelKey{ jobs[i].geom.fmt, jobs[i].geom.k, 1 }].push_back(i);
        }
        for (auto & kv : groups) {
            Ctx & c = get_ctx(kv.first);
            std::vector<Job *> js;
            int64_t max_tok = 0;
            for (size_t i : kv.second) {
                js.push_back(&get_job(jobs[i], c));
                max_tok = std::max(max_tok, jobs[i].act->n_tok);
            }
            if (std::set<Job *>(js.begin(), js.end()).size() != js.size()) {
                // the same weights used twice in one batch share their A/Y BOs: run the jobs one after the other
                for (size_t i : kv.second) {
                    if (!run_locked(&jobs[i], 1)) return false;
                }
                continue;
            }
            for (int64_t t = 0; t < max_tok; ++t) {
                std::vector<size_t> act;     // indices into kv.second that still have token t
                for (size_t q = 0; q < kv.second.size(); ++q) {
                    if (jobs[kv.second[q]].act->n_tok > t) act.push_back(q);
                }
                // 1. activations
                double t0 = now_s();
                for (size_t q : act) {
                    const MulMatJob & mj = jobs[kv.second[q]];
                    Job & j = *js[q];
                    pack_act_bo(*mj.act, mj.geom, t, 1, j.a.map<uint8_t *>());
                    j.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    st_.bytes_w += double(mj.geom.bytes);
                }
                st_.t_pack_a += now_s() - t0;
                // 2. execute
                const bool switched = last_ctx_ != &c;
                last_ctx_ = &c;
                t0 = now_s();
                if (act.size() == 1) {
                    Job & j = *js[act[0]];
                    j.run.start();
                    j.run.wait2();
                } else {
                    std::vector<Job *> key;
                    for (size_t q : act) key.push_back(js[q]);
                    auto it = c.runlists.find(key);
                    if (it == c.runlists.end()) {
                        xrt::runlist rl(c.hw);
                        for (Job * j : key) rl.add(j->run);
                        it = c.runlists.emplace(key, std::move(rl)).first;
                    }
                    it->second.execute();
                    it->second.wait();
                }
                st_.t_wait += now_s() - t0;
                if (switched) { st_.n_switch++; st_.t_wait_switch += now_s() - t0; }
                st_.runs += act.size();
                st_.batches += 1;
                // 3. results
                t0 = now_s();
                for (size_t q : act) {
                    const MulMatJob & mj = jobs[kv.second[q]];
                    Job & j = *js[q];
                    sync_from_device(j.y);
                    unpermute(mj, j, t);
                }
                st_.t_unpermute += now_s() - t0;
            }
        }
        return true;
    }

    // Y: f32 [col 8][G][core 4][16]  ->  row = ((col*4 + k)*G + g)*16 + i
    static void unpermute(const MulMatJob & mj, Job & j, int64_t t) {
        const layout::Geom & g = mj.geom;
        const float * y = j.y.map<const float *>();
        float * dst = reinterpret_cast<float *>(reinterpret_cast<char *>(mj.dst) + size_t(t) * mj.dst_row_stride);
        const int64_t G = g.groups;
        for (int64_t c = 0; c < 8; ++c) {
            for (int64_t gg = 0; gg < G; ++gg) {
                for (int64_t k = 0; k < 4; ++k) {
                    const int64_t row0 = ((c * 4 + k) * G + gg) * 16;
                    if (row0 >= g.n) continue;
                    const int64_t cnt = std::min<int64_t>(16, g.n - row0);
                    std::memcpy(dst + row0, y + ((c * G + gg) * 4 + k) * 16, size_t(cnt) * sizeof(float));
                }
            }
        }
    }

    std::mutex mu_;
    xrt::device dev_;
    bool ok_ = false;
    uint64_t tick_ = 0;
    const Ctx * last_ctx_ = nullptr;
    std::map<KernelKey, std::unique_ptr<Ctx>> ctxs_;
    std::map<JobKey, std::unique_ptr<Job>> jobs_;
    ExecStats st_;
};

} // namespace

std::unique_ptr<NpuExecutor> make_xrt_executor() {
    return std::make_unique<XrtExecutor>();
}

} // namespace xdna2

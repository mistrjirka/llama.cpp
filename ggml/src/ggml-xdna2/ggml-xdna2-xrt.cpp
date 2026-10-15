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
#include "ggml-xdna2-kernels.h"

namespace xdna2 {

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr size_t kMaxRunlist = 128;     // dispatches per xrt::runlist

struct SJob;

// The ONE hw context of the process: unified decode kernel (all qtypes, all K, all N). Instruction BOs are per (chain, act, g) words.
struct Ctx {
    xrt::xclbin                       xb;
    xrt::hw_context                   hw;
    xrt::kernel                       kern;
    std::map<std::tuple<int64_t, int64_t, int64_t>, xrt::bo> instr;
    std::map<std::tuple<int64_t, int64_t, int64_t>, uint32_t> instr_words;
    std::map<std::vector<SJob *>, xrt::runlist> runlists;
};

// One dispatch: one split of one 2D weight matrix.
struct SJob {
    layout::Geom  geom;
    int64_t       split = 0;
    const Arena * arena = nullptr;
    size_t        w_off = 0;
    xrt::bo       w, a, y;
    size_t        a_bytes = 0;
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
        if (ctx_) ctx_->runlists.clear();
        jobs_.clear();
        ctx_.reset();
    }

    bool available() const override { return ok_; }

    std::unique_ptr<Arena> alloc_weight_arena(size_t bytes) override {
        return std::make_unique<XrtBoArena>(dev_, bytes);
    }

    void release_arena(const Arena * arena) override {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = jobs_.begin(); it != jobs_.end();) {
            if (std::get<0>(it->first) == arena) {
                unbind(it->second);
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
    using Splits = std::vector<std::unique_ptr<SJob>>;

    void unbind(Splits & sj) {
        for (auto & j : sj) {
            if (!ctx_) continue;
            for (auto it = ctx_->runlists.begin(); it != ctx_->runlists.end();) {
                bool uses = false;
                for (SJob * x : it->first) uses |= x == j.get();
                it = uses ? ctx_->runlists.erase(it) : std::next(it);
            }
            j->run = xrt::run();
        }
    }

    Ctx & get_ctx() {
        if (ctx_) return *ctx_;
        const double t0 = now_s();
        KernelCache & kc = kernel_cache();
        kc.request();
        const double tw = now_s();
        if (!kc.wait()) {
            throw std::runtime_error("the NPU decode kernel is not available (build failed, see the kernel cache)");
        }
        st_.t_kernel_wait += now_s() - tw;
        auto c = std::make_unique<Ctx>();
        c->xb = xrt::xclbin(kc.xclbin_path());
        dev_.register_xclbin(c->xb);
        c->hw = xrt::hw_context(dev_, c->xb.get_uuid());
        std::string kn;
        for (auto & k : c->xb.get_kernels()) {
            if (k.get_name().rfind("MLIR_AIE", 0) == 0) { kn = k.get_name(); break; }
        }
        if (kn.empty()) throw std::runtime_error("no MLIR_AIE kernel in " + kc.xclbin_path());
        c->kern = xrt::kernel(c->hw, kn);
        ctx_ = std::move(c);
        st_.t_setup += now_s() - t0;
        return *ctx_;
    }

    const xrt::bo & get_instr(Ctx & c, const layout::Geom & g, int64_t s, size_t a_bytes) {
        const auto key = std::make_tuple(int64_t(g.sp_chain_words(s)), int64_t(a_bytes / 4), g.sp_groups(s));
        auto it = c.instr.find(key);
        if (it != c.instr.end()) return it->second;
        std::vector<uint32_t> w;
        if (!kernel_cache().insts_for(std::get<0>(key), std::get<1>(key), std::get<2>(key), w)) {
            throw std::runtime_error("cannot derive the instruction stream of the decode kernel for chain_words=" +
                                     std::to_string(std::get<0>(key)));
        }
        xrt::bo bi(dev_, w.size() * 4, xrt::bo::flags::cacheable, c.kern.group_id(1));
        std::memcpy(bi.map<char *>(), w.data(), w.size() * 4);
        bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        c.instr_words[key] = uint32_t(w.size());
        return c.instr.emplace(key, std::move(bi)).first->second;
    }

    Splits & get_job(const MulMatJob & mj, Ctx & c) {
        const layout::Geom & g = mj.geom;
        const JobKey key{ mj.arena, mj.w_off, int(g.fmt), g.n, g.k };
        auto it = jobs_.find(key);
        if (it != jobs_.end()) return it->second;
        const double t0 = now_s();
        const XrtBoArena * xa = dynamic_cast<const XrtBoArena *>(mj.arena);
        if (!xa) throw std::runtime_error("weights are not in an NPU buffer");
        Splits sj;
        for (int64_t s = 0; s < g.nsplit; ++s) {
            auto j = std::make_unique<SJob>();
            j->geom = g;
            j->split = s;
            j->arena = mj.arena;
            j->w_off = mj.w_off;
            j->w = xrt::bo(xa->bo(), g.sp_bytes(s), mj.w_off + g.sp_off(s));
            j->a_bytes = act_bo_bytes(g, 1);
            j->a = xrt::ext::bo{dev_, j->a_bytes};
            const size_t yb = size_t(g.sp_groups(s)) * layout::ROW_GRAN * sizeof(float);
            j->y = xrt::ext::bo{dev_, yb};
            std::memset(j->a.map<char *>(), 0, j->a_bytes);
            j->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            std::memset(j->y.map<char *>(), 0, yb);
            j->y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            const xrt::bo & bi = get_instr(c, g, s, j->a_bytes);
            xrt::run r(c.kern);
            r.set_arg(0, 3);
            r.set_arg(1, bi);
            r.set_arg(2, c.instr_words[std::make_tuple(int64_t(g.sp_chain_words(s)), int64_t(j->a_bytes / 4), g.sp_groups(s))]);
            r.set_arg(3, j->w);
            r.set_arg(4, j->a);
            r.set_arg(5, j->y);
            j->run = std::move(r);
            sj.push_back(std::move(j));
        }
        st_.t_setup += now_s() - t0;
        return jobs_.emplace(key, std::move(sj)).first->second;
    }

    void execute(Ctx & c, const std::vector<SJob *> & key) {
        if (key.size() == 1) {
            key[0]->run.start();
            key[0]->run.wait2();
            return;
        }
        auto it = c.runlists.find(key);
        if (it == c.runlists.end()) {
            xrt::runlist rl(c.hw);
            for (SJob * j : key) rl.add(j->run);
            it = c.runlists.emplace(key, std::move(rl)).first;
        }
        it->second.execute();
        it->second.wait();
    }

    bool run_locked(const MulMatJob * jobs, size_t n_jobs) {
        Ctx & c = get_ctx();
        // all matmuls of the batch are independent: ONE runlist per token in one hw context (no context switch ever)
        std::vector<Splits *> js;
        int64_t max_tok = 0;
        {
            std::set<Splits *> seen;
            bool dup = false;
            for (size_t i = 0; i < n_jobs; ++i) {
                js.push_back(&get_job(jobs[i], c));
                dup |= !seen.insert(js.back()).second;
                max_tok = std::max(max_tok, jobs[i].act->n_tok);
            }
            if (dup) {
                // the same weights used twice in one batch share their A/Y BOs: run the jobs one after the other
                for (size_t i = 0; i < n_jobs; ++i) {
                    if (!run_locked(&jobs[i], 1)) return false;
                }
                return true;
            }
        }
        for (int64_t t = 0; t < max_tok; ++t) {
            std::vector<size_t> act;     // matmuls that still have token t
            for (size_t q = 0; q < n_jobs; ++q) {
                if (jobs[q].act->n_tok > t) act.push_back(q);
            }
            // 1. activations
            double t0 = now_s();
            std::vector<SJob *> disp;
            for (size_t q : act) {
                const MulMatJob & mj = jobs[q];
                for (auto & sj : *js[q]) {
                    pack_act_bo(*mj.act, mj.geom, sj->split, t, sj->a.map<uint8_t *>());
                    sj->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    disp.push_back(sj.get());
                }
                st_.bytes_w += double(mj.geom.bytes);
            }
            st_.t_pack_a += now_s() - t0;
            // 2. execute
            t0 = now_s();
            for (size_t b = 0; b < disp.size(); b += kMaxRunlist) {
                const size_t e = std::min(disp.size(), b + kMaxRunlist);
                execute(c, std::vector<SJob *>(disp.begin() + b, disp.begin() + e));
                st_.batches += 1;
            }
            st_.t_wait += now_s() - t0;
            st_.runs += disp.size();
            // 3. results
            t0 = now_s();
            for (size_t q : act) {
                for (auto & sj : *js[q]) {
                    sync_from_device(sj->y);
                    unpermute(jobs[q], *sj, t);
                }
            }
            st_.t_unpermute += now_s() - t0;
        }
        return true;
    }

    // Y of a split: f32 [col 8][Gs][core 4][16]  ->  row = split_row0 + ((col*4 + k)*Gs + g)*16 + i
    static void unpermute(const MulMatJob & mj, SJob & j, int64_t t) {
        const layout::Geom & g = mj.geom;
        const float * y = j.y.map<const float *>();
        float * dst = reinterpret_cast<float *>(reinterpret_cast<char *>(mj.dst) + size_t(t) * mj.dst_row_stride);
        const int64_t G = g.sp_groups(j.split), row_base = g.sp_row0(j.split);
        for (int64_t c = 0; c < 8; ++c) {
            for (int64_t gg = 0; gg < G; ++gg) {
                for (int64_t k = 0; k < 4; ++k) {
                    const int64_t row0 = row_base + ((c * 4 + k) * G + gg) * 16;
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
    std::unique_ptr<Ctx> ctx_;
    std::map<JobKey, Splits> jobs_;
    ExecStats st_;
};

} // namespace

std::unique_ptr<NpuExecutor> make_xrt_executor() {
    return std::make_unique<XrtExecutor>();
}

} // namespace xdna2

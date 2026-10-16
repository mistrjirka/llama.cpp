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
#include "ggml-xdna2-bfp.h"

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
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, xrt::bo> instr;          // (chain_words, a_words, g, stride_words)
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, uint32_t> instr_words;
    std::map<std::vector<SJob *>, xrt::runlist> runlists;
};

// The PREFILL hw context (second xclbin): instruction BOs are per (chain, a, o) words.
struct PfCtx {
    xrt::xclbin                       xb;
    xrt::hw_context                   hw;
    xrt::kernel                       kern;
    std::map<std::tuple<int64_t, int64_t, int64_t>, xrt::bo> instr;
};

// Activation + output BOs shared by all prefill dispatches of one shape (a dispatch is a tile; PF_MAX_TILES tiles per wave).
struct PfPool {
    size_t  a_tile = 0, a_stride = 0, y_tile = 0;
    int     next_slot = 0;                  // round robin over the PF_MAX_TILES tile slots (at most 2 dispatches in flight)
    xrt::bo a, y;
    std::vector<xrt::bo> a_sub, y_sub;      // per tile
};

// One prefill job split: the weight sub-BO and one xrt::run per tile slot (arguments set once).
struct PfSplit {
    layout::Geom geom;
    int64_t      split = 0;
    xrt::bo      w;
    PfPool *     pool = nullptr;
    std::vector<xrt::run> runs;             // per tile slot, created lazily
};

// One dispatch: one split of one 2D weight matrix.
struct SJob {
    layout::Geom  geom;
    int64_t       split = 0;
    const Arena * arena = nullptr;
    size_t        w_off = 0;
    xrt::bo       w, a, y;
    size_t        a_bytes = 0;
    xrt::run      run;        // member of runlists (cannot be started on its own)
    xrt::run      run_solo;   // same arguments, started on its own
    uint32_t      nwords = 0;
    // decode split: the NPU part of this split (a prefix of every chain: G_npu groups per core, nbk_npu records per row)
    bool          active = true;
    int64_t       G_npu = 0, nbk_npu = 0, n_w = 0;
    bool          add = false;      // the rows also get a CPU part (K split): add instead of store
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
        split_jobs_.clear();
        pf_jobs_.clear();
        pf_pools_.clear();
        pf_ctx_.reset();
        jobs_.clear();
        ctx_.reset();
    }

    bool available() const override { return ok_; }

    std::unique_ptr<Arena> alloc_weight_arena(size_t bytes) override {
        return std::make_unique<XrtBoArena>(dev_, bytes);
    }

    void release_arena(const Arena * arena) override {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = pf_jobs_.begin(); it != pf_jobs_.end();) {
            if (std::get<0>(it->first) == arena) it = pf_jobs_.erase(it); else ++it;
        }
        for (auto it = jobs_.begin(); it != jobs_.end();) {
            if (std::get<0>(it->first) == arena) {
                unbind(it->second);
                it = jobs_.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = split_jobs_.begin(); it != split_jobs_.end();) {
            if (std::get<0>(it->first) == arena) {
                unbind(it->second);
                it = split_jobs_.erase(it);
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
            j->run_solo = xrt::run();
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

    const xrt::bo & get_instr(Ctx & c, int64_t chain_words, size_t a_bytes, int64_t groups, int64_t stride_words) {
        const auto key = std::make_tuple(chain_words, int64_t(a_bytes / 4), groups, stride_words);
        auto it = c.instr.find(key);
        if (it != c.instr.end()) return it->second;
        std::vector<uint32_t> w;
        if (!kernel_cache().insts_for(std::get<0>(key), std::get<1>(key), std::get<2>(key), w, std::get<3>(key))) {
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
            const xrt::bo & bi = get_instr(c, int64_t(g.sp_chain_words(s)), j->a_bytes, g.sp_groups(s), int64_t(g.sp_chain_words(s)));
            const auto ikey = std::make_tuple(int64_t(g.sp_chain_words(s)), int64_t(j->a_bytes / 4), g.sp_groups(s), int64_t(g.sp_chain_words(s)));
            xrt::run r(c.kern);
            r.set_arg(0, 3);
            r.set_arg(1, bi);
            r.set_arg(2, c.instr_words[ikey]);
            r.set_arg(3, j->w);
            r.set_arg(4, j->a);
            r.set_arg(5, j->y);
            j->run = std::move(r);
            {
                xrt::run r2(c.kern);
                r2.set_arg(0, 3);
                r2.set_arg(1, bi);
                r2.set_arg(2, c.instr_words[ikey]);
                r2.set_arg(3, j->w);
                r2.set_arg(4, j->a);
                r2.set_arg(5, j->y);
                j->run_solo = std::move(r2);
            }
            sj.push_back(std::move(j));
        }
        st_.t_setup += now_s() - t0;
        return jobs_.emplace(key, std::move(sj)).first->second;
    }

    void execute(Ctx & c, const std::vector<SJob *> & key) {
        if (key.size() == 1) {
            key[0]->run_solo.start();
            key[0]->run_solo.wait2();
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
        if (last_ctx_ != 1) { if (last_ctx_ != 0) st_.pf_switches++; last_ctx_ = 1; }
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


    // ---------------------------------------------------------------- decode split path (NPU prefix of every chain, CPU does the rest)
    Splits & get_split_job(const CpuDecodeJob & cj, Ctx & c) {
        const layout::Geom & g = cj.geom;
        const JobKey key{ cj.arena, cj.w_off, int(g.fmt), g.n, g.k };
        auto it = split_jobs_.find(key);
        if (it != split_jobs_.end()) return it->second;
        const double t0 = now_s();
        const XrtBoArena * xa = dynamic_cast<const XrtBoArena *>(cj.arena);
        if (!xa) throw std::runtime_error("weights are not in an NPU buffer");
        Splits sj;
        for (int64_t s = 0; s < g.nsplit; ++s) {
            const SplitPlan p = cj.plan[s];
            auto j = std::make_unique<SJob>();
            j->geom = g;
            j->split = s;
            j->arena = cj.arena;
            j->w_off = cj.w_off;
            if (!p.npu()) {
                j->active = false;
                sj.push_back(std::move(j));
                continue;
            }
            const int64_t Gs = g.sp_groups(s);
            j->G_npu = std::min<int64_t>(p.sb_npu * layout::GSB, Gs);
            j->nbk_npu = p.kb_npu;
            j->add = p.kb_npu < g.nbk;
            const size_t lane_bytes = g.lane_bytes_groups_k(j->G_npu, j->nbk_npu);
            j->n_w = int64_t(lane_bytes / layout::CHUNK);
            const int64_t chain_words = j->n_w * layout::CHUNK;            // n_w chunks x 4 lanes x CHUNK B / 4
            j->w = xrt::bo(xa->bo(), g.sp_bytes(s), cj.w_off + g.sp_off(s));
            j->a_bytes = layout::act_chunks(g.fmt, j->nbk_npu * 256) * layout::N_LANES * layout::CHUNK;
            j->a = xrt::ext::bo{dev_, j->a_bytes};
            const size_t yb = size_t(j->G_npu) * layout::ROW_GRAN * sizeof(float);
            j->y = xrt::ext::bo{dev_, yb};
            std::memset(j->a.map<char *>(), 0, j->a_bytes);
            j->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            std::memset(j->y.map<char *>(), 0, yb);
            j->y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            const xrt::bo & bi = get_instr(c, chain_words, j->a_bytes, j->G_npu, int64_t(g.sp_chain_words(s)));
            const auto ikey = std::make_tuple(chain_words, int64_t(j->a_bytes / 4), j->G_npu, int64_t(g.sp_chain_words(s)));
            j->nwords = c.instr_words[ikey];
            for (xrt::run * r : { &j->run, &j->run_solo }) {
                xrt::run rr(c.kern);
                rr.set_arg(0, 3);
                rr.set_arg(1, bi);
                rr.set_arg(2, j->nwords);
                rr.set_arg(3, j->w);
                rr.set_arg(4, j->a);
                rr.set_arg(5, j->y);
                *r = std::move(rr);
            }
            sj.push_back(std::move(j));
        }
        st_.t_setup += now_s() - t0;
        return split_jobs_.emplace(key, std::move(sj)).first->second;
    }

    bool decode_split_begin(const CpuDecodeJob * jobs, size_t n) override {
        std::lock_guard<std::mutex> lk(mu_);
        try {
            Ctx & c = get_ctx();
            if (last_ctx_ != 1) { if (last_ctx_ != 0) st_.pf_switches++; last_ctx_ = 1; }
            sp_disp_.clear();
            sp_jobs_.clear();
            double t0 = now_s();
            for (size_t i = 0; i < n; ++i) {
                bool any = false;
                for (int64_t s = 0; jobs[i].plan && s < jobs[i].geom.nsplit; ++s) any |= jobs[i].plan[s].npu();
                if (!any) continue;
                Splits & sj = get_split_job(jobs[i], c);
                for (auto & j : sj) {
                    if (!j->active) continue;
                    cpu_pack_npu_act(jobs[i].geom.fmt, jobs[i].geom.k, j->nbk_npu * 256, j->G_npu, j->n_w, cpu_decode_act(i), j->a.map<uint8_t *>());
                    j->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    sp_disp_.push_back(j.get());
                    sp_jobs_.push_back(i);
                    st_.sp_bytes_w += double(jobs[i].geom.lane_bytes_groups_k(j->G_npu, j->nbk_npu)) * layout::N_CORES;
                }
            }
            st_.sp_t_pack += now_s() - t0;
            sp_cjobs_ = jobs;
            if (sp_disp_.empty()) return true;
            if (sp_disp_.size() > kMaxRunlist) throw std::runtime_error("too many dispatches in one decode batch");
            t0 = now_s();
            if (sp_disp_.size() == 1) {
                sp_disp_[0]->run_solo.start();
            } else {
                auto it = c.runlists.find(sp_disp_);
                if (it == c.runlists.end()) {
                    xrt::runlist rl(c.hw);
                    for (SJob * j : sp_disp_) rl.add(j->run);
                    it = c.runlists.emplace(sp_disp_, std::move(rl)).first;
                }
                it->second.execute();
            }
            st_.sp_t_start += now_s() - t0;
            st_.sp_calls++;
            st_.sp_runs += sp_disp_.size();
            return true;
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("ggml-xdna2: NPU decode split begin failed: %s\n", e.what());
            return false;
        }
    }

    bool decode_split_end() override {
        std::lock_guard<std::mutex> lk(mu_);
        try {
            if (sp_disp_.empty()) return true;
            double t0 = now_s();
            if (sp_disp_.size() == 1) {
                static const bool poll = std::getenv("XD_POLL") != nullptr;
                if (poll) {
                    while (sp_disp_[0]->run_solo.state() != ERT_CMD_STATE_COMPLETED) {}
                } else {
                    sp_disp_[0]->run_solo.wait2();
                }
            } else {
                ctx_->runlists.find(sp_disp_)->second.wait();
            }
            st_.sp_t_wait += now_s() - t0;
            t0 = now_s();
            for (size_t q = 0; q < sp_disp_.size(); ++q) {
                SJob & j = *sp_disp_[q];
                const CpuDecodeJob & cj = sp_cjobs_[sp_jobs_[q]];
                sync_from_device(j.y);
                unpermute_part(cj.geom, j, cj.dst);
            }
            st_.sp_t_merge += now_s() - t0;
            sp_disp_.clear();
            return true;
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("ggml-xdna2: NPU decode split end failed: %s\n", e.what());
            return false;
        }
    }

    // Y of the NPU part of a split: f32 [col 8][G_npu][core 4][16]  ->  row = split_row0 + ((col*4 + k)*Gs + g)*16 + i  (g < G_npu)
    static void unpermute_part(const layout::Geom & g, SJob & j, float * dst) {
        const float * y = j.y.map<const float *>();
        const int64_t Gs = g.sp_groups(j.split), G = j.G_npu, row_base = g.sp_row0(j.split);
        for (int64_t c = 0; c < 8; ++c) {
            for (int64_t gg = 0; gg < G; ++gg) {
                for (int64_t k = 0; k < 4; ++k) {
                    const int64_t row0 = row_base + ((c * 4 + k) * Gs + gg) * 16;
                    if (row0 >= g.n) continue;
                    const int64_t cnt = std::min<int64_t>(16, g.n - row0);
                    const float * src = y + ((c * G + gg) * 4 + k) * 16;
                    if (j.add) {
                        for (int64_t i = 0; i < cnt; ++i) dst[row0 + i] += src[i];
                    } else {
                        std::memcpy(dst + row0, src, size_t(cnt) * sizeof(float));
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- prefill path
    PfCtx & get_pf_ctx() {
        if (pf_ctx_) return *pf_ctx_;
        const double t0 = now_s();
        KernelCache & kc = prefill_kernel_cache();
        kc.request();
        const double tw = now_s();
        if (!kc.wait()) {
            throw std::runtime_error("the NPU prefill kernel is not available (build failed, see the kernel cache)");
        }
        st_.t_kernel_wait += now_s() - tw;
        auto c = std::make_unique<PfCtx>();
        c->xb = xrt::xclbin(kc.xclbin_path());
        dev_.register_xclbin(c->xb);
        c->hw = xrt::hw_context(dev_, c->xb.get_uuid());
        std::string kn;
        for (auto & k : c->xb.get_kernels()) {
            if (k.get_name().rfind("MLIR_AIE", 0) == 0) { kn = k.get_name(); break; }
        }
        if (kn.empty()) throw std::runtime_error("no MLIR_AIE kernel in " + kc.xclbin_path());
        c->kern = xrt::kernel(c->hw, kn);
        pf_ctx_ = std::move(c);
        st_.pf_t_setup += now_s() - t0;
        return *pf_ctx_;
    }

    PfPool & get_pf_pool(PfCtx & c, size_t a_tile, size_t y_tile) {
        const auto key = std::make_pair(a_tile, y_tile);
        auto it = pf_pools_.find(key);
        if (it != pf_pools_.end()) return *it->second;
        auto p = std::make_unique<PfPool>();
        p->a_tile = a_tile;
        p->a_stride = (a_tile + 4095) / 4096 * 4096;
        p->y_tile = y_tile;
        p->a = xrt::ext::bo{dev_, p->a_stride * PF_MAX_TILES};
        p->y = xrt::ext::bo{dev_, y_tile * PF_MAX_TILES};
        std::memset(p->a.map<char *>(), 0, p->a_stride * PF_MAX_TILES);
        p->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::memset(p->y.map<char *>(), 0, y_tile * PF_MAX_TILES);
        p->y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for (int t = 0; t < PF_MAX_TILES; ++t) {
            p->a_sub.emplace_back(p->a, a_tile, p->a_stride * t);
            p->y_sub.emplace_back(p->y, y_tile, y_tile * t);
        }
        (void) c;
        return *pf_pools_.emplace(key, std::move(p)).first->second;
    }

    const xrt::bo & get_pf_instr(PfCtx & c, int64_t chain_words, int64_t a_words, int64_t o_words, uint32_t & nwords) {
        const auto key = std::make_tuple(chain_words, a_words, o_words);
        auto it = c.instr.find(key);
        if (it == c.instr.end()) {
            std::vector<uint32_t> w;
            if (!prefill_kernel_cache().insts_for(chain_words, a_words, o_words, w)) {
                throw std::runtime_error("cannot derive the instruction stream of the prefill kernel for chain_words=" + std::to_string(chain_words));
            }
            xrt::bo bi(dev_, w.size() * 4, xrt::bo::flags::cacheable, c.kern.group_id(1));
            std::memcpy(bi.map<char *>(), w.data(), w.size() * 4);
            bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            pf_instr_words_[key] = uint32_t(w.size());
            it = c.instr.emplace(key, std::move(bi)).first;
        }
        nwords = pf_instr_words_[key];
        return it->second;
    }

    using PfSplits = std::vector<std::unique_ptr<PfSplit>>;
    PfSplits & get_pf_job(const PrefillJob & pj, PfCtx & c) {
        const layout::Geom & g = pj.geom;
        const JobKey key{ pj.arena, pj.w_off, int(g.fmt), g.n, g.k };
        auto it = pf_jobs_.find(key);
        if (it != pf_jobs_.end()) return it->second;
        const double t0 = now_s();
        const XrtBoArena * xa = dynamic_cast<const XrtBoArena *>(pj.arena);
        if (!xa) throw std::runtime_error("weights are not in an NPU buffer");
        PfSplits sj;
        for (int64_t s = 0; s < g.nsplit; ++s) {
            auto j = std::make_unique<PfSplit>();
            j->geom = g;
            j->split = s;
            j->w = xrt::bo(xa->bo(), g.sp_bytes(s), pj.w_off + g.sp_off(s));
            const int64_t nsb = g.sp_nsb(s);
            j->pool = &get_pf_pool(c, pf_a_tile_bytes(g.nbk, nsb), pf_y_tile_bytes(nsb));
            j->runs.resize(PF_MAX_TILES);
            sj.push_back(std::move(j));
        }
        st_.pf_t_setup += now_s() - t0;
        return pf_jobs_.emplace(key, std::move(sj)).first->second;
    }

    xrt::run & get_pf_run(PfCtx & c, PfSplit & j, int t) {
        if (j.runs[t]) return j.runs[t];
        const layout::Geom & g = j.geom;
        const int64_t nsb = g.sp_nsb(j.split);
        uint32_t nw = 0;
        const xrt::bo & bi = get_pf_instr(c, int64_t(g.sp_chain_words(j.split)), int64_t(j.pool->a_tile / 4), nsb * 4 * layout::GSB * 1024, nw);
        xrt::run r(c.kern);
        r.set_arg(0, 3);
        r.set_arg(1, bi);
        r.set_arg(2, nw);
        r.set_arg(3, j.w);
        r.set_arg(4, j.pool->a_sub[t]);
        r.set_arg(5, j.pool->y_sub[t]);
        j.runs[t] = std::move(r);
        return j.runs[t];
    }

    bool prefill_available() override {
        if (!prefill_enabled()) return false;
        return prefill_kernel_cache().available();
    }

    bool run_prefill(const PrefillJob * jobs, size_t n_jobs) override {
        std::lock_guard<std::mutex> lk(mu_);
        try {
            return run_prefill_locked(jobs, n_jobs);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("ggml-xdna2: NPU prefill run failed: %s\n", e.what());
            return false;
        }
    }

    // One dispatch = one 64-token tile of one split of one job. The units of a call are software pipelined: while the NPU runs unit u-1 the host
    // encodes/replicates the activations of unit u and unpermutes the result of unit u-2 (at most two dispatches are in flight).
    struct PfUnit { size_t job; PfSplit * sp; int tile; int slot; };

    bool run_prefill_locked(const PrefillJob * jobs, size_t n_jobs) {
        PfCtx & c = get_pf_ctx();
        if (last_ctx_ != 2) { if (last_ctx_ != 0) st_.pf_switches++; last_ctx_ = 2; }
        std::vector<PfSplits *> js;
        for (size_t i = 0; i < n_jobs; ++i) js.push_back(&get_pf_job(jobs[i], c));
        // jobs sharing the same input rows share the encoded slabs (per tile, encoded on first use)
        struct Enc { const float * x; size_t stride; int64_t n_tok, k; std::vector<std::vector<uint8_t>> tiles; std::vector<char> done; };
        std::vector<Enc> encs;
        std::vector<size_t> enc_of(n_jobs);
        for (size_t i = 0; i < n_jobs; ++i) {
            size_t e = encs.size();
            for (size_t q = 0; q < encs.size(); ++q) {
                if (encs[q].x == jobs[i].x && encs[q].stride == jobs[i].x_stride && encs[q].n_tok == jobs[i].n_tok && encs[q].k == jobs[i].geom.k) { e = q; break; }
            }
            if (e == encs.size()) {
                const int64_t nt = (jobs[i].n_tok + PF_TILE_TOK - 1) / PF_TILE_TOK;
                encs.push_back(Enc{ jobs[i].x, jobs[i].x_stride, jobs[i].n_tok, jobs[i].geom.k, std::vector<std::vector<uint8_t>>(nt), std::vector<char>(nt, 0) });
            }
            enc_of[i] = e;
        }
        std::vector<PfUnit> units;
        for (size_t i = 0; i < n_jobs; ++i) {
            const int64_t nt = (jobs[i].n_tok + PF_TILE_TOK - 1) / PF_TILE_TOK;
            for (auto & sp : *js[i]) for (int64_t t = 0; t < nt; ++t) units.push_back(PfUnit{ i, sp.get(), int(t), -1 });
        }
        auto finish = [&](PfUnit & u) {
            double t0 = now_s();
            u.sp->runs[u.slot].wait2();
            st_.pf_t_npu += now_s() - t0;      // blocked on the NPU (the host work of the next unit already ran during the dispatch)
            t0 = now_s();
            PfPool & pool = *u.sp->pool;
            sync_from_device(pool.y, pool.y_tile, pool.y_tile * u.slot);
            st_.pf_t_ysync += now_s() - t0;
            const PrefillJob & pj = jobs[u.job];
            pf_unpermute_tile(pool.y.map<const float *>() + (pool.y_tile / sizeof(float)) * u.slot, u.tile, pj.n_tok, pj.geom, u.sp->split, pj.dst, pj.dst_row_stride);
            st_.pf_t_unperm += now_s() - t0;
            st_.pf_bytes_y += double(pool.y_tile);
        };
        PfUnit * prev = nullptr;
        for (size_t ui = 0; ui < units.size(); ++ui) {
            PfUnit & u = units[ui];
            const PrefillJob & pj = jobs[u.job];
            const layout::Geom & g = pj.geom;
            PfPool & pool = *u.sp->pool;
            const int64_t nsb = g.sp_nsb(u.sp->split);
            // 1. encode (once per input and tile)
            double t0 = now_s();
            Enc & en = encs[enc_of[u.job]];
            if (!en.done[u.tile]) {
                en.tiles[u.tile].resize(size_t(g.nbk) * PF_SLAB_BYTES);
                pf_encode_tile(en.tiles[u.tile].data(), pj.x, pj.x_stride, pj.n_tok, g.k, u.tile);
                en.done[u.tile] = 1;
                st_.pf_bytes_x += double(g.k) * PF_TILE_TOK * sizeof(float);
            }
            st_.pf_t_encode += now_s() - t0;
            // 2. A BO tile slot
            t0 = now_s();
            u.slot = pool.next_slot;
            pool.next_slot = (pool.next_slot + 1) % PF_MAX_TILES;
            pf_build_a_tile(pool.a.map<uint8_t *>() + pool.a_stride * u.slot, en.tiles[u.tile].data(), g.nbk, nsb, layout::fmt_info(g.fmt).qt, g.sp_groups(u.sp->split));
            st_.pf_t_abuild += now_s() - t0;
            st_.pf_bytes_a += double(pool.a_tile);
            pool.a.sync(XCL_BO_SYNC_BO_TO_DEVICE, pool.a_tile, pool.a_stride * u.slot);
            st_.pf_t_abo += now_s() - t0;
            // 3. dispatch
            get_pf_run(c, *u.sp, u.slot).start();
            st_.pf_dispatches++;
            st_.pf_bytes_w += double(g.sp_bytes(u.sp->split));
            // 4. finish the previous dispatch while this one runs
            if (prev) finish(*prev);
            prev = &u;
        }
        if (prev) finish(*prev);
        for (size_t i = 0; i < n_jobs; ++i) st_.pf_jobs++;
        st_.pf_waves++;
        return true;
    }

    std::mutex mu_;
    xrt::device dev_;
    bool ok_ = false;
    std::unique_ptr<Ctx> ctx_;
    std::map<JobKey, Splits> jobs_;
    std::map<JobKey, Splits> split_jobs_;
    std::vector<SJob *> sp_disp_;          // dispatches of the decode batch in flight
    std::vector<size_t> sp_jobs_;          // job index of each
    const CpuDecodeJob * sp_cjobs_ = nullptr;
    std::unique_ptr<PfCtx> pf_ctx_;
    std::map<std::pair<size_t, size_t>, std::unique_ptr<PfPool>> pf_pools_;
    std::map<JobKey, PfSplits> pf_jobs_;
    std::map<std::tuple<int64_t, int64_t, int64_t>, uint32_t> pf_instr_words_;
    int last_ctx_ = 0;      // 0 none, 1 decode, 2 prefill
    ExecStats st_;
};

} // namespace

std::unique_ptr<NpuExecutor> make_xrt_executor() {
    return std::make_unique<XrtExecutor>();
}

} // namespace xdna2

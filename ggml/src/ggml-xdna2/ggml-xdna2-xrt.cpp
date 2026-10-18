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
#include <numeric>
#include <mutex>
#include <tuple>
#include <vector>

#include <sys/resource.h>

#include "ggml-xdna2-sync.h"
#include "ggml-xdna2-kernels.h"
#include "ggml-xdna2-bfp.h"
#include "ggml-xdna2-attn.h"

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
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int>, xrt::bo> instr;     // (chain_words, a_words, a_rep, o_words, tiles)
};

// The ATTENTION hw context (third xclbin): instruction BOs per (pairs, nb, kvmax).
struct FaCtx {
    xrt::xclbin                       xb;
    xrt::hw_context                   hw;
    xrt::kernel                       kern;
    std::map<std::tuple<int, int64_t, int64_t>, std::pair<xrt::bo, uint32_t>> instr;
};

// bfp16 shadow of one layer's KV cache: K and V BOs [kv head][cap_rows / 16 blocks][obj]. A block is `full` once it was encoded from 16 valid
// cache rows; its fingerprint then detects a later rewrite of those rows (another sequence, a context shift, a new prompt of the same length).
struct FaShadow {
    std::string           variant;          // FaShape::variant() of the encoding (a cache address may be reused with another shape)
    int64_t               cap_rows = 0;
    xrt::bo               k, v;
    std::vector<char>     full;
    std::vector<uint64_t> fp;
};

// Activation + output BOs shared by all prefill dispatches of one shape. A dispatch ("unit") runs up to pf_max_tiles() consecutive 64-token tiles of
// one split; two units are in flight, so there are two slots, each with room for pf_max_tiles() A tiles and Y tiles.
struct PfPool {
    size_t  a_tile = 0, y_tile = 0, a_slot = 0, y_slot = 0;
    int     next_slot = 0;
    xrt::bo a, y;
    std::vector<xrt::bo> a_sub, y_sub;      // per slot
};

// One prefill job split: the weight sub-BO and one xrt::run per (slot, tiles), created lazily (arguments set once).
struct PfSplit {
    layout::Geom geom;
    int64_t      split = 0;
    xrt::bo      w;
    PfPool *     pool = nullptr;
    std::map<std::pair<int, int>, xrt::run> runs;
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
            // XRT locks a 64 MiB heap when it opens the device; the usual 8 MiB default RLIMIT_MEMLOCK makes that fail
            rlimit rl{};
            if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < (rlim_t(128) << 20)) {
                GGML_LOG_ERROR("ggml-xdna2: the locked-memory limit is %llu KiB, XRT needs more: run 'ulimit -l unlimited' first "
                               "(raise the hard limit in /etc/security/limits.d if needed). Falling back to the CPU.\n",
                               (unsigned long long) (rl.rlim_cur >> 10));
            }
        }
    }

    ~XrtExecutor() override {
        std::lock_guard<std::mutex> lk(mu_);
        if (ctx_) ctx_->runlists.clear();
        split_jobs_.clear();
        pf_jobs_.clear();
        pf_pools_.clear();
        pf_ctx_.reset();
        fa_shadows_.clear();
        fa_ctxs_.clear();
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
            j->G_npu = std::min<int64_t>(p.sb_npu * g.gsb, Gs);
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
                    cpu_pack_npu_act(jobs[i].geom.fmt, jobs[i].geom.k, j->nbk_npu * 256, j->G_npu, j->n_w, jobs[i].geom.gsb, cpu_decode_act(i), j->a.map<uint8_t *>());
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

    static int pf_max_tiles() {
        static const int t = [] {
            const char * e = std::getenv("GGML_XDNA2_PF_TILES");
            int v = e && *e ? std::atoi(e) : 4;      // measured: 4 beats 1 (dispatch cost) and 16 (coarser host/NPU overlap)
            int p = 1;
            while (p * 2 <= std::min(std::max(v, 1), 32)) p *= 2;
            return p;
        }();
        return t;
    }

    PfPool & get_pf_pool(PfCtx & c, size_t a_tile, size_t y_tile) {
        const auto key = std::make_pair(a_tile, y_tile);
        auto it = pf_pools_.find(key);
        if (it != pf_pools_.end()) return *it->second;
        auto p = std::make_unique<PfPool>();
        const size_t tm = size_t(pf_max_tiles());
        p->a_tile = a_tile;
        p->y_tile = y_tile;
        p->a_slot = (a_tile * tm + 4095) / 4096 * 4096;
        p->y_slot = (y_tile * tm + 4095) / 4096 * 4096;
        p->a = xrt::ext::bo{dev_, p->a_slot * 2};
        p->y = xrt::ext::bo{dev_, p->y_slot * 2};
        std::memset(p->a.map<char *>(), 0, p->a_slot * 2);
        p->a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::memset(p->y.map<char *>(), 0, p->y_slot * 2);
        p->y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for (int t = 0; t < 2; ++t) {
            p->a_sub.emplace_back(p->a, p->a_slot, p->a_slot * t);
            p->y_sub.emplace_back(p->y, p->y_slot, p->y_slot * t);
        }
        (void) c;
        return *pf_pools_.emplace(key, std::move(p)).first->second;
    }

    // a_words: A slab words of one superblock pass (read a_rep = nsb times by the DMA)
    const xrt::bo & get_pf_instr(PfCtx & c, int64_t chain_words, int64_t a_words, int64_t a_rep, int64_t o_words, int tiles, uint32_t & nwords) {
        const auto key = std::make_tuple(chain_words, a_words, a_rep, o_words, tiles);
        auto it = c.instr.find(key);
        if (it == c.instr.end()) {
            std::vector<uint32_t> w;
            if (!prefill_kernel_cache().insts_for(chain_words, a_words, o_words, w, 0, a_rep, tiles)) {
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
            j->pool = &get_pf_pool(c, pf_a_tile_bytes(g.nbk), pf_y_tile_bytes(nsb));
            sj.push_back(std::move(j));
        }
        st_.pf_t_setup += now_s() - t0;
        return pf_jobs_.emplace(key, std::move(sj)).first->second;
    }

    xrt::run & get_pf_run(PfCtx & c, PfSplit & j, int slot, int tiles) {
        const auto key = std::make_pair(slot, tiles);
        auto it = j.runs.find(key);
        if (it != j.runs.end()) return it->second;
        const layout::Geom & g = j.geom;
        const int64_t nsb = g.sp_nsb(j.split);
        uint32_t nw = 0;
        const xrt::bo & bi = get_pf_instr(c, int64_t(g.sp_chain_words(j.split)), int64_t(g.nbk * PF_SLAB_BYTES / 4), nsb, nsb * 4 * layout::GSB_MAX * 1024, tiles, nw);
        xrt::run r(c.kern);
        r.set_arg(0, 3);
        r.set_arg(1, bi);
        r.set_arg(2, nw);
        r.set_arg(3, j.w);
        r.set_arg(4, j.pool->a_sub[slot]);
        r.set_arg(5, j.pool->y_sub[slot]);
        return j.runs.emplace(key, std::move(r)).first->second;
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

    // One dispatch ("unit") = up to pf_max_tiles() consecutive 64-token tiles of one split of one job (the NPU re-streams the split's weights per
    // tile, but the fixed cost of a dispatch is paid once). The units of a call are software pipelined: while the NPU runs unit u-1 the host
    // encodes/builds the A tiles of unit u and unpermutes the result of unit u-2 (at most two units are in flight, one per pool slot).
    struct PfUnit { size_t job; PfSplit * sp; int tile0; int tiles; int slot; };

    bool run_prefill_locked(const PrefillJob * jobs, size_t n_jobs) {
        PfCtx & c = get_pf_ctx();
        if (last_ctx_ != 2) { if (last_ctx_ != 0) st_.pf_switches++; last_ctx_ = 2; }
        std::vector<PfSplits *> js;
        for (size_t i = 0; i < n_jobs; ++i) js.push_back(&get_pf_job(jobs[i], c));
        // jobs sharing the same input rows share the encoded slabs (per tile, encoded on first use)
        struct Enc { const float * x; size_t stride; int64_t n_tok, k; std::vector<uint8_t> slabs; std::vector<char> done; };   // slabs: [tile][nbk][slab]
        std::vector<Enc> encs;
        std::vector<size_t> enc_of(n_jobs);
        for (size_t i = 0; i < n_jobs; ++i) {
            size_t e = encs.size();
            for (size_t q = 0; q < encs.size(); ++q) {
                if (encs[q].x == jobs[i].x && encs[q].stride == jobs[i].x_stride && encs[q].n_tok == jobs[i].n_tok && encs[q].k == jobs[i].geom.k) { e = q; break; }
            }
            if (e == encs.size()) {
                const int64_t nt = (jobs[i].n_tok + PF_TILE_TOK - 1) / PF_TILE_TOK;
                encs.push_back(Enc{ jobs[i].x, jobs[i].x_stride, jobs[i].n_tok, jobs[i].geom.k, std::vector<uint8_t>(size_t(nt) * size_t(jobs[i].geom.nbk) * PF_SLAB_BYTES),
                                    std::vector<char>(nt, 0) });
            }
            enc_of[i] = e;
        }
        std::vector<PfUnit> units;
        const int tmax = pf_max_tiles();
        for (size_t i = 0; i < n_jobs; ++i) {
            const int nt = int((jobs[i].n_tok + PF_TILE_TOK - 1) / PF_TILE_TOK);
            for (auto & sp : *js[i]) {
                for (int t0 = 0; t0 < nt;) {
                    int T = tmax;
                    while (T > nt - t0) T /= 2;
                    units.push_back(PfUnit{ i, sp.get(), t0, T, -1 });
                    t0 += T;
                }
            }
        }
        auto finish = [&](PfUnit & u) {
            double t0 = now_s();
            get_pf_run(c, *u.sp, u.slot, u.tiles).wait2();
            st_.pf_t_npu += now_s() - t0;      // blocked on the NPU (the host work of the next unit already ran during the dispatch)
            t0 = now_s();
            PfPool & pool = *u.sp->pool;
            sync_from_device(pool.y, pool.y_tile * size_t(u.tiles), pool.y_slot * size_t(u.slot));
            st_.pf_t_ysync += now_s() - t0;
            const PrefillJob & pj = jobs[u.job];
            const float * y = pool.y.map<const float *>() + (pool.y_slot / sizeof(float)) * size_t(u.slot);
            pf_unpermute_tiles(y, pool.y_tile / sizeof(float), u.tile0, u.tiles, pj.n_tok, pj.geom, u.sp->split, pj.dst, pj.dst_row_stride,
                               pj.epi, pj.epi_other);
            st_.pf_t_unperm += now_s() - t0;
            st_.pf_bytes_y += double(pool.y_tile) * u.tiles;
        };
        PfUnit * prev = nullptr;
        int next_slot = 0;
        for (size_t ui = 0; ui < units.size(); ++ui) {
            PfUnit & u = units[ui];
            const PrefillJob & pj = jobs[u.job];
            const layout::Geom & g = pj.geom;
            PfPool & pool = *u.sp->pool;
            const int64_t nsb = g.sp_nsb(u.sp->split);
            // 1. encode (once per input and tile)
            double t0 = now_s();
            Enc & en = encs[enc_of[u.job]];
            const size_t tile_bytes = size_t(g.nbk) * PF_SLAB_BYTES;
            for (int t = u.tile0; t < u.tile0 + u.tiles;) {            // runs of not yet encoded tiles, each in one parallel region
                if (en.done[t]) { ++t; continue; }
                int e = t;
                while (e < u.tile0 + u.tiles && !en.done[e]) en.done[e++] = 1;
                pf_encode_tiles(en.slabs.data() + tile_bytes * size_t(t), tile_bytes, pj.x, pj.x_stride, pj.n_tok, g.k, t, e - t);
                st_.pf_bytes_x += double(g.k) * PF_TILE_TOK * sizeof(float) * (e - t);
                t = e;
            }
            st_.pf_t_encode += now_s() - t0;
            // 2. A BO slot: the tiles back to back (stride a_tile)
            t0 = now_s();
            u.slot = next_slot;
            next_slot ^= 1;
            uint8_t * abase = pool.a.map<uint8_t *>() + pool.a_slot * size_t(u.slot);
            for (int t = 0; t < u.tiles; ++t) {
                pf_build_a_tile(abase + pool.a_tile * size_t(t), en.slabs.data() + tile_bytes * size_t(u.tile0 + t), g.nbk, nsb, layout::fmt_info(g.fmt).qt,
                                g.sp_groups(u.sp->split), g.gsb);
            }
            st_.pf_t_abuild += now_s() - t0;
            st_.pf_bytes_a += double(pool.a_tile) * u.tiles;
            pool.a.sync(XCL_BO_SYNC_BO_TO_DEVICE, pool.a_tile * size_t(u.tiles), pool.a_slot * size_t(u.slot));
            st_.pf_t_abo += now_s() - t0;
            // 3. dispatch (the slot of u was last used by u-2, finished in the previous iteration)
            get_pf_run(c, *u.sp, u.slot, u.tiles).start();
            st_.pf_dispatches++;
            st_.pf_bytes_w += double(g.sp_bytes(u.sp->split)) * u.tiles;
            // 4. finish the previous unit while this one runs
            if (prev) finish(*prev);
            prev = &u;
        }
        if (prev) finish(*prev);
        for (size_t i = 0; i < n_jobs; ++i) st_.pf_jobs++;
        st_.pf_waves++;
        return true;
    }

    // ---------------------------------------------------------------- attention path
    FaCtx & get_fa_ctx(const FaShape & sh) {
        auto & slot = fa_ctxs_[sh.variant()];
        if (slot) return *slot;
        const double t0 = now_s();
        KernelCache & kc = attention_kernel_cache(sh);
        kc.request();
        const double tw = now_s();
        if (!kc.wait()) {
            throw std::runtime_error("the NPU attention kernel " + sh.variant() + " is not available (build failed, see the kernel cache)");
        }
        st_.t_kernel_wait += now_s() - tw;
        auto c = std::make_unique<FaCtx>();
        c->xb = xrt::xclbin(kc.xclbin_path());
        dev_.register_xclbin(c->xb);
        c->hw = xrt::hw_context(dev_, c->xb.get_uuid());
        std::string kn;
        for (auto & k : c->xb.get_kernels()) {
            if (k.get_name().rfind("MLIR_AIE", 0) == 0) { kn = k.get_name(); break; }
        }
        if (kn.empty()) throw std::runtime_error("no MLIR_AIE kernel in " + kc.xclbin_path());
        c->kern = xrt::kernel(c->hw, kn);
        slot = std::move(c);
        st_.fa_t_setup += now_s() - t0;
        return *slot;
    }

    std::pair<xrt::bo, uint32_t> & get_fa_instr(FaCtx & c, const FaShape & sh, int units, int64_t nb, int64_t kvmax) {
        const auto key = std::make_tuple(units, nb, kvmax);
        auto it = c.instr.find(key);
        if (it != c.instr.end()) return it->second;
        std::vector<uint32_t> w;
        if (!attention_kernel_cache(sh).fa_insts_for(units, nb, kvmax, w)) {
            throw std::runtime_error("cannot derive the instruction stream of the attention kernel");
        }
        xrt::bo bi(dev_, w.size() * 4, xrt::bo::flags::cacheable, c.kern.group_id(1));
        std::memcpy(bi.map<char *>(), w.data(), w.size() * 4);
        bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return c.instr.emplace(key, std::make_pair(std::move(bi), uint32_t(w.size()))).first->second;
    }

    FaShadow & get_fa_shadow(const FaShape & sh, const void * id, int64_t rows) {
        FaShadow & s = fa_shadows_[id];
        if (s.variant != sh.variant()) {
            s = FaShadow{};
            s.variant = sh.variant();
        }
        if (s.cap_rows < rows) {
            const double t0 = now_s();
            const int64_t cap = (rows + 255) / 256 * 256;
            const size_t bytes = size_t(sh.nkv) * size_t(cap / FA_BLK) * sh.obj();
            s.k = xrt::ext::bo{dev_, bytes};
            s.v = xrt::ext::bo{dev_, bytes};
            s.cap_rows = cap;
            s.full.assign(size_t(cap / FA_BLK), 0);
            s.fp.assign(size_t(cap / FA_BLK), 0);
            st_.fa_t_setup += now_s() - t0;
        }
        return s;
    }

    // per-call Q (+ header) and output regions, one per dispatch (4 KiB aligned sub-BOs)
    static size_t fa_q_region(const FaShape & sh, int units) { return (size_t(units) * (sh.unit_qb() + sh.obj()) + 4095) / 4096 * 4096; }
    void ensure_fa_io(const FaShape & sh, int64_t total_units) {
        const size_t qb = fa_q_region(sh, int(total_units)) + 5 * 4096, ob = size_t(total_units) * sh.unit_ob();
        if (fa_qbuf_bytes_ < qb) { fa_qbuf_ = xrt::ext::bo{dev_, qb}; fa_qbuf_bytes_ = qb; }
        if (fa_obuf_bytes_ < ob) { fa_obuf_ = xrt::ext::bo{dev_, ob}; fa_obuf_bytes_ = ob; }
    }

    bool attention_available(const FaShape & sh) override {
        if (!attention_enabled() || !sh.supported()) return false;
        return attention_kernel_cache(sh).available();
    }

    bool run_attention(const AttnJob & job) override {
        std::lock_guard<std::mutex> lk(mu_);
        try {
            return run_attention_locked(job);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("ggml-xdna2: NPU attention run failed: %s\n", e.what());
            return false;
        }
    }

    // The queries p0 .. p0 + n_tok - 1 are covered by 16-query tiles from pos0 = p0 rounded down to 16, cpk tiles per unit; unit i streams the
    // K/V blocks 0 .. pos0/16 + cpk (i + 1) - 1. The units are cut into dispatches of 16/8/4/2/1 units, all queued at once (the cores run them
    // back to back).
    bool run_attention_locked(const AttnJob & j) {
        const FaShape & sh = j.shape;
        const int cpk = sh.cpk(), uq = sh.unit_q(), nc = sh.nc(), grp = sh.grp();
        const size_t OBJ = sh.obj();
        FaCtx & c = get_fa_ctx(sh);
        if (last_ctx_ != 3 || last_fa_variant_ != sh.variant()) { if (last_ctx_ != 0) st_.fa_switches++; last_ctx_ = 3; last_fa_variant_ = sh.variant(); }
        st_.fa_calls++;
        const int64_t pos0 = j.p0 / FA_BLK * FA_BLK;
        const int64_t n_tiles = (j.p0 + j.n_tok - pos0 + FA_BLK - 1) / FA_BLK;
        const int64_t units = (n_tiles + cpk - 1) / cpk;
        const int64_t d0 = pos0 / FA_BLK;
        const int64_t nb_total = d0 + cpk * units;
        const int64_t n_valid = j.p0 + j.n_tok;                   // cells with data (the mask lets query n_tok-1 see them all)
        // sliding window: wb blocks before the q block are visible. A dispatch streams K/V from its first needed block, rounded down so that
        // the sub-BO offset stays 4 KiB aligned; blocks before the window of the first query are never read (and not encoded)
        const int64_t wb = j.window / FA_BLK;
        const int64_t b_align = int64_t(4096 / std::gcd<size_t>(sh.obj(), 4096));
        auto first_block = [&](int64_t d_first) { return wb ? std::max<int64_t>(0, d_first - wb) / b_align * b_align : 0; };
        const int64_t b_min = first_block(d0);
        FaShadow & s = get_fa_shadow(sh, j.kv_id, std::max(j.kv_cap, nb_total * FA_BLK));
        const int64_t cap_blocks = s.cap_rows / FA_BLK;

        // 1. shadow KV: (re)encode the blocks that are new, partial, or whose rows changed
        double t0 = now_s();
        std::vector<char> todo(size_t(nb_total), 0);
        const size_t row_bytes = size_t(sh.nkv) * size_t(sh.d) * 2;
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(pf_threads())
#endif
        for (int64_t b = b_min; b < nb_total; ++b) {
            if (j.rot > 0 || !s.full[b] || (b + 1) * FA_BLK > n_valid || b >= j.p0 / FA_BLK) { todo[b] = 1; continue; }   // rotated: re-encode all
            if (fa_block_fingerprint(j.k, j.v, j.kv_row_stride, row_bytes, b) != s.fp[b]) todo[b] = 1;
        }
        int64_t lo = nb_total, hi = -1, n_enc = 0;
        for (int64_t b = 0; b < nb_total; ++b) if (todo[b]) { lo = std::min(lo, b); hi = b; n_enc++; }
        if (hi >= 0) {
            uint8_t * kb = s.k.map<uint8_t *>(), * vb = s.v.map<uint8_t *>();
#if defined(_OPENMP)
            #pragma omp parallel for collapse(2) schedule(dynamic, 4) num_threads(pf_threads())
#endif
            for (int64_t b = lo; b <= hi; ++b) {
                for (int g = 0; g < sh.nkv; ++g) {
                    if (!todo[b]) continue;
                    const size_t off = (size_t(g) * cap_blocks + size_t(b)) * OBJ;
                    fa_encode_kv_block(sh, kb + off, vb + off, j.k, j.v, j.bf16, j.kv_row_stride, j.kv_head_stride, g, b, n_valid, j.rot, j.n_cells);
                }
            }
            for (int64_t b = lo; b <= hi; ++b) {
                if (!todo[b]) continue;
                s.full[b] = j.rot == 0 && (b + 1) * FA_BLK <= n_valid;       // a rotated encoding is never reused
                if (s.full[b]) s.fp[b] = fa_block_fingerprint(j.k, j.v, j.kv_row_stride, row_bytes, b);
            }
            for (int g = 0; g < sh.nkv; ++g) {
                const size_t off = (size_t(g) * cap_blocks + size_t(lo)) * OBJ, len = size_t(hi - lo + 1) * OBJ;
                s.k.sync(XCL_BO_SYNC_BO_TO_DEVICE, len, off);
                s.v.sync(XCL_BO_SYNC_BO_TO_DEVICE, len, off);
            }
        }
        st_.fa_blocks_enc += uint64_t(n_enc);
        st_.fa_t_kv += now_s() - t0;

        // 2. dispatch plan and Q objects + headers
        t0 = now_s();
        struct Disp { int64_t first; int units; size_t qoff, ooff; int64_t b0; xrt::run run; };
        std::vector<Disp> ds;
        size_t qoff = 0, ooff = 0;
        for (int64_t p = 0; p < units;) {
            int r = 16;
            while (r > units - p) r /= 2;
            ds.push_back(Disp{ p, r, qoff, ooff, first_block(d0 + cpk * p), xrt::run() });
            qoff += fa_q_region(sh, r);
            ooff += size_t(r) * sh.unit_ob();
            p += r;
        }
        ensure_fa_io(sh, units);
        uint8_t * qb = fa_qbuf_.map<uint8_t *>();
        const float prescale = j.scale * 1.4426950408889634f;
#if defined(_OPENMP)
        #pragma omp parallel for collapse(2) schedule(static) num_threads(pf_threads())
#endif
        for (int64_t p = 0; p < units; ++p) {
            for (int o = 0; o < sh.nkv * nc; ++o) {
                size_t di = 0;
                while (di + 1 < ds.size() && ds[di + 1].first <= p) di++;
                const Disp & d = ds[di];
                const int64_t r = p - d.first;
                const int g = o / nc, me = o % nc, qi = me / 4, hh = me % 4;
                fa_encode_q(sh, qb + d.qoff + (size_t(r) * sh.nkv + size_t(g)) * nc * OBJ + size_t(me) * OBJ, j.q, j.q_tok_stride, j.q_head_stride,
                            j.n_tok, j.p0, pos0 + uq * p + FA_BLK * qi, hh < grp ? grp * g + hh : sh.nh, prescale);
            }
        }
        for (auto & d : ds) {
            for (int r = 0; r < d.units; ++r) {
                int32_t * hdr = reinterpret_cast<int32_t *>(qb + d.qoff + size_t(d.units) * sh.unit_qb() + size_t(r) * OBJ);
                hdr[0] = int32_t(d0 + cpk * (d.first + r) + cpk - d.b0);     // block indices relative to the first streamed block
                hdr[1] = int32_t(d0 + cpk * (d.first + r) - d.b0);
                hdr[2] = int32_t(wb);
            }
        }
        fa_qbuf_.sync(XCL_BO_SYNC_BO_TO_DEVICE, qoff, 0);
        st_.fa_t_q += now_s() - t0;

        // 3. queue all dispatches, then collect them in order (the scatter of one overlaps the next)
        t0 = now_s();
        std::vector<xrt::bo> subs;
        subs.reserve(4 * ds.size());
        for (auto & d : ds) {
            auto & in = get_fa_instr(c, sh, d.units, d0 + cpk * d.first + cpk - d.b0, s.cap_rows);
            subs.emplace_back(fa_qbuf_, fa_q_region(sh, d.units), d.qoff);
            subs.emplace_back(fa_obuf_, size_t(d.units) * sh.unit_ob(), d.ooff);
            const size_t kv_off = size_t(d.b0) * OBJ, kv_bytes = size_t(sh.nkv) * size_t(cap_blocks) * OBJ;
            if (d.b0 > 0) {
                subs.emplace_back(s.k, kv_bytes - kv_off, kv_off);
                subs.emplace_back(s.v, kv_bytes - kv_off, kv_off);
            }
            d.run = xrt::run(c.kern);
            d.run.set_arg(0, 3);
            d.run.set_arg(1, in.first);
            d.run.set_arg(2, in.second);
            const size_t qs = subs.size() - (d.b0 > 0 ? 4 : 2);
            d.run.set_arg(3, subs[qs]);
            if (d.b0 > 0) {
                d.run.set_arg(4, subs[qs + 2]);
                d.run.set_arg(5, subs[qs + 3]);
            } else {
                d.run.set_arg(4, s.k);
                d.run.set_arg(5, s.v);
            }
            d.run.set_arg(6, subs[qs + 1]);
            d.run.start();
            st_.fa_dispatches++;
        }
        st_.fa_t_setup += now_s() - t0;
        for (auto & d : ds) {
            t0 = now_s();
            d.run.wait2();
            st_.fa_t_npu += now_s() - t0;
            t0 = now_s();
            sync_from_device(fa_obuf_, size_t(d.units) * sh.unit_ob(), d.ooff);
            fa_scatter_out(sh, fa_obuf_.map<const float *>() + d.ooff / sizeof(float), d.units, pos0 + uq * d.first, j.p0, j.n_tok, j.dst,
                           j.dst_tok_stride, j.dst_head_stride);
            st_.fa_t_out += now_s() - t0;
        }
        st_.fa_pairs += uint64_t(units);
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
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int>, uint32_t> pf_instr_words_;
    std::map<std::string, std::unique_ptr<FaCtx>> fa_ctxs_;
    std::string last_fa_variant_;
    std::map<const void *, FaShadow> fa_shadows_;
    xrt::bo fa_qbuf_, fa_obuf_;
    size_t fa_qbuf_bytes_ = 0, fa_obuf_bytes_ = 0;
    int last_ctx_ = 0;      // 0 none, 1 decode, 2 prefill, 3 attention
    ExecStats st_;
};

} // namespace

std::unique_ptr<NpuExecutor> make_xrt_executor() {
    return std::make_unique<XrtExecutor>();
}

} // namespace xdna2

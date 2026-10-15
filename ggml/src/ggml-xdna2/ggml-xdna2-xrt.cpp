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
    std::map<std::tuple<int64_t, int64_t, int64_t>, xrt::bo> instr;
    std::map<std::tuple<int64_t, int64_t, int64_t>, uint32_t> instr_words;
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
            {
                xrt::run r2(c.kern);
                r2.set_arg(0, 3);
                r2.set_arg(1, bi);
                r2.set_arg(2, c.instr_words[std::make_tuple(int64_t(g.sp_chain_words(s)), int64_t(j->a_bytes / 4), g.sp_groups(s))]);
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

    bool run_prefill_locked(const PrefillJob * jobs, size_t n_jobs) {
        PfCtx & c = get_pf_ctx();
        if (last_ctx_ != 2) { if (last_ctx_ != 0) st_.pf_switches++; last_ctx_ = 2; }
        std::vector<PfSplits *> js;
        for (size_t i = 0; i < n_jobs; ++i) js.push_back(&get_pf_job(jobs[i], c));
        // jobs sharing the same input rows share the encoded slabs
        std::vector<std::vector<uint8_t>> enc(n_jobs);
        std::vector<int> enc_of(n_jobs, -1);
        int64_t max_tiles = 0;
        for (size_t i = 0; i < n_jobs; ++i) max_tiles = std::max(max_tiles, (jobs[i].n_tok + PF_TILE_TOK - 1) / PF_TILE_TOK);
        for (int64_t w0 = 0; w0 < max_tiles; w0 += PF_MAX_TILES) {
            st_.pf_waves++;
            for (size_t i = 0; i < n_jobs; ++i) {
                const PrefillJob & pj = jobs[i];
                const int64_t ntiles_all = (pj.n_tok + PF_TILE_TOK - 1) / PF_TILE_TOK;
                if (w0 >= ntiles_all) continue;
                const int T = int(std::min<int64_t>(PF_MAX_TILES, ntiles_all - w0));
                const layout::Geom & g = pj.geom;
                // 1. encode (shared with an earlier job of this call that has the same input)
                double t0 = now_s();
                int src = -1;
                for (size_t q = 0; q < i; ++q) {
                    if (jobs[q].x == pj.x && jobs[q].x_stride == pj.x_stride && jobs[q].n_tok == pj.n_tok && jobs[q].geom.k == g.k) { src = int(q); break; }
                }
                if (src >= 0 && enc_of[src] >= 0) {
                    enc_of[i] = enc_of[src];
                } else {
                    enc[i].resize(size_t(T) * size_t(g.nbk) * PF_SLAB_BYTES);
                    pf_encode_acts(enc[i].data(), pj.x, pj.x_stride, pj.n_tok, g.k, int(w0), T);
                    enc_of[i] = int(i);
                }
                // (the cache entry of the sharing job may belong to a previous wave: re-encode when the wave changed)
                st_.pf_t_encode += now_s() - t0;
                const std::vector<uint8_t> & slabs = enc[enc_of[i]];
                for (auto & sp : *js[i]) {
                    PfSplit & j = *sp;
                    const int64_t nsb = g.sp_nsb(j.split);
                    // 2. A BO
                    t0 = now_s();
                    uint8_t * abo = j.pool->a.map<uint8_t *>();
                    pf_build_a_bo(abo, j.pool->a_stride, slabs.data(), T, g.nbk, nsb, layout::fmt_info(g.fmt).qt, g.sp_groups(j.split));
                    j.pool->a.sync(XCL_BO_SYNC_BO_TO_DEVICE, j.pool->a_stride * (T - 1) + j.pool->a_tile, 0);
                    st_.pf_t_abo += now_s() - t0;
                    // 3. dispatch the T tiles
                    t0 = now_s();
                    for (int t = 0; t < T; ++t) get_pf_run(c, j, t).start();
                    for (int t = 0; t < T; ++t) j.runs[t].wait2();
                    st_.pf_t_npu += now_s() - t0;
                    st_.pf_dispatches += T;
                    st_.pf_bytes_w += double(g.sp_bytes(j.split)) * T;
                    // 4. results
                    t0 = now_s();
                    sync_from_device(j.pool->y, j.pool->y_tile * T);
                    pf_unpermute(j.pool->y.map<const float *>(), T, int(w0), pj.n_tok, g, j.split, pj.dst, pj.dst_row_stride);
                    st_.pf_t_unperm += now_s() - t0;
                }
                st_.pf_jobs++;
            }
            // encoded slabs belong to one wave
            for (auto & e : enc) e.clear();
            std::fill(enc_of.begin(), enc_of.end(), -1);
        }
        return true;
    }

    std::mutex mu_;
    xrt::device dev_;
    bool ok_ = false;
    std::unique_ptr<Ctx> ctx_;
    std::map<JobKey, Splits> jobs_;
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

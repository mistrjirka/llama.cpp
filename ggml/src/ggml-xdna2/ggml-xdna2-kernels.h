#pragma once

// NPU kernel cache with build-at-load. Dependency free (std only) so it is shared by the backend and the xdna2-kernel-tool.
//
// There is exactly ONE decode kernel (xclbin): kernels/qku.cc + genu.py. The qtype (Q4_K/Q5_K/Q6_K/Q8_0), K (<= layout::KMAX),
// and the number of groups are RUNTIME values (activation header, see qku.cc), so all weights of a model run in ONE hw context.
// Only the instruction stream depends on three host values, each linear (zero intercept) in exactly one 32-bit word set:
//     chain_words   weight words per chain of a dispatch  (layout::Geom::sp_chain_words)
//     a_words       words of the activation BO
//     g             16-row groups per core                (output words)
// The build step produces the xclbin plus the base stream (kRefChain, kRefA, kRefG) and one stream per varied parameter; the backend
// derives the stream of a dispatch by scaling the words that differ (insts_for; checked bit identical against native builds).
//
// Build step: kernels/build_kernel.sh (IRON generator genu.py + kernel qku.cc + peano + aiecc, lab stage5), stored in the cache dir:
//
//   <cache_dir>/<src_hash>/unified/kernel.xclbin
//                                  insts_base.bin insts_cw.bin insts_aw.bin insts_g.bin     (reference instruction streams)
//   <cache_dir>/<src_hash>/unified.failed.log                                               (only after a failed build)
//
//   src_hash = 64-bit FNV-1a of the build step's sources (qku.cc, genu.py, build_kernel.sh, env.sh), so editing a kernel or
//   the generator never reuses stale binaries. (The toolchain version is not part of the key.)
//
// Atomicity: a build runs in <cache_dir>/<src_hash>/.tmp.<pid>.<n>.unified/ and the finished directory is rename()d into
// place, so the kernel directory is either absent or complete, also with several processes sharing the cache.

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ggml-xdna2-layout.h"

namespace xdna2 {

constexpr int64_t kRefChain = 2048;   // reference values of the instruction stream parameters (genu.py --chain_words/--a_words/--g)
constexpr int64_t kRefA     = 2048;
constexpr int64_t kRefG     = 2;

class KernelCache {
public:
    struct Config {
        std::string cache_dir;
        std::string kernel_src_dir;   // directory with build_kernel.sh, qku.cc, genu.py, env.sh
        std::string toolchain_env;    // mlir-aie python environment (bin/aiecc, bin/python, lib/python*/site-packages/llvm-aie)
        int         max_jobs = 4;     // unused (one kernel), kept for tools
        std::function<void(int level /*0 info, 1 warn, 2 error*/, const std::string &)> log;
    };

    explicit KernelCache(Config cfg);
    ~KernelCache();

    // external toolchain + build scripts are installed
    bool toolchain_present() const;
    // the unified kernel can run this weight geometry (K % 256 == 0, K <= layout::KMAX; any N: larger matrices are split)
    static bool supports(const layout::Geom & geom);
    // the kernel is in the cache
    bool cached() const;
    // cached, or buildable with the installed toolchain
    bool available() const;

    std::string dir() const;
    std::string xclbin_path() const { return dir() + "/kernel.xclbin"; }

    // Instruction stream (words) of a dispatch, derived from the reference streams.
    bool insts_for(int64_t chain_words, int64_t a_words, int64_t g, std::vector<uint32_t> & out);
    // the pure patching rule (also used by the tool/tests): the words where base and var differ are linear in the parameter
    // (zero intercept): w(p) = base * p / p0. Applied on top of `inout`.
    static bool patch_param(const std::vector<uint32_t> & base, const std::vector<uint32_t> & var, int64_t p0, int64_t p1, int64_t p,
                            std::vector<uint32_t> & inout);
    std::string src_hash() const { return hash_; }

    // queue a background build if the kernel is not cached (no-op otherwise)
    void request();
    // block until the kernel is cached or its build failed; true = usable
    bool wait();
    // block until the queued build finished; returns the number of failures
    int wait_all();

    struct Stats { int built = 0, failed = 0; double sum_seconds = 0, wall_seconds = 0; };
    Stats stats() const;

private:
    enum class State { None, Building, Done, Failed };
    void worker();
    bool run_build(double & seconds);
    void say(int level, const std::string & msg) const;

    Config                       cfg_;
    std::string                  hash_;
    mutable std::mutex           mu_;
    mutable bool                 known_cached_ = false;
    std::vector<std::vector<uint32_t>> refs_;     // base, cw, aw, g
    std::condition_variable      cv_;
    State                        state_ = State::None;
    std::thread                  thread_;
    Stats                        stats_;
    int                          tmp_counter_ = 0;
};

} // namespace xdna2

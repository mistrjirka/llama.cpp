#pragma once

// NPU kernel cache with build-at-load. Dependency free (std only) so it is shared by the backend and the
// xdna2-kernel-tool.
//
// A kernel (xclbin + instruction stream) is specialised per (qtype, N_pad of the dispatch chunk, K, M activation vectors).
// It is produced by the external build step kernels/build_kernel.sh (IRON generator gen_qk.py + peano + aiecc, the lab's
// stage2 toolchain) and stored in the cache directory:
//
//   <cache_dir>/<src_hash>/<qtype>_n<N>_k<K>_m<M>/kernel.xclbin
//                                                  insts.bin
//                                                  meta.txt        (generator META line)
//                                                  build.log       (compiler output)
//   <cache_dir>/<src_hash>/<qtype>_n<N>_k<K>_m<M>.failed.log      (only after a failed build)
//
//   src_hash = 64-bit FNV-1a of the build step's sources (qk.cc, gen_qk.py, build_kernel.sh, env.sh), so editing a kernel or
//   the generator never reuses stale binaries. (The toolchain version is not part of the key.)
//
// Atomicity: a build runs in <cache_dir>/<src_hash>/.tmp.<pid>.<n>.<name>/ and the finished directory is rename()d into
// place, so a kernel directory is either absent or complete, also with several processes sharing the cache.
// Two processes may still build the same kernel at the same time (wasted work, not a correctness issue).

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ggml-xdna2-layout.h"

namespace xdna2 {

struct KernelKey {
    layout::Fmt fmt = layout::Fmt::Q4_K;
    int64_t     n_pad = 0;      // rows of the dispatch chunk (multiple of 512)
    int64_t     k = 0;
    int         m = 1;          // activation vectors per dispatch

    std::string name() const;   // e.g. Q4_K_n12288_k4096_m1
    bool operator<(const KernelKey & o) const;
};

class KernelCache {
public:
    struct Config {
        std::string cache_dir;
        std::string kernel_src_dir;   // directory with build_kernel.sh, qk.cc, gen_qk.py, env.sh
        std::string toolchain_env;    // mlir-aie python environment (bin/aiecc, bin/python, lib/python*/site-packages/llvm-aie)
        int         max_jobs = 4;
        std::function<void(int level /*0 info, 1 warn, 2 error*/, const std::string &)> log;
    };

    explicit KernelCache(Config cfg);
    ~KernelCache();

    // external toolchain + build scripts are installed
    bool toolchain_present() const;
    // the generator can produce this kernel (Q4_K / Q6_K, K % 4096 == 0, Q6_K only M == 1, M <= 8)
    static bool generator_supports(const KernelKey & key);
    // already in the cache
    bool cached(const KernelKey & key) const;
    // cached, or buildable with the installed toolchain
    bool available(const KernelKey & key) const;

    std::string dir_of(const KernelKey & key) const;
    std::string src_hash() const { return hash_; }

    // queue a background build if the kernel is not cached (no-op otherwise); at most max_jobs run at a time
    void request(const KernelKey & key);
    // block until the kernel is cached or its build failed; true = usable
    bool wait(const KernelKey & key);
    // block until all queued builds finished; returns the number of failures
    int wait_all();

    struct Stats { int built = 0, failed = 0; double sum_seconds = 0, wall_seconds = 0; };
    Stats stats() const;

private:
    enum class State { Queued, Building, Done, Failed };
    void worker();
    bool run_build(const KernelKey & key, double & seconds);
    void say(int level, const std::string & msg) const;

    Config                       cfg_;
    std::string                  hash_;
    mutable std::mutex           mu_;
    std::condition_variable      cv_;
    std::map<KernelKey, State>   state_;
    std::vector<KernelKey>       queue_;
    std::vector<std::thread>     threads_;
    int                          active_workers_ = 0;
    int                          pending_ = 0;
    Stats                        stats_;
    double                       batch_t0_ = 0;
    Stats                        batch_;
    int                          tmp_counter_ = 0;
};

} // namespace xdna2

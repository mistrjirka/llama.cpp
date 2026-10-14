#pragma once

// NPU kernel cache with build-at-load. Dependency free (std only) so it is shared by the backend and the
// xdna2-kernel-tool.
//
// A kernel (xclbin + instruction stream) is specific to (qtype, K, M activation vectors) and serves ANY N % 512 == 0: only the
// instruction stream depends on N, and its 68 N dependent words are linear in N (stage3 RESULTS.md). The build step therefore
// produces the xclbin plus the instruction streams for two reference sizes (kRefN0, kRefN1) and the backend derives the stream
// for the actual N by scaling the words that differ (bit identical to a native build, checked by `xdna2-kernel-tool patch`).
//
// It is produced by the external build step kernels/build_kernel.sh (IRON generator gen3.py + kernel qk3.cc + peano + aiecc, the
// lab's stage3 toolchain) and stored in the cache directory:
//
//   <cache_dir>/<src_hash>/<qtype>_k<K>_m<M>/kernel.xclbin
//                                              insts_n1024.bin insts_n2048.bin     (reference instruction streams)
//                                              meta.txt                            (generator META line)
//   <cache_dir>/<src_hash>/<qtype>_k<K>_m<M>.failed.log                            (only after a failed build)
//
//   src_hash = 64-bit FNV-1a of the build step's sources (qk3.cc, gen3.py, build_kernel.sh, env.sh), so editing a kernel or
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

constexpr int64_t kRefN0 = 1024;
constexpr int64_t kRefN1 = 2048;

struct KernelKey {
    layout::Fmt fmt = layout::Fmt::Q4_K;
    int64_t     k = 0;
    int         m = 1;          // activation vectors per dispatch

    std::string name() const;   // e.g. Q4_K_k4096_m1
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
    // the generator can produce this kernel (K % 256 == 0 and K <= 12288 (validated range), Q4_K M <= 4, others M == 1)
    static bool generator_supports(const KernelKey & key);
    // already in the cache
    bool cached(const KernelKey & key) const;
    // cached, or buildable with the installed toolchain
    bool available(const KernelKey & key) const;

    std::string dir_of(const KernelKey & key) const;
    std::string xclbin_path(const KernelKey & key) const { return dir_of(key) + "/kernel.xclbin"; }

    // Instruction stream (words) of a cached kernel for n_pad rows (n_pad % 512 == 0), derived from the two reference streams.
    bool insts_for(const KernelKey & key, int64_t n_pad, std::vector<uint32_t> & out);
    // the pure patching rule (also used by the tool/tests): w(N) = w0 * N / N0 for the words where a and b differ
    static bool patch_insts(const std::vector<uint32_t> & a, int64_t n0, const std::vector<uint32_t> & b, int64_t n1, int64_t n,
                            std::vector<uint32_t> & out);
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
    mutable std::map<KernelKey, bool> known_cached_;   // positive cache of the stat()s
    std::map<KernelKey, std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> refs_;
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

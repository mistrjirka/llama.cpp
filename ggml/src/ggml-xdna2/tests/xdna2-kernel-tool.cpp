// xdna2-kernel-tool: exercise the NPU kernel cache without the backend (does not touch the NPU).
//
//   xdna2-kernel-tool [--cache DIR] [--toolchain DIR] [--src DIR] [--list] [SPEC...]
//   SPEC = QTYPE:KxN   in GGUF shape order (K, N), e.g. Q4_K:4096x12288  Q6_K:4096x248320  -> geometry (splits, bytes) and support check
//   SPEC = ornith      the unique weight shapes of Ornith-1.5-9B
//   xdna2-kernel-tool patch CHAIN_WORDS ACT_WORDS G out.bin   write the instruction stream derived from the cached reference streams
//
// The unified decode kernel serves every shape, so there is exactly one kernel to build; the wall time is reported.
// Running it twice shows the warm (cached) path.

#include "ggml-xdna2-kernels.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace xdna2;

#ifndef GGML_XDNA2_KERNEL_DIR
#define GGML_XDNA2_KERNEL_DIR ""
#endif
#ifndef GGML_XDNA2_MLIR_AIE_ENV
#define GGML_XDNA2_MLIR_AIE_ENV ""
#endif

static bool parse_fmt(const std::string & s, layout::Fmt & f) {
    for (int i = 0; i < int(layout::Fmt::COUNT); ++i) {
        if (s == layout::fmt_info(layout::Fmt(i)).name) { f = layout::Fmt(i); return true; }
    }
    return false;
}

int main(int argc, char ** argv) {
    KernelCache::Config cfg;
    const char * home = std::getenv("HOME");
    const char * ce = std::getenv("GGML_XDNA2_CACHE_DIR");
    cfg.cache_dir = ce && *ce ? ce : std::string(home ? home : ".") + "/.cache/ggml-xdna2";
    cfg.kernel_src_dir = GGML_XDNA2_KERNEL_DIR;
    cfg.toolchain_env = GGML_XDNA2_MLIR_AIE_ENV;
    cfg.log = [](int level, const std::string & m) { std::printf("[%s] %s\n", level == 0 ? "info" : level == 1 ? "warn" : "ERROR", m.c_str()); std::fflush(stdout); };
    bool list_only = false;
    std::vector<std::string> specs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cache" && i + 1 < argc) cfg.cache_dir = argv[++i];
        else if (a == "--toolchain" && i + 1 < argc) cfg.toolchain_env = argv[++i];
        else if (a == "--src" && i + 1 < argc) cfg.kernel_src_dir = argv[++i];
        else if (a == "--list") list_only = true;
        else if (a == "ornith") {
            for (const char * s : { "Q4_K:4096x12288", "Q4_K:12288x4096", "Q4_K:4096x4096", "Q4_K:4096x8192", "Q4_K:4096x1024",
                                    "Q6_K:4096x8192", "Q6_K:12288x4096", "Q6_K:4096x1024", "Q6_K:4096x248320",
                                    "Q8_0:4096x4096", "Q5_K:4096x1024", "Q5_K:4096x4096" }) specs.push_back(s);
        } else specs.push_back(a);
    }
    if (specs.size() == 5 && specs[0] == "patch") {
        KernelCache pc(cfg);
        std::vector<uint32_t> w;
        if (!pc.insts_for(std::atoll(specs[1].c_str()), std::atoll(specs[2].c_str()), std::atoll(specs[3].c_str()), w)) {
            std::fprintf(stderr, "patch failed (kernel not cached or a stream word is not linear in its parameter)\n");
            return 1;
        }
        FILE * f = std::fopen(specs[4].c_str(), "wb");
        std::fwrite(w.data(), 4, w.size(), f);
        std::fclose(f);
        return 0;
    }

    KernelCache cache(cfg);
    std::printf("cache dir      : %s\nsource hash    : %s\ntoolchain      : %s (%s)\nkernel         : %s\n", cfg.cache_dir.c_str(),
                cache.src_hash().c_str(), cfg.toolchain_env.c_str(), cache.toolchain_present() ? "present" : "MISSING",
                cache.cached() ? "cached" : cache.toolchain_present() ? "missing (buildable)" : "missing, NO TOOLCHAIN: weights would stay on the CPU");
    int unsupported = 0;
    for (const std::string & s : specs) {
        const size_t c = s.find(':'), x = s.find('x');
        layout::Fmt fmt;
        if (c == std::string::npos || x == std::string::npos || !parse_fmt(s.substr(0, c), fmt)) { std::fprintf(stderr, "bad spec %s\n", s.c_str()); return 2; }
        const int64_t k = std::atoll(s.substr(c + 1, x - c - 1).c_str()), n = std::atoll(s.c_str() + x + 1);
        const layout::Geom g = layout::make_geom(fmt, n, k);
        const bool sup = KernelCache::supports(g);
        std::printf("%-6s K=%-6lld N=%-7lld N_pad %-7lld %2lld split(s)  %.1f MB device (GGUF %.1f MB)  %s\n", s.substr(0, c).c_str(), (long long) k,
                    (long long) n, (long long) g.n_pad, (long long) g.nsplit, g.bytes / 1e6, g.gguf_bytes / 1e6, sup ? "supported" : "NOT SUPPORTED");
        unsupported += !sup;
    }
    if (list_only) return 0;

    const auto t0 = std::chrono::steady_clock::now();
    cache.request();
    int failed = cache.wait_all();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto st = cache.stats();
    std::printf("request+wait: %.2f s wall, built %d, failed %d, build work %.1f s\n", wall, st.built, st.failed, st.sum_seconds);
    if (!cache.cached()) { std::printf("MISSING unified kernel\n"); ++failed; }
    return failed ? 1 : 0;
}

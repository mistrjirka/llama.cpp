#include "ggml-xdna2-kernels.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include <dirent.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char ** environ;

namespace xdna2 {

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool exists(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

bool executable(const std::string & p) { return access(p.c_str(), X_OK) == 0; }

void mkdirs(const std::string & path) {
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            mkdir(path.substr(0, i).c_str(), 0755);
        }
    }
}

void rmtree(const std::string & path) {
    DIR * d = opendir(path.c_str());
    if (!d) {
        unlink(path.c_str());
        return;
    }
    while (dirent * e = readdir(d)) {
        if (!std::strcmp(e->d_name, ".") || !std::strcmp(e->d_name, "..")) continue;
        const std::string p = path + "/" + e->d_name;
        struct stat st;
        if (lstat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) rmtree(p); else unlink(p.c_str());
    }
    closedir(d);
    rmdir(path.c_str());
}

std::string read_file(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

const std::vector<const char *> kBuildFilesDecode  = { "build_kernel.sh", "env.sh", "qku.cc", "genu.py" };
const std::vector<const char *> kBuildFilesPrefill = { "build_prefill.sh", "env.sh", "pf.cc", "genp.py", "aie_kernel_utils.h" };

const std::vector<const char *> & build_files(Product p) { return p == Product::Prefill ? kBuildFilesPrefill : kBuildFilesDecode; }

std::string compute_hash(const std::string & dir, Product prod) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const std::string & s) {
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    };
    for (const char * f : build_files(prod)) {
        mix(f);
        mix(read_file(dir + "/" + f));
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return buf;
}

} // namespace

KernelCache::KernelCache(Config cfg) : cfg_(std::move(cfg)) {
    hash_ = compute_hash(cfg_.kernel_src_dir, cfg_.product);
}

KernelCache::~KernelCache() {
    if (thread_.joinable()) thread_.join();
}

void KernelCache::say(int level, const std::string & msg) const {
    if (cfg_.log) cfg_.log(level, msg);
}

bool KernelCache::toolchain_present() const {
    if (cfg_.toolchain_env.empty()) return false;
    for (const char * f : build_files(cfg_.product)) {
        if (!exists(cfg_.kernel_src_dir + "/" + f)) return false;
    }
    return executable(cfg_.toolchain_env + "/bin/aiecc") && executable(cfg_.toolchain_env + "/bin/python") && executable("/bin/bash");
}

bool KernelCache::supports(const layout::Geom & g) {
    return layout::k_supported(g.k) && g.n > 0;
}

static const char * product_dir(Product p) { return p == Product::Prefill ? "prefill" : "unified"; }
std::string KernelCache::dir() const { return cfg_.cache_dir + "/" + hash_ + "/" + product_dir(cfg_.product); }

static const std::vector<const char *> kInstFilesDecode  = { "insts_base.bin", "insts_cw.bin", "insts_aw.bin", "insts_g.bin", "insts_sw.bin" };
static const std::vector<const char *> kInstFilesPrefill = { "insts_base.bin", "insts_cw.bin", "insts_aw.bin", "insts_ow.bin" };
static const std::vector<const char *> & inst_files(Product p) { return p == Product::Prefill ? kInstFilesPrefill : kInstFilesDecode; }

bool KernelCache::cached() const {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (known_cached_) return true;
    }
    bool ok = exists(dir() + "/kernel.xclbin");
    for (const char * f : inst_files(cfg_.product)) ok = ok && exists(dir() + "/" + f);
    if (ok) {
        std::lock_guard<std::mutex> lk(mu_);
        known_cached_ = true;
    }
    return ok;
}

bool KernelCache::available() const {
    return cached() || toolchain_present();
}

bool KernelCache::patch_param(const std::vector<uint32_t> & base, const std::vector<uint32_t> & var, int64_t p0, int64_t p1, int64_t p,
                              std::vector<uint32_t> & inout) {
    if (base.size() != var.size() || inout.size() != base.size() || p0 <= 0 || p1 <= 0 || p <= 0) return false;
    for (size_t i = 0; i < base.size(); ++i) {
        if (base[i] == var[i]) continue;
        // BD lengths/offsets in words/bytes: linear in the parameter with zero intercept
        if (int64_t(base[i]) * p1 != int64_t(var[i]) * p0) return false;
        if (inout[i] != base[i]) return false;            // a word must depend on ONE parameter only
        const int64_t v = int64_t(base[i]) * p;
        if (v % p0 != 0 || v / p0 > int64_t(UINT32_MAX)) return false;
        inout[i] = uint32_t(v / p0);
    }
    return true;
}

static bool read_words(const std::string & path, std::vector<uint32_t> & w) {
    std::string s = read_file(path);
    if (s.empty() || s.size() % 4) return false;
    w.resize(s.size() / 4);
    std::memcpy(w.data(), s.data(), s.size());
    return true;
}

bool KernelCache::insts_for(int64_t chain_words, int64_t a_words, int64_t g, std::vector<uint32_t> & out, int64_t stride_words) {
    const int64_t refThird = cfg_.product == Product::Prefill ? kRefO : kRefG;
    if (!cached()) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (refs_.empty()) {
        std::vector<std::vector<uint32_t>> r(inst_files(cfg_.product).size());
        for (size_t i = 0; i < r.size(); ++i) {
            if (!read_words(dir() + "/" + inst_files(cfg_.product)[i], r[i])) return false;
        }
        refs_ = std::move(r);
    }
    out = refs_[0];
    bool ok = patch_param(refs_[0], refs_[1], kRefChain, 2 * kRefChain, chain_words, out) &&
              patch_param(refs_[0], refs_[2], kRefA, 2 * kRefA, a_words, out) &&
              patch_param(refs_[0], refs_[3], refThird, 2 * refThird, g, out);
    if (ok && cfg_.product == Product::Decode) {
        // the offsets of the chains (i * stride) are linear in the stride parameter, the lengths in chain_words
        ok = patch_param(refs_[0], refs_[4], kRefChain, 2 * kRefChain, stride_words > 0 ? stride_words : chain_words, out);
    }
    return ok;
}

void KernelCache::request() {
    if (cached()) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (state_ == State::Building || state_ == State::Done) return;
    if (!toolchain_present()) {
        state_ = State::Failed;
        return;
    }
    state_ = State::Building;
    if (thread_.joinable()) thread_.join();
    thread_ = std::thread([this] { worker(); });
}

void KernelCache::worker() {
    double secs = 0;
    const bool ok = run_build(secs);
    {
        std::lock_guard<std::mutex> lk(mu_);
        state_ = ok ? State::Done : State::Failed;
        (ok ? stats_.built : stats_.failed) += 1;
        stats_.sum_seconds += secs;
        stats_.wall_seconds += secs;
    }
    cv_.notify_all();
}

bool KernelCache::run_build(double & seconds) {
    const double t0 = now_s();
    const std::string root = cfg_.cache_dir + "/" + hash_;
    mkdirs(root);
    int serial;
    {
        std::lock_guard<std::mutex> lk(mu_);
        serial = tmp_counter_++;
    }
    const std::string tmp = root + "/.tmp." + std::to_string(getpid()) + "." + std::to_string(serial) + "." + product_dir(cfg_.product);
    const std::string fin = dir();
    mkdirs(tmp);

    const std::string script = cfg_.kernel_src_dir + (cfg_.product == Product::Prefill ? "/build_prefill.sh" : "/build_kernel.sh");
    const char * argv[] = { "bash", script.c_str(), cfg_.toolchain_env.c_str(), tmp.c_str(), nullptr };

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    const std::string log = tmp + "/build.log";
    posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    pid_t pid = 0;
    say(0, cfg_.product == Product::Prefill ? "building the prefill kernel" : "building the unified decode kernel");
    int rc = posix_spawn(&pid, "/bin/bash", &fa, nullptr, const_cast<char * const *>(argv), environ);
    posix_spawn_file_actions_destroy(&fa);
    bool ok = false;
    if (rc == 0) {
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        ok = WIFEXITED(status) && WEXITSTATUS(status) == 0 && exists(tmp + "/kernel.xclbin");
        for (const char * f : inst_files(cfg_.product)) ok = ok && exists(tmp + "/" + f);
    }
    seconds = now_s() - t0;
    char tbuf[64];
    std::snprintf(tbuf, sizeof(tbuf), "%.1f s", seconds);

    if (ok) {
        if (rename(tmp.c_str(), fin.c_str()) != 0) {
            // someone else stored the same kernel first: keep theirs
            ok = exists(fin + "/kernel.xclbin");
            rmtree(tmp);
        }
        if (ok) say(0, std::string("built the ") + (cfg_.product == Product::Prefill ? "prefill" : "unified decode") + " kernel in " + tbuf);
    } else {
        const std::string failed = fin + ".failed.log";
        std::string txt = read_file(log);
        if (txt.size() > 4000) txt = txt.substr(txt.size() - 4000);
        std::ofstream(failed) << txt;
        rmtree(tmp);
        say(2, std::string("build of the ") + product_dir(cfg_.product) + " kernel FAILED after " + tbuf + " (see " + failed + ")");
    }
    return ok;
}

bool KernelCache::wait() {
    if (cached()) return true;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return state_ != State::Building; });
    lk.unlock();
    return cached();
}

int KernelCache::wait_all() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return state_ != State::Building; });
    return state_ == State::Failed ? 1 : 0;
}

KernelCache::Stats KernelCache::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
}

} // namespace xdna2

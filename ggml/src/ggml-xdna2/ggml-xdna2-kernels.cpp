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

const char * const kBuildFiles[] = { "build_kernel.sh", "env.sh", "qk.cc", "gen_qk.py" };

std::string compute_hash(const std::string & dir) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const std::string & s) {
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    };
    for (const char * f : kBuildFiles) {
        mix(f);
        mix(read_file(dir + "/" + f));
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return buf;
}

} // namespace

std::string KernelKey::name() const {
    return std::string(layout::fmt_info(fmt).name) + "_n" + std::to_string(n_pad) + "_k" + std::to_string(k) + "_m" + std::to_string(m);
}

bool KernelKey::operator<(const KernelKey & o) const {
    if (fmt != o.fmt) return int(fmt) < int(o.fmt);
    if (n_pad != o.n_pad) return n_pad < o.n_pad;
    if (k != o.k) return k < o.k;
    return m < o.m;
}

KernelCache::KernelCache(Config cfg) : cfg_(std::move(cfg)) {
    if (cfg_.max_jobs < 1) cfg_.max_jobs = 1;
    hash_ = compute_hash(cfg_.kernel_src_dir);
}

KernelCache::~KernelCache() {
    wait_all();
    for (auto & t : threads_) {
        if (t.joinable()) t.join();
    }
}

void KernelCache::say(int level, const std::string & msg) const {
    if (cfg_.log) cfg_.log(level, msg);
}

bool KernelCache::toolchain_present() const {
    if (cfg_.toolchain_env.empty()) return false;
    for (const char * f : kBuildFiles) {
        if (!exists(cfg_.kernel_src_dir + "/" + f)) return false;
    }
    return executable(cfg_.toolchain_env + "/bin/aiecc") && executable(cfg_.toolchain_env + "/bin/python") && executable("/bin/bash");
}

bool KernelCache::generator_supports(const KernelKey & key) {
    // gen_qk.py: K % 4096 == 0, RPC % 16 == 0 (N_pad is a multiple of 512 by construction); Q6_K kernel is M=1 only
    if (key.fmt != layout::Fmt::Q4_K && key.fmt != layout::Fmt::Q6_K) return false;
    if (key.k <= 0 || key.k % 4096 != 0) return false;
    if (key.n_pad <= 0 || key.n_pad % layout::ROW_GRAN != 0) return false;
    if (key.m < 1 || key.m > 8) return false;
    if (key.fmt == layout::Fmt::Q6_K && key.m != 1) return false;
    return true;
}

static std::string KernelKey_dir(const std::string & cache_dir, const std::string & hash, const KernelKey & key) {
    return cache_dir + "/" + hash + "/" + key.name();
}

std::string KernelCache::dir_of(const KernelKey & key) const { return KernelKey_dir(cfg_.cache_dir, hash_, key); }

bool KernelCache::cached(const KernelKey & key) const {
    const std::string d = dir_of(key);
    return exists(d + "/kernel.xclbin") && exists(d + "/insts.bin");
}

bool KernelCache::available(const KernelKey & key) const {
    return cached(key) || (generator_supports(key) && toolchain_present());
}

void KernelCache::request(const KernelKey & key) {
    if (cached(key)) return;
    std::lock_guard<std::mutex> lk(mu_);
    auto it = state_.find(key);
    if (it != state_.end() && it->second != State::Failed) return;   // queued/building/done
    if (!generator_supports(key) || !toolchain_present()) {
        state_[key] = State::Failed;
        return;
    }
    state_[key] = State::Queued;
    queue_.push_back(key);
    if (pending_ == 0) {
        batch_t0_ = now_s();
        batch_ = Stats();
    }
    ++pending_;
    if (active_workers_ < cfg_.max_jobs) {
        ++active_workers_;
        threads_.emplace_back([this] { worker(); });
    }
}

void KernelCache::worker() {
    for (;;) {
        KernelKey key;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (queue_.empty()) {
                --active_workers_;
                return;
            }
            key = queue_.front();
            queue_.erase(queue_.begin());
            state_[key] = State::Building;
        }
        double secs = 0;
        const bool ok = run_build(key, secs);
        {
            std::lock_guard<std::mutex> lk(mu_);
            state_[key] = ok ? State::Done : State::Failed;
            (ok ? stats_.built : stats_.failed) += 1;
            (ok ? batch_.built : batch_.failed) += 1;
            stats_.sum_seconds += secs;
            batch_.sum_seconds += secs;
            if (--pending_ == 0) {
                batch_.wall_seconds = now_s() - batch_t0_;
                stats_.wall_seconds += batch_.wall_seconds;
                char buf[256];
                std::snprintf(buf, sizeof(buf), "built %d kernel(s), %d failed, %.1f s wall (%.1f s of build work, up to %d parallel)",
                              batch_.built, batch_.failed, batch_.wall_seconds, batch_.sum_seconds, cfg_.max_jobs);
                say(batch_.failed ? 1 : 0, buf);
            }
        }
        cv_.notify_all();
    }
}

bool KernelCache::run_build(const KernelKey & key, double & seconds) {
    const double t0 = now_s();
    const std::string root = cfg_.cache_dir + "/" + hash_;
    mkdirs(root);
    int serial;
    {
        std::lock_guard<std::mutex> lk(mu_);
        serial = tmp_counter_++;
    }
    const std::string name = key.name();
    const std::string tmp = root + "/.tmp." + std::to_string(getpid()) + "." + std::to_string(serial) + "." + name;
    const std::string fin = root + "/" + name;
    mkdirs(tmp);

    const std::string script = cfg_.kernel_src_dir + "/build_kernel.sh";
    const std::string qt = key.fmt == layout::Fmt::Q4_K ? "4" : "6";
    const std::string n = std::to_string(key.n_pad), k = std::to_string(key.k), m = std::to_string(key.m);
    const char * argv[] = { "bash", script.c_str(), cfg_.toolchain_env.c_str(), tmp.c_str(), qt.c_str(), n.c_str(), k.c_str(), m.c_str(), nullptr };

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    const std::string log = tmp + "/build.log";
    posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    pid_t pid = 0;
    say(0, "building " + name);
    int rc = posix_spawn(&pid, "/bin/bash", &fa, nullptr, const_cast<char * const *>(argv), environ);
    posix_spawn_file_actions_destroy(&fa);
    bool ok = false;
    if (rc == 0) {
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        ok = WIFEXITED(status) && WEXITSTATUS(status) == 0 && exists(tmp + "/kernel.xclbin") && exists(tmp + "/insts.bin");
    }
    seconds = now_s() - t0;
    char tbuf[64];
    std::snprintf(tbuf, sizeof(tbuf), "%.1f s", seconds);

    if (ok) {
        if (rename(tmp.c_str(), fin.c_str()) != 0) {
            // someone else stored the same kernel first: keep theirs
            ok = exists(fin + "/kernel.xclbin") && exists(fin + "/insts.bin");
            rmtree(tmp);
        }
        if (ok) say(0, "built " + name + " in " + tbuf);
    } else {
        const std::string failed = fin + ".failed.log";
        std::string txt = read_file(log);
        if (txt.size() > 4000) txt = txt.substr(txt.size() - 4000);
        std::ofstream(failed) << txt;
        rmtree(tmp);
        say(2, "build of " + name + " FAILED after " + tbuf + " (see " + failed + ")");
    }
    return ok;
}

bool KernelCache::wait(const KernelKey & key) {
    if (cached(key)) return true;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] {
        auto it = state_.find(key);
        return it == state_.end() || it->second == State::Done || it->second == State::Failed;
    });
    lk.unlock();
    return cached(key);
}

int KernelCache::wait_all() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return pending_ == 0; });
    int failed = 0;
    for (auto & kv : state_) failed += kv.second == State::Failed;
    return failed;
}

KernelCache::Stats KernelCache::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
}

} // namespace xdna2

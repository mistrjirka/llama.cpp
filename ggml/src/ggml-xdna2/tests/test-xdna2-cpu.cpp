// Unit test + microbenchmark of the CPU decode GEMV that reads the NPU weight layout v2 directly (ggml-xdna2-cpu.cpp).
//
//   test-xdna2-cpu                  accuracy of all four formats vs. an f64 reference (ggml dequantization x f32 activations)
//   test-xdna2-cpu --bench [MB]     bandwidth on a large matrix set (default 1024 MB of weights, cycled)
//   test-xdna2-cpu --split          check that the superblock split (NPU prefix / CPU suffix) reproduces the full result

#include "ggml.h"
#include "ggml-xdna2-cpu.h"
#include "ggml-xdna2-bfp.h"
#include "ggml-xdna2-layout.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/mman.h>

extern "C" void quantize_row_q8_K_ref(const float * x, void * y, int64_t k);

using namespace xdna2;
using namespace xdna2::layout;

static ggml_type gt(Fmt f) {
    switch (f) { case Fmt::Q8_0: return GGML_TYPE_Q8_0; case Fmt::Q4_K: return GGML_TYPE_Q4_K; case Fmt::Q5_K: return GGML_TYPE_Q5_K; default: return GGML_TYPE_Q6_K; }
}

static void fp16_put(uint8_t * p, float v) { const ggml_fp16_t h = ggml_fp32_to_fp16(v); std::memcpy(p, &h, 2); }

// random GGUF blocks with sane scales
static std::vector<uint8_t> random_matrix(const Geom & g, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(g.gguf_bytes);
    for (auto & x : v) x = uint8_t(rng());
    std::uniform_real_distribution<float> sc(0.0005f, 0.02f);
    const size_t nblk = g.gguf_bytes / g.block_bytes;
    for (size_t b = 0; b < nblk; ++b) {
        uint8_t * p = v.data() + b * g.block_bytes;
        switch (g.fmt) {
            case Fmt::Q4_K: case Fmt::Q5_K: fp16_put(p, sc(rng)); fp16_put(p + 2, sc(rng)); break;
            case Fmt::Q6_K: fp16_put(p + 208, sc(rng)); break;
            default: fp16_put(p, sc(rng)); break;
        }
    }
    return v;
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static int g_fail = 0;

// sb_from > 0: the NPU would do the first sb_from superblocks (rows must stay untouched); kb_npu >= 0 (single-superblock matrices): the NPU would do the
// first kb_npu records of every row, the CPU result is the K-suffix partial sum
static void test_shape(Fmt fmt, int64_t n, int64_t k, uint32_t seed, int64_t sb_from = 0, int64_t kb_npu = -1) {
    const Geom g = make_geom(fmt, n, k);
    const std::vector<uint8_t> src = random_matrix(g, seed);
    std::vector<uint8_t> dev(g.bytes, 0);
    pack(g, dev.data(), src.data());
    std::mt19937 rng(seed + 1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(k);
    for (auto & a : x) a = nd(rng);
    // reference: ggml dequantization, f64 accumulation
    const auto * tr = ggml_get_type_traits(gt(fmt));
    std::vector<float> row(k);
    std::vector<double> ref(n);
    for (int64_t i = 0; i < n; ++i) {
        tr->to_float(src.data() + size_t(i) * g.row_bytes, row.data(), k);
        double s = 0;
        for (int64_t j = (kb_npu >= 0 ? kb_npu * 256 : 0); j < k; ++j) s += double(row[j]) * double(x[j]);
        ref[i] = s;
    }
    // reference 2: the same math with ggml's own activation quantization (exact in f64): isolates the GEMV from the quantization noise
    std::vector<double> ref2(n);
    {
        std::vector<float> xd(k);
        if (fmt == Fmt::Q8_0) {
            const auto * atr = ggml_get_type_traits(GGML_TYPE_Q8_0);
            std::vector<uint8_t> qa(ggml_row_size(GGML_TYPE_Q8_0, k));
            atr->from_float_ref(x.data(), qa.data(), k);
            atr->to_float(qa.data(), xd.data(), k);
        } else {
            std::vector<uint8_t> qa(size_t(k / 256) * 292);
            quantize_row_q8_K_ref(x.data(), qa.data(), k);
            for (int64_t b = 0; b < k / 256; ++b) {
                float d; std::memcpy(&d, qa.data() + b * 292, 4);
                for (int j = 0; j < 256; ++j) xd[b * 256 + j] = d * float(int8_t(qa[b * 292 + 4 + j]));
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            tr->to_float(src.data() + size_t(i) * g.row_bytes, row.data(), k);
            double s = 0;
            for (int64_t j = (kb_npu >= 0 ? kb_npu * 256 : 0); j < k; ++j) s += double(row[j]) * double(xd[j]);
            ref2[i] = s;
        }
    }
    std::vector<float> y(n, 12345.0f);
    CpuDecodeJob job;
    std::vector<SplitPlan> plan(size_t(g.nsplit), SplitPlan{ sb_from, kb_npu >= 0 ? kb_npu : g.nbk });
    job.geom = g; job.w = dev.data(); job.x = x.data(); job.dst = y.data(); job.plan = (sb_from > 0 || kb_npu >= 0) ? plan.data() : nullptr;
    cpu_decode_run(&job, 1);
    double num = 0, den = 0, maxe = 0, num2 = 0;
    int64_t nrows_checked = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (kb_npu >= g.nbk) { if (y[i] != 12345.0f) { std::printf("FAIL row %lld touched\n", (long long) i); ++g_fail; } continue; }
        if (sb_from > 0 && kb_npu < 0) {
            // rows of the NPU prefix must be untouched
            const int64_t Gs = g.sp_groups(i / (GMAX * ROW_GRAN));
            const int64_t rr = i - g.sp_row0(i / (GMAX * ROW_GRAN));
            const int64_t grp = (rr % (Gs * GROUP_ROWS)) / GROUP_ROWS;
            if (grp / GSB < sb_from) { if (y[i] != 12345.0f) { std::printf("FAIL prefix row %lld touched\n", (long long) i); ++g_fail; } continue; }
        }
        const double e = double(y[i]) - ref[i];
        num2 += (double(y[i]) - ref2[i]) * (double(y[i]) - ref2[i]);
        num += e * e; den += ref[i] * ref[i]; maxe = std::max(maxe, std::fabs(e)); ++nrows_checked;
    }
    const double nmse = den > 0 ? num / den : num;
    const double nmse2 = den > 0 ? num2 / den : num2;
    const bool ok = (nrows_checked == 0 && kb_npu >= g.nbk) || (nmse2 < 1e-9 && nmse < 5e-4 && nrows_checked > 0);
    std::printf("%-5s N=%-6lld K=%-5lld sb_from=%lld  rows %lld  NMSE vs f32 %.3e  vs same-quant-acts %.3e  max|e| %.3e  %s\n", fmt_info(fmt).name, (long long) n, (long long) k,
                (long long) sb_from, (long long) nrows_checked, nmse, nmse2, maxe, ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
}

static void bench(size_t mb, int threads_note) {
    (void) threads_note;
    struct Case { Fmt fmt; int64_t n, k; };
    const Case cases[] = { { Fmt::Q4_K, 9216, 2560 }, { Fmt::Q4_K, 12288, 4096 }, { Fmt::Q5_K, 2560, 4096 }, { Fmt::Q6_K, 2560, 9216 }, { Fmt::Q8_0, 2560, 4096 },
                           { Fmt::Q6_K, 8192, 4096 } };
    size_t ncases = std::getenv("XD_CASES") ? size_t(std::atoi(std::getenv("XD_CASES"))) : sizeof(cases) / sizeof(cases[0]);
    for (const Case & c : cases) {
        if (ncases-- == 0) break;
        const Geom g = make_geom(c.fmt, c.n, c.k);
        const size_t copies = std::max<size_t>(2, mb * 1000000 / g.bytes);
        std::vector<uint8_t> blk = random_matrix(g, 7);
        std::vector<uint8_t> one(g.bytes, 0);
        pack(g, one.data(), blk.data());
        uint8_t * devp = static_cast<uint8_t *>(mmap(nullptr, copies * g.bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (std::getenv("XD_NOTHP")) madvise(devp, copies * g.bytes, MADV_NOHUGEPAGE);
        struct { uint8_t * p; uint8_t * data() { return p; } } dev{ devp };
        for (size_t i = 0; i < copies; ++i) std::memcpy(dev.data() + i * g.bytes, one.data(), g.bytes);
        std::vector<float> x(c.k), y(c.n);
        std::mt19937 rng(3);
        for (auto & a : x) a = float(rng() % 1000) / 500.0f - 1.0f;
        std::vector<CpuDecodeJob> jobs(copies);
        for (size_t i = 0; i < copies; ++i) { jobs[i].geom = g; jobs[i].w = dev.data() + i * g.bytes; jobs[i].x = x.data(); jobs[i].dst = y.data(); }
        cpu_decode_run(jobs.data(), 1);   // warm up
        double best = 1e9;
        for (int rep = 0; rep < 3; ++rep) {
            const double t0 = now();
            for (size_t i = 0; i < copies; ++i) cpu_decode_run(&jobs[i], 1);
            best = std::min(best, now() - t0);
        }
        double bestb = 1e9;
        const size_t B = std::getenv("XD_BATCH") ? size_t(std::atoi(std::getenv("XD_BATCH"))) : 1;
        if (B > 1) {
            for (int rep = 0; rep < 3; ++rep) {
                const double t0 = now();
                for (size_t i = 0; i + B <= copies; i += B) cpu_decode_run(&jobs[i], B);
                bestb = std::min(bestb, now() - t0);
            }
            best = bestb * double(copies) / double(copies / B * B);
        }
        std::printf("bench %-5s %6lld x %-5lld  %zu copies  %.1f MB/matrix  %.3f ms/matrix  %.1f GB/s (threads %d)\n", fmt_info(c.fmt).name, (long long) c.n, (long long) c.k,
                    copies, g.bytes / 1e6, best / copies * 1e3, double(g.bytes) * copies / best / 1e9, pf_threads());
    }
}

int main(int argc, char ** argv) {
    if (!cpu_decode_available()) { std::printf("AVX-512 VNNI/VBMI not available: skipped\n"); return 0; }
    if (argc > 1 && std::string(argv[1]) == "--bench") { bench(argc > 2 ? size_t(std::atoi(argv[2])) : 1024, 0); return 0; }
    const Fmt fmts[] = { Fmt::Q4_K, Fmt::Q5_K, Fmt::Q6_K, Fmt::Q8_0 };
    uint32_t seed = 1;
    for (Fmt f : fmts) {
        test_shape(f, 512, 256, seed++);
        test_shape(f, 1024, 2560, seed++);
        test_shape(f, 2560, 4096, seed++);
        test_shape(f, 1000, 1024, seed++);        // N not a multiple of 512 (padding rows)
        test_shape(f, 9216, 2560, seed++);
        test_shape(f, 70000, 512, seed++);        // several splits (GMAX groups each)
        test_shape(f, 2560, 12288, seed++);
        test_shape(f, 9216, 2560, seed++, 1);
        test_shape(f, 9216, 2560, seed++, 3);
        test_shape(f, 70000, 512, seed++, 2);
        test_shape(f, 2560, 4096, seed++, 1, 6);       // K split of a one-superblock matrix: CPU = K suffix of all rows
        test_shape(f, 1024, 2560, seed++, 1, 9);
        test_shape(f, 2560, 4096, seed++, 1, 16);      // NPU does everything: nothing left for the CPU (rows untouched)
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL OK\n", g_fail);
    return g_fail ? 1 : 0;
}

// Unit test of the device weight layouts (ggml-xdna2-layout.h) for Q8_0, Q4_K, Q5_K, Q6_K:
// round trip, chunked/unaligned writes, padding. Bit equality with the lab packer (stage3 pack3.py) is checked
// with --compare (see tests/check-layout-vs-lab.sh).
//
//   test-xdna2-layout                         run all tests
//   test-xdna2-layout --compare FMT N K blk.bin expected.bin
//        pack the GGUF-format blocks in blk.bin and compare with expected.bin (W_bo as written by pack3.py)

#include "ggml-xdna2-layout.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace xdna2::layout;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static std::vector<uint8_t> random_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(n);
    for (auto & x : v) x = uint8_t(rng());
    return v;
}

// bytes of the scale/d field of a GGUF block (must be zero for padding)
static void scale_field(Fmt f, size_t & off, size_t & len) {
    switch (f) {
        case Fmt::Q4_K: case Fmt::Q5_K: off = 0; len = 4; break;
        case Fmt::Q6_K: off = 208; len = 2; break;
        default: off = 0; len = 2; break;
    }
}

static void test_shape(Fmt fmt, int64_t n, int64_t k, uint32_t seed) {
    const FmtInfo & fi = fmt_info(fmt);
    const Geom g = make_geom(fmt, n, k);
    CHECK(g.bytes == size_t(g.n_pad) * size_t(k / 256) * fi.rec_bytes / 16);
    CHECK(g.gguf_bytes == size_t(n) * size_t(k / fi.elems) * fi.block_bytes);
    CHECK(g.n_pad % 512 == 0 && g.bytes >= g.gguf_bytes);
    if (n % 512 == 0) CHECK(g.bytes == g.gguf_bytes);                       // weight records are exactly GGUF sized
    CHECK(fi.rec_bytes == 16 * size_t(fi.rec_blocks) * fi.block_bytes);

    const std::vector<uint8_t> src = random_bytes(g.gguf_bytes, seed);

    // 1. round trip, independent of the initial device content
    for (uint8_t fill : {uint8_t(0), uint8_t(0xAA)}) {
        std::vector<uint8_t> dev(g.bytes, fill), back(g.gguf_bytes, 0x5C);
        pack(g, dev.data(), src.data());
        unpack(g, dev.data(), back.data());
        CHECK(back == src);
    }

    std::vector<uint8_t> dev(g.bytes, 0);
    pack(g, dev.data(), src.data());

    // 2. padded rows have a zero scale
    size_t bad = 0;
    size_t so, sl;
    scale_field(fmt, so, sl);
    for (int64_t r = g.n; r < g.n_pad; ++r) {
        for (int64_t b = 0; b < g.kb; ++b) {
            uint8_t blk[MAX_BLOCK_BYTES];
            const uint8_t * rec = dev.data() + g.group_base(r) + size_t(b / g.rec_blocks) * g.rec_stride;
            detail::unpack_block(fmt, rec, int(b % g.rec_blocks), int(r % 16), blk);
            for (size_t i = 0; i < sl; ++i) bad += blk[so + i] != 0;
        }
    }
    CHECK(bad == 0);

    // 3. chunked, arbitrarily aligned writes give the same device bytes; random partial reads match
    std::mt19937 rng(seed + 1);
    std::vector<uint8_t> dev2(g.bytes, 0);
    for (size_t off = 0; off < g.gguf_bytes;) {
        const size_t len = std::min<size_t>(g.gguf_bytes - off, 1 + rng() % 7001);
        pack_range(g, dev2.data(), src.data() + off, off, len);
        off += len;
    }
    CHECK(dev2 == dev);
    for (int i = 0; i < 20; ++i) {
        const size_t off = rng() % g.gguf_bytes;
        const size_t len = 1 + rng() % (g.gguf_bytes - off);
        std::vector<uint8_t> part(len);
        unpack_range(g, dev.data(), part.data(), off, len);
        CHECK(std::equal(part.begin(), part.end(), src.begin() + off));
    }
}

static void test_slices(Fmt fmt, int64_t k) {
    // tensor with 3 slices, written in odd chunks across slice boundaries
    const Geom g = make_geom(fmt, 700, k);
    const size_t slices = 3;
    const std::vector<uint8_t> src = random_bytes(g.gguf_bytes * slices, 7);
    std::vector<uint8_t> dev(g.bytes * slices, 0), back(src.size());
    std::mt19937 rng(9);
    for (size_t off = 0; off < src.size();) {
        const size_t len = std::min<size_t>(src.size() - off, 1 + rng() % 100000);
        tensor_pack_range(g, dev.data(), src.data() + off, off, len);
        off += len;
    }
    tensor_unpack_range(g, dev.data(), back.data(), 0, back.size());
    CHECK(back == src);
}

static int compare(const std::string & fname, int64_t n, int64_t k, const char * blk, const char * exp) {
    Fmt fmt = Fmt::Q4_K;
    bool found = false;
    for (int i = 0; i < int(Fmt::COUNT); ++i) if (fname == fmt_info(Fmt(i)).name) { fmt = Fmt(i); found = true; }
    if (!found) { std::printf("unknown format %s\n", fname.c_str()); return 2; }
    auto rd = [](const char * p) {
        std::vector<uint8_t> v;
        FILE * f = std::fopen(p, "rb");
        if (!f) {
            return v;
        }
        std::fseek(f, 0, SEEK_END);
        v.resize(size_t(std::ftell(f)));
        std::fseek(f, 0, SEEK_SET);
        if (std::fread(v.data(), 1, v.size(), f) != v.size()) {
            v.clear();
        }
        std::fclose(f);
        return v;
    };
    const Geom g = make_geom(fmt, n, k);
    auto src = rd(blk), want = rd(exp);
    if (src.size() != g.gguf_bytes || want.size() != g.bytes) { std::printf("size mismatch %zu/%zu %zu/%zu\n", src.size(), g.gguf_bytes, want.size(), g.bytes); return 2; }
    std::vector<uint8_t> dev(g.bytes, 0);
    pack(g, dev.data(), src.data());
    const bool ok = dev == want;
    std::printf("compare %s vs lab packer N=%ld K=%ld: %s\n", fname.c_str(), long(n), long(k), ok ? "IDENTICAL" : "MISMATCH");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc == 7 && std::string(argv[1]) == "--compare") {
        return compare(argv[2], std::atoll(argv[3]), std::atoll(argv[4]), argv[5], argv[6]);
    }
    const int64_t Ns[] = {1, 16, 17, 100, 512, 513, 1000, 1537, 2048};
    const int64_t Ks[] = {256, 512, 768, 1536, 4096};
    int cases = 0;
    uint32_t seed = 1;
    for (int f = 0; f < int(Fmt::COUNT); ++f) {
        const Fmt fmt = Fmt(f);
        for (int64_t n : Ns) {
            for (int64_t k : Ks) {
                if (n * k > 1537 * 1536 && k > 1536) continue; // keep the runtime short
                test_shape(fmt, n, k, seed++);
                ++cases;
            }
        }
        test_slices(fmt, 512);
        std::printf("  %s: ok so far (%d failures)\n", fmt_info(fmt).name, g_fail);
    }
    // Ornith output.weight (Q6_K, K=4096, N=248320): one dispatch, no padding
    {
        const Geom g = make_geom(Fmt::Q6_K, 248320, 4096);
        CHECK(g.bytes == g.gguf_bytes && g.bytes == size_t(248320) * 16 * 210);
    }
    std::printf("test-xdna2-layout: %d shapes + slices, %s\n", cases, g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}

// Unit test of the device weight layouts (ggml-xdna2-layout.h) for Q8_0, Q4_K, Q5_K, Q6_K:
// round trip, chunked/unaligned writes, padding, record straddling over chunk boundaries, multi-split matrices.
// Bit equality with the lab packer (stage5 packu.py) is checked with --compare (see tests/check-layout-vs-lab.py).
//
//   test-xdna2-layout                         run all tests
//   test-xdna2-layout --compare FMT N K blk.bin expected.bin
//        pack the GGUF-format blocks in blk.bin and compare with expected.bin (W_bo as written by packu.py)

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
    const size_t payload = size_t(g.n_pad) * size_t(k / 256) * fi.rec_bytes / 16;       // N_pad rows of records, no chunk padding
    CHECK(g.gguf_bytes == size_t(n) * size_t(k / fi.elems) * fi.block_bytes);
    CHECK(g.n_pad % 512 == 0 && g.bytes >= g.gguf_bytes && g.bytes >= payload);
    {   // only the tail of each slab (sb, kb) is padded to a whole chunk
        size_t padmax = 0;
        for (int64_t sp = 0; sp < g.nsplit; ++sp) padmax += size_t(N_CORES) * size_t(g.nbk) * size_t(g.sp_nsb(sp)) * CHUNK;
        CHECK(g.bytes - payload <= padmax);
    }
    {
        size_t sum = 0;
        for (int64_t sp = 0; sp < g.nsplit; ++sp) { CHECK(g.sp_off(sp) == sum); sum += g.sp_bytes(sp); CHECK(g.sp_bytes(sp) % (N_CHAINS * N_LANES * CHUNK) == 0); }
        CHECK(sum == g.bytes);
    }
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
            const RecPtr rec = locate(g, dev.data(), r, b / g.rec_blocks);
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

// every record of a core stream is reachable; records that straddle chunk boundaries exist and are contiguous after gather;
// records of a slab are back to back, slabs are chunk aligned
static void test_straddle(Fmt fmt, int64_t n, int64_t k) {
    const Geom g = make_geom(fmt, n, k);
    std::vector<uint8_t> dev(g.bytes, 0);
    size_t straddle = 0, total = 0;
    for (int64_t row = 0; row < g.n_pad; row += GROUP_ROWS) {
        for (int64_t r = 0; r < g.nbk; ++r) {
            const RecPtr p = locate(g, dev.data(), row, r);
            ++total;
            if (p.straddles(g.rec_bytes)) {
                ++straddle;
                const size_t room = CHUNK - (p.s0 % CHUNK);
                CHECK(p(room) == p(room - 1) + 1 + (N_LANES - 1) * CHUNK);      // continuation = next chunk of the same lane
            }
            CHECK(p(0) >= dev.data() && p(g.rec_bytes - 1) < dev.data() + g.bytes);
            const int64_t grp = (row % (GMAX * ROW_GRAN)) % (g.sp_groups(row / (GMAX * ROW_GRAN)) * 16 * 1) / 16;   // group inside the split stream of its core
            (void) grp;
        }
    }
    CHECK(straddle > 0 || g.rec_bytes * size_t(g.nbk) * size_t(g.groups) < CHUNK);
    // slab structure: records of one slab consecutive, slab start chunk aligned
    {
        const int64_t sp = 0, Gs = g.sp_groups(0);
        for (int64_t sb = 0; sb < g.sp_nsb(sp); ++sb) {
            for (int64_t kb = 0; kb < g.nbk; ++kb) {
                const RecPtr first = locate(g, dev.data(), (sb * g.gsb) * GROUP_ROWS, kb);
                CHECK(first.s0 % CHUNK == 0);
                for (int64_t gl = 1; gl < g.sb_groups(sp, sb); ++gl) {
                    const RecPtr q = locate(g, dev.data(), (sb * g.gsb + gl) * GROUP_ROWS, kb);
                    CHECK(q.s0 == first.s0 + size_t(gl) * g.rec_bytes);
                }
            }
        }
        (void) Gs;
    }
    std::printf("  straddle %s N=%ld K=%ld: %zu of %zu records straddle a chunk\n", fmt_info(fmt).name, long(n), long(k), straddle, total);
}

// a matrix with several splits: round trip, and the split regions are disjoint and in order
static void test_splits(Fmt fmt, int64_t n, int64_t k) {
    const Geom g = make_geom(fmt, n, k);
    CHECK(g.nsplit >= 2);
    const std::vector<uint8_t> src = random_bytes(g.gguf_bytes, 77);
    std::vector<uint8_t> dev(g.bytes, 0), back(g.gguf_bytes, 0x5C);
    pack(g, dev.data(), src.data());
    unpack(g, dev.data(), back.data());
    CHECK(back == src);
    // a row of the last split must live in the last split's region
    const RecPtr p = locate(g, dev.data(), g.n_pad - 1, g.nbk - 1);
    CHECK(p(0) >= dev.data() + g.sp_off(g.nsplit - 1));
    std::printf("  splits %s N=%ld K=%ld: %ld splits, %.2f MB, round trip %s\n", fmt_info(fmt).name, long(n), long(k), long(g.nsplit), g.bytes / 1e6, back == src ? "ok" : "FAILED");
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

// --pack FMT N K blk.bin out.bin: write the device layout of the GGUF blocks (lab tooling)
static int pack_file(const std::string & fname, int64_t n, int64_t k, const char * blk, const char * out) {
    Fmt fmt = Fmt::Q4_K;
    for (int i = 0; i < int(Fmt::COUNT); ++i) if (fname == fmt_info(Fmt(i)).name) fmt = Fmt(i);
    const Geom g = make_geom(fmt, n, k);
    std::vector<uint8_t> src(g.gguf_bytes), dev(g.bytes, 0);
    FILE * f = std::fopen(blk, "rb");
    if (!f || std::fread(src.data(), 1, src.size(), f) != src.size()) { std::printf("cannot read %zu bytes from %s\n", src.size(), blk); return 2; }
    std::fclose(f);
    pack(g, dev.data(), src.data());
    f = std::fopen(out, "wb");
    std::fwrite(dev.data(), 1, dev.size(), f);
    std::fclose(f);
    std::printf("packed %s N=%ld K=%ld: %zu -> %zu bytes, nsplit=%ld Gs=%ld sp_nw=%zu chain_words=%zu nsb=%ld\n", fname.c_str(), long(n), long(k), src.size(), dev.size(),
                long(g.nsplit), long(g.sp_groups(0)), g.sp_nw(0), g.sp_chain_words(0), long(g.sp_nsb(0)));
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 7 && std::string(argv[1]) == "--pack") {
        return pack_file(argv[2], std::atoll(argv[3]), std::atoll(argv[4]), argv[5], argv[6]);
    }
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
    for (int f = 0; f < int(Fmt::COUNT); ++f) {
        test_straddle(Fmt(f), 1024, 2560);
        test_straddle(Fmt(f), 512, 256);
    }
    test_splits(Fmt::Q4_K, 70000, 256);
    test_splits(Fmt::Q6_K, 66000, 256);
    test_splits(Fmt::Q8_0, 65536 + 512, 256);
    // Ornith output.weight (Q6_K, K=4096, N=248320): 4 splits (GMAX=128 groups = 65536 rows each), < 5 % slab padding
    {
        const Geom g = make_geom(Fmt::Q6_K, 248320, 4096);
        CHECK(g.nsplit == 4 && g.sp_groups(3) == g.groups - 3 * GMAX);
        CHECK(g.bytes >= size_t(248320) * 16 * 210 && g.bytes < size_t(248320) * 16 * 210 * 1050 / 1000);
    }
    std::printf("test-xdna2-layout: %d shapes + slices, %s\n", cases, g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}

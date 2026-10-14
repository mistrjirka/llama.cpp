// Unit test of the Q4_0 device layout (ggml-xdna2-layout.h): round trip, chunked writes, padding,
// and an independent reference packer written in the style of the stage0 prep_q4.py.
//
//   test-xdna2-layout                         run all tests
//   test-xdna2-layout --compare N K blk.bin expected.bin
//        pack the GGUF-format blocks in blk.bin and compare with expected.bin (all 32 core streams,
//        produced by a python script that uses prep_q4.py's packing code)

#include "ggml-xdna2-layout.h"

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <random>
#include <vector>

using namespace xdna2::layout;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// Independent packer: follows prep_q4.py literally (explicit [g, kb, k, n] transpose, nibble pairing, records).
static std::vector<uint8_t> reference_pack(const Q4Geom & g, const std::vector<uint8_t> & gguf) {
    std::vector<uint8_t> out(g.bytes, 0);
    const int64_t KB = g.kb_pad;
    // logical arrays, padded: nib[row][kb][k] (already q^8), scale bytes sc[row][kb][2]
    std::vector<uint8_t> nib(size_t(g.n_pad) * KB * 32, 0), sc(size_t(g.n_pad) * KB * 2, 0);
    for (int64_t r = 0; r < g.n; ++r) {
        for (int64_t b = 0; b < g.kb; ++b) {
            const uint8_t * blk = gguf.data() + (size_t(r) * g.kb + b) * 18;
            sc[(size_t(r) * KB + b) * 2]     = blk[0];
            sc[(size_t(r) * KB + b) * 2 + 1] = blk[1];
            for (int j = 0; j < 16; ++j) {
                nib[(size_t(r) * KB + b) * 32 + j]      = (blk[2 + j] & 15) ^ 8;
                nib[(size_t(r) * KB + b) * 32 + j + 16] = (blk[2 + j] >> 4) ^ 8;
            }
        }
    }
    for (int64_t core = 0; core < 32; ++core) {
        for (int64_t gr = 0; gr < g.groups_per_core; ++gr) {
            const int64_t r0 = core * g.rpc + gr * 16;
            for (int64_t rec = 0; rec < g.recs_per_group; ++rec) {
                uint8_t * R = out.data() + size_t(core) * g.core_bytes + size_t(gr * g.recs_per_group + rec) * 2304;
                // nibble stream: per block b: t[k][n] flattened as k*16+n (k = 0..31), 512 nibbles -> 256 B
                for (int b = 0; b < 8; ++b) {
                    const int64_t kb = rec * 8 + b;
                    std::vector<uint8_t> t(512);
                    for (int k = 0; k < 32; ++k)
                        for (int n = 0; n < 16; ++n)
                            t[size_t(k) * 16 + n] = nib[(size_t(r0 + n) * KB + kb) * 32 + k];
                    for (int i = 0; i < 256; ++i) R[b * 256 + i] = t[2 * i] | (t[2 * i + 1] << 4);
                    for (int n = 0; n < 16; ++n) {
                        R[2048 + (b * 16 + n) * 2]     = sc[(size_t(r0 + n) * KB + kb) * 2];
                        R[2048 + (b * 16 + n) * 2 + 1] = sc[(size_t(r0 + n) * KB + kb) * 2 + 1];
                    }
                }
            }
        }
    }
    return out;
}

static std::vector<uint8_t> random_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(n);
    for (auto & x : v) x = uint8_t(rng());
    return v;
}

static void test_shape(int64_t n, int64_t k, uint32_t seed) {
    const Q4Geom g = q4_geom(n, k);
    CHECK(g.bytes == size_t(g.n_pad) * size_t(g.kb_pad) * 18);
    CHECK(g.gguf_bytes == size_t(n) * size_t(k / 32) * 18);
    CHECK(g.n_pad % 512 == 0 && g.kb_pad % 8 == 0 && g.bytes >= g.gguf_bytes);

    const std::vector<uint8_t> src = random_bytes(g.gguf_bytes, seed);

    // 1. round trip, independent of the initial device content
    for (uint8_t fill : {uint8_t(0), uint8_t(0xAA)}) {
        std::vector<uint8_t> dev(g.bytes, fill), back(g.gguf_bytes, 0x5C);
        q4_0_pack(g, dev.data(), src.data());
        q4_0_unpack(g, dev.data(), back.data());
        CHECK(back == src);
    }

    // 2. same as the independent reference packer
    std::vector<uint8_t> dev(g.bytes, 0);
    q4_0_pack(g, dev.data(), src.data());
    CHECK(dev == reference_pack(g, src));

    // 3. padding is untouched (zero scale and zero nibbles)
    size_t nonzero_padding = 0;
    for (int64_t r = 0; r < g.n_pad; ++r) {
        for (int64_t b = 0; b < g.kb_pad; ++b) {
            if (r < g.n && b < g.kb) continue;
            const uint8_t * rec = dev.data() + g.rec_offset(r, b);
            const int bb = int(b % 8), nn = int(r % 16);
            nonzero_padding += rec[detail::sc_off(bb, nn)] != 0 || rec[detail::sc_off(bb, nn) + 1] != 0;
            for (int e = 0; e < 32; ++e)
                nonzero_padding += ((rec[detail::nib_off(bb, nn, e)] >> ((nn & 1) * 4)) & 15) != 0;
        }
    }
    CHECK(nonzero_padding == 0);

    // 4. chunked, arbitrarily aligned writes give the same device bytes; random partial reads match
    std::mt19937 rng(seed + 1);
    std::vector<uint8_t> dev2(g.bytes, 0);
    for (size_t off = 0; off < g.gguf_bytes;) {
        const size_t len = std::min<size_t>(g.gguf_bytes - off, 1 + rng() % 7001);
        q4_0_pack_range(g, dev2.data(), src.data() + off, off, len);
        off += len;
    }
    CHECK(dev2 == dev);
    for (int i = 0; i < 20; ++i) {
        const size_t off = rng() % g.gguf_bytes;
        const size_t len = 1 + rng() % (g.gguf_bytes - off);
        std::vector<uint8_t> part(len);
        q4_0_unpack_range(g, dev.data(), part.data(), off, len);
        CHECK(std::equal(part.begin(), part.end(), src.begin() + off));
    }
}

static void test_slices() {
    // tensor with 3 slices, written in odd chunks across slice boundaries
    const Q4Geom g = q4_geom(700, 288);
    const size_t slices = 3;
    const std::vector<uint8_t> src = random_bytes(g.gguf_bytes * slices, 7);
    std::vector<uint8_t> dev(g.bytes * slices, 0), back(src.size());
    std::mt19937 rng(9);
    for (size_t off = 0; off < src.size();) {
        const size_t len = std::min<size_t>(src.size() - off, 1 + rng() % 100000);
        q4_0_tensor_pack_range(g, dev.data(), src.data() + off, off, len);
        off += len;
    }
    q4_0_tensor_unpack_range(g, dev.data(), back.data(), 0, back.size());
    CHECK(back == src);
}

static int compare(int64_t n, int64_t k, const char * blk, const char * exp) {
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
    const Q4Geom g = q4_geom(n, k);
    auto src = rd(blk), want = rd(exp);
    if (src.size() != g.gguf_bytes || want.size() != g.bytes) { std::printf("size mismatch %zu/%zu %zu/%zu\n", src.size(), g.gguf_bytes, want.size(), g.bytes); return 2; }
    std::vector<uint8_t> dev(g.bytes, 0);
    q4_0_pack(g, dev.data(), src.data());
    const bool ok = dev == want;
    std::printf("compare vs prep_q4.py N=%ld K=%ld: %s\n", long(n), long(k), ok ? "IDENTICAL" : "MISMATCH");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc == 6 && std::string(argv[1]) == "--compare") {
        return compare(std::atoll(argv[2]), std::atoll(argv[3]), argv[4], argv[5]);
    }
    const int64_t Ns[] = {1, 16, 17, 100, 512, 513, 1000, 1537, 2048};
    const int64_t Ks[] = {32, 64, 256, 288, 800, 1536, 4096};
    uint32_t seed = 1;
    int cases = 0;
    for (int64_t n : Ns) {
        for (int64_t k : Ks) {
            if (n * k > 1537 * 1536 && k > 1536) continue; // keep the runtime short
            test_shape(n, k, seed++);
            ++cases;
        }
    }
    test_slices();
    std::printf("test-xdna2-layout: %d shapes + slices, %s\n", cases, g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}

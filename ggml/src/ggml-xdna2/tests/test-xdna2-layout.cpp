// Unit test of the device weight layouts (ggml-xdna2-layout.h) for Q4_0, Q8_0, Q4_K, Q5_K, Q6_K:
// round trip, chunked/unaligned writes, dispatch chunks, padding, and (Q4_0) an independent reference packer
// written in the style of the stage0 prep_q4.py.
//
//   test-xdna2-layout                         run all tests
//   test-xdna2-layout --compare FMT N K blk.bin expected.bin
//        pack the GGUF-format blocks in blk.bin and compare with expected.bin (the 32 core streams of the single
//        chunk, produced by a python script that uses the lab's prep_q4.py / pack_qk.py packing code)

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

// Independent Q4_0 packer: follows prep_q4.py literally (explicit [g, kb, k, n] transpose, nibble pairing, records).
static std::vector<uint8_t> reference_pack_q4_0(const Geom & g, const std::vector<uint8_t> & gguf) {
    std::vector<uint8_t> out(g.bytes, 0);
    const int64_t KB = g.kb_pad;
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
    for (int ci = 0; ci < g.nchunks; ++ci) {
        const Chunk & ch = g.chunks[ci];
        for (int64_t core = 0; core < 32; ++core) {
            for (int64_t gr = 0; gr < ch.groups_per_core; ++gr) {
                const int64_t r0 = ch.row0 + core * ch.rpc + gr * 16;
                for (int64_t rec = 0; rec < g.recs_per_group; ++rec) {
                    uint8_t * R = out.data() + ch.off + size_t(core) * ch.core_bytes + size_t(gr * g.recs_per_group + rec) * 2304;
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
    }
    return out;
}

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

static void test_shape(Fmt fmt, int64_t n, int64_t k, uint32_t seed, size_t max_dispatch = MAX_DISPATCH_BYTES) {
    const FmtInfo & fi = fmt_info(fmt);
    const Geom g = make_geom(fmt, n, k, max_dispatch);
    CHECK(g.bytes == size_t(g.n_pad) * size_t(g.kb_pad) * fi.block_bytes);
    CHECK(g.gguf_bytes == size_t(n) * size_t(k / fi.elems) * fi.block_bytes);   // weight records are exactly GGUF sized
    CHECK(g.n_pad % 512 == 0 && g.kb_pad % fi.rec_blocks == 0 && g.bytes >= g.gguf_bytes);
    CHECK(fi.rec_bytes == 16 * size_t(fi.rec_blocks) * fi.block_bytes);
    size_t csum = 0; int64_t rsum = 0;
    for (int i = 0; i < g.nchunks; ++i) {
        CHECK(g.chunks[i].rows % 512 == 0 && g.chunks[i].off == csum && g.chunks[i].row0 == rsum);
        csum += g.chunks[i].bytes; rsum += g.chunks[i].rows;
        if (max_dispatch == MAX_DISPATCH_BYTES) CHECK(g.chunks[i].bytes <= max_dispatch);
    }
    CHECK(csum == g.bytes && rsum == g.n_pad);

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

    // 2. Q4_0 only: same as the independent reference packer
    if (fmt == Fmt::Q4_0) CHECK(dev == reference_pack_q4_0(g, src));

    // 3. padded rows / blocks have a zero scale
    size_t bad = 0;
    size_t so, sl;
    scale_field(fmt, so, sl);
    for (int64_t r = 0; r < g.n_pad; ++r) {
        for (int64_t b = 0; b < g.kb_pad; ++b) {
            if (r < g.n && b < g.kb) continue;
            uint8_t blk[MAX_BLOCK_BYTES];
            const uint8_t * rec = dev.data() + g.group_base(r) + size_t(b / g.rec_blocks) * g.rec_bytes;
            detail::unpack_block(fmt, rec, int(b % g.rec_blocks), int(r % 16), blk);
            for (size_t i = 0; i < sl; ++i) bad += blk[so + i] != 0;
        }
    }
    CHECK(bad == 0);

    // 4. chunked, arbitrarily aligned writes give the same device bytes; random partial reads match
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
    Fmt fmt = Fmt::Q4_0;
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
    int cases = 0;
    uint32_t seed = 1;
    for (int f = 0; f < int(Fmt::COUNT); ++f) {
        const Fmt fmt = Fmt(f);
        const std::vector<int64_t> Ks = fmt_info(fmt).elems == 32 ? std::vector<int64_t>{32, 64, 256, 288, 800, 1536, 4096}
                                                                  : std::vector<int64_t>{256, 512, 768, 1536, 4096};
        for (int64_t n : Ns) {
            for (int64_t k : Ks) {
                if (n * k > 1537 * 1536 && k > 1536) continue; // keep the runtime short
                test_shape(fmt, n, k, seed++);
                ++cases;
            }
        }
        // multi-chunk geometry (limit shrunk to ~0.6 MB so that small tensors split into several dispatches)
        for (int64_t n : {2048, 5000}) {
            const int64_t k = fmt_info(fmt).elems == 32 ? 800 : 768;
            const Geom g = make_geom(fmt, n, k, 600000);
            CHECK(g.nchunks > 1);
            test_shape(fmt, n, k, seed++, 600000);
            ++cases;
        }
        test_slices(fmt, fmt_info(fmt).elems == 32 ? 288 : 512);
        std::printf("  %s: ok so far (%d failures)\n", fmt_info(fmt).name, g_fail);
    }
    // Ornith output.weight (Q6_K, K=4096, N=248320) must split into two <=512 MB dispatches
    {
        const Geom g = make_geom(Fmt::Q6_K, 248320, 4096);
        CHECK(g.nchunks == 2 && g.chunks[0].rows == 124416 && g.chunks[1].rows == 123904);
        CHECK(g.gguf_bytes == size_t(248320) * 16 * 210 && g.bytes == g.gguf_bytes);   // N is a multiple of 512: no padding
    }
    std::printf("test-xdna2-layout: %d shapes + slices, %s\n", cases, g_fail ? "FAILED" : "PASSED");
    return g_fail ? 1 : 0;
}

// Standalone cache-store tests using an in-memory state transport.
// Build with: c++ -std=c++17 -O2 -Iinclude -Iggml/include -I. 
// -Itools/server tools/server/server-cache-store.cpp 
// tools/server/tests/integration/test_cache_block_store.cpp -o /tmp/cache-store-test
#include "server-cache-store.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <random>
#include <stdexcept>

struct llama_context {
    std::vector<std::vector<uint8_t>> regions;
};

size_t llama_state_seq_get_stream(llama_context * ctx, llama_seq_id,
        llama_state_stream_write_cb cb, void * user) {
    size_t total = 0;
    for (auto & region : ctx->regions) {
        if (!cb(user, region.data(), region.size(), true)) { return 0; }
        total += region.size();
    }
    return total;
}
size_t llama_state_seq_set_stream(llama_context * ctx, llama_seq_id,
        llama_state_stream_read_cb cb, void * user) {
    size_t total = 0;
    for (auto & region : ctx->regions) {
        if (!cb(user, region.data(), region.size())) { return 0; }
        total += region.size();
    }
    return total;
}

int main() {
    constexpr size_t mib = 1024*1024;
    std::mt19937 rng(42);
    llama_context a, b, out;
    a.regions.resize(4);
    for (auto & v : a.regions) {
        v.resize(mib);
        for (auto & x : v) { x = uint8_t(rng()); }
    }
    b = a;
    b.regions.back().resize(mib*2);
    for (size_t i=mib; i<2*mib; ++i) { b.regions.back()[i] = uint8_t(rng()); }
    server_cache_pool pool;
    const auto admit = [&](size_t n) { return pool.size() + n <= 6*mib; };
    {
        server_cache_blob x, y;
        assert(x.capture(&a,0,4*mib,pool,admit));
        assert(y.capture(&b,0,5*mib,pool,admit));
        assert(pool.payload_size() == 5*mib);
        out = b;
        for (auto & v : out.regions) { std::fill(v.begin(), v.end(), 0); }
        assert(y.restore(&out,0) && out.regions == b.regions);
        out = a;
        for (auto & v : out.regions) { std::fill(v.begin(), v.end(), 0); }
        assert(x.restore(&out,0) && out.regions == a.regions);
        {
            server_cache_blob rejected;
            std::vector<uint8_t> v(2*mib);
            for (auto & c : v) { c = uint8_t(rng()); }
            assert(!rejected.capture_bytes(v.data(),v.size(),pool,admit));
            assert(pool.size() <= 6*mib);
        }
        assert(pool.payload_size() == 5*mib);
        x = {};
        assert(pool.payload_size() == 5*mib);
        y = {};
        assert(pool.size() == 0 && pool.blocks() == 0);
    }
    for (int i=0; i<1000; ++i) {
        server_cache_blob blob;
        a.regions[0][0] = uint8_t(i);
        assert(blob.capture(&a,0,4*mib,pool,admit));
    }
    assert(pool.size() == 0 && pool.blocks() == 0);
    pool.trim_unused();
    std::puts("PASS: exact reconstruction, budget rejection, and 1,000 allocation/eviction cycles; no retained pool bytes or index entries");
}

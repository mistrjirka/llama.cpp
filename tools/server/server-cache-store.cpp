#include "server-cache-store.h"
#define XXH_INLINE_ALL
#include "vendor/hash/xxhash/xxhash.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

server_cache_ref server_cache_pool::intern(const uint8_t * data, size_t n, const server_cache_admit & admit) {
    const uint64_t key = XXH3_64bits(data, n);
    auto range = index.equal_range(key);
    for (auto it = range.first; it != range.second; ++it) {
        auto ref = it->second.lock();
        // Compare bytes as well as hashes. No collision can change state.
        if (ref && ref->bytes.size() == n && std::memcmp(ref->bytes.data(), data, n) == 0) {
            return ref;
        }
    }
    if (!admit(n + block_overhead)) { return {}; }
    auto raw = std::make_unique<server_cache_block>();
    raw->bytes.assign(data, data + n);
    const size_t charge = raw->bytes.capacity() + block_overhead;
    if (charge > n + block_overhead && !admit(charge)) { return {}; }
    accounted += charge;
    payload += n;
    server_cache_ref ref(raw.release(), [this, key, charge, n](const server_cache_block * block) {
        auto range = index.equal_range(key);
        for (auto it = range.first; it != range.second;) {
            if (it->second.expired()) { it = index.erase(it); }
            else { ++it; }
        }
        accounted -= charge;
        released_since_trim += charge;
        payload -= n;
        delete block;
    });
    index.emplace(key, ref);
    return ref;
}

void server_cache_pool::trim_unused() {
    // glibc can retain freed 128 KiB allocations in arenas. Reclaim unused
    // heap pages after substantial eviction, not on every request.
    if (released_since_trim >= 128 * 1024 * 1024) {
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        released_since_trim = 0;
    }
}

namespace {
constexpr size_t block_size = 128 * 1024;
struct writer {
    server_cache_blob & blob;
    server_cache_pool & pool;
    const server_cache_admit & admit;
    std::vector<uint8_t> pending;
    size_t expected;

    writer(server_cache_blob & b, server_cache_pool & p, const server_cache_admit & a, size_t n)
        : blob(b), pool(p), admit(a), expected(n) {
        pending.reserve(block_size);
    }

    bool flush() {
        if (pending.empty()) { return true; }
        if (blob.refs.size() == blob.refs.capacity()) {
            const size_t capacity = std::max<size_t>(16, blob.refs.capacity() * 2);
            // Charge the new allocation before reserve, including the brief
            // overlap with the old vector. The pool admission may evict data.
            if (!admit(capacity * sizeof(server_cache_ref))) { return false; }
            blob.refs.reserve(capacity);
        }
        auto ref = pool.intern(pending.data(), pending.size(), admit);
        if (!ref) { return false; }
        blob.logical += pending.size();
        blob.refs.push_back(std::move(ref));
        pending.clear();
        return true;
    }
    bool write(const uint8_t * data, size_t n, bool boundary) {
        if (blob.logical > expected || pending.size() > expected - blob.logical ||
            n > expected - blob.logical - pending.size()) { return false; }
        if (boundary && !flush()) { return false; }
        while (n) {
            const size_t take = std::min(n, block_size - pending.size());
            pending.insert(pending.end(), data, data + take);
            data += take;
            n -= take;
            if (pending.size() == block_size && !flush()) { return false; }
        }
        return true;
    }
    static bool callback(void * user, const uint8_t * data, size_t n, bool boundary) {
        return static_cast<writer *>(user)->write(data, n, boundary);
    }
};

struct reader {
    const server_cache_blob & blob;
    size_t index = 0, offset = 0, consumed = 0;
    bool read(uint8_t * data, size_t n) {
        if (consumed > blob.logical || n > blob.logical - consumed) { return false; }
        while (n) {
            if (index >= blob.refs.size()) { return false; }
            const auto & bytes = blob.refs[index]->bytes;
            const size_t take = std::min(n, bytes.size() - offset);
            std::memcpy(data, bytes.data() + offset, take);
            data += take;
            n -= take;
            offset += take;
            consumed += take;
            if (offset == bytes.size()) { ++index; offset = 0; }
        }
        return true;
    }
    static bool callback(void * user, uint8_t * data, size_t n) {
        return static_cast<reader *>(user)->read(data, n);
    }
};
}

bool server_cache_blob::capture(llama_context * ctx, llama_seq_id id, size_t expected,
                                server_cache_pool & pool, const server_cache_admit & admit) {
    if (!expected) { return false; }
    writer w(*this, pool, admit, expected);
    const size_t n = llama_state_seq_get_stream(ctx, id, writer::callback, &w);
    return n == expected && w.flush() && logical == expected;
}

bool server_cache_blob::capture_bytes(const uint8_t * data, size_t n, server_cache_pool & pool,
                                      const server_cache_admit & admit) {
    if (!n) { return true; }
    writer w(*this, pool, admit, n);
    return w.write(data, n, false) && w.flush() && logical == n;
}

bool server_cache_blob::capture_input(size_t n, const std::function<void(uint8_t *, size_t)> & read,
                                      server_cache_pool & pool, const server_cache_admit & admit) {
    if (!n) { return true; }
    writer w(*this, pool, admit, n);
    std::vector<uint8_t> buffer(std::min(n, block_size));
    size_t remaining = n;
    while (remaining) {
        const size_t take = std::min(remaining, buffer.size());
        read(buffer.data(), take);
        remaining -= take;
        if (!w.write(buffer.data(), take, false)) { return false; }
    }
    return w.flush() && logical == n;
}

bool server_cache_blob::restore(llama_context * ctx, llama_seq_id id, llama_seq_id prefix_id, llama_pos prefix_pos) const {
    if (empty()) { return false; }
    reader r{*this};
    const size_t n = prefix_id >= 0 && prefix_pos > 0
            ? llama_state_seq_set_stream_prefix(ctx, id, prefix_id, prefix_pos, reader::callback, &r)
            : llama_state_seq_set_stream(ctx, id, reader::callback, &r);
    return n == logical && r.consumed == logical;
}

bool server_cache_blob::restore_bytes(std::vector<uint8_t> & out) const {
    out.resize(logical);
    reader r{*this};
    return r.read(out.data(), out.size()) && r.consumed == logical;
}

#pragma once
#include <cstdint>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include "llama.h"

// The pool owns no strong references. Releasing the last recipe reference
// removes the index entry and releases its bytes immediately.
struct server_cache_block {
    std::vector<uint8_t> bytes;
};
using server_cache_ref = std::shared_ptr<const server_cache_block>;
using server_cache_admit = std::function<bool(size_t)>;

class server_cache_pool {
public:
    server_cache_pool() = default;
    server_cache_pool(const server_cache_pool &) = delete;
    server_cache_pool & operator=(const server_cache_pool &) = delete;
    size_t size() const { return accounted; }
    size_t payload_size() const { return payload; }
    size_t blocks() const { return index.size(); }
    void trim_unused();
    server_cache_ref intern(const uint8_t * data, size_t n, const server_cache_admit & admit);
private:
    // Includes a conservative charge for the vector, shared_ptr control
    // block, tree node, and allocation bookkeeping.
    static constexpr size_t block_overhead = 256;
    std::multimap<uint64_t, std::weak_ptr<const server_cache_block>> index;
    size_t accounted = 0;
    size_t payload = 0;
    size_t released_since_trim = 0;
};

struct server_cache_blob {
    std::vector<server_cache_ref> refs;
    size_t logical = 0;
    size_t size() const { return logical; }
    size_t overhead() const { return refs.capacity() * sizeof(server_cache_ref); }
    bool empty() const { return logical == 0; }

    bool capture(llama_context * ctx, llama_seq_id id, size_t expected,
                 server_cache_pool & pool, const server_cache_admit & admit);
    bool capture_bytes(const uint8_t * data, size_t n, server_cache_pool & pool,
                       const server_cache_admit & admit);
    bool restore(llama_context * ctx, llama_seq_id id) const;
    bool restore_bytes(std::vector<uint8_t> & out) const;
};

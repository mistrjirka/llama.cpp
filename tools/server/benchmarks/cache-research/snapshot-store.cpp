// Standalone exact snapshot-storage experiment. No inference code is modified.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xxhash.h>

using byte = uint8_t;
using Clock = std::chrono::steady_clock;

struct Mapping {
    int fd = -1;
    size_t size = 0;
    const byte * data = nullptr;
    std::string name;
    explicit Mapping(std::string path) : name(std::move(path)) {
        fd = open(name.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("open: " + name);
        struct stat st {};
        if (fstat(fd, &st) || st.st_size <= 0) { close(fd); throw std::runtime_error("stat: " + name); }
        size = size_t(st.st_size);
        void * p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) { close(fd); throw std::runtime_error("mmap: " + name); }
        data = static_cast<const byte *>(p);
    }
    ~Mapping() { if (data) munmap(const_cast<byte *>(data), size); if (fd >= 0) close(fd); }
    Mapping(const Mapping &) = delete;
    Mapping & operator=(const Mapping &) = delete;
};

struct Key {
    uint64_t hash;
    size_t size;
    bool operator==(const Key & other) const { return hash == other.hash && size == other.size; }
};
struct KeyHash {
    size_t operator()(const Key & k) const { return size_t(k.hash ^ (k.size * 0x9e3779b97f4a7c15ULL)); }
};
struct Block {
    const byte * data;
    size_t size;
    std::unique_ptr<byte[]> owned;
};
struct Recipe {
    const Mapping * original;
    std::vector<size_t> chunks;
};
struct Range { size_t start, bytes, row; };

static uint32_t read_u32(const Mapping & f, size_t & pos) {
    if (pos > f.size || f.size - pos < 4) throw std::runtime_error("short u32");
    uint32_t v;
    std::memcpy(&v, f.data + pos, 4);
    pos += 4;
    return v;
}
static uint64_t read_u64(const Mapping & f, size_t & pos) {
    if (pos > f.size || f.size - pos < 8) throw std::runtime_error("short u64");
    uint64_t v;
    std::memcpy(&v, f.data + pos, 8);
    pos += 8;
    return v;
}
static void skip(const Mapping & f, size_t & pos, size_t n) {
    if (pos > f.size || f.size - pos < n) throw std::runtime_error("range overflow");
    pos += n;
}

// An offline parser for the *standard contiguous KV serializer* used by
// llama_state_seq_save_file v4. Other layouts must use a generic chunker.
// Future production tensor-aligned storage should use serializer callbacks
// instead of parsing a particular on-disk format.
static std::vector<Range> standard_kv_ranges(const Mapping & f, size_t ext) {
    size_t pos = 0;
    if (read_u32(f, pos) != 0x67677371u || read_u32(f, pos) != 4)
        throw std::runtime_error("not a v4 sequence snapshot");
    const auto token_words = read_u32(f, pos);
    skip(f, pos, size_t(token_words) * 4);
    if (read_u32(f, pos) != 1) throw std::runtime_error("not a single KV stream");
    const auto cells = read_u32(f, pos);
    if (!cells || cells > 1048576) throw std::runtime_error("bad cell count");
    int32_t prev = -1;
    for (uint32_t i = 0; i < cells; ++i) {
        const auto p = int32_t(read_u32(f, pos));
        const auto seqs = read_u32(f, pos);
        if (p < 0 || p < prev || seqs != 1) throw std::runtime_error("unexpected cell metadata");
        prev = p;
        skip(f, pos, ext);
        if (read_u32(f, pos) != 0) throw std::runtime_error("not slot zero");
    }
    const auto v_trans = read_u32(f, pos);
    const auto layers = read_u32(f, pos);
    if (v_trans || !layers || layers > 256) throw std::runtime_error("unsupported KV layout");
    // State format writes n_rot_k and n_rot_v as uint32_t.
    skip(f, pos, 8);
    std::vector<Range> ranges;
    for (uint32_t i = 0; i < 2 * layers; ++i) {
        const auto type = read_u32(f, pos);
        const auto row = read_u64(f, pos);
        if (type > 100 || !row || row > (1u << 20) || row > (f.size - pos) / cells)
            throw std::runtime_error("invalid tensor record");
        const auto bytes = size_t(row) * cells;
        ranges.push_back({pos, bytes, size_t(row)});
        skip(f, pos, bytes);
    }
    return ranges;
}

static std::vector<Range> tensor_ranges(const Mapping & f) {
    std::vector<Range> result;
    int matches = 0;
    for (size_t ext : {size_t(0), size_t(12)}) {
        try { auto r = standard_kv_ranges(f, ext); result = std::move(r); ++matches; }
        catch (const std::runtime_error &) {}
    }
    if (matches != 1) throw std::runtime_error("unsupported or ambiguous tensor layout: " + f.name);
    return result;
}

struct Store {
    bool materialize;
    std::unordered_multimap<Key, size_t, KeyHash> index;
    std::vector<Block> blocks;
    std::vector<Recipe> recipes;
    size_t unique = 0, logical = 0;
    explicit Store(bool own) : materialize(own) {}

    void add(Recipe & r, const byte * p, size_t n) {
        if (!n) return;
        Key k{XXH3_64bits(p, n), n};
        auto [first, last] = index.equal_range(k);
        for (auto it = first; it != last; ++it) {
            const auto id = it->second;
            // A hash alone is never sufficient: compare the actual bytes.
            if (std::memcmp(blocks[id].data, p, n) == 0) {
                r.chunks.push_back(id);
                logical += n;
                return;
            }
        }
        Block b{p, n, nullptr};
        if (materialize) {
            b.owned.reset(new byte[n]);
            std::memcpy(b.owned.get(), p, n);
            b.data = b.owned.get();
        }
        const auto id = blocks.size();
        blocks.push_back(std::move(b));
        index.emplace(k, id);
        r.chunks.push_back(id);
        unique += n;
        logical += n;
    }

    void fixed(Recipe & r, const byte * p, size_t n, size_t chunk) {
        while (n) {
            size_t take = std::min(n, chunk);
            add(r, p, take);
            p += take;
            n -= take;
        }
    }

    static const std::array<uint64_t, 256> & gear_table() {
        static const auto table = [] {
            std::array<uint64_t, 256> t {};
            uint64_t x = 0xa0761d6478bd642fULL;
            for (auto & v : t) {
                x += 0x9e3779b97f4a7c15ULL;
                uint64_t z = x;
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                v = z ^ (z >> 31);
            }
            return t;
        }();
        return table;
    }

    // Content-defined Gear boundaries. Generic byte-level resynchronization,
    // not a claim to implement a specific FastCDC paper's tuned algorithm.
    void gear(Recipe & r, const byte * p, size_t n, size_t avg) {
        const auto & t = gear_table();
        const size_t min_chunk = avg / 4, max_chunk = avg * 4;
        const uint64_t mask = uint64_t(avg - 1);
        while (n) {
            size_t i = std::min(n, min_chunk);
            uint64_t h = 0;
            const size_t cap = std::min(n, max_chunk);
            while (i < cap) {
                h = (h << 1) + t[p[i++]];
                if ((h & mask) == 0) break;
            }
            add(r, p, i);
            p += i;
            n -= i;
        }
    }

    void ingest(const Mapping & f, const std::string & mode, size_t size) {
        Recipe r{&f, {}};
        if (mode == "fixed") {
            fixed(r, f.data, f.size, size);
        } else if (mode == "gear") {
            gear(r, f.data, f.size, size);
        } else if (mode == "tensor") {
            const auto ranges = tensor_ranges(f);
            size_t pos = 0;
            for (auto & tr : ranges) {
                gear(r, f.data + pos, tr.start - pos, 256 * 1024);
                fixed(r, f.data + tr.start, tr.bytes, tr.row * size);
                pos = tr.start + tr.bytes;
            }
            gear(r, f.data + pos, f.size - pos, 256 * 1024);
        } else {
            throw std::runtime_error("unknown mode");
        }
        recipes.push_back(std::move(r));
    }

    double verify() const {
        const auto start = Clock::now();
        // Materialize the stream in bounded scratch windows. Check each
        // reconstructed byte, including metadata and recurrent checkpoints.
        size_t max_block = 0;
        for (const auto & b : blocks) max_block = std::max(max_block, b.size);
        std::vector<byte> scratch(max_block);
        for (const auto & r : recipes) {
            size_t pos = 0;
            for (const auto id : r.chunks) {
                const auto & b = blocks.at(id);
                if (b.size > r.original->size - pos) throw std::runtime_error("restore length mismatch");
                std::memcpy(scratch.data(), b.data, b.size);
                if (std::memcmp(scratch.data(), r.original->data + pos, b.size))
                    throw std::runtime_error("lossless restore failed");
                pos += b.size;
            }
            if (pos != r.original->size) throw std::runtime_error("restore size mismatch");
        }
        return std::chrono::duration<double>(Clock::now() - start).count();
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc < 5) throw std::runtime_error("usage: snapshot-store fixed|gear|tensor size own|mmap files...");
        const std::string mode = argv[1];
        const size_t size = std::stoull(argv[2]);
        const bool own = std::string(argv[3]) == "own";
        if (!size || (mode == "gear" && (size < 256 || (size & (size - 1)))))
            throw std::runtime_error("invalid chunk size");
        std::vector<std::unique_ptr<Mapping>> files;
        for (int i = 4; i < argc; ++i) files.push_back(std::make_unique<Mapping>(argv[i]));
        Store store(own);
        const auto start = Clock::now();
        for (const auto & f : files) store.ingest(*f, mode, size);
        const double save_s = std::chrono::duration<double>(Clock::now() - start).count();
        const double restore_s = store.verify();
        size_t refs = 0;
        for (const auto & r : store.recipes) refs += r.chunks.size();
        std::cout << "{\"mode\":\"" << mode << "\",\"chunk\":" << size
                  << ",\"owned\":" << (own ? "true" : "false")
                  << ",\"snapshots\":" << files.size() << ",\"logical_bytes\":" << store.logical
                  << ",\"unique_bytes\":" << store.unique << ",\"blocks\":" << store.blocks.size()
                  << ",\"refs\":" << refs << ",\"saved_pct\":" << 100.0 * (1.0 - double(store.unique)/store.logical)
                  << ",\"ingest_s\":" << save_s << ",\"restore_verify_s\":" << restore_s
                  << ",\"verified\":true}" << std::endl;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

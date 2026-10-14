#pragma once

// Arena = the storage behind one ggml backend buffer (one allocation, host mapped).
//
//   HostArena   anonymous mmap (zero filled, page aligned). Used while there is no NPU executor
//               (host reference compute path) and for tests.
//   XrtBoArena  ONE xrt::bo with host mapping (flags host_only, memory group of the kernel it will be used
//               with). The tensors are sub-BOs of it: xrt::bo(arena.bo(), size, offset_in_arena).
//               It is created by the NpuExecutor (which owns the xrt::device and knows the kernel's
//               memory group), see ggml-xdna2-npu.h.
//
// The arena base is always 4096 aligned and the memory is zero initialised (padding of repacked
// tensors must have a zero scale).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <sys/mman.h>

#ifdef GGML_XDNA2_HAVE_XRT
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include "ggml-xdna2-sync.h"
#endif

namespace xdna2 {

constexpr size_t ARENA_ALIGN = 4096;

class Arena {
public:
    virtual ~Arena() = default;
    virtual uint8_t * base() const = 0;
    virtual size_t    size() const = 0;
    // make host writes in [off, off+len) visible to the device / device writes visible to the host
    virtual void sync_to_device(size_t off, size_t len) = 0;
    virtual void sync_from_device(size_t off, size_t len) = 0;
    virtual const char * kind() const = 0;
};

class HostArena final : public Arena {
public:
    explicit HostArena(size_t size) {
        size_ = (size + ARENA_ALIGN - 1) / ARENA_ALIGN * ARENA_ALIGN;
        void * p = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            throw std::runtime_error("xdna2: mmap failed");
        }
        base_ = static_cast<uint8_t *>(p);
    }
    ~HostArena() override { munmap(base_, size_); }
    uint8_t * base() const override { return base_; }
    size_t    size() const override { return size_; }
    void sync_to_device(size_t, size_t) override {}
    void sync_from_device(size_t, size_t) override {}
    const char * kind() const override { return "host"; }
private:
    uint8_t * base_ = nullptr;
    size_t    size_ = 0;
};

#ifdef GGML_XDNA2_HAVE_XRT
class XrtBoArena final : public Arena {
public:
    // memory_group: kernel->group_id(<arg index of the weight buffer>)
    XrtBoArena(const xrt::device & dev, size_t size, xrt::memory_group memory_group)
        : bo_(dev, (size + ARENA_ALIGN - 1) / ARENA_ALIGN * ARENA_ALIGN, xrt::bo::flags::host_only, memory_group) {
        base_ = bo_.map<uint8_t *>();
        size_ = bo_.size();
        // BO memory is not guaranteed to be zero; padding of repacked tensors needs zero scales
        std::memset(base_, 0, size_);
        xdna2::sync_to_device(bo_, size_);
    }
    uint8_t * base() const override { return base_; }
    size_t    size() const override { return size_; }
    void sync_to_device(size_t off, size_t len) override { xdna2::sync_to_device(bo_, len, off); }
    void sync_from_device(size_t off, size_t len) override { xdna2::sync_from_device(bo_, len, off); }
    const char * kind() const override { return "xrt-bo"; }
    xrt::bo & bo() { return bo_; }
private:
    xrt::bo   bo_;
    uint8_t * base_ = nullptr;
    size_t    size_ = 0;
};
#endif

} // namespace xdna2

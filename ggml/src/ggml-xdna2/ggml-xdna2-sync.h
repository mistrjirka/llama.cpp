#pragma once

// Cache coherency helpers for host-mapped XRT buffer objects.
// (Copied from the ggml-xdna prototype: CLFLUSH-based BO syncs on some XDNA shims return
//  without fencing subsequent CPU loads, so an explicit MFENCE is added after reading back.)

#ifdef GGML_XDNA2_HAVE_XRT

#include <xrt/xrt_bo.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace xdna2 {

inline void sync_to_device(xrt::bo & buffer, size_t bytes, size_t offset = 0) {
    buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, offset);
}

inline void sync_from_device(xrt::bo & buffer, size_t bytes, size_t offset = 0) {
    buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, offset);
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    // Some XDNA shims return from CLFLUSH without fencing subsequent CPU loads.
    _mm_mfence();
#endif
}

inline void sync_from_device(xrt::bo & buffer) {
    sync_from_device(buffer, buffer.size());
}

} // namespace xdna2

#endif // GGML_XDNA2_HAVE_XRT

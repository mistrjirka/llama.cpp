// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "moe_trace_support.inc"

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_moe_promotion_worker;

// First-stage persistent MoE cache. Each selected source tensor owns a dedicated
// backend buffer with the original full expert layout. Only expert rows that are
// actually routed are uploaded, and uploaded rows remain valid across scheduler
// graph resets. This intentionally prioritizes an exact persistence boundary
// before introducing compact slot remapping and eviction.
struct ggml_backend_sched_expert_cache_entry {
    const struct ggml_tensor * source = nullptr;
    std::string source_name;
    int backend_id = -1;
    ggml_backend_buffer_t buffer = nullptr;
    size_t allocation_size = 0;
    int64_t n_expert = 0;
    size_t expert_size = 0;
    std::unique_ptr<ggml_tensor> persistent_tensor;
    std::vector<uint8_t> resident;
};

struct ggml_backend_sched_expert_cache {
    size_t budget_bytes = 0;
    size_t allocated_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bytes_uploaded = 0;
    uint64_t bytes_avoided = 0;
    uint64_t attach_attempts = 0;
    uint64_t rejected_host_target = 0;
    uint64_t rejected_budget = 0;
    bool allocation_failed = false;
    bool managed = false;
    bool vmm = false;
    bool profile = false;
    void (*vmm_promote)(ggml_backend_buffer_t buffer, size_t offset, size_t size) = nullptr;
    uint64_t profile_selective_inputs = 0;
    uint64_t profile_ids_reads = 0;
    uint64_t profile_copy_groups = 0;
    uint64_t profile_copy_bytes = 0;
    uint64_t profile_copy_groups_prefill = 0;
    uint64_t profile_copy_bytes_prefill = 0;
    uint64_t profile_copy_groups_decode = 0;
    uint64_t profile_copy_bytes_decode = 0;
    uint64_t profile_dynamic_copy_groups = 0;
    uint64_t profile_dynamic_copy_bytes = 0;
    uint64_t profile_dynamic_copy_issue_ns = 0;
    uint64_t profile_experts_requested = 0;
    uint64_t profile_missing_experts = 0;
    uint64_t profile_input_sync_ns = 0;
    uint64_t profile_target_sync_calls = 0;
    uint64_t profile_target_sync_ns = 0;
    uint64_t profile_dynamic_target_sync_calls = 0;
    uint64_t profile_dynamic_target_sync_ns = 0;
    uint64_t profile_dynamic_hot_splits = 0;
    uint64_t profile_dynamic_hot_skipped = 0;
    uint64_t profile_dynamic_hot_nodes_skipped = 0;
    uint64_t profile_dynamic_hot_promotion_inputs = 0;
    uint64_t profile_dynamic_zero_fills = 0;
    uint64_t profile_dynamic_zero_bytes = 0;
    uint64_t profile_dynamic_hot_compute_calls = 0;
    uint64_t profile_dynamic_hot_compute_ns = 0;
    uint64_t profile_dynamic_cold_compute_calls = 0;
    uint64_t profile_dynamic_cold_compute_ns = 0;
    uint64_t profile_dynamic_decision_sync_calls = 0;
    uint64_t profile_dynamic_decision_sync_ns = 0;
    uint64_t profile_ids_sync_ns = 0;
    uint64_t profile_attach_ns = 0;
    uint64_t profile_copy_issue_ns = 0;
    uint64_t profile_transfer_h2d_groups = 0;
    uint64_t profile_transfer_h2d_bytes = 0;
    uint64_t profile_transfer_d2h_groups = 0;
    uint64_t profile_transfer_d2h_bytes = 0;
    uint64_t profile_transfer_d2d_groups = 0;
    uint64_t profile_transfer_d2d_bytes = 0;
    uint64_t profile_transfer_h2h_groups = 0;
    uint64_t profile_transfer_h2h_bytes = 0;
    uint64_t profile_transfer_async_groups = 0;
    uint64_t profile_transfer_fallback_groups = 0;
    uint64_t profile_transfer_issue_ns = 0;
    uint64_t profile_transfer_source_sync_ns = 0;
    uint64_t profile_transfer_destination_sync_ns = 0;
    uint64_t profile_submit_cpu_calls = 0;
    uint64_t profile_submit_cpu_ns = 0;
    uint64_t profile_submit_gpu_calls = 0;
    uint64_t profile_submit_gpu_ns = 0;
    uint64_t profile_submit_dynamic_hot_calls = 0;
    uint64_t profile_submit_dynamic_hot_ns = 0;
    uint64_t profile_submit_dynamic_cold_calls = 0;
    uint64_t profile_submit_dynamic_cold_ns = 0;
    ggml_backend_moe_promotion_worker * promotion_worker = nullptr;
    std::vector<ggml_backend_sched_expert_cache_entry> entries;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    ggml_backend_sched_expert_cache * expert_cache;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static ggml_backend_sched_expert_cache_entry * ggml_backend_sched_expert_cache_find(
        ggml_backend_sched_t sched,
        const struct ggml_tensor * source,
        int backend_id) {
    if (sched->expert_cache == nullptr) {
        return nullptr;
    }
    for (auto & entry : sched->expert_cache->entries) {
        if (entry.backend_id == backend_id &&
            entry.source_name == source->name) {
            entry.source = source;
            return &entry;
        }
    }
    return nullptr;
}

using ggml_backend_dev_managed_buffer_type_t = ggml_backend_buffer_type_t (*)(ggml_backend_dev_t device);
using ggml_backend_dev_expert_vmm_buffer_type_t = ggml_backend_buffer_type_t (*)(ggml_backend_dev_t device);
using ggml_backend_expert_vmm_promote_t = void (*)(ggml_backend_buffer_t buffer, size_t offset, size_t size);

static ggml_backend_buffer_type_t ggml_backend_sched_expert_cache_buffer_type(
        ggml_backend_sched_t sched,
        int backend_id) {
    ggml_backend_buffer_type_t buft = sched->bufts[backend_id];
    if (sched->expert_cache == nullptr || (!sched->expert_cache->managed && !sched->expert_cache->vmm)) {
        return buft;
    }

    ggml_backend_dev_t device = sched->backends[backend_id]->device;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);

    if (sched->expert_cache->vmm) {
        auto get_vmm = (ggml_backend_dev_expert_vmm_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_expert_vmm_buffer_type");
        if (get_vmm != nullptr) {
            ggml_backend_buffer_type_t vmm = get_vmm(device);
            if (vmm != nullptr) {
                if (sched->expert_cache->vmm_promote == nullptr) {
                    sched->expert_cache->vmm_promote = (ggml_backend_expert_vmm_promote_t)
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_expert_vmm_promote");
                }
                return vmm;
            }
        }
        GGML_LOG_WARN("expert-cache: backend %s does not provide expert VMM buffers; using ordinary device memory\n",
            ggml_backend_name(sched->backends[backend_id]));
        return buft;
    }

    auto get_managed = (ggml_backend_dev_managed_buffer_type_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_managed_buffer_type");
    if (get_managed == nullptr) {
        GGML_LOG_WARN("expert-cache: backend %s does not provide managed buffers; using ordinary device memory\n",
            ggml_backend_name(sched->backends[backend_id]));
        return buft;
    }

    ggml_backend_buffer_type_t managed = get_managed(device);
    return managed != nullptr ? managed : buft;
}

struct ggml_backend_sched_static_map_registry {
    std::mutex mutex;
    std::string loaded_path;
    std::vector<std::vector<int32_t>> experts_by_layer;
    std::unordered_map<std::string, std::vector<int32_t>> slot_maps;
};

static ggml_backend_sched_static_map_registry & ggml_backend_sched_static_map_registry_get() {
    static ggml_backend_sched_static_map_registry registry;
    return registry;
}

static void ggml_backend_sched_static_map_load(
        ggml_backend_sched_static_map_registry & registry,
        const std::string & path) {
    if (registry.loaded_path == path) {
        return;
    }
    registry.loaded_path = path;
    registry.experts_by_layer.clear();
    registry.slot_maps.clear();
    if (path.empty()) {
        return;
    }

    std::ifstream input(path);
    if (!input) {
        GGML_LOG_WARN("expert-cache: unable to open static expert map %s; using identity mapping\n", path.c_str());
        return;
    }
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream stream(line);
        int layer = -1;
        stream >> layer;
        if (layer < 0) {
            continue;
        }
        if ((size_t) layer >= registry.experts_by_layer.size()) {
            registry.experts_by_layer.resize((size_t) layer + 1);
        }
        int expert = -1;
        while (stream >> expert) {
            registry.experts_by_layer[(size_t) layer].push_back(expert);
        }
    }
}

static const std::vector<int32_t> & ggml_backend_sched_static_slot_map(
        const ggml_tensor * source,
        int32_t n_slots,
        int32_t n_source_experts) {
    auto & registry = ggml_backend_sched_static_map_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);

    const char * path_env = getenv("GGML_MOE_STATIC_SPLIT_MAP");
    const std::string path = path_env != nullptr ? path_env : "";
    ggml_backend_sched_static_map_load(registry, path);

    const char * layer_suffix = strrchr(source->name, '-');
    const int layer = layer_suffix != nullptr ? atoi(layer_suffix + 1) : -1;
    const std::string key = path + ":" + std::to_string(layer) + ":" +
        std::to_string(n_slots) + ":" + std::to_string(n_source_experts);
    auto found = registry.slot_maps.find(key);
    if (found != registry.slot_maps.end()) {
        return found->second;
    }

    std::vector<int32_t> mapping;
    mapping.reserve((size_t) n_slots);
    std::vector<uint8_t> used((size_t) n_source_experts, 0);
    if (layer >= 0 && (size_t) layer < registry.experts_by_layer.size()) {
        for (int32_t expert : registry.experts_by_layer[(size_t) layer]) {
            if ((int32_t) mapping.size() >= n_slots) {
                break;
            }
            if (expert < 0 || expert >= n_source_experts || used[(size_t) expert]) {
                continue;
            }
            mapping.push_back(expert);
            used[(size_t) expert] = 1;
        }
    }
    for (int32_t expert = 0; expert < n_source_experts && (int32_t) mapping.size() < n_slots; ++expert) {
        if (!used[(size_t) expert]) {
            mapping.push_back(expert);
        }
    }
    GGML_ASSERT((int32_t) mapping.size() == n_slots);
    return registry.slot_maps.emplace(key, std::move(mapping)).first->second;
}

enum ggml_backend_moe_dynamic_slot_state {
    GGML_BACKEND_MOE_SLOT_EMPTY = 0,
    GGML_BACKEND_MOE_SLOT_COPYING,
    GGML_BACKEND_MOE_SLOT_READY,
};

struct ggml_backend_moe_dynamic_slot {
    int32_t expert = -1;
    ggml_backend_moe_dynamic_slot_state state = GGML_BACKEND_MOE_SLOT_EMPTY;
    uint8_t components_enqueued = 0;
    uint8_t components_completed = 0;
    uint64_t promotion_bundle_id = 0;
    uint64_t promotion_queued_mono_ns = 0;
    uint64_t prediction_step = 0;
    int32_t prediction_distance = 0;
    uint64_t admitted_step = 0;
    uint64_t last_used_step = 0;
    uint32_t hit_count = 0;
    bool protected_segment = false;
};

// Runtime binding between one canonical expert component and the persistent
// compact GPU tensor that owns that component's slot array. Graph attachment
// establishes these pointers once; an urgent predictor request can then issue
// all three component copies immediately instead of waiting for the target
// layer to be encountered again by the scheduler.
struct ggml_backend_moe_dynamic_component_binding {
    bool valid = false;
    const uint8_t * source_base = nullptr;
    size_t source_stride = 0;
    ggml_tensor * destination = nullptr;
    size_t destination_stride = 0;
    bool source_pinned = false;
    ggml_backend_moe_promotion_worker * worker = nullptr;
    ggml_backend_sched_expert_cache * cache = nullptr;
    uint8_t * resident = nullptr;
};

struct ggml_backend_moe_dynamic_layer {
    int32_t n_expert = 0;
    int32_t n_slots = 0;
    uint64_t step = 0;
    bool active = false;
    bool static_frozen = false;
    double observed_coverage = 0.0;
    uint64_t admissions = 0;
    uint64_t resident_hits = 0;
    uint64_t resident_misses = 0;
    uint64_t ready_hits = 0;
    uint64_t cpu_misses = 0;
    uint64_t split_decisions = 0;
    uint64_t split_enabled = 0;
    uint64_t decode_steps = 0;
    uint64_t prefill_batches = 0;
    uint64_t prefill_ready_routes = 0;
    uint64_t prefill_total_routes = 0;
    uint64_t full_gpu_layer_steps = 0;
    uint64_t cpu_dependent_layer_steps = 0;
    int32_t last_ready_routes = 0;
    int32_t last_gpu_routes = 0;
    int32_t last_total_routes = 0;
    bool gpu_split_enabled = false;
    std::vector<uint64_t> ready_route_hist;
    std::vector<uint64_t> gpu_route_hist;
    std::vector<ggml_backend_moe_dynamic_slot> slots;
    std::vector<int32_t> slot_by_expert;
    std::vector<double> score;
    std::vector<uint64_t> last_score_step;
    std::vector<int32_t> previous_selected;
    std::vector<uint16_t> transition_counts;
    // Same-token cross-layer predictors. The target layer owns the model:
    // distance-one maps routes from layer L-1 to L, while distance-two maps
    // routes from L-2 to L and provides an extra layer of upload lead time.
    std::vector<uint16_t> cross_layer_transition_counts;
    std::vector<uint32_t> cross_layer_source_counts;
    std::vector<uint16_t> cross_layer2_transition_counts;
    std::vector<uint32_t> cross_layer2_source_counts;
    // Bounded sparse-context approximation: 512 hashed buckets for pairs
    // (expert at L-2, expert at L-1), each with target-expert counts.
    std::vector<uint16_t> cross_joint_transition_counts;
    std::vector<uint32_t> cross_joint_source_counts;
    uint64_t last_cross_decay_step = 0;
    std::array<ggml_backend_moe_dynamic_component_binding, 3> component_bindings;
    uint64_t prediction_steps = 0;
    uint64_t prediction_admissions = 0;
    uint64_t cross_prediction_steps = 0;
    uint64_t cross_prediction_admissions = 0;
};

struct ggml_backend_moe_dynamic_token_coverage {
    uint64_t layers_seen = 0;
    uint64_t full_gpu_layers = 0;
    uint64_t cpu_dependent_layers = 0;
    uint64_t gpu_routes = 0;
    uint64_t total_routes = 0;
};

struct ggml_backend_moe_dynamic_current_routes {
    uint64_t step = 0;
    std::vector<int32_t> selected;
};

struct ggml_backend_moe_dynamic_registry {
    std::mutex mutex;
    std::unordered_map<int32_t, ggml_backend_moe_dynamic_layer> layers;
    std::unordered_map<uint64_t, ggml_backend_moe_dynamic_token_coverage> token_coverage;
    std::unordered_map<int32_t, ggml_backend_moe_dynamic_current_routes> current_routes;
    uint64_t admission_step = 0;
    int32_t admissions_this_step = 0;
    uint64_t warm_start_step = 0;
    int32_t warm_start_admissions_this_step = 0;
    uint64_t predictor_step = 0;
    int32_t predictor_admissions_this_step = 0;
    uint64_t cross_predictor_step = 0;
    int32_t cross_predictor_admissions_this_step = 0;
    uint64_t next_promotion_bundle_id = 1;
};

static bool ggml_backend_moe_dynamic_issue_urgent_locked(
        ggml_backend_moe_dynamic_registry & registry,
        int32_t layer_id,
        int32_t slot,
        int32_t expert,
        int32_t distance,
        uint64_t step);

struct ggml_backend_moe_dynamic_trace_state {
    std::mutex mutex;
    FILE * file = nullptr;
    std::string path;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    uint64_t sequence = 0;
    uint64_t pending = 0;
};

static ggml_backend_moe_dynamic_trace_state & ggml_backend_moe_dynamic_trace_state_get() {
    static ggml_backend_moe_dynamic_trace_state state;
    return state;
}

static uint64_t ggml_backend_moe_dynamic_mono_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void ggml_backend_moe_dynamic_tracef(const char * format, ...) {
    const char * path_env = getenv("GGML_MOE_DYNAMIC_TRACE");
    if (path_env == nullptr || path_env[0] == '\0') {
        return;
    }

    const char * compact_env = getenv("GGML_MOE_DYNAMIC_TRACE_COMPACT");
    const bool compact = compact_env != nullptr && compact_env[0] != '\0' && atoi(compact_env) != 0;
    std::string compact_payload;
    if (compact) {
        va_list args;
        va_start(args, format);
        va_list args_copy;
        va_copy(args_copy, args);
        const int required = vsnprintf(nullptr, 0, format, args);
        va_end(args);
        if (required < 0) {
            va_end(args_copy);
            return;
        }
        std::vector<char> buffer((size_t) required + 1);
        vsnprintf(buffer.data(), buffer.size(), format, args_copy);
        va_end(args_copy);
        compact_payload.assign(buffer.data(), (size_t) required);

        static const char * retained_events[] = {
            "\"event\":\"route\"",
            "\"event\":\"route_batch\"",
            "\"event\":\"observe\"",
            "\"event\":\"layer_activate\"",
            "\"event\":\"warm_start\"",
            "\"event\":\"admit\"",
            "\"event\":\"evict\"",
            "\"event\":\"spill_evict\"",
            "\"event\":\"ready_after_completion\"",
            "\"event\":\"urgent_bundle_queued\"",
            "\"event\":\"cross_layer_predict_prefetch\"",
            "\"event\":\"cross_layer_decay\"",
            "\"event\":\"worker_transfer_verify\"",
            "\"event\":\"phase_marker\"",
        };
        bool retained = false;
        for (const char * event : retained_events) {
            if (compact_payload.find(event) != std::string::npos) {
                retained = true;
                break;
            }
        }
        if (!retained) {
            return;
        }
    }

    auto & state = ggml_backend_moe_dynamic_trace_state_get();
    std::lock_guard<std::mutex> lock(state.mutex);
    const std::string path = path_env;
    if (state.file == nullptr || state.path != path) {
        if (state.file != nullptr) {
            fclose(state.file);
        }
        state.file = fopen(path.c_str(), "w");
        state.path = path;
        state.start = std::chrono::steady_clock::now();
        state.sequence = 0;
        state.pending = 0;
        if (state.file == nullptr) {
            GGML_LOG_WARN("moe-dynamic-trace: unable to open %s\n", path.c_str());
            return;
        }
    }

    const uint64_t elapsed_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - state.start).count();
    const uint64_t wall_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    fprintf(state.file,
        "{\"seq\":%" PRIu64 ",\"us\":%" PRIu64 ",\"wall_us\":%" PRIu64 ",",
        state.sequence++, elapsed_us, wall_us);
    if (compact) {
        fputs(compact_payload.c_str(), state.file);
    } else {
        va_list args;
        va_start(args, format);
        vfprintf(state.file, format, args);
        va_end(args);
    }
    fputs("}\n", state.file);
    state.pending++;
    const char * flush_every_env = getenv("GGML_MOE_DYNAMIC_TRACE_FLUSH_EVERY");
    const int32_t flush_every = std::max<int32_t>(
        1, flush_every_env != nullptr ? atoi(flush_every_env) : 256);
    if (state.pending >= (uint64_t) flush_every) {
        fflush(state.file);
        state.pending = 0;
    }
}

void ggml_backend_moe_dynamic_trace_marker(const char * marker) {
    GGML_ASSERT(marker != nullptr);
    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"phase_marker\",\"marker\":\"%s\"",
        marker);
}

static const char * ggml_backend_moe_dynamic_slot_state_name(ggml_backend_moe_dynamic_slot_state state) {
    switch (state) {
        case GGML_BACKEND_MOE_SLOT_EMPTY:   return "empty";
        case GGML_BACKEND_MOE_SLOT_COPYING: return "copying";
        case GGML_BACKEND_MOE_SLOT_READY:   return "ready";
    }
    return "unknown";
}

static const char * ggml_backend_moe_dynamic_split_kind(const ggml_backend_sched_split * split) {
    bool hot = false;
    bool cold = false;
    for (int node_index = 0; node_index < split->graph.n_nodes; ++node_index) {
        const ggml_tensor * node = split->graph.nodes[node_index];
        hot |= strstr(node->name, "ffn_moe_dynamic_hot") != nullptr;
        cold |= strstr(node->name, "ffn_moe_dynamic_cold") != nullptr;
        for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
            const ggml_tensor * source = node->src[source_index];
            if (source == nullptr) {
                continue;
            }
            hot |= strstr(source->name, "ffn_moe_dynamic_hot") != nullptr;
            cold |= strstr(source->name, "ffn_moe_dynamic_cold") != nullptr;
        }
    }
    if (hot && cold) {
        return "mixed";
    }
    if (hot) {
        return "hot";
    }
    if (cold) {
        return "cold";
    }
    return nullptr;
}

static bool ggml_backend_moe_dynamic_tensor_name_contains(
        const ggml_tensor * tensor,
        const char * needle) {
    for (const ggml_tensor * current = tensor; current != nullptr; current = current->view_src) {
        if (strstr(current->name, needle) != nullptr) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_moe_dynamic_split_has_compute(
        const ggml_backend_sched_split * split,
        const char * weight_name_prefix) {
    for (int node_index = 0; node_index < split->graph.n_nodes; ++node_index) {
        const ggml_tensor * node = split->graph.nodes[node_index];
        if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] == nullptr) {
            continue;
        }
        if (ggml_backend_moe_dynamic_tensor_name_contains(node->src[0], weight_name_prefix)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_moe_dynamic_split_has_hot_compute(
        const ggml_backend_sched_split * split) {
    return ggml_backend_moe_dynamic_split_has_compute(split, "ffn_moe_dynamic_hot_");
}

static bool ggml_backend_moe_dynamic_split_has_cold_compute(
        const ggml_backend_sched_split * split) {
    return ggml_backend_moe_dynamic_split_has_compute(split, "ffn_moe_dynamic_cold_");
}

static int32_t ggml_backend_moe_dynamic_layer_from_name(const char * name) {
    if (name == nullptr || strstr(name, "ffn_moe_dynamic_") == nullptr) {
        return -1;
    }
    const char * suffix = strrchr(name, '-');
    if (suffix == nullptr || suffix[1] == '\0') {
        return -1;
    }
    char * end = nullptr;
    const long parsed = strtol(suffix + 1, &end, 10);
    if (end == suffix + 1 || (*end != '\0' && *end != '#') || parsed < 0 || parsed > INT32_MAX) {
        return -1;
    }
    return (int32_t) parsed;
}

static int32_t ggml_backend_moe_dynamic_split_layer(const ggml_backend_sched_split * split) {
    for (int node_index = 0; node_index < split->graph.n_nodes; ++node_index) {
        const ggml_tensor * node = split->graph.nodes[node_index];
        int32_t layer = ggml_backend_moe_dynamic_layer_from_name(node->name);
        if (layer >= 0) {
            return layer;
        }
        for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
            const ggml_tensor * source = node->src[source_index];
            if (source == nullptr) {
                continue;
            }
            layer = ggml_backend_moe_dynamic_layer_from_name(source->name);
            if (layer >= 0) {
                return layer;
            }
        }
    }
    return -1;
}

static ggml_backend_moe_dynamic_registry & ggml_backend_moe_dynamic_registry_get() {
    static ggml_backend_moe_dynamic_registry registry;
    return registry;
}

static bool ggml_backend_moe_dynamic_layer_split_state(
        int32_t layer_id,
        bool * enabled,
        int32_t * ready_routes,
        int32_t * gpu_routes,
        int32_t * total_routes) {
    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto found = registry.layers.find(layer_id);
    if (found == registry.layers.end() || found->second.split_decisions == 0) {
        return false;
    }
    const auto & layer = found->second;
    if (enabled != nullptr) {
        *enabled = layer.gpu_split_enabled;
    }
    if (ready_routes != nullptr) {
        *ready_routes = layer.last_ready_routes;
    }
    if (gpu_routes != nullptr) {
        *gpu_routes = layer.last_gpu_routes;
    }
    if (total_routes != nullptr) {
        *total_routes = layer.last_total_routes;
    }
    return true;
}

static bool ggml_backend_moe_dynamic_tensor_is_skipped(
        const std::unordered_set<const ggml_tensor *> & skipped,
        const ggml_tensor * tensor) {
    for (const ggml_tensor * current = tensor; current != nullptr; current = current->view_src) {
        if (skipped.find(current) != skipped.end()) {
            return true;
        }
    }
    return false;
}

static double ggml_backend_moe_dynamic_env_double(const char * name, double fallback) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const double parsed = strtod(value, &end);
    return end != value && *end == '\0' && std::isfinite(parsed) ? parsed : fallback;
}

static int32_t ggml_backend_moe_dynamic_env_i32(const char * name, int32_t fallback) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < INT32_MIN || parsed > INT32_MAX) {
        return fallback;
    }
    return (int32_t) parsed;
}

static double ggml_backend_moe_dynamic_current_score(
        const ggml_backend_moe_dynamic_layer & layer,
        int32_t expert,
        double decay) {
    const uint64_t elapsed = layer.step - layer.last_score_step[(size_t) expert];
    return layer.score[(size_t) expert] * std::pow(decay, (double) elapsed);
}

static ggml_backend_moe_dynamic_layer & ggml_backend_moe_dynamic_get_layer(
        ggml_backend_moe_dynamic_registry & registry,
        int32_t layer_id,
        int32_t n_expert,
        int32_t n_slots) {
    auto & layer = registry.layers[layer_id];
    if (layer.n_expert != n_expert || layer.n_slots != n_slots) {
        layer = {};
        layer.n_expert = n_expert;
        layer.n_slots = n_slots;
        layer.slots.resize((size_t) n_slots);
        layer.slot_by_expert.assign((size_t) n_expert, -1);
        layer.score.assign((size_t) n_expert, 0.0);
        layer.last_score_step.assign((size_t) n_expert, 0);
        layer.previous_selected.clear();
        layer.transition_counts.assign((size_t) n_expert * (size_t) n_expert, 0);
        layer.cross_layer_transition_counts.assign((size_t) n_expert * (size_t) n_expert, 0);
        layer.cross_layer_source_counts.assign((size_t) n_expert, 0);
        layer.cross_layer2_transition_counts.assign((size_t) n_expert * (size_t) n_expert, 0);
        layer.cross_layer2_source_counts.assign((size_t) n_expert, 0);
        layer.cross_joint_transition_counts.assign((size_t) 512 * (size_t) n_expert, 0);
        layer.cross_joint_source_counts.assign((size_t) 512, 0);
        layer.ready_route_hist.assign((size_t) n_expert + 1, 0);
        layer.gpu_route_hist.assign((size_t) n_expert + 1, 0);
    }
    return layer;
}

static int32_t ggml_backend_moe_dynamic_spill_slots(
        const ggml_backend_moe_dynamic_layer & layer) {
    return std::clamp<int32_t>(
        ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_SPILL_SLOTS", 0),
        0,
        std::max<int32_t>(0, layer.n_slots - 1));
}

static int32_t ggml_backend_moe_dynamic_regular_slot_limit(
        const ggml_backend_moe_dynamic_layer & layer) {
    return layer.n_slots - ggml_backend_moe_dynamic_spill_slots(layer);
}

static std::vector<int32_t> ggml_backend_moe_dynamic_static_slot_map(
        int32_t layer_id,
        int32_t n_expert,
        int32_t n_slots) {
    const char * path_env = getenv("GGML_MOE_DYNAMIC_STATIC_MAP");
    if (path_env == nullptr || path_env[0] == '\0') {
        return {};
    }

    auto & map_registry = ggml_backend_sched_static_map_registry_get();
    std::lock_guard<std::mutex> map_lock(map_registry.mutex);
    const std::string path = path_env;
    ggml_backend_sched_static_map_load(map_registry, path);

    std::vector<int32_t> mapping;
    mapping.reserve((size_t) n_slots);
    std::vector<uint8_t> used((size_t) n_expert, 0);
    if (layer_id >= 0 && (size_t) layer_id < map_registry.experts_by_layer.size()) {
        for (int32_t expert : map_registry.experts_by_layer[(size_t) layer_id]) {
            if ((int32_t) mapping.size() >= n_slots) {
                break;
            }
            if (expert < 0 || expert >= n_expert || used[(size_t) expert]) {
                continue;
            }
            mapping.push_back(expert);
            used[(size_t) expert] = 1;
        }
    }
    for (int32_t expert = 0; expert < n_expert && (int32_t) mapping.size() < n_slots; ++expert) {
        if (!used[(size_t) expert]) {
            mapping.push_back(expert);
        }
    }
    GGML_ASSERT((int32_t) mapping.size() == n_slots);
    return mapping;
}

bool ggml_backend_moe_dynamic_prepare_static_map(
        int32_t layer_id,
        int32_t n_expert,
        int32_t n_slots) {
    GGML_ASSERT(layer_id >= 0 && n_expert > 0 && n_slots > 0 && n_slots <= n_expert);
    const char * path_env = getenv("GGML_MOE_DYNAMIC_STATIC_MAP");
    std::vector<int32_t> mapping = ggml_backend_moe_dynamic_static_slot_map(
        layer_id, n_expert, n_slots);
    if (mapping.empty()) {
        return false;
    }

    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto & layer = ggml_backend_moe_dynamic_get_layer(
        registry, layer_id, n_expert, n_slots);
    if (layer.static_frozen) {
        for (int32_t slot = 0; slot < n_slots; ++slot) {
            GGML_ASSERT(layer.slots[(size_t) slot].expert == mapping[(size_t) slot]);
        }
        return true;
    }

    layer.slot_by_expert.assign((size_t) n_expert, -1);
    for (int32_t slot = 0; slot < n_slots; ++slot) {
        const int32_t expert = mapping[(size_t) slot];
        auto & entry = layer.slots[(size_t) slot];
        entry = {};
        entry.expert = expert;
        entry.state = GGML_BACKEND_MOE_SLOT_COPYING;
        layer.slot_by_expert[(size_t) expert] = slot;
    }
    layer.active = true;
    layer.static_frozen = true;

    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"static_map_prepare\",\"layer\":%d,\"experts\":%d,"
        "\"slots\":%d,\"path\":\"%s\"",
        layer_id, n_expert, n_slots, path_env);
    GGML_LOG_INFO(
        "moe-static-frozen-map: layer=%d slots=%d/%d path=%s\n",
        layer_id, n_slots, n_expert, path_env);
    return true;
}

static void ggml_backend_moe_dynamic_update_cross_layer_model(
        ggml_backend_moe_dynamic_layer & target_layer,
        const std::vector<int32_t> & source_selected,
        const std::vector<int32_t> & target_selected,
        int32_t distance) {
    auto & transitions = distance == 1
        ? target_layer.cross_layer_transition_counts
        : target_layer.cross_layer2_transition_counts;
    auto & source_counts = distance == 1
        ? target_layer.cross_layer_source_counts
        : target_layer.cross_layer2_source_counts;
    GGML_ASSERT(distance == 1 || distance == 2);
    GGML_ASSERT(transitions.size() == (size_t) target_layer.n_expert * (size_t) target_layer.n_expert);
    GGML_ASSERT(source_counts.size() == (size_t) target_layer.n_expert);

    for (int32_t source : source_selected) {
        if (source < 0 || source >= target_layer.n_expert) {
            continue;
        }
        if (source_counts[(size_t) source] != UINT32_MAX) {
            source_counts[(size_t) source]++;
        }
        for (int32_t target : target_selected) {
            if (target < 0 || target >= target_layer.n_expert) {
                continue;
            }
            auto & count = transitions[(size_t) source * (size_t) target_layer.n_expert + (size_t) target];
            if (count != UINT16_MAX) {
                count++;
            }
        }
    }
}

static size_t ggml_backend_moe_dynamic_joint_bucket(int32_t expert_lminus2, int32_t expert_lminus1) {
    // 512 buckets keeps the per-layer model bounded at 256 KiB while retaining
    // enough context diversity for the observed eight-by-eight source sets.
    uint32_t value = (uint32_t) expert_lminus2 * 0x9e3779b1u;
    value ^= (uint32_t) expert_lminus1 * 0x85ebca6bu + 0x27d4eb2du;
    value ^= value >> 16;
    return (size_t) (value & 511u);
}

static void ggml_backend_moe_dynamic_maybe_decay_cross_layer_model(
        ggml_backend_moe_dynamic_layer & layer,
        uint64_t step) {
    const int32_t interval = std::max<int32_t>(0,
        ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_CROSS_LAYER_DECAY_INTERVAL", 0));
    if (interval == 0 || step < layer.last_cross_decay_step + (uint64_t) interval) {
        return;
    }
    auto decay_u16 = [](std::vector<uint16_t> & values) {
        for (auto & value : values) {
            value = (uint16_t) ((value + 1u) / 2u);
        }
    };
    auto decay_u32 = [](std::vector<uint32_t> & values) {
        for (auto & value : values) {
            value = (value + 1u) / 2u;
        }
    };
    decay_u16(layer.cross_layer_transition_counts);
    decay_u32(layer.cross_layer_source_counts);
    decay_u16(layer.cross_layer2_transition_counts);
    decay_u32(layer.cross_layer2_source_counts);
    if (ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_CROSS_LAYER_INTERACTION", 0) != 0) {
        decay_u16(layer.cross_joint_transition_counts);
        decay_u32(layer.cross_joint_source_counts);
    }
    layer.last_cross_decay_step = step;
    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"cross_layer_decay\",\"step\":%" PRIu64
        ",\"interval\":%d",
        step, interval);
}

static void ggml_backend_moe_dynamic_update_joint_model(
        ggml_backend_moe_dynamic_layer & target_layer,
        const std::vector<int32_t> & source_lminus2,
        const std::vector<int32_t> & source_lminus1,
        const std::vector<int32_t> & target_selected) {
    if (ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_CROSS_LAYER_INTERACTION", 0) == 0) {
        return;
    }
    std::array<uint8_t, 512> seen{};
    for (int32_t expert2 : source_lminus2) {
        if (expert2 < 0 || expert2 >= target_layer.n_expert) {
            continue;
        }
        for (int32_t expert1 : source_lminus1) {
            if (expert1 < 0 || expert1 >= target_layer.n_expert) {
                continue;
            }
            const size_t bucket = ggml_backend_moe_dynamic_joint_bucket(expert2, expert1);
            if (seen[bucket]) {
                continue;
            }
            seen[bucket] = 1;
            auto & observations = target_layer.cross_joint_source_counts[bucket];
            if (observations != UINT32_MAX) {
                observations++;
            }
            for (int32_t target : target_selected) {
                if (target < 0 || target >= target_layer.n_expert) {
                    continue;
                }
                auto & count = target_layer.cross_joint_transition_counts[
                    bucket * (size_t) target_layer.n_expert + (size_t) target];
                if (count != UINT16_MAX) {
                    count++;
                }
            }
        }
    }
}

static int32_t ggml_backend_moe_dynamic_cross_layer_prefetch(
        ggml_backend_moe_dynamic_registry & registry,
        int32_t source_layer_id,
        int32_t target_layer_id,
        int32_t distance,
        uint64_t step,
        const std::vector<int32_t> & source_selected) {
    GGML_ASSERT(distance == 1 || distance == 2);
    const int32_t per_layer_limit = std::max<int32_t>(0,
        ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_PER_LAYER", 0));
    const int32_t total_limit = std::max<int32_t>(0,
        ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_CROSS_LAYER_PREFETCH_TOTAL", 8));
    const int32_t min_observations = std::max<int32_t>(1,
        ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_CROSS_LAYER_MIN_OBSERVATIONS", 8));
    if (per_layer_limit == 0 || total_limit == 0 || source_selected.empty()) {
        return 0;
    }

    auto found = registry.layers.find(target_layer_id);
    if (found == registry.layers.end()) {
        return 0;
    }
    auto & target_layer = found->second;
    if (!target_layer.active || target_layer.static_frozen || target_layer.n_slots <= 0) {
        return 0;
    }
    ggml_backend_moe_dynamic_maybe_decay_cross_layer_model(target_layer, step);

    if (registry.cross_predictor_step != step) {
        registry.cross_predictor_step = step;
        registry.cross_predictor_admissions_this_step = 0;
    }
    if (registry.cross_predictor_admissions_this_step >= total_limit) {
        return 0;
    }
    // Rotate the global budget across target layers. Without this gate, the
    // first layers encountered in graph order would consume every token's
    // speculative upload budget.
    if ((int32_t) registry.layers.size() > total_limit) {
        std::vector<int32_t> layer_ids;
        layer_ids.reserve(registry.layers.size());
        for (const auto & item : registry.layers) {
            layer_ids.push_back(item.first);
        }
        std::sort(layer_ids.begin(), layer_ids.end());
        const size_t start = (size_t) ((step - 1) * (uint64_t) total_limit) % layer_ids.size();
        bool eligible = false;
        for (int32_t offset = 0; offset < total_limit; ++offset) {
            if (layer_ids[(start + (size_t) offset) % layer_ids.size()] == target_layer_id) {
                eligible = true;
                break;
            }
        }
        if (!eligible) {
            return 0;
        }
    }

    const auto & transitions = distance == 1
        ? target_layer.cross_layer_transition_counts
        : target_layer.cross_layer2_transition_counts;
    const auto & source_counts = distance == 1
        ? target_layer.cross_layer_source_counts
        : target_layer.cross_layer2_source_counts;
    std::vector<double> scores((size_t) target_layer.n_expert, 0.0);
    int32_t eligible_sources = 0;
    for (int32_t source : source_selected) {
        if (source < 0 || source >= target_layer.n_expert) {
            continue;
        }
        const uint32_t observations = source_counts[(size_t) source];
        if (observations < (uint32_t) min_observations) {
            continue;
        }
        eligible_sources++;
        for (int32_t expert = 0; expert < target_layer.n_expert; ++expert) {
            const uint16_t count = transitions[
                (size_t) source * (size_t) target_layer.n_expert + (size_t) expert];
            if (count > 0) {
                scores[(size_t) expert] += (double) count / (double) observations;
            }
        }
    }
    int32_t eligible_joint_buckets = 0;
    if (distance == 1 &&
        ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_CROSS_LAYER_INTERACTION", 0) != 0) {
        auto earlier = registry.current_routes.find(source_layer_id - 1);
        if (earlier != registry.current_routes.end() && earlier->second.step == step) {
            const double interaction_weight = ggml_backend_moe_dynamic_env_double(
                "GGML_MOE_DYNAMIC_CROSS_LAYER_INTERACTION_WEIGHT", 0.25);
            std::array<uint8_t, 512> seen{};
            for (int32_t expert2 : earlier->second.selected) {
                for (int32_t expert1 : source_selected) {
                    const size_t bucket = ggml_backend_moe_dynamic_joint_bucket(expert2, expert1);
                    if (seen[bucket]) {
                        continue;
                    }
                    seen[bucket] = 1;
                    const uint32_t observations = target_layer.cross_joint_source_counts[bucket];
                    if (observations < (uint32_t) min_observations) {
                        continue;
                    }
                    eligible_joint_buckets++;
                    for (int32_t expert = 0; expert < target_layer.n_expert; ++expert) {
                        const uint16_t count = target_layer.cross_joint_transition_counts[
                            bucket * (size_t) target_layer.n_expert + (size_t) expert];
                        if (count > 0) {
                            scores[(size_t) expert] += interaction_weight *
                                (double) count / (double) observations;
                        }
                    }
                }
            }
        }
    }
    if (eligible_sources == 0 && eligible_joint_buckets == 0) {
        return 0;
    }

    std::vector<std::pair<double, int32_t>> predicted;
    predicted.reserve((size_t) target_layer.n_expert);
    for (int32_t expert = 0; expert < target_layer.n_expert; ++expert) {
        if (target_layer.slot_by_expert[(size_t) expert] < 0 && scores[(size_t) expert] > 0.0) {
            predicted.emplace_back(scores[(size_t) expert], expert);
        }
    }
    std::sort(predicted.begin(), predicted.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first > rhs.first;
        }
        return lhs.second < rhs.second;
    });

    const double minimum_prediction_score = ggml_backend_moe_dynamic_env_double(
        "GGML_MOE_DYNAMIC_CROSS_LAYER_MIN_SCORE", 0.0);
    int32_t admitted = 0;
    for (const auto & prediction : predicted) {
        if (prediction.first < minimum_prediction_score) {
            break;
        }
        if (admitted >= per_layer_limit ||
            registry.cross_predictor_admissions_this_step >= total_limit) {
            break;
        }
        int32_t destination = -1;
        int32_t victim_expert = -1;
        const int32_t spill_slots = ggml_backend_moe_dynamic_spill_slots(target_layer);
        const int32_t spill_begin = target_layer.n_slots - spill_slots;
        const int32_t search_begin = spill_slots > 0 ? spill_begin : 0;
        for (int32_t slot_index = search_begin; slot_index < target_layer.n_slots; ++slot_index) {
            if (target_layer.slots[(size_t) slot_index].state == GGML_BACKEND_MOE_SLOT_EMPTY) {
                destination = slot_index;
                break;
            }
        }
        if (destination < 0 && spill_slots > 0) {
            uint64_t oldest_use = UINT64_MAX;
            for (int32_t slot_index = spill_begin; slot_index < target_layer.n_slots; ++slot_index) {
                const auto & candidate = target_layer.slots[(size_t) slot_index];
                if (candidate.state != GGML_BACKEND_MOE_SLOT_READY ||
                    candidate.admitted_step >= step) {
                    continue;
                }
                const uint64_t last_use = candidate.last_used_step > 0
                    ? candidate.last_used_step
                    : candidate.admitted_step;
                if (destination < 0 || last_use < oldest_use) {
                    destination = slot_index;
                    victim_expert = candidate.expert;
                    oldest_use = last_use;
                }
            }
        }
        if (destination < 0) {
            break;
        }

        const int32_t expert = prediction.second;
        if (victim_expert >= 0) {
            target_layer.slot_by_expert[(size_t) victim_expert] = -1;
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"spill_evict\",\"layer\":%d,\"step\":%" PRIu64
                ",\"slot\":%d,\"expert\":%d,\"replacement\":%d,"
                "\"distance\":%d",
                target_layer_id, step, destination, victim_expert, expert, distance);
        }
        auto & slot = target_layer.slots[(size_t) destination];
        slot.expert = expert;
        slot.state = GGML_BACKEND_MOE_SLOT_COPYING;
        slot.components_enqueued = 0;
        slot.components_completed = 0;
        slot.promotion_bundle_id = 0;
        slot.promotion_queued_mono_ns = 0;
        slot.prediction_step = 0;
        slot.prediction_distance = 0;
        slot.admitted_step = step;
        slot.last_used_step = 0;
        slot.hit_count = 0;
        slot.protected_segment = false;
        target_layer.slot_by_expert[(size_t) expert] = destination;
        target_layer.admissions++;
        target_layer.cross_prediction_admissions++;
        admitted++;
        registry.cross_predictor_admissions_this_step++;
        const bool urgent_queued = ggml_backend_moe_dynamic_issue_urgent_locked(
            registry, target_layer_id, destination, expert, distance, step);
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"admit\",\"policy\":\"cross_layer_transition_prefetch\","
            "\"source_layer\":%d,\"layer\":%d,\"distance\":%d,"
            "\"step\":%" PRIu64 ",\"expert\":%d,\"prediction_score\":%.8g,"
            "\"eligible_sources\":%d,\"eligible_joint_buckets\":%d,"
            "\"slot\":%d,\"victim_expert\":%d,\"spill_slots\":%d,"
            "\"urgent_queued\":%s",
            source_layer_id, target_layer_id, distance, step, expert,
            prediction.first, eligible_sources, eligible_joint_buckets, destination,
            victim_expert, spill_slots, urgent_queued ? "true" : "false");
    }
    target_layer.cross_prediction_steps++;
    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"cross_layer_predict_prefetch\",\"source_layer\":%d,"
        "\"layer\":%d,\"distance\":%d,\"step\":%" PRIu64
        ",\"candidates\":%zu,\"admitted\":%d,\"eligible_sources\":%d,"
        "\"eligible_joint_buckets\":%d,\"per_layer_limit\":%d,"
        "\"global_admitted\":%d,\"global_limit\":%d",
        source_layer_id, target_layer_id, distance, step, predicted.size(), admitted,
        eligible_sources, eligible_joint_buckets, per_layer_limit,
        registry.cross_predictor_admissions_this_step, total_limit);
    return admitted;
}

void ggml_backend_moe_dynamic_observe_routes(
        int32_t layer_id,
        int32_t n_expert,
        int32_t n_slots,
        int32_t n_routes_per_token,
        const int32_t * ids,
        int64_t n_ids) {
    GGML_ASSERT(layer_id >= 0 && n_expert > 0 && n_slots > 0 && n_slots <= n_expert);
    GGML_ASSERT(n_routes_per_token > 0 && n_ids % n_routes_per_token == 0);
    GGML_ASSERT(ids != nullptr && n_ids > 0);

    const double decay = ggml_backend_moe_dynamic_env_double("GGML_MOE_DYNAMIC_DECAY", 0.98);
    const int32_t warmup_steps = std::max<int32_t>(1,
        ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_LAYER_WARMUP", 16));
    const double activation_coverage = ggml_backend_moe_dynamic_env_double(
        "GGML_MOE_DYNAMIC_LAYER_COVERAGE", 0.50);

    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto & layer = ggml_backend_moe_dynamic_get_layer(
        registry, layer_id, n_expert, n_slots);
    const bool prefill_batch = n_ids > n_routes_per_token;
    if (layer.active && !prefill_batch) {
        return;
    }
    layer.step++;
    if (prefill_batch) {
        layer.prefill_batches++;
        layer.prefill_total_routes += (uint64_t) n_ids;
    }

    for (int64_t index = 0; index < n_ids; ++index) {
        const int32_t expert = ids[index];
        GGML_ASSERT(expert >= 0 && expert < n_expert);
        const double prior = ggml_backend_moe_dynamic_current_score(layer, expert, decay);
        layer.score[(size_t) expert] = prior + 1.0;
        layer.last_score_step[(size_t) expert] = layer.step;
    }

    std::vector<double> scores((size_t) n_expert);
    double total = 0.0;
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        const double score = ggml_backend_moe_dynamic_current_score(layer, expert, decay);
        scores[(size_t) expert] = score;
        total += score;
    }
    const int32_t top_count = std::min<int32_t>(n_slots, n_expert);
    if (top_count < n_expert) {
        std::nth_element(
            scores.begin(), scores.begin() + top_count, scores.end(), std::greater<double>());
    }
    double top_total = 0.0;
    for (int32_t index = 0; index < top_count; ++index) {
        top_total += scores[(size_t) index];
    }
    layer.observed_coverage = total > 0.0 ? top_total / total : 0.0;

    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"observe\",\"phase\":\"%s\",\"layer\":%d,\"step\":%" PRIu64
        ",\"tokens\":%" PRIi64 ",\"routes\":%" PRIi64
        ",\"coverage\":%.8g,\"threshold\":%.8g,\"warmup\":%d",
        prefill_batch ? "prefill" : "decode",
        layer_id, layer.step, n_ids / n_routes_per_token, n_ids,
        layer.observed_coverage, activation_coverage, warmup_steps);

    if (layer.step >= (uint64_t) warmup_steps &&
        layer.observed_coverage >= activation_coverage) {
        layer.active = true;
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"layer_activate\",\"layer\":%d,\"step\":%" PRIu64
            ",\"coverage\":%.8g,\"threshold\":%.8g,\"slots\":%d",
            layer_id, layer.step, layer.observed_coverage, activation_coverage, n_slots);
    }
}

bool ggml_backend_moe_dynamic_layer_is_active(
        int32_t layer_id,
        int32_t n_expert,
        int32_t n_slots) {
    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto found = registry.layers.find(layer_id);
    return found != registry.layers.end() &&
        found->second.n_expert == n_expert &&
        found->second.n_slots == n_slots &&
        found->second.active;
}

bool ggml_backend_moe_dynamic_get_slot_map(
        int32_t layer_id,
        int32_t * slot_map,
        int32_t n_expert) {
    GGML_ASSERT(slot_map != nullptr && n_expert > 0);
    std::fill(slot_map, slot_map + n_expert, -1);
    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto found = registry.layers.find(layer_id);
    if (found == registry.layers.end() || found->second.n_expert != n_expert) {
        return false;
    }
    auto & layer = found->second;
    const bool gpu_route_map =
        ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_GPU_ROUTE_MAP", 0) != 0;
    if (gpu_route_map && !layer.static_frozen) {
        const int32_t warm_start_per_layer = std::max<int32_t>(0,
            ggml_backend_moe_dynamic_env_i32(
                "GGML_MOE_DYNAMIC_GPU_MAP_WARM_START_PER_LAYER",
                std::min<int32_t>(16, layer.n_slots)));
        int32_t resident_or_copying = 0;
        for (const auto & slot : layer.slots) {
            if (slot.state != GGML_BACKEND_MOE_SLOT_EMPTY) {
                resident_or_copying++;
            }
        }
        if (resident_or_copying < warm_start_per_layer && layer.prefill_batches > 0) {
            std::vector<int32_t> ranked((size_t) n_expert);
            std::iota(ranked.begin(), ranked.end(), 0);
            std::sort(ranked.begin(), ranked.end(), [&](int32_t lhs, int32_t rhs) {
                if (layer.score[(size_t) lhs] != layer.score[(size_t) rhs]) {
                    return layer.score[(size_t) lhs] > layer.score[(size_t) rhs];
                }
                return lhs < rhs;
            });
            for (int32_t expert : ranked) {
                if (resident_or_copying >= warm_start_per_layer) {
                    break;
                }
                if (layer.score[(size_t) expert] <= 0.0 ||
                    layer.slot_by_expert[(size_t) expert] >= 0) {
                    continue;
                }
                int32_t destination = -1;
                for (int32_t slot_index = 0;
                     slot_index < ggml_backend_moe_dynamic_regular_slot_limit(layer);
                     ++slot_index) {
                    if (layer.slots[(size_t) slot_index].state == GGML_BACKEND_MOE_SLOT_EMPTY) {
                        destination = slot_index;
                        break;
                    }
                }
                if (destination < 0) {
                    break;
                }
                auto & slot = layer.slots[(size_t) destination];
                slot.expert = expert;
                slot.state = GGML_BACKEND_MOE_SLOT_COPYING;
                slot.components_enqueued = 0;
                slot.components_completed = 0;
                slot.promotion_bundle_id = 0;
                slot.promotion_queued_mono_ns = 0;
                slot.prediction_step = 0;
                slot.prediction_distance = 0;
                slot.admitted_step = layer.step;
                slot.last_used_step = 0;
                slot.hit_count = 0;
                slot.protected_segment = false;
                layer.slot_by_expert[(size_t) expert] = destination;
                layer.admissions++;
                resident_or_copying++;
                ggml_backend_moe_dynamic_tracef(
                    "\"event\":\"admit\",\"policy\":\"gpu_map_prefill_static\","
                    "\"layer\":%d,\"step\":%" PRIu64 ",\"expert\":%d,"
                    "\"score\":%.8g,\"slot\":%d,\"victim_expert\":-1",
                    layer_id, layer.step, expert, layer.score[(size_t) expert], destination);
            }
            layer.active = true;
        }
    }
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        const int32_t slot = layer.slot_by_expert[(size_t) expert];
        if (slot >= 0 && layer.slots[(size_t) slot].state == GGML_BACKEND_MOE_SLOT_READY) {
            slot_map[expert] = slot;
        }
    }
    return true;
}

void ggml_backend_moe_dynamic_split_routes(
        int32_t layer_id,
        int32_t n_expert,
        int32_t n_slots,
        int32_t n_routes_per_token,
        const int32_t * ids,
        int64_t n_ids,
        int32_t * hot_ids,
        int32_t * cold_ids) {
    GGML_ASSERT(layer_id >= 0 && n_expert > 0 && n_slots > 0 && n_slots <= n_expert);
    GGML_ASSERT(n_routes_per_token > 0 && n_ids % n_routes_per_token == 0);
    GGML_ASSERT(ids != nullptr && hot_ids != nullptr && cold_ids != nullptr && n_ids > 0);

    const double decay = ggml_backend_moe_dynamic_env_double("GGML_MOE_DYNAMIC_DECAY", 0.98);
    const double threshold = ggml_backend_moe_dynamic_env_double("GGML_MOE_DYNAMIC_THRESHOLD", 3.0);
    const double admission_ratio = ggml_backend_moe_dynamic_env_double("GGML_MOE_DYNAMIC_ADMISSION_RATIO", 1.10);
    const int32_t min_hot_routes = std::max<int32_t>(1, (int32_t)
        ggml_backend_moe_dynamic_env_double("GGML_MOE_DYNAMIC_MIN_HOT_ROUTES", 1.0));

    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto & layer = ggml_backend_moe_dynamic_get_layer(
        registry, layer_id, n_expert, n_slots);
    layer.active = true;
    layer.step++;
    const bool decode_step = n_ids == n_routes_per_token;
    const bool first_decode_after_prefill =
        decode_step && layer.decode_steps == 0 && layer.prefill_batches > 0;

    std::vector<uint8_t> selected((size_t) n_expert, 0);
    std::vector<double> selected_scores((size_t) n_ids, 0.0);
    int32_t best_candidate = -1;
    double best_candidate_score = -INFINITY;
    int32_t ready_routes = 0;
    const uint64_t route_decision_mono_ns = ggml_backend_moe_dynamic_mono_ns();

    for (int64_t index = 0; index < n_ids; ++index) {
        const int32_t expert = ids[index];
        GGML_ASSERT(expert >= 0 && expert < n_expert);
        selected[(size_t) expert] = 1;

        const double prior = ggml_backend_moe_dynamic_current_score(layer, expert, decay);
        const double updated = prior + 1.0;
        selected_scores[(size_t) index] = updated;
        layer.score[(size_t) expert] = updated;
        layer.last_score_step[(size_t) expert] = layer.step;

        const int32_t slot = layer.slot_by_expert[(size_t) expert];
        if (slot >= 0 && layer.slots[(size_t) slot].state == GGML_BACKEND_MOE_SLOT_READY) {
            auto & resident = layer.slots[(size_t) slot];
            resident.last_used_step = layer.step;
            resident.hit_count++;
            const uint32_t protected_hits = (uint32_t) std::max<int32_t>(2,
                ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_PROTECTED_HITS", 2));
            resident.protected_segment = resident.hit_count >= protected_hits;
            hot_ids[index] = slot;
            cold_ids[index] = -1;
            ready_routes++;
        } else {
            hot_ids[index] = -1;
            cold_ids[index] = expert;
            if (slot < 0 && updated >= threshold && updated > best_candidate_score) {
                best_candidate = expert;
                best_candidate_score = updated;
            }
        }
    }

    if (first_decode_after_prefill && !layer.static_frozen) {
        const int32_t warm_start_per_layer = std::max<int32_t>(0,
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_WARM_START_PER_LAYER", 0));
        const int32_t warm_start_total = std::max<int32_t>(0,
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_WARM_START_TOTAL", 128));
        if (registry.warm_start_step != layer.step) {
            registry.warm_start_step = layer.step;
            registry.warm_start_admissions_this_step = 0;
        }

        std::vector<int32_t> ranked((size_t) n_expert);
        std::iota(ranked.begin(), ranked.end(), 0);
        std::sort(ranked.begin(), ranked.end(), [&](int32_t lhs, int32_t rhs) {
            const double lhs_score = ggml_backend_moe_dynamic_current_score(layer, lhs, decay);
            const double rhs_score = ggml_backend_moe_dynamic_current_score(layer, rhs, decay);
            if (lhs_score != rhs_score) {
                return lhs_score > rhs_score;
            }
            return lhs < rhs;
        });

        int32_t admitted = 0;
        for (int32_t expert : ranked) {
            if (admitted >= warm_start_per_layer ||
                registry.warm_start_admissions_this_step >= warm_start_total) {
                break;
            }
            if (layer.slot_by_expert[(size_t) expert] >= 0) {
                continue;
            }
            const double score = ggml_backend_moe_dynamic_current_score(layer, expert, decay);
            if (score <= 0.0) {
                break;
            }
            int32_t destination = -1;
            for (int32_t slot_index = 0;
                 slot_index < ggml_backend_moe_dynamic_regular_slot_limit(layer);
                 ++slot_index) {
                if (layer.slots[(size_t) slot_index].state == GGML_BACKEND_MOE_SLOT_EMPTY) {
                    destination = slot_index;
                    break;
                }
            }
            if (destination < 0) {
                break;
            }
            auto & slot = layer.slots[(size_t) destination];
            slot.expert = expert;
            slot.state = GGML_BACKEND_MOE_SLOT_COPYING;
            slot.components_enqueued = 0;
            slot.components_completed = 0;
            slot.promotion_bundle_id = 0;
            slot.promotion_queued_mono_ns = 0;
            slot.prediction_step = 0;
            slot.prediction_distance = 0;
            slot.admitted_step = layer.step;
            slot.last_used_step = 0;
            slot.hit_count = 0;
            slot.protected_segment = false;
            layer.slot_by_expert[(size_t) expert] = destination;
            layer.admissions++;
            admitted++;
            registry.warm_start_admissions_this_step++;
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"admit\",\"policy\":\"prefill_warm_start\","
                "\"layer\":%d,\"step\":%" PRIu64 ",\"expert\":%d,"
                "\"score\":%.8g,\"slot\":%d,\"victim_expert\":-1",
                layer_id, layer.step, expert, score, destination);
        }
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"warm_start\",\"layer\":%d,\"step\":%" PRIu64
            ",\"admitted\":%d,\"per_layer_limit\":%d,\"global_admitted\":%d,"
            "\"global_limit\":%d,\"prefill_batches\":%" PRIu64,
            layer_id, layer.step, admitted, warm_start_per_layer,
            registry.warm_start_admissions_this_step, warm_start_total,
            layer.prefill_batches);
        if (best_candidate >= 0 && layer.slot_by_expert[(size_t) best_candidate] >= 0) {
            best_candidate = -1;
            best_candidate_score = -INFINITY;
        }
    }

    if (decode_step) {
        std::vector<int32_t> current_selected;
        current_selected.reserve((size_t) n_routes_per_token);
        for (int64_t index = 0; index < n_ids; ++index) {
            if (std::find(current_selected.begin(), current_selected.end(), ids[index]) == current_selected.end()) {
                current_selected.push_back(ids[index]);
            }
        }

        // Train the target layer's same-token cross-layer models from routes
        // already observed earlier in this token. Predictions were made before
        // this target route was available, so this remains prequential.
        ggml_backend_moe_dynamic_maybe_decay_cross_layer_model(layer, layer.step);
        for (int32_t distance = 1; distance <= 2; ++distance) {
            auto source_found = registry.current_routes.find(layer_id - distance);
            if (source_found != registry.current_routes.end() &&
                source_found->second.step == layer.step) {
                ggml_backend_moe_dynamic_update_cross_layer_model(
                    layer, source_found->second.selected, current_selected, distance);
            }
        }
        auto source_lminus1 = registry.current_routes.find(layer_id - 1);
        auto source_lminus2 = registry.current_routes.find(layer_id - 2);
        if (source_lminus1 != registry.current_routes.end() &&
            source_lminus2 != registry.current_routes.end() &&
            source_lminus1->second.step == layer.step &&
            source_lminus2->second.step == layer.step) {
            ggml_backend_moe_dynamic_update_joint_model(
                layer,
                source_lminus2->second.selected,
                source_lminus1->second.selected,
                current_selected);
        }
        registry.current_routes[layer_id] = { layer.step, current_selected };

        // The original implementation issues distance two first. That gives the
        // farther target more lead, but it often lets that copy occupy the single
        // promotion stream before the distance-one request is even enqueued. Keep
        // the old order as the default and expose a controlled near-deadline-first
        // experiment rather than changing benchmark semantics silently.
        const bool near_deadline_first = ggml_backend_moe_dynamic_env_i32(
            "GGML_MOE_DYNAMIC_CROSS_LAYER_NEAR_FIRST", 0) != 0;
        if (near_deadline_first) {
            ggml_backend_moe_dynamic_cross_layer_prefetch(
                registry, layer_id, layer_id + 1, 1, layer.step, current_selected);
            ggml_backend_moe_dynamic_cross_layer_prefetch(
                registry, layer_id, layer_id + 2, 2, layer.step, current_selected);
        } else {
            ggml_backend_moe_dynamic_cross_layer_prefetch(
                registry, layer_id, layer_id + 2, 2, layer.step, current_selected);
            ggml_backend_moe_dynamic_cross_layer_prefetch(
                registry, layer_id, layer_id + 1, 1, layer.step, current_selected);
        }

        if (!layer.previous_selected.empty()) {
            for (int32_t prior : layer.previous_selected) {
                for (int32_t current : current_selected) {
                    auto & count = layer.transition_counts[
                        (size_t) prior * (size_t) n_expert + (size_t) current];
                    if (count != UINT16_MAX) {
                        count++;
                    }
                }
            }
        }

        const int32_t predict_per_layer = std::max<int32_t>(0,
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_PREDICT_PREFETCH_PER_LAYER", 0));
        const int32_t predict_total = std::max<int32_t>(0,
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_PREDICT_PREFETCH_TOTAL", 32));
        if (registry.predictor_step != layer.step) {
            registry.predictor_step = layer.step;
            registry.predictor_admissions_this_step = 0;
        }

        if (!layer.static_frozen && predict_per_layer > 0 &&
            registry.predictor_admissions_this_step < predict_total) {
            std::vector<std::pair<uint64_t, int32_t>> predicted;
            predicted.reserve((size_t) n_expert);
            for (int32_t expert = 0; expert < n_expert; ++expert) {
                if (layer.slot_by_expert[(size_t) expert] >= 0) {
                    continue;
                }
                uint64_t transition_score = 0;
                for (int32_t source : current_selected) {
                    transition_score += layer.transition_counts[
                        (size_t) source * (size_t) n_expert + (size_t) expert];
                }
                if (transition_score > 0) {
                    predicted.emplace_back(transition_score, expert);
                }
            }
            std::sort(predicted.begin(), predicted.end(), [](const auto & lhs, const auto & rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first > rhs.first;
                }
                return lhs.second < rhs.second;
            });

            int32_t admitted = 0;
            for (const auto & prediction : predicted) {
                if (admitted >= predict_per_layer ||
                    registry.predictor_admissions_this_step >= predict_total) {
                    break;
                }
                int32_t destination = -1;
                for (int32_t slot_index = 0;
                     slot_index < ggml_backend_moe_dynamic_regular_slot_limit(layer);
                     ++slot_index) {
                    if (layer.slots[(size_t) slot_index].state == GGML_BACKEND_MOE_SLOT_EMPTY) {
                        destination = slot_index;
                        break;
                    }
                }
                if (destination < 0) {
                    break;
                }
                const int32_t expert = prediction.second;
                auto & slot = layer.slots[(size_t) destination];
                slot.expert = expert;
                slot.state = GGML_BACKEND_MOE_SLOT_COPYING;
                slot.components_enqueued = 0;
                slot.components_completed = 0;
                slot.promotion_bundle_id = 0;
                slot.promotion_queued_mono_ns = 0;
                slot.prediction_step = 0;
                slot.prediction_distance = 0;
                slot.admitted_step = layer.step;
                slot.last_used_step = 0;
                slot.hit_count = 0;
                slot.protected_segment = false;
                layer.slot_by_expert[(size_t) expert] = destination;
                layer.admissions++;
                layer.prediction_admissions++;
                admitted++;
                registry.predictor_admissions_this_step++;
                ggml_backend_moe_dynamic_tracef(
                    "\"event\":\"admit\",\"policy\":\"same_layer_transition_prefetch\","
                    "\"layer\":%d,\"step\":%" PRIu64 ",\"expert\":%d,"
                    "\"prediction_score\":%" PRIu64 ",\"slot\":%d,\"victim_expert\":-1",
                    layer_id, layer.step, expert, prediction.first, destination);
            }
            layer.prediction_steps++;
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"predict_prefetch\",\"layer\":%d,\"step\":%" PRIu64
                ",\"candidates\":%zu,\"admitted\":%d,\"per_layer_limit\":%d,"
                "\"global_admitted\":%d,\"global_limit\":%d",
                layer_id, layer.step, predicted.size(), admitted, predict_per_layer,
                registry.predictor_admissions_this_step, predict_total);
        }
        layer.previous_selected = std::move(current_selected);
        if (best_candidate >= 0 && layer.slot_by_expert[(size_t) best_candidate] >= 0) {
            best_candidate = -1;
            best_candidate_score = -INFINITY;
        }
    }

    const bool use_gpu_routes = ready_routes >= min_hot_routes;
    if (!use_gpu_routes) {
        for (int64_t index = 0; index < n_ids; ++index) {
            hot_ids[index] = -1;
            cold_ids[index] = ids[index];
        }
    }
    const int32_t gpu_route_count = use_gpu_routes ? ready_routes : 0;
    GGML_ASSERT(ready_routes >= 0 && ready_routes <= n_ids);
    GGML_ASSERT(gpu_route_count >= 0 && gpu_route_count <= n_ids);
    layer.split_decisions++;
    layer.split_enabled += use_gpu_routes;
    layer.last_ready_routes = ready_routes;
    layer.last_gpu_routes = gpu_route_count;
    layer.last_total_routes = (int32_t) n_ids;
    layer.gpu_split_enabled = use_gpu_routes;
    layer.resident_hits += (uint64_t) ready_routes;
    layer.resident_misses += (uint64_t) (n_ids - ready_routes);
    layer.ready_hits += (uint64_t) gpu_route_count;
    layer.cpu_misses += (uint64_t) (n_ids - gpu_route_count);

    if (decode_step) {
        layer.decode_steps++;
        layer.full_gpu_layer_steps += gpu_route_count == n_ids;
        layer.cpu_dependent_layer_steps += gpu_route_count < n_ids;
        layer.ready_route_hist[(size_t) ready_routes]++;
        layer.gpu_route_hist[(size_t) gpu_route_count]++;

        auto & token_coverage = registry.token_coverage[layer.step];
        token_coverage.layers_seen++;
        token_coverage.full_gpu_layers += gpu_route_count == n_ids;
        token_coverage.cpu_dependent_layers += gpu_route_count < n_ids;
        token_coverage.gpu_routes += (uint64_t) gpu_route_count;
        token_coverage.total_routes += (uint64_t) n_ids;
    } else {
        layer.prefill_batches++;
        layer.prefill_ready_routes += (uint64_t) ready_routes;
        layer.prefill_total_routes += (uint64_t) n_ids;
    }

    if (decode_step) {
        std::ostringstream trace;
        trace << "\"event\":\"route\",\"phase\":\"decode\",\"layer\":" << layer_id
              << ",\"step\":" << layer.step
              << ",\"route_decision_mono_ns\":" << route_decision_mono_ns
              << ",\"selected\":[";
        for (int64_t index = 0; index < n_ids; ++index) {
            if (index > 0) {
                trace << ',';
            }
            trace << ids[index];
        }
        trace << "],\"hot_slots\":[";
        for (int64_t index = 0; index < n_ids; ++index) {
            if (index > 0) {
                trace << ',';
            }
            trace << hot_ids[index];
        }
        trace << "],\"cold_experts\":[";
        for (int64_t index = 0; index < n_ids; ++index) {
            if (index > 0) {
                trace << ',';
            }
            trace << cold_ids[index];
        }
        trace << "],\"scores\":[";
        for (int64_t index = 0; index < n_ids; ++index) {
            if (index > 0) {
                trace << ',';
            }
            trace << selected_scores[(size_t) index];
        }
        trace << "],\"ready_available\":" << ready_routes
              << ",\"gpu_routes\":" << gpu_route_count
              << ",\"min_hot_routes\":" << min_hot_routes
              << ",\"best_candidate\":" << best_candidate
              << ",\"best_candidate_score\":";
        if (std::isfinite(best_candidate_score)) {
            trace << best_candidate_score;
        } else {
            trace << "null";
        }
        ggml_backend_moe_dynamic_tracef("%s", trace.str().c_str());
    } else {
        const int32_t unique_experts = (int32_t)
            std::count(selected.begin(), selected.end(), (uint8_t) 1);
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"route_batch\",\"phase\":\"prefill\",\"layer\":%d,"
            "\"step\":%" PRIu64 ",\"tokens\":%" PRIi64 ",\"routes\":%" PRIi64
            ",\"unique_experts\":%d,\"ready_available\":%d,\"gpu_routes\":%d,"
            "\"min_hot_routes\":%d,\"best_candidate\":%d,\"best_candidate_score\":%.8g",
            layer_id, layer.step, n_ids / n_routes_per_token, n_ids,
            unique_experts, ready_routes, gpu_route_count, min_hot_routes,
            best_candidate, best_candidate_score);
    }

    // Admit at most one candidate per layer and token. A separate rotating global
    // cap prevents all host layers from promoting simultaneously after two touches.
    // The startup-static ceiling experiment is immutable by construction.
    if (layer.static_frozen || best_candidate < 0) {
        return;
    }

    const int32_t max_admissions_per_token =
        ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_MAX_ADMISSIONS_PER_TOKEN", 8);
    if (registry.admission_step != layer.step) {
        registry.admission_step = layer.step;
        registry.admissions_this_step = 0;
    }
    if (max_admissions_per_token == 0) {
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"reject\",\"layer\":%d,\"step\":%" PRIu64
            ",\"expert\":%d,\"score\":%.8g,\"reason\":\"global_admissions_disabled\"",
            layer_id, layer.step, best_candidate, best_candidate_score);
        return;
    }
    if (max_admissions_per_token > 0) {
        std::vector<int32_t> layer_ids;
        layer_ids.reserve(registry.layers.size());
        for (const auto & item : registry.layers) {
            layer_ids.push_back(item.first);
        }
        std::sort(layer_ids.begin(), layer_ids.end());

        bool eligible = layer_ids.size() <= (size_t) max_admissions_per_token;
        if (!eligible && !layer_ids.empty()) {
            const size_t start = (size_t) ((layer.step - 1) * (uint64_t) max_admissions_per_token) %
                layer_ids.size();
            for (int32_t offset = 0; offset < max_admissions_per_token; ++offset) {
                if (layer_ids[(start + (size_t) offset) % layer_ids.size()] == layer_id) {
                    eligible = true;
                    break;
                }
            }
        }
        if (!eligible || registry.admissions_this_step >= max_admissions_per_token) {
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"reject\",\"layer\":%d,\"step\":%" PRIu64
                ",\"expert\":%d,\"score\":%.8g,\"reason\":\"global_admission_cap\","
                "\"admissions_this_step\":%d,\"max_admissions\":%d",
                layer_id, layer.step, best_candidate, best_candidate_score,
                registry.admissions_this_step, max_admissions_per_token);
            return;
        }
    }

    int32_t destination = -1;
    const int32_t regular_slot_limit = ggml_backend_moe_dynamic_regular_slot_limit(layer);
    for (int32_t slot = 0; slot < regular_slot_limit; ++slot) {
        if (layer.slots[(size_t) slot].state == GGML_BACKEND_MOE_SLOT_EMPTY) {
            destination = slot;
            break;
        }
    }

    int32_t victim_expert = -1;
    double victim_score = INFINITY;
    bool victim_protected = true;
    uint64_t victim_last_used = UINT64_MAX;
    uint32_t victim_hit_count = 0;
    if (destination < 0) {
        const bool segmented_lru =
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_SEGMENTED_LRU", 1) != 0;
        for (int32_t slot = 0; slot < regular_slot_limit; ++slot) {
            const auto & candidate_slot = layer.slots[(size_t) slot];
            if (candidate_slot.state != GGML_BACKEND_MOE_SLOT_READY ||
                selected[(size_t) candidate_slot.expert]) {
                continue;
            }
            const double score = ggml_backend_moe_dynamic_current_score(
                layer, candidate_slot.expert, decay);
            const uint64_t last_used = candidate_slot.last_used_step > 0
                ? candidate_slot.last_used_step
                : candidate_slot.admitted_step;
            const bool better_segment = segmented_lru &&
                candidate_slot.protected_segment != victim_protected &&
                !candidate_slot.protected_segment;
            const bool same_segment = !segmented_lru ||
                candidate_slot.protected_segment == victim_protected;
            const bool better_recency = same_segment &&
                (last_used < victim_last_used ||
                 (last_used == victim_last_used && score < victim_score));
            if (destination < 0 || better_segment || better_recency) {
                destination = slot;
                victim_expert = candidate_slot.expert;
                victim_score = score;
                victim_protected = candidate_slot.protected_segment;
                victim_last_used = last_used;
                victim_hit_count = candidate_slot.hit_count;
            }
        }
        if (destination < 0) {
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"reject\",\"layer\":%d,\"step\":%" PRIu64
                ",\"expert\":%d,\"score\":%.8g,\"reason\":\"no_safe_victim\"",
                layer_id, layer.step, best_candidate, best_candidate_score);
            return;
        }
        if (best_candidate_score < victim_score * admission_ratio) {
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"reject\",\"layer\":%d,\"step\":%" PRIu64
                ",\"expert\":%d,\"score\":%.8g,\"victim_expert\":%d,"
                "\"victim_score\":%.8g,\"ratio\":%.8g,\"reason\":\"candidate_not_better\"",
                layer_id, layer.step, best_candidate, best_candidate_score,
                victim_expert, victim_score, admission_ratio);
            return;
        }
    }

    if (victim_expert >= 0) {
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"evict\",\"layer\":%d,\"step\":%" PRIu64
            ",\"slot\":%d,\"expert\":%d,\"victim_score\":%.8g,"
            "\"victim_protected\":%s,\"victim_hits\":%u,"
            "\"victim_last_used\":%" PRIu64 ",\"replacement\":%d,"
            "\"replacement_score\":%.8g",
            layer_id, layer.step, destination, victim_expert, victim_score,
            victim_protected ? "true" : "false", victim_hit_count,
            victim_last_used, best_candidate, best_candidate_score);
        layer.slot_by_expert[(size_t) victim_expert] = -1;
    }
    auto & slot = layer.slots[(size_t) destination];
    slot.expert = best_candidate;
    slot.state = GGML_BACKEND_MOE_SLOT_COPYING;
    slot.components_enqueued = 0;
    slot.components_completed = 0;
    slot.promotion_bundle_id = 0;
    slot.promotion_queued_mono_ns = 0;
    slot.prediction_step = 0;
    slot.prediction_distance = 0;
    slot.admitted_step = layer.step;
    slot.last_used_step = 0;
    slot.hit_count = 0;
    slot.protected_segment = false;
    layer.slot_by_expert[(size_t) best_candidate] = destination;
    layer.admissions++;
    registry.admissions_this_step++;
    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"admit\",\"layer\":%d,\"step\":%" PRIu64
        ",\"expert\":%d,\"score\":%.8g,\"slot\":%d,\"victim_expert\":%d",
        layer_id, layer.step, best_candidate, best_candidate_score, destination, victim_expert);
}

int32_t ggml_backend_moe_dynamic_slot_expert(int32_t layer_id, int32_t slot) {
    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto found = registry.layers.find(layer_id);
    if (found == registry.layers.end() || slot < 0 || slot >= found->second.n_slots) {
        return -1;
    }
    return found->second.slots[(size_t) slot].expert;
}

bool ggml_backend_moe_dynamic_slot_needs_component(
        int32_t layer_id,
        int32_t slot,
        int32_t component) {
    GGML_ASSERT(component >= 0 && component < 3);
    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto found = registry.layers.find(layer_id);
    if (found == registry.layers.end() || slot < 0 || slot >= found->second.n_slots) {
        return false;
    }
    const auto & entry = found->second.slots[(size_t) slot];
    return entry.state == GGML_BACKEND_MOE_SLOT_COPYING &&
        (entry.components_enqueued & (uint8_t) (1u << component)) == 0;
}

void ggml_backend_moe_dynamic_slot_component_enqueued(
        int32_t layer_id,
        int32_t slot,
        int32_t component) {
    GGML_ASSERT(component >= 0 && component < 3);
    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto found = registry.layers.find(layer_id);
    if (found == registry.layers.end() || slot < 0 || slot >= found->second.n_slots) {
        return;
    }
    auto & entry = found->second.slots[(size_t) slot];
    if (entry.state != GGML_BACKEND_MOE_SLOT_COPYING) {
        return;
    }
    entry.components_enqueued |= (uint8_t) (1u << component);
    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"component_enqueued\",\"layer\":%d,\"slot\":%d,"
        "\"expert\":%d,\"component\":%d,\"components\":%u",
        layer_id, slot, entry.expert, component, (unsigned) entry.components_enqueued);
    if (entry.components_enqueued == 0x7u) {
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"components_queued\",\"layer\":%d,\"slot\":%d,"
            "\"expert\":%d,\"state\":\"copying\"",
            layer_id, slot, entry.expert);
    }
}

static void ggml_backend_moe_dynamic_slot_component_completed(
        int32_t layer_id,
        int32_t slot,
        int32_t expert,
        int32_t component,
        uint64_t completion_mono_ns = 0) {
    GGML_ASSERT(component >= 0 && component < 3);
    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto found = registry.layers.find(layer_id);
    if (found == registry.layers.end() || slot < 0 || slot >= found->second.n_slots) {
        return;
    }
    auto & entry = found->second.slots[(size_t) slot];
    if (entry.state != GGML_BACKEND_MOE_SLOT_COPYING || entry.expert != expert) {
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"stale_completion\",\"layer\":%d,\"slot\":%d,"
            "\"expected_expert\":%d,\"completed_expert\":%d,\"component\":%d",
            layer_id, slot, entry.expert, expert, component);
        return;
    }
    if (completion_mono_ns == 0) {
        completion_mono_ns = ggml_backend_moe_dynamic_mono_ns();
    }
    entry.components_completed |= (uint8_t) (1u << component);
    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"component_completed\",\"layer\":%d,\"slot\":%d,"
        "\"expert\":%d,\"component\":%d,\"components\":%u,"
        "\"bundle_id\":%" PRIu64 ",\"completion_mono_ns\":%" PRIu64,
        layer_id, slot, expert, component, (unsigned) entry.components_completed,
        entry.promotion_bundle_id, completion_mono_ns);
    if (entry.components_completed == 0x7u) {
        entry.state = GGML_BACKEND_MOE_SLOT_READY;
        const double prediction_to_ready_us = entry.promotion_queued_mono_ns > 0
            ? (completion_mono_ns - entry.promotion_queued_mono_ns) / 1000.0
            : 0.0;
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"ready_after_completion\",\"layer\":%d,\"slot\":%d,"
            "\"expert\":%d,\"bundle_id\":%" PRIu64
            ",\"prediction_distance\":%d,\"prediction_step\":%" PRIu64
            ",\"queued_mono_ns\":%" PRIu64 ",\"ready_mono_ns\":%" PRIu64
            ",\"prediction_to_ready_us\":%.3f",
            layer_id, slot, expert, entry.promotion_bundle_id,
            entry.prediction_distance, entry.prediction_step,
            entry.promotion_queued_mono_ns, completion_mono_ns,
            prediction_to_ready_us);
    }
}

void ggml_backend_moe_dynamic_reset(void) {
    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.layers.clear();
    registry.token_coverage.clear();
    registry.current_routes.clear();
    registry.admission_step = 0;
    registry.admissions_this_step = 0;
    registry.warm_start_step = 0;
    registry.warm_start_admissions_this_step = 0;
    registry.predictor_step = 0;
    registry.predictor_admissions_this_step = 0;
    registry.cross_predictor_step = 0;
    registry.cross_predictor_admissions_this_step = 0;
    registry.next_promotion_bundle_id = 1;
}

static void ggml_backend_moe_dynamic_log_summary() {
    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    uint64_t admissions = 0;
    uint64_t resident_hits = 0;
    uint64_t resident_misses = 0;
    uint64_t ready_hits = 0;
    uint64_t cpu_misses = 0;
    uint64_t ready_slots = 0;
    uint64_t copying_slots = 0;
    uint64_t split_decisions = 0;
    uint64_t split_enabled = 0;
    uint64_t decode_steps = 0;
    uint64_t prefill_batches = 0;
    uint64_t prefill_ready_routes = 0;
    uint64_t prefill_total_routes = 0;
    uint64_t full_gpu_layer_steps = 0;
    uint64_t cpu_dependent_layer_steps = 0;
    std::vector<uint64_t> ready_route_hist;
    std::vector<uint64_t> gpu_route_hist;
    for (const auto & layer_item : registry.layers) {
        const auto & layer = layer_item.second;
        admissions += layer.admissions;
        resident_hits += layer.resident_hits;
        resident_misses += layer.resident_misses;
        ready_hits += layer.ready_hits;
        cpu_misses += layer.cpu_misses;
        split_decisions += layer.split_decisions;
        split_enabled += layer.split_enabled;
        decode_steps += layer.decode_steps;
        prefill_batches += layer.prefill_batches;
        prefill_ready_routes += layer.prefill_ready_routes;
        prefill_total_routes += layer.prefill_total_routes;
        full_gpu_layer_steps += layer.full_gpu_layer_steps;
        cpu_dependent_layer_steps += layer.cpu_dependent_layer_steps;
        if (ready_route_hist.size() < layer.ready_route_hist.size()) {
            ready_route_hist.resize(layer.ready_route_hist.size(), 0);
        }
        if (gpu_route_hist.size() < layer.gpu_route_hist.size()) {
            gpu_route_hist.resize(layer.gpu_route_hist.size(), 0);
        }
        for (size_t routes = 0; routes < layer.ready_route_hist.size(); ++routes) {
            ready_route_hist[routes] += layer.ready_route_hist[routes];
        }
        for (size_t routes = 0; routes < layer.gpu_route_hist.size(); ++routes) {
            gpu_route_hist[routes] += layer.gpu_route_hist[routes];
        }
        for (const auto & slot : layer.slots) {
            ready_slots += slot.state == GGML_BACKEND_MOE_SLOT_READY;
            copying_slots += slot.state == GGML_BACKEND_MOE_SLOT_COPYING;
        }
    }

    const uint64_t expected_layers = registry.layers.size();
    uint64_t complete_tokens = 0;
    uint64_t full_gpu_expert_path_tokens = 0;
    uint64_t complete_token_cpu_dependent_layers = 0;
    for (const auto & token_item : registry.token_coverage) {
        const auto & coverage = token_item.second;
        if (coverage.layers_seen != expected_layers) {
            continue;
        }
        complete_tokens++;
        full_gpu_expert_path_tokens += coverage.full_gpu_layers == expected_layers;
        complete_token_cpu_dependent_layers += coverage.cpu_dependent_layers;
    }

    auto format_route_hist = [](const std::vector<uint64_t> & hist) {
        std::ostringstream result;
        result << '[';
        bool first = true;
        for (size_t routes = 0; routes < hist.size(); ++routes) {
            if (hist[routes] == 0) {
                continue;
            }
            if (!first) {
                result << ',';
            }
            first = false;
            result << routes << ':' << hist[routes];
        }
        result << ']';
        return result.str();
    };

    GGML_LOG_INFO(
        "moe-dynamic-cache: layers=%zu admissions=%" PRIu64
        " resident-hits=%" PRIu64 " resident-misses=%" PRIu64 " resident-hit-rate=%.4f"
        " executed-gpu-routes=%" PRIu64 " cpu-routes=%" PRIu64 " execution-hit-rate=%.4f"
        " ready-slots=%" PRIu64 " copying-slots=%" PRIu64
        " split-enabled=%" PRIu64 "/%" PRIu64 "\n",
        registry.layers.size(),
        admissions,
        resident_hits,
        resident_misses,
        resident_hits + resident_misses > 0
            ? (double) resident_hits / (double) (resident_hits + resident_misses)
            : 0.0,
        ready_hits,
        cpu_misses,
        ready_hits + cpu_misses > 0
            ? (double) ready_hits / (double) (ready_hits + cpu_misses)
            : 0.0,
        ready_slots,
        copying_slots,
        split_enabled,
        split_decisions);
    GGML_LOG_INFO(
        "moe-dynamic-coverage: full-gpu-expert-layers=%" PRIu64 "/%" PRIu64
        " cpu-dependent-layers=%" PRIu64 " full-gpu-expert-path-tokens=%" PRIu64
        "/%" PRIu64 " mean-cpu-dependent-layers=%.4f"
        " prefill-batches=%" PRIu64 " prefill-ready-routes=%" PRIu64 "/%" PRIu64
        " ready-route-hist=%s gpu-route-hist=%s\n",
        full_gpu_layer_steps,
        decode_steps,
        cpu_dependent_layer_steps,
        full_gpu_expert_path_tokens,
        complete_tokens,
        complete_tokens > 0
            ? (double) complete_token_cpu_dependent_layers / (double) complete_tokens
            : 0.0,
        prefill_batches,
        prefill_ready_routes,
        prefill_total_routes,
        format_route_hist(ready_route_hist).c_str(),
        format_route_hist(gpu_route_hist).c_str());
    if (ggml_backend_moe_dynamic_env_i32("GGML_EXPERT_CACHE_PROFILE", 0) != 0) {
        std::vector<int32_t> layer_ids;
        layer_ids.reserve(registry.layers.size());
        for (const auto & item : registry.layers) {
            layer_ids.push_back(item.first);
        }
        std::sort(layer_ids.begin(), layer_ids.end());
        for (int32_t layer_id : layer_ids) {
            const auto & layer = registry.layers.at(layer_id);
            uint64_t layer_ready = 0;
            uint64_t layer_copying = 0;
            for (const auto & slot : layer.slots) {
                layer_ready += slot.state == GGML_BACKEND_MOE_SLOT_READY;
                layer_copying += slot.state == GGML_BACKEND_MOE_SLOT_COPYING;
            }
            GGML_LOG_INFO(
                "moe-dynamic-layer: layer=%d admissions=%" PRIu64
                " resident-hits=%" PRIu64 " resident-misses=%" PRIu64 " resident-hit-rate=%.4f"
                " executed-gpu-routes=%" PRIu64 " cpu-routes=%" PRIu64 " execution-hit-rate=%.4f"
                " ready-slots=%" PRIu64 " copying-slots=%" PRIu64
                " split-enabled=%" PRIu64 "/%" PRIu64
                " full-gpu-expert-layers=%" PRIu64 "/%" PRIu64
                " prefill-batches=%" PRIu64 " prefill-ready-routes=%" PRIu64 "/%" PRIu64
                " ready-route-hist=%s gpu-route-hist=%s\n",
                layer_id,
                layer.admissions,
                layer.resident_hits,
                layer.resident_misses,
                layer.resident_hits + layer.resident_misses > 0
                    ? (double) layer.resident_hits / (double) (layer.resident_hits + layer.resident_misses)
                    : 0.0,
                layer.ready_hits,
                layer.cpu_misses,
                layer.ready_hits + layer.cpu_misses > 0
                    ? (double) layer.ready_hits / (double) (layer.ready_hits + layer.cpu_misses)
                    : 0.0,
                layer_ready,
                layer_copying,
                layer.split_enabled,
                layer.split_decisions,
                layer.full_gpu_layer_steps,
                layer.decode_steps,
                layer.prefill_batches,
                layer.prefill_ready_routes,
                layer.prefill_total_routes,
                format_route_hist(layer.ready_route_hist).c_str(),
                format_route_hist(layer.gpu_route_hist).c_str());
        }
    }
}

struct ggml_backend_moe_promotion_job {
    uint64_t job_id = 0;
    uint64_t bundle_id = 0;
    uint64_t queued_mono_ns = 0;
    uint64_t prediction_step = 0;
    int32_t prediction_distance = 0;
    int32_t layer = -1;
    int32_t slot = -1;
    int32_t expert = -1;
    int32_t component = -1;
    const uint8_t * source = nullptr;
    ggml_tensor * destination = nullptr;
    size_t destination_offset = 0;
    size_t bytes = 0;
    bool source_pinned = false;
    bool urgent = false;
    std::chrono::steady_clock::time_point queued_at;
};

struct ggml_backend_moe_promotion_worker {
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_expert_cache * cache = nullptr;
    ggml_backend_buffer_t staging_buffer = nullptr;
    uint8_t * staging_base = nullptr;
    size_t staging_slots = 0;
    size_t staging_stride = 0;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<ggml_backend_moe_promotion_job> queue;
    bool stop = false;
    std::thread thread;

    uint64_t jobs = 0;
    uint64_t batches = 0;
    uint64_t urgent_jobs = 0;
    uint64_t urgent_batches = 0;
    uint64_t bytes = 0;
    uint64_t host_issue_ns = 0;
    uint64_t staging_copy_ns = 0;
    uint64_t staging_bytes = 0;
    uint64_t sync_ns = 0;
    uint64_t queue_delay_ns = 0;
    size_t max_queue_depth = 0;
    uint64_t next_job_id = 1;
    uint64_t next_batch_id = 1;
    int32_t verify_per_shape = 0;
    uint64_t verified_jobs = 0;
    uint64_t verification_failures = 0;
    std::map<std::pair<int32_t, size_t>, int32_t> verified_shape_counts;

    ggml_backend_moe_promotion_worker(
            ggml_backend_dev_t device,
            ggml_backend_sched_expert_cache * cache)
        : backend(ggml_backend_dev_init(device, nullptr)), cache(cache) {
        if (backend == nullptr) {
            return;
        }

        const int32_t requested_slots = std::max<int32_t>(1,
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_STAGING_SLOTS", 24));
        const int32_t requested_mib = std::max<int32_t>(1,
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_STAGING_MIB", 96));
        verify_per_shape = std::max<int32_t>(0,
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_VERIFY_PROMOTIONS_PER_SHAPE", 0));
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(device);
        if (host_buft != nullptr) {
            const size_t staging_bytes_total = (size_t) requested_mib * 1024 * 1024;
            staging_buffer = ggml_backend_buft_alloc_buffer(host_buft, staging_bytes_total);
            if (staging_buffer != nullptr) {
                staging_base = (uint8_t *) ggml_backend_buffer_get_base(staging_buffer);
                staging_slots = (size_t) requested_slots;
                staging_stride = staging_bytes_total / staging_slots;
                if (staging_base == nullptr || staging_stride == 0) {
                    ggml_backend_buffer_free(staging_buffer);
                    staging_buffer = nullptr;
                    staging_base = nullptr;
                    staging_slots = 0;
                    staging_stride = 0;
                }
            }
        }

        GGML_LOG_INFO(
            "moe-promotion-worker: staging=%s slots=%zu stride=%.2f MiB verify-per-shape=%d\n",
            staging_buffer != nullptr ? "pinned" : "direct",
            staging_slots,
            staging_stride / 1024.0 / 1024.0,
            verify_per_shape);
        thread = std::thread([this]() { run(); });
    }

    ~ggml_backend_moe_promotion_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        if (thread.joinable()) {
            thread.join();
        }
        if (cache != nullptr) {
            cache->profile_dynamic_copy_issue_ns += host_issue_ns;
            cache->profile_copy_issue_ns += host_issue_ns;
        }
        GGML_LOG_INFO(
            "moe-promotion-worker: jobs=%" PRIu64 " batches=%" PRIu64
            " urgent-jobs=%" PRIu64 " urgent-batches=%" PRIu64
            " bytes=%.2f MiB staging-bytes=%.2f MiB staging-copy=%.3f ms"
            " host-issue=%.3f ms sync=%.3f ms queue-delay=%.3f ms max-depth=%zu\n",
            jobs,
            batches,
            urgent_jobs,
            urgent_batches,
            bytes / 1024.0 / 1024.0,
            staging_bytes / 1024.0 / 1024.0,
            staging_copy_ns / 1.0e6,
            host_issue_ns / 1.0e6,
            sync_ns / 1.0e6,
            queue_delay_ns / 1.0e6,
            max_queue_depth);
        if (verify_per_shape > 0) {
            GGML_LOG_INFO(
                "moe-promotion-worker-verify: checked=%" PRIu64 " failures=%" PRIu64
                " shapes=%zu per-shape-limit=%d\n",
                verified_jobs,
                verification_failures,
                verified_shape_counts.size(),
                verify_per_shape);
        }
        if (staging_buffer != nullptr) {
            ggml_backend_buffer_free(staging_buffer);
            staging_buffer = nullptr;
            staging_base = nullptr;
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }

    bool valid() const {
        return backend != nullptr;
    }

    bool enqueue_batch(std::vector<ggml_backend_moe_promotion_job> jobs_to_enqueue, bool urgent) {
        if (backend == nullptr || jobs_to_enqueue.empty()) {
            return false;
        }
        const auto queued_at = std::chrono::steady_clock::now();
        const uint64_t queued_mono_ns = ggml_backend_moe_dynamic_mono_ns();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stop) {
                return false;
            }
            for (auto & job : jobs_to_enqueue) {
                job.job_id = next_job_id++;
                job.queued_at = queued_at;
                if (job.queued_mono_ns == 0) {
                    job.queued_mono_ns = queued_mono_ns;
                }
                job.urgent = urgent;
            }
            if (urgent) {
                // Push the complete bundle atomically at the front while
                // preserving component order. This prevents the worker from
                // draining gate/up/down as three separately synchronized
                // batches and gives current-token requests priority over
                // background cache maintenance.
                for (auto it = jobs_to_enqueue.rbegin(); it != jobs_to_enqueue.rend(); ++it) {
                    queue.push_front(std::move(*it));
                }
            } else {
                for (auto & job : jobs_to_enqueue) {
                    queue.push_back(std::move(job));
                }
            }
            max_queue_depth = std::max(max_queue_depth, queue.size());
        }
        cv.notify_one();
        return true;
    }

    bool enqueue(ggml_backend_moe_promotion_job job) {
        std::vector<ggml_backend_moe_promotion_job> jobs_to_enqueue;
        jobs_to_enqueue.push_back(std::move(job));
        return enqueue_batch(std::move(jobs_to_enqueue), false);
    }

    void run() {
#ifdef __linux__
        const int32_t worker_cpu =
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_WORKER_CPU", -1);
        if (worker_cpu >= 0 && worker_cpu < CPU_SETSIZE) {
            cpu_set_t affinity;
            CPU_ZERO(&affinity);
            CPU_SET(worker_cpu, &affinity);
            const int result = pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            GGML_LOG_INFO(
                "moe-promotion-worker: cpu=%d affinity=%s\n",
                worker_cpu,
                result == 0 ? "set" : "failed");
        }
#endif
        struct transfer_timing {
            uint64_t job_start_ns = 0;
            uint64_t staging_begin_ns = 0;
            uint64_t staging_end_ns = 0;
            uint64_t h2d_issue_begin_ns = 0;
            uint64_t h2d_issue_end_ns = 0;
            bool used_staging = false;
        };

        for (;;) {
            std::vector<ggml_backend_moe_promotion_job> batch;
            uint64_t batch_id = 0;
            uint64_t batch_capture_ns = 0;
            size_t batch_urgent_jobs = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this]() { return stop || !queue.empty(); });
                if (stop && queue.empty()) {
                    break;
                }
                batch_id = next_batch_id++;
                batch_capture_ns = ggml_backend_moe_dynamic_mono_ns();
                batch.reserve(queue.size());
                while (!queue.empty()) {
                    batch_urgent_jobs += queue.front().urgent;
                    batch.push_back(std::move(queue.front()));
                    queue.pop_front();
                }
            }

            // A staging slot cannot be reused until the H2D transfer that reads it
            // has completed. Process arbitrarily large queues in bounded chunks,
            // one pinned slot per pageable source job, and synchronize once per
            // chunk. Pinned model sources bypass staging entirely.
            const size_t chunk_capacity = staging_slots > 0 ? staging_slots : batch.size();
            for (size_t chunk_begin = 0; chunk_begin < batch.size(); chunk_begin += chunk_capacity) {
                const size_t chunk_end = std::min(batch.size(), chunk_begin + chunk_capacity);
                const size_t chunk_size = chunk_end - chunk_begin;
                std::vector<transfer_timing> timings(chunk_size);
                bool urgent_chunk = false;

                for (size_t index = chunk_begin; index < chunk_end; ++index) {
                    const auto & job = batch[index];
                    auto & timing = timings[index - chunk_begin];
                    urgent_chunk = urgent_chunk || job.urgent;
                    timing.job_start_ns = ggml_backend_moe_dynamic_mono_ns();
                    const uint64_t queue_delay = timing.job_start_ns - job.queued_mono_ns;
                    queue_delay_ns += queue_delay;

                    const uint8_t * transfer_source = job.source;
                    timing.staging_begin_ns = timing.job_start_ns;
                    timing.staging_end_ns = timing.job_start_ns;
                    if (!job.source_pinned && staging_base != nullptr && job.bytes <= staging_stride) {
                        uint8_t * staging = staging_base + (index - chunk_begin) * staging_stride;
                        timing.staging_begin_ns = ggml_backend_moe_dynamic_mono_ns();
                        memcpy(staging, job.source, job.bytes);
                        timing.staging_end_ns = ggml_backend_moe_dynamic_mono_ns();
                        const uint64_t staging_ns = timing.staging_end_ns - timing.staging_begin_ns;
                        staging_copy_ns += staging_ns;
                        staging_bytes += job.bytes;
                        transfer_source = staging;
                        timing.used_staging = true;
                    }

                    timing.h2d_issue_begin_ns = ggml_backend_moe_dynamic_mono_ns();
                    ggml_backend_tensor_set_async(
                        backend,
                        job.destination,
                        transfer_source,
                        job.destination_offset,
                        job.bytes);
                    timing.h2d_issue_end_ns = ggml_backend_moe_dynamic_mono_ns();
                    host_issue_ns += timing.h2d_issue_end_ns - timing.job_start_ns;
                }

                const uint64_t sync_begin_ns = ggml_backend_moe_dynamic_mono_ns();
                ggml_backend_synchronize(backend);
                const uint64_t sync_end_ns = ggml_backend_moe_dynamic_mono_ns();
                const uint64_t batch_sync_ns = sync_end_ns - sync_begin_ns;
                sync_ns += batch_sync_ns;

                uint64_t completion_mono_ns = sync_end_ns;
                if (verify_per_shape > 0) {
                    for (size_t index = chunk_begin; index < chunk_end; ++index) {
                        const auto & job = batch[index];
                        const auto shape = std::make_pair(
                            job.component + (job.urgent ? 3 : 0), job.bytes);
                        int32_t & shape_count = verified_shape_counts[shape];
                        if (shape_count >= verify_per_shape) {
                            continue;
                        }
                        std::vector<uint8_t> readback(job.bytes);
                        const uint64_t verify_begin_ns = ggml_backend_moe_dynamic_mono_ns();
                        ggml_backend_tensor_get_async(
                            backend,
                            job.destination,
                            readback.data(),
                            job.destination_offset,
                            job.bytes);
                        ggml_backend_synchronize(backend);
                        const uint64_t verify_end_ns = ggml_backend_moe_dynamic_mono_ns();
                        size_t first_mismatch = job.bytes;
                        for (size_t offset = 0; offset < job.bytes; ++offset) {
                            if (readback[offset] != job.source[offset]) {
                                first_mismatch = offset;
                                break;
                            }
                        }
                        const bool match = first_mismatch == job.bytes;
                        shape_count++;
                        verified_jobs++;
                        verification_failures += !match;
                        if (!match) {
                            GGML_LOG_ERROR(
                                "moe-promotion-worker-verify: mismatch job=%" PRIu64
                                " layer=%d slot=%d expert=%d component=%d bytes=%zu offset=%zu\n",
                                job.job_id, job.layer, job.slot, job.expert,
                                job.component, job.bytes, first_mismatch);
                        }
                        ggml_backend_moe_dynamic_tracef(
                            "\"event\":\"worker_transfer_verify\",\"job_id\":%" PRIu64
                            ",\"bundle_id\":%" PRIu64 ",\"layer\":%d,\"slot\":%d,"
                            "\"expert\":%d,\"component\":%d,\"bytes\":%zu,"
                            "\"match\":%s,\"first_mismatch\":%zu,\"duration_us\":%.3f",
                            job.job_id, job.bundle_id, job.layer, job.slot, job.expert,
                            job.component, job.bytes, match ? "true" : "false",
                            first_mismatch, (verify_end_ns - verify_begin_ns) / 1000.0);
                    }
                    completion_mono_ns = ggml_backend_moe_dynamic_mono_ns();
                }

                for (size_t index = chunk_begin; index < chunk_end; ++index) {
                    const auto & job = batch[index];
                    const auto & timing = timings[index - chunk_begin];
                    ggml_backend_moe_dynamic_slot_component_completed(
                        job.layer, job.slot, job.expert, job.component, completion_mono_ns);
                    ggml_backend_moe_dynamic_tracef(
                        "\"event\":\"worker_transfer_timing\","
                        "\"job_id\":%" PRIu64 ",\"bundle_id\":%" PRIu64
                        ",\"batch_id\":%" PRIu64 ",\"batch_jobs\":%zu,"
                        "\"batch_urgent_jobs\":%zu,\"chunk_begin\":%zu,\"chunk_jobs\":%zu,"
                        "\"layer\":%d,\"slot\":%d,\"expert\":%d,\"component\":%d,"
                        "\"prediction_distance\":%d,\"prediction_step\":%" PRIu64
                        ",\"bytes\":%zu,\"urgent\":%s,\"staged\":%s,"
                        "\"queued_mono_ns\":%" PRIu64 ",\"batch_capture_mono_ns\":%" PRIu64
                        ",\"job_start_mono_ns\":%" PRIu64
                        ",\"staging_begin_mono_ns\":%" PRIu64
                        ",\"staging_end_mono_ns\":%" PRIu64
                        ",\"h2d_issue_begin_mono_ns\":%" PRIu64
                        ",\"h2d_issue_end_mono_ns\":%" PRIu64
                        ",\"sync_begin_mono_ns\":%" PRIu64
                        ",\"sync_end_mono_ns\":%" PRIu64
                        ",\"queue_delay_us\":%.3f,\"captured_wait_us\":%.3f,"
                        "\"staging_copy_us\":%.3f,\"h2d_api_us\":%.3f,"
                        "\"issue_to_sync_end_us\":%.3f,\"prediction_to_sync_end_us\":%.3f",
                        job.job_id,
                        job.bundle_id,
                        batch_id,
                        batch.size(),
                        batch_urgent_jobs,
                        chunk_begin,
                        chunk_size,
                        job.layer,
                        job.slot,
                        job.expert,
                        job.component,
                        job.prediction_distance,
                        job.prediction_step,
                        job.bytes,
                        job.urgent ? "true" : "false",
                        timing.used_staging ? "true" : "false",
                        job.queued_mono_ns,
                        batch_capture_ns,
                        timing.job_start_ns,
                        timing.staging_begin_ns,
                        timing.staging_end_ns,
                        timing.h2d_issue_begin_ns,
                        timing.h2d_issue_end_ns,
                        sync_begin_ns,
                        sync_end_ns,
                        (timing.job_start_ns - job.queued_mono_ns) / 1000.0,
                        (timing.job_start_ns - batch_capture_ns) / 1000.0,
                        (timing.staging_end_ns - timing.staging_begin_ns) / 1000.0,
                        (timing.h2d_issue_end_ns - timing.h2d_issue_begin_ns) / 1000.0,
                        (sync_end_ns - timing.h2d_issue_begin_ns) / 1000.0,
                        (sync_end_ns - job.queued_mono_ns) / 1000.0);
                    jobs++;
                    urgent_jobs += job.urgent;
                    bytes += job.bytes;
                }
                batches++;
                urgent_batches += urgent_chunk;
            }
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"worker_batch_completed\",\"batch_id\":%" PRIu64
                ",\"jobs\":%zu,\"urgent_jobs\":%zu,\"capture_mono_ns\":%" PRIu64
                ",\"complete_mono_ns\":%" PRIu64 ",\"duration_us\":%.3f",
                batch_id,
                batch.size(),
                batch_urgent_jobs,
                batch_capture_ns,
                ggml_backend_moe_dynamic_mono_ns(),
                (ggml_backend_moe_dynamic_mono_ns() - batch_capture_ns) / 1000.0);
        }
    }
};

static void ggml_backend_moe_dynamic_register_component_binding(
        int32_t layer_id,
        int32_t component,
        const ggml_tensor * source,
        ggml_tensor * destination,
        size_t destination_stride,
        ggml_backend_moe_promotion_worker * worker,
        ggml_backend_sched_expert_cache * cache,
        uint8_t * resident) {
    GGML_ASSERT(layer_id >= 0 && component >= 0 && component < 3);
    GGML_ASSERT(source != nullptr && destination != nullptr);
    auto & registry = ggml_backend_moe_dynamic_registry_get();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto found = registry.layers.find(layer_id);
    if (found == registry.layers.end()) {
        return;
    }
    auto & binding = found->second.component_bindings[(size_t) component];
    binding.valid = true;
    binding.source_base = (const uint8_t *) source->data;
    binding.source_stride = source->nb[2];
    binding.destination = destination;
    binding.destination_stride = destination_stride;
    binding.source_pinned = source->buffer != nullptr &&
        strstr(ggml_backend_buffer_name(source->buffer), "CUDA_Host") != nullptr;
    binding.worker = worker;
    binding.cache = cache;
    binding.resident = resident;
    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"component_binding\",\"layer\":%d,\"component\":%d,"
        "\"source_stride\":%zu,\"destination_stride\":%zu,"
        "\"source_pinned\":%s,\"worker\":%s",
        layer_id, component, binding.source_stride, binding.destination_stride,
        binding.source_pinned ? "true" : "false", worker != nullptr ? "true" : "false");
}

static bool ggml_backend_moe_dynamic_issue_urgent_locked(
        ggml_backend_moe_dynamic_registry & registry,
        int32_t layer_id,
        int32_t slot,
        int32_t expert,
        int32_t distance,
        uint64_t step) {
    const char * enabled_env = getenv("GGML_MOE_DYNAMIC_URGENT_PREDICT_UPLOAD");
    if (enabled_env == nullptr || enabled_env[0] == '\0' || atoi(enabled_env) == 0) {
        return false;
    }
    auto found = registry.layers.find(layer_id);
    if (found == registry.layers.end() || slot < 0 || slot >= found->second.n_slots) {
        return false;
    }
    auto & layer = found->second;
    auto & slot_entry = layer.slots[(size_t) slot];
    if (slot_entry.state != GGML_BACKEND_MOE_SLOT_COPYING ||
        slot_entry.expert != expert || slot_entry.components_enqueued != 0) {
        return false;
    }

    const uint64_t bundle_id = registry.next_promotion_bundle_id++;
    const uint64_t queued_mono_ns = ggml_backend_moe_dynamic_mono_ns();
    ggml_backend_moe_promotion_worker * worker = nullptr;
    ggml_backend_sched_expert_cache * cache = nullptr;
    size_t bundle_bytes = 0;
    std::vector<ggml_backend_moe_promotion_job> jobs;
    jobs.reserve(3);
    for (int32_t component = 0; component < 3; ++component) {
        const auto & binding = layer.component_bindings[(size_t) component];
        if (!binding.valid || binding.worker == nullptr || binding.cache == nullptr ||
            binding.source_base == nullptr || binding.destination == nullptr ||
            binding.source_stride == 0 || binding.destination_stride == 0) {
            return false;
        }
        if (worker == nullptr) {
            worker = binding.worker;
            cache = binding.cache;
        } else if (worker != binding.worker || cache != binding.cache) {
            return false;
        }
        ggml_backend_moe_promotion_job job;
        job.bundle_id = bundle_id;
        job.queued_mono_ns = queued_mono_ns;
        job.prediction_step = step;
        job.prediction_distance = distance;
        job.layer = layer_id;
        job.slot = slot;
        job.expert = expert;
        job.component = component;
        job.source = binding.source_base + (size_t) expert * binding.source_stride;
        job.destination = binding.destination;
        job.destination_offset = (size_t) slot * binding.destination_stride;
        job.bytes = binding.source_stride;
        job.source_pinned = binding.source_pinned;
        job.urgent = true;
        bundle_bytes += job.bytes;
        jobs.push_back(std::move(job));
    }

    if (!worker->enqueue_batch(std::move(jobs), true)) {
        return false;
    }

    slot_entry.components_enqueued = 0x7u;
    slot_entry.promotion_bundle_id = bundle_id;
    slot_entry.promotion_queued_mono_ns = queued_mono_ns;
    slot_entry.prediction_step = step;
    slot_entry.prediction_distance = distance;
    for (int32_t component = 0; component < 3; ++component) {
        const auto & binding = layer.component_bindings[(size_t) component];
        if (binding.resident != nullptr) {
            binding.resident[(size_t) slot] = 1;
        }
        cache->bytes_uploaded += binding.source_stride;
        if (component == 0) {
            cache->misses++;
        }
        if (cache->profile) {
            cache->profile_copy_groups++;
            cache->profile_copy_bytes += binding.source_stride;
            cache->profile_copy_groups_decode++;
            cache->profile_copy_bytes_decode += binding.source_stride;
            cache->profile_dynamic_copy_groups++;
            cache->profile_dynamic_copy_bytes += binding.source_stride;
        }
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"component_enqueued\",\"policy\":\"urgent_predict\","
            "\"layer\":%d,\"slot\":%d,\"expert\":%d,\"component\":%d,"
            "\"components\":7,\"bundle_id\":%" PRIu64,
            layer_id, slot, expert, component, bundle_id);
    }
    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"urgent_bundle_queued\",\"layer\":%d,\"slot\":%d,"
        "\"expert\":%d,\"distance\":%d,\"step\":%" PRIu64
        ",\"bundle_id\":%" PRIu64 ",\"queued_mono_ns\":%" PRIu64
        ",\"bytes\":%zu",
        layer_id, slot, expert, distance, step, bundle_id, queued_mono_ns, bundle_bytes);
    return true;
}

static bool ggml_backend_moe_async_promotion_enabled() {
    const char * value = getenv("GGML_MOE_DYNAMIC_ASYNC_PROMOTION");
    return value == nullptr || value[0] == '\0' || atoi(value) != 0;
}

static ggml_backend_sched_expert_cache_entry * ggml_backend_sched_expert_cache_attach(
        ggml_backend_sched_t sched,
        const struct ggml_tensor * source,
        int backend_id,
        struct ggml_tensor * tensor_copy) {
    ggml_backend_sched_expert_cache * cache = sched->expert_cache;
    if (cache == nullptr || cache->budget_bytes == 0 || cache->allocation_failed) {
        return nullptr;
    }
    if (getenv("GGML_MOE_DYNAMIC_SPLIT_SLOTS") != nullptr) {
        if (strncmp(source->name, "ffn_moe_dynamic_hot_", 20) != 0) {
            return nullptr;
        }
    } else if (getenv("GGML_MOE_STATIC_SPLIT_SLOTS") != nullptr &&
               strncmp(source->name, "ffn_moe_static_hot_", 19) != 0) {
        return nullptr;
    }
    cache->attach_attempts++;

    ggml_backend_sched_expert_cache_entry * entry =
        ggml_backend_sched_expert_cache_find(sched, source, backend_id);

    if (entry == nullptr) {
        ggml_backend_buffer_type_t buft = ggml_backend_sched_expert_cache_buffer_type(sched, backend_id);
        if (ggml_backend_buft_is_host(buft)) {
            cache->rejected_host_target++;
            return nullptr;
        }

        const size_t allocation_size = ggml_backend_buft_get_alloc_size(buft, tensor_copy);
        if (!cache->vmm && allocation_size > cache->budget_bytes - cache->allocated_bytes) {
            cache->rejected_budget++;
            return nullptr;
        }

        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, allocation_size);
        if (buffer == nullptr) {
            GGML_LOG_WARN("expert-cache: failed to allocate %zu bytes on backend %s; disabling further cache allocations\n",
                allocation_size, ggml_backend_name(sched->backends[backend_id]));
            cache->allocation_failed = true;
            return nullptr;
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(buffer, 0);

        ggml_backend_sched_expert_cache_entry new_entry;
        new_entry.source = source;
        new_entry.source_name = source->name;
        new_entry.backend_id = backend_id;
        new_entry.buffer = buffer;
        new_entry.allocation_size = allocation_size;
        new_entry.n_expert = source->ne[2];
        new_entry.expert_size = source->nb[2];
        new_entry.persistent_tensor = std::make_unique<ggml_tensor>(*tensor_copy);
        new_entry.persistent_tensor->buffer = nullptr;
        new_entry.persistent_tensor->data = nullptr;
        new_entry.persistent_tensor->view_src = nullptr;
        new_entry.persistent_tensor->view_offs = 0;
        new_entry.persistent_tensor->extra = nullptr;
        enum ggml_status persistent_status = ggml_backend_tensor_alloc(
            buffer,
            new_entry.persistent_tensor.get(),
            ggml_backend_buffer_get_base(buffer));
        GGML_ASSERT(persistent_status == GGML_STATUS_SUCCESS);
        new_entry.resident.resize((size_t) new_entry.n_expert, 0);

        cache->entries.push_back(std::move(new_entry));
        cache->allocated_bytes += allocation_size;
        entry = &cache->entries.back();

        GGML_LOG_INFO("expert-cache: allocated %.2f MiB for %s on %s (total %.2f / %.2f MiB)\n",
            allocation_size / 1024.0 / 1024.0,
            source->name,
            ggml_backend_name(sched->backends[backend_id]),
            cache->allocated_bytes / 1024.0 / 1024.0,
            cache->budget_bytes / 1024.0 / 1024.0);
    }

    GGML_ASSERT(entry->n_expert == source->ne[2]);
    GGML_ASSERT(entry->expert_size == source->nb[2]);
    GGML_ASSERT(entry->allocation_size >= ggml_backend_buft_get_alloc_size(entry->buffer->buft, tensor_copy));
    GGML_ASSERT(tensor_copy->view_src == nullptr);

    if (tensor_copy->buffer != entry->buffer) {
        // The graph allocator may already have assigned transient arena storage.
        // Rebind this tensor object to the cache-owned buffer before any upload or
        // kernel launch. The arena allocation remains owned by the allocator and
        // is simply unused for this tensor execution.
        tensor_copy->buffer = nullptr;
        tensor_copy->data = nullptr;
        enum ggml_status status = ggml_backend_tensor_alloc(
            entry->buffer, tensor_copy, ggml_backend_buffer_get_base(entry->buffer));
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    return entry;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);

                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;
    const uint64_t moe_trace_call = ggml_moe_trace_begin_call();

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;
    std::vector<ggml_bitset_t> missing_ids;
    std::unordered_set<const ggml_tensor *> skipped_dynamic_tensors;

    auto backend_type_name = [](ggml_backend_t backend) {
        const enum ggml_backend_dev_type type =
            ggml_backend_dev_type(ggml_backend_get_device(backend));
        switch (type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU: return "CPU";
            case GGML_BACKEND_DEVICE_TYPE_GPU: return "GPU";
            case GGML_BACKEND_DEVICE_TYPE_IGPU: return "IGPU";
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
            case GGML_BACKEND_DEVICE_TYPE_META: return "META";
        }
        return "UNKNOWN";
    };
    auto profile_transfer = [&](int split_id,
                                const ggml_tensor * tensor,
                                ggml_backend_t source_backend,
                                ggml_backend_t destination_backend,
                                size_t bytes,
                                bool async_copy,
                                bool fallback_copy,
                                uint64_t issue_ns,
                                uint64_t source_sync_ns,
                                uint64_t destination_sync_ns) {
        if (sched->expert_cache == nullptr || !sched->expert_cache->profile) {
            return;
        }
        const enum ggml_backend_dev_type source_type =
            ggml_backend_dev_type(ggml_backend_get_device(source_backend));
        const enum ggml_backend_dev_type destination_type =
            ggml_backend_dev_type(ggml_backend_get_device(destination_backend));
        if (source_type == GGML_BACKEND_DEVICE_TYPE_CPU &&
            destination_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            sched->expert_cache->profile_transfer_h2d_groups++;
            sched->expert_cache->profile_transfer_h2d_bytes += bytes;
        } else if (source_type == GGML_BACKEND_DEVICE_TYPE_GPU &&
                   destination_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            sched->expert_cache->profile_transfer_d2h_groups++;
            sched->expert_cache->profile_transfer_d2h_bytes += bytes;
        } else if (source_type == GGML_BACKEND_DEVICE_TYPE_GPU &&
                   destination_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            sched->expert_cache->profile_transfer_d2d_groups++;
            sched->expert_cache->profile_transfer_d2d_bytes += bytes;
        } else {
            sched->expert_cache->profile_transfer_h2h_groups++;
            sched->expert_cache->profile_transfer_h2h_bytes += bytes;
        }
        sched->expert_cache->profile_transfer_async_groups += async_copy;
        sched->expert_cache->profile_transfer_fallback_groups += fallback_copy;
        sched->expert_cache->profile_transfer_issue_ns += issue_ns;
        sched->expert_cache->profile_transfer_source_sync_ns += source_sync_ns;
        sched->expert_cache->profile_transfer_destination_sync_ns += destination_sync_ns;
        ggml_backend_moe_dynamic_tracef(
            "\"event\":\"split_transfer\",\"split\":%d,\"tensor\":\"%s\","
            "\"source_backend\":\"%s\",\"source_type\":\"%s\","
            "\"destination_backend\":\"%s\",\"destination_type\":\"%s\","
            "\"bytes\":%zu,\"async\":%s,\"fallback\":%s,"
            "\"issue_us\":%.3f,\"source_sync_us\":%.3f,\"destination_sync_us\":%.3f",
            split_id,
            tensor != nullptr ? tensor->name : "",
            ggml_backend_name(source_backend), backend_type_name(source_backend),
            ggml_backend_name(destination_backend), backend_type_name(destination_backend),
            bytes,
            async_copy ? "true" : "false",
            fallback_copy ? "true" : "false",
            issue_ns / 1000.0,
            source_sync_ns / 1000.0,
            destination_sync_ns / 1000.0);
    };

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];
        const char * dynamic_split_kind = ggml_backend_moe_dynamic_split_kind(split);
        const bool dynamic_hot_compute_split =
            ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            ggml_backend_moe_dynamic_split_has_hot_compute(split);
        const bool dynamic_cold_compute_split =
            ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU &&
            dynamic_split_kind != nullptr && strcmp(dynamic_split_kind, "mixed") == 0;
        const int32_t dynamic_layer =
            dynamic_split_kind != nullptr || dynamic_hot_compute_split || dynamic_cold_compute_split
                ? ggml_backend_moe_dynamic_split_layer(split)
                : -1;

        bool dynamic_split_enabled = true;
        int32_t dynamic_ready_routes = 0;
        int32_t dynamic_gpu_routes = 0;
        int32_t dynamic_total_routes = 0;
        bool dynamic_split_state_known = dynamic_layer >= 0 &&
            ggml_backend_moe_dynamic_layer_split_state(
                dynamic_layer,
                &dynamic_split_enabled,
                &dynamic_ready_routes,
                &dynamic_gpu_routes,
                &dynamic_total_routes);

        bool dynamic_decision_synchronized = false;
        if (dynamic_hot_compute_split && dynamic_layer >= 0 && !dynamic_split_state_known) {
            ggml_tensor * route_ids = nullptr;
            auto consider_route_tensor = [&](ggml_tensor * root) {
                for (ggml_tensor * candidate = root; candidate != nullptr; candidate = candidate->view_src) {
                    if (strstr(candidate->name, "ffn_moe_dynamic_hot_ids") != nullptr) {
                        route_ids = candidate;
                        return;
                    }
                }
            };
            for (int node_index = 0; node_index < split->graph.n_nodes && route_ids == nullptr; ++node_index) {
                ggml_tensor * node = split->graph.nodes[node_index];
                consider_route_tensor(node);
                for (int source_index = 0; source_index < GGML_MAX_SRC && route_ids == nullptr; ++source_index) {
                    if (node->src[source_index] != nullptr) {
                        consider_route_tensor(node->src[source_index]);
                    }
                }
            }
            if (route_ids != nullptr) {
                ggml_tensor * route_sync_tensor = route_ids;
                for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
                    ggml_tensor * copied_input = tensor_copy(
                        split->inputs[input_id], split_backend_id, sched->cur_copy);
                    for (ggml_tensor * candidate = route_ids; candidate != nullptr; candidate = candidate->view_src) {
                        if (candidate == copied_input) {
                            route_sync_tensor = split->inputs[input_id];
                            break;
                        }
                    }
                    if (route_sync_tensor == split->inputs[input_id]) {
                        break;
                    }
                }
                ggml_backend_t route_backend = ggml_backend_sched_get_tensor_backend(sched, route_sync_tensor);
                if (route_backend != nullptr) {
                    const auto sync_start = std::chrono::steady_clock::now();
                    ggml_backend_synchronize(route_backend);
                    const uint64_t sync_ns = (uint64_t)
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - sync_start).count();
                    dynamic_decision_synchronized = true;
                    if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                        sched->expert_cache->profile_dynamic_decision_sync_calls++;
                        sched->expert_cache->profile_dynamic_decision_sync_ns += sync_ns;
                    }
                    dynamic_split_state_known = ggml_backend_moe_dynamic_layer_split_state(
                        dynamic_layer,
                        &dynamic_split_enabled,
                        &dynamic_ready_routes,
                        &dynamic_gpu_routes,
                        &dynamic_total_routes);
                    ggml_backend_moe_dynamic_tracef(
                        "\"event\":\"hot_decision_sync\",\"split\":%d,\"layer\":%d,"
                        "\"tensor\":\"%s\",\"sync_tensor\":\"%s\","
                        "\"backend\":\"%s\",\"duration_us\":%.3f,\"state_known\":%s",
                        split_id, dynamic_layer, route_ids->name, route_sync_tensor->name,
                        ggml_backend_name(route_backend), sync_ns / 1000.0,
                        dynamic_split_state_known ? "true" : "false");
                }
            } else {
                ggml_backend_moe_dynamic_tracef(
                    "\"event\":\"hot_decision_missing\",\"split\":%d,\"layer\":%d,"
                    "\"backend\":\"%s\",\"nodes\":%d,\"inputs\":%d",
                    split_id, dynamic_layer, ggml_backend_name(split_backend),
                    split->graph.n_nodes, split->n_inputs);
            }
        }

        const bool skip_dynamic_hot_split =
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_SKIP_HOT_BRANCH", 1) != 0 &&
            dynamic_hot_compute_split && dynamic_split_state_known && !dynamic_split_enabled;
        const bool force_gpu_only =
            ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_FORCE_GPU_ONLY", 0) != 0;
        const bool skip_dynamic_cold_split =
            dynamic_cold_compute_split && dynamic_split_state_known &&
            (force_gpu_only ||
             (ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_SKIP_COLD_BRANCH", 1) != 0 &&
              dynamic_total_routes > 0 && dynamic_gpu_routes == dynamic_total_routes));
        if (sched->expert_cache != nullptr && sched->expert_cache->profile && dynamic_hot_compute_split) {
            sched->expert_cache->profile_dynamic_hot_splits++;
            sched->expert_cache->profile_dynamic_hot_skipped += skip_dynamic_hot_split;
            if (skip_dynamic_hot_split) {
                sched->expert_cache->profile_dynamic_hot_nodes_skipped += split->graph.n_nodes;
            }
        }
        if (skip_dynamic_hot_split) {
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"hot_split_skip\",\"split\":%d,\"layer\":%d,"
                "\"backend\":\"%s\",\"nodes\":%d,\"inputs\":%d,"
                "\"ready_routes\":%d,\"gpu_routes\":%d",
                split_id, dynamic_layer, ggml_backend_name(split_backend),
                split->graph.n_nodes, split->n_inputs,
                dynamic_ready_routes, dynamic_gpu_routes);
        }
        if (dynamic_split_kind != nullptr) {
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"split_prepare\",\"split\":%d,\"kind\":\"%s\","
                "\"backend\":\"%s\",\"nodes\":%d,\"inputs\":%d,"
                "\"layer\":%d,\"hot_compute\":%s,\"decision_sync\":%s,"
                "\"state_known\":%s,\"enabled\":%s",
                split_id, dynamic_split_kind, ggml_backend_name(split_backend),
                split->graph.n_nodes, split->n_inputs, dynamic_layer,
                dynamic_hot_compute_split ? "true" : "false",
                dynamic_decision_synchronized ? "true" : "false",
                dynamic_split_state_known ? "true" : "false",
                dynamic_split_enabled ? "true" : "false");
        }

        if (skip_dynamic_cold_split) {
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"cold_split_skip\",\"split\":%d,\"layer\":%d,"
                "\"backend\":\"%s\",\"nodes\":%d,\"inputs\":%d,"
                "\"ready_routes\":%d,\"gpu_routes\":%d,\"total_routes\":%d",
                split_id, dynamic_layer, ggml_backend_name(split_backend),
                split->graph.n_nodes, split->n_inputs,
                dynamic_ready_routes, dynamic_gpu_routes, dynamic_total_routes);
            for (int node_index = 0; node_index < split->graph.n_nodes; ++node_index) {
                skipped_dynamic_tensors.insert(split->graph.nodes[node_index]);
            }
            continue;
        }

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
            uint64_t input_target_sync_ns = 0;

            if (ggml_backend_moe_dynamic_tensor_is_skipped(skipped_dynamic_tensors, input)) {
                const auto zero_target_sync_start = std::chrono::steady_clock::now();
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                input_target_sync_ns = (uint64_t)
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - zero_target_sync_start).count();
                if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                    sched->expert_cache->profile_transfer_destination_sync_ns += input_target_sync_ns;
                }
                const size_t zero_bytes = ggml_nbytes(input_cpy);
                ggml_backend_tensor_memset(input_cpy, 0, 0, zero_bytes);
                if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                    sched->expert_cache->profile_dynamic_zero_fills++;
                    sched->expert_cache->profile_dynamic_zero_bytes += zero_bytes;
                }
                ggml_backend_moe_dynamic_tracef(
                    "\"event\":\"skipped_input_zero\",\"split\":%d,"
                    "\"tensor\":\"%s\",\"backend\":\"%s\",\"bytes\":%zu",
                    split_id, input->name, ggml_backend_name(split_backend), zero_bytes);
                continue;
            }

            const bool dynamic_hot_weight_input = skip_dynamic_hot_split &&
                strncmp(input->name, "ffn_moe_dynamic_hot_", 20) == 0 &&
                input->buffer != nullptr &&
                ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                ggml_backend_buffer_is_host(input->buffer);
            if (skip_dynamic_hot_split && !dynamic_hot_weight_input) {
                continue;
            }
            if (dynamic_hot_weight_input) {
                if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                    sched->expert_cache->profile_dynamic_hot_promotion_inputs++;
                }
                ggml_backend_moe_dynamic_tracef(
                    "\"event\":\"hot_split_promotion_input\",\"split\":%d,"
                    "\"layer\":%d,\"tensor\":\"%s\"",
                    split_id, dynamic_layer, input->name);
            }

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                const auto target_sync_start = std::chrono::steady_clock::now();
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                input_target_sync_ns = (uint64_t)
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - target_sync_start).count();
                const auto copy_issue_start = std::chrono::steady_clock::now();
                ggml_backend_tensor_copy(input, input_cpy);
                const uint64_t copy_issue_ns = (uint64_t)
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - copy_issue_start).count();
                profile_transfer(
                    split_id, input, input_backend, split_backend, ggml_nbytes(input_cpy),
                    false, true, copy_issue_ns, 0, input_target_sync_ns);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                const bool dynamic_hot_input =
                    strncmp(input->name, "ffn_moe_dynamic_hot_", 20) == 0;
                const auto target_sync_start = std::chrono::steady_clock::now();
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                input_target_sync_ns = (uint64_t)
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - target_sync_start).count();
                if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                    sched->expert_cache->profile_target_sync_calls++;
                    sched->expert_cache->profile_target_sync_ns += input_target_sync_ns;
                    if (dynamic_hot_input) {
                        sched->expert_cache->profile_dynamic_target_sync_calls++;
                        sched->expert_cache->profile_dynamic_target_sync_ns += input_target_sync_ns;
                    }
                }
                if (dynamic_hot_input) {
                    ggml_backend_moe_dynamic_tracef(
                        "\"event\":\"target_sync\",\"tensor\":\"%s\","
                        "\"backend\":\"%s\",\"duration_us\":%.3f",
                        input->name, ggml_backend_name(split_backend), input_target_sync_ns / 1000.0);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = nullptr;
                if (ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer)) {
                    for (int node_index = 0; node_index < split->graph.n_nodes; ++node_index) {
                        ggml_tensor * candidate = split->graph.nodes[node_index];
                        if (candidate->op == GGML_OP_MUL_MAT_ID && candidate->src[0] == input_cpy) {
                            node = candidate;
                            break;
                        }
                    }
                }
                if (node != nullptr) {
                    const bool gpu_route_map_node = ggml_get_op_params_i32(node, 2) > 0;

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];
                    ggml_moe_trace_weight(moe_trace_call, input, n_expert, expert_size);
                    if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                        sched->expert_cache->profile_selective_inputs++;
                    }

                    if (!gpu_route_map_node) {
                        const auto input_sync_start = std::chrono::steady_clock::now();
                        ggml_backend_synchronize(input_backend);
                        if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                            sched->expert_cache->profile_input_sync_ns += (uint64_t)
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - input_sync_start).count();
                        }
                    }

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // If the IDs tensor is also an input of the split and has not been copied yet,
                    // use the original tensor. Promotion-only preparation never consumes route IDs,
                    // but searching all inputs makes the diagnostic mapping explicit.
                    const int first_ids_input = skip_dynamic_hot_split ? 0 : input_id + 1;
                    for (int i = first_ids_input; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (skip_dynamic_hot_split || gpu_route_map_node) {
                        // Promotion-only preparation and GPU route mapping are driven by
                        // registry slot state. Canonical route IDs must not be interpreted
                        // as compact slot IDs by scheduler-local selective copying.
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        prev_ids_tensor = nullptr;
                        if (gpu_route_map_node) {
                            ggml_backend_moe_dynamic_tracef(
                                "\"event\":\"gpu_route_map_selective_copy_skip\","
                                "\"tensor\":\"%s\",\"backend\":\"%s\"",
                                input->name, ggml_backend_name(split_backend));
                        }
                    } else if (ids_tensor != prev_ids_tensor) {
                        const auto ids_sync_start = std::chrono::steady_clock::now();
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);
                        if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                            sched->expert_cache->profile_ids_reads++;
                            sched->expert_cache->profile_ids_sync_ns += (uint64_t)
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - ids_sync_start).count();
                        }

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                if (id < 0) {
                                    GGML_ASSERT(node->op == GGML_OP_MUL_MAT_ID &&
                                        ggml_get_op_params_i32(node, 0) != 0 && id == -1);
                                    continue;
                                }
                                GGML_ASSERT(id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        ggml_moe_trace_route(moe_trace_call, split_backend, input, ids_tensor, ids, n_expert);
                        prev_ids_tensor = ids_tensor;
                    }

                    if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                        for (int32_t expert_id = 0; expert_id < n_expert; ++expert_id) {
                            if (ggml_bitset_get(used_ids.data(), expert_id)) {
                                sched->expert_cache->profile_experts_requested++;
                            }
                        }
                    }

                    const auto attach_start = std::chrono::steady_clock::now();
                    ggml_backend_sched_expert_cache_entry * cache_entry =
                        ggml_backend_sched_expert_cache_attach(sched, input, split_backend_id, input_cpy);
                    if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                        sched->expert_cache->profile_attach_ns += (uint64_t)
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - attach_start).count();
                    }

                    if (cache_entry != nullptr && input_cpy->buffer == cache_entry->buffer &&
                        sched->expert_cache->vmm && sched->expert_cache->vmm_promote != nullptr &&
                        ids_tensor->ne[1] == 1) {
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        int32_t first_id = -1;
                        int32_t last_id = -1;
                        auto promote_range = [&]() {
                            if (first_id < 0) {
                                return;
                            }
                            const size_t offset = (size_t) first_id * expert_size;
                            const size_t bytes = (size_t) (last_id - first_id + 1) * expert_size +
                                (last_id < n_expert - 1 ? padding : 0);
                            sched->expert_cache->vmm_promote(cache_entry->buffer, offset, bytes);
                        };
                        for (int32_t expert_id = 0; expert_id < n_expert; ++expert_id) {
                            if (!ggml_bitset_get(used_ids.data(), expert_id) ||
                                !cache_entry->resident[(size_t) expert_id]) {
                                continue;
                            }
                            if (first_id < 0) {
                                first_id = last_id = expert_id;
                            } else if (expert_id == last_id + 1) {
                                last_id = expert_id;
                            } else {
                                promote_range();
                                first_id = last_id = expert_id;
                            }
                        }
                        promote_range();
                    }

                    const bool static_compact =
                        strncmp(input->name, "ffn_moe_static_hot_", 19) == 0 &&
                        input->view_src != nullptr;
                    const bool dynamic_compact =
                        strncmp(input->name, "ffn_moe_dynamic_hot_", 20) == 0 &&
                        input->view_src != nullptr;
                    const bool compact = static_compact || dynamic_compact;
                    const ggml_tensor * copy_source = compact ? input->view_src : input;
                    ggml_backend_t copy_source_backend = ggml_backend_sched_get_tensor_backend(
                        sched, const_cast<ggml_tensor *>(copy_source));
                    if (copy_source_backend == nullptr) {
                        copy_source_backend = input_backend;
                    }
                    const std::vector<int32_t> * static_slot_map = nullptr;
                    int32_t compact_layer = -1;
                    int32_t compact_component = -1;
                    if (compact) {
                        GGML_ASSERT(copy_source->ne[2] >= input->ne[2]);
                        GGML_ASSERT(copy_source->nb[2] == expert_size);
                        const char * layer_suffix = strrchr(input->name, '-');
                        compact_layer = layer_suffix != nullptr ? atoi(layer_suffix + 1) : -1;
                    }
                    if (static_compact) {
                        static_slot_map = &ggml_backend_sched_static_slot_map(
                            input, (int32_t) input->ne[2], (int32_t) copy_source->ne[2]);
                    } else if (dynamic_compact) {
                        if (strstr(input->name, "_gate_weights") != nullptr) {
                            compact_component = 0;
                        } else if (strstr(input->name, "_up_weights") != nullptr) {
                            compact_component = 1;
                        } else if (strstr(input->name, "_down_weights") != nullptr) {
                            compact_component = 2;
                        }
                        GGML_ASSERT(compact_layer >= 0 && compact_component >= 0);
                    }

                    if (dynamic_compact && cache_entry != nullptr &&
                        input_cpy->buffer == cache_entry->buffer) {
                        GGML_ASSERT(cache_entry->n_expert == n_expert);
                        GGML_ASSERT(cache_entry->persistent_tensor != nullptr);
                        ggml_backend_moe_dynamic_register_component_binding(
                            compact_layer,
                            compact_component,
                            copy_source,
                            cache_entry->persistent_tensor.get(),
                            expert_size,
                            sched->expert_cache->promotion_worker,
                            sched->expert_cache,
                            cache_entry->resident.data());
                        for (int32_t slot = 0; slot < n_expert; ++slot) {
                            if (!ggml_backend_moe_dynamic_slot_needs_component(
                                    compact_layer, slot, compact_component)) {
                                continue;
                            }
                            const int32_t source_expert =
                                ggml_backend_moe_dynamic_slot_expert(compact_layer, slot);
                            GGML_ASSERT(source_expert >= 0 && source_expert < copy_source->ne[2]);
                            const size_t source_offset = (size_t) source_expert * copy_source->nb[2];
                            const size_t destination_offset = (size_t) slot * expert_size;
                            ggml_backend_moe_dynamic_slot_component_enqueued(
                                compact_layer, slot, compact_component);

                            bool queued = false;
                            uint64_t promotion_issue_ns = 0;
                            uint64_t promotion_sync_ns = 0;
                            const auto promotion_issue_start = std::chrono::steady_clock::now();
                            if (sched->expert_cache->promotion_worker != nullptr) {
                                ggml_backend_moe_promotion_job job;
                                job.layer = compact_layer;
                                job.slot = slot;
                                job.expert = source_expert;
                                job.component = compact_component;
                                job.source = (const uint8_t *) copy_source->data + source_offset;
                                job.source_pinned = strstr(
                                    ggml_backend_buffer_name(copy_source->buffer), "CUDA_Host") != nullptr;
                                job.destination = cache_entry->persistent_tensor.get();
                                job.destination_offset = destination_offset;
                                job.bytes = expert_size;
                                queued = sched->expert_cache->promotion_worker->enqueue(std::move(job));
                            }
                            promotion_issue_ns = (uint64_t)
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - promotion_issue_start).count();

                            if (!queued) {
                                ggml_backend_moe_dynamic_tracef(
                                    "\"event\":\"fallback_transfer_begin\",\"layer\":%d,"
                                    "\"slot\":%d,\"expert\":%d,\"component\":%d,"
                                    "\"bytes\":%zu,\"source_buffer\":\"%s\"",
                                    compact_layer,
                                    slot,
                                    source_expert,
                                    compact_component,
                                    expert_size,
                                    ggml_backend_buffer_name(copy_source->buffer));
                                const auto issue_start = std::chrono::steady_clock::now();
                                ggml_backend_tensor_set_async(
                                    split_backend,
                                    cache_entry->persistent_tensor.get(),
                                    (const uint8_t *) copy_source->data + source_offset,
                                    destination_offset,
                                    expert_size);
                                const uint64_t issue_ns = (uint64_t)
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - issue_start).count();
                                const auto sync_start = std::chrono::steady_clock::now();
                                ggml_backend_synchronize(split_backend);
                                const uint64_t sync_ns = (uint64_t)
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - sync_start).count();
                                promotion_issue_ns = issue_ns;
                                promotion_sync_ns = sync_ns;
                                ggml_backend_moe_dynamic_tracef(
                                    "\"event\":\"fallback_transfer_completed\",\"layer\":%d,"
                                    "\"slot\":%d,\"expert\":%d,\"component\":%d,"
                                    "\"bytes\":%zu,\"host_call_us\":%.3f,\"sync_us\":%.3f",
                                    compact_layer,
                                    slot,
                                    source_expert,
                                    compact_component,
                                    expert_size,
                                    issue_ns / 1000.0,
                                    sync_ns / 1000.0);
                                ggml_backend_moe_dynamic_slot_component_completed(
                                    compact_layer, slot, source_expert, compact_component);
                                if (sched->expert_cache->profile) {
                                    sched->expert_cache->profile_dynamic_copy_issue_ns += issue_ns;
                                    sched->expert_cache->profile_copy_issue_ns += issue_ns;
                                }
                            }

                            profile_transfer(
                                split_id, input, copy_source_backend, split_backend, expert_size,
                                queued, !queued, promotion_issue_ns, 0, promotion_sync_ns);
                            cache_entry->resident[(size_t) slot] = 1;
                            sched->expert_cache->bytes_uploaded += expert_size;
                            if (compact_component == 0) {
                                sched->expert_cache->misses++;
                            }
                            if (sched->expert_cache->profile) {
                                sched->expert_cache->profile_copy_groups++;
                                sched->expert_cache->profile_copy_bytes += expert_size;
                                sched->expert_cache->profile_copy_groups_decode++;
                                sched->expert_cache->profile_copy_bytes_decode += expert_size;
                                sched->expert_cache->profile_dynamic_copy_groups++;
                                sched->expert_cache->profile_dynamic_copy_bytes += expert_size;
                            }
                        }
                    }

                    if (dynamic_compact && cache_entry != nullptr &&
                        input_cpy->buffer == cache_entry->buffer) {
                        // Dynamic compact tensors are backed by the persistent slot cache. The
                        // dedicated component loop above is the only path that populates those
                        // slots, and a slot is exposed to routing only after all three components
                        // are READY. Falling through to the generic route-driven selective copy
                        // recopies already-resident expert weights on every executed hot split.
                        continue;
                    }

                    const std::vector<ggml_bitset_t> * ids_to_copy = &used_ids;
                    if (cache_entry != nullptr && input_cpy->buffer == cache_entry->buffer) {
                        GGML_ASSERT(cache_entry->n_expert == n_expert);
                        GGML_ASSERT(cache_entry->expert_size == expert_size);

                        missing_ids.clear();
                        missing_ids.resize(ggml_bitset_size(n_expert));
                        for (int32_t expert_id = 0; expert_id < n_expert; ++expert_id) {
                            if (!ggml_bitset_get(used_ids.data(), expert_id)) {
                                continue;
                            }
                            if (cache_entry->resident[(size_t) expert_id]) {
                                sched->expert_cache->hits++;
                                sched->expert_cache->bytes_avoided += expert_size;
                            } else {
                                ggml_bitset_set(missing_ids.data(), expert_id);
                                cache_entry->resident[(size_t) expert_id] = 1;
                                sched->expert_cache->misses++;
                                sched->expert_cache->bytes_uploaded += expert_size;
                                if (sched->expert_cache->profile) {
                                    sched->expert_cache->profile_missing_experts++;
                                }
                            }
                        }
                        ids_to_copy = &missing_ids;
                    }

                    const bool decode_selective_copy = ids_tensor->ne[1] == 1;
                    bool any_to_copy = false;
                    for (int32_t expert_id = 0; expert_id < n_expert; ++expert_id) {
                        if (ggml_bitset_get(ids_to_copy->data(), expert_id)) {
                            any_to_copy = true;
                            break;
                        }
                    }
                    if (!any_to_copy) {
                        continue;
                    }

                    if (compact) {
                        if (static_compact) {
                            GGML_ASSERT(static_slot_map != nullptr);
                        }
                        for (int32_t slot = 0; slot < n_expert; ++slot) {
                            if (!ggml_bitset_get(ids_to_copy->data(), slot)) {
                                continue;
                            }
                            const int32_t source_expert = static_compact
                                ? (*static_slot_map)[(size_t) slot]
                                : ggml_backend_moe_dynamic_slot_expert(compact_layer, slot);
                            GGML_ASSERT(source_expert >= 0 && source_expert < copy_source->ne[2]);
                            const size_t source_offset = (size_t) source_expert * copy_source->nb[2];
                            const size_t destination_offset = (size_t) slot * expert_size;
                            const auto copy_issue_start = std::chrono::steady_clock::now();
                            ggml_backend_tensor_set_async(
                                split_backend,
                                input_cpy,
                                (const uint8_t *) copy_source->data + source_offset,
                                destination_offset,
                                expert_size);
                            const uint64_t copy_issue_ns = (uint64_t)
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - copy_issue_start).count();
                            ggml_backend_moe_dynamic_tracef(
                                "\"event\":\"selective_copy\",\"call\":%" PRIu64
                                ",\"phase\":\"%s\",\"tensor\":\"%s\",\"kind\":\"compact_fallback\","
                                "\"slot\":%d,\"source_expert\":%d,\"bytes\":%zu,"
                                "\"host_call_us\":%.3f,\"cache_attached\":%s",
                                moe_trace_call, decode_selective_copy ? "decode" : "prefill",
                                input->name, slot, source_expert, expert_size, copy_issue_ns / 1000.0,
                                cache_entry != nullptr ? "true" : "false");
                            if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                                sched->expert_cache->profile_copy_groups++;
                                sched->expert_cache->profile_copy_bytes += expert_size;
                                if (decode_selective_copy) {
                                    sched->expert_cache->profile_copy_groups_decode++;
                                    sched->expert_cache->profile_copy_bytes_decode += expert_size;
                                } else {
                                    sched->expert_cache->profile_copy_groups_prefill++;
                                    sched->expert_cache->profile_copy_bytes_prefill += expert_size;
                                }
                                sched->expert_cache->profile_copy_issue_ns += copy_issue_ns;
                            }
                            profile_transfer(
                                split_id, input, copy_source_backend, split_backend, expert_size,
                                true, false, copy_issue_ns, 0, 0);
                        }
                    } else {
                        // group consecutive experts and copy them together
                        auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                            const size_t expert_offset = first_id * expert_size;
                            const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                            const size_t padding = std::min<size_t>(expert_size, 512);
                            const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                            const auto copy_issue_start = std::chrono::steady_clock::now();
                            ggml_backend_tensor_set_async(
                                split_backend,
                                input_cpy,
                                (const uint8_t *) input->data + expert_offset,
                                expert_offset,
                                // Copy a bit extra to ensure there are no NaNs in the padding of the last expert.
                                // This is necessary for MMQ in the CUDA backend.
                                expert_size_copy + padding_end);
                            const size_t copy_bytes = expert_size_copy + padding_end;
                            const uint64_t copy_issue_ns = (uint64_t)
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - copy_issue_start).count();
                            ggml_backend_moe_dynamic_tracef(
                                "\"event\":\"selective_copy\",\"call\":%" PRIu64
                                ",\"phase\":\"%s\",\"tensor\":\"%s\",\"kind\":\"ordinary\","
                                "\"first_expert\":%d,\"last_expert\":%d,\"bytes\":%zu,"
                                "\"host_call_us\":%.3f,\"cache_attached\":%s",
                                moe_trace_call, decode_selective_copy ? "decode" : "prefill",
                                input->name, first_id, last_id, copy_bytes, copy_issue_ns / 1000.0,
                                cache_entry != nullptr ? "true" : "false");
                            if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                                sched->expert_cache->profile_copy_groups++;
                                sched->expert_cache->profile_copy_bytes += copy_bytes;
                                if (decode_selective_copy) {
                                    sched->expert_cache->profile_copy_groups_decode++;
                                    sched->expert_cache->profile_copy_bytes_decode += copy_bytes;
                                } else {
                                    sched->expert_cache->profile_copy_groups_prefill++;
                                    sched->expert_cache->profile_copy_bytes_prefill += copy_bytes;
                                }
                                sched->expert_cache->profile_copy_issue_ns += copy_issue_ns;
                            }
                            profile_transfer(
                                split_id, input, copy_source_backend, split_backend, copy_bytes,
                                true, false, copy_issue_ns, 0, 0);
                        };

                        int id = 0;
                        while (!ggml_bitset_get(ids_to_copy->data(), id)) {
                            id++;
                        }
                        int32_t first_id = id;
                        int32_t last_id = first_id;

                        for (++id; id < n_expert; ++id) {
                            if (!ggml_bitset_get(ids_to_copy->data(), id)) {
                                continue;
                            }

                            if (id == last_id + 1) {
                                last_id = id;
                                continue;
                            }

                            copy_experts(first_id, last_id);

                            first_id = id;
                            last_id = id;
                        }
                        copy_experts(first_id, last_id);
                    }
                } else {
                    // Try an asynchronous backend copy first. Record the direction and every
                    // explicit synchronization in the fallback path without serializing successful copies.
                    const auto transfer_issue_start = std::chrono::steady_clock::now();
                    const bool async_copied = split_backend->iface.cpy_tensor_async != nullptr &&
                        split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy);
                    uint64_t transfer_issue_ns = (uint64_t)
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - transfer_issue_start).count();
                    uint64_t source_sync_ns = 0;
                    uint64_t destination_sync_ns = input_target_sync_ns;
                    if (!async_copied) {
                        const auto source_sync_start = std::chrono::steady_clock::now();
                        ggml_backend_synchronize(input_backend);
                        source_sync_ns = (uint64_t)
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - source_sync_start).count();

                        const auto destination_sync_start = std::chrono::steady_clock::now();
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        destination_sync_ns += (uint64_t)
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - destination_sync_start).count();

                        const auto fallback_copy_start = std::chrono::steady_clock::now();
                        ggml_backend_tensor_copy(input, input_cpy);
                        transfer_issue_ns += (uint64_t)
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - fallback_copy_start).count();
                    }
                    profile_transfer(
                        split_id, input, input_backend, split_backend, ggml_nbytes(input_cpy),
                        async_copied, !async_copied, transfer_issue_ns,
                        source_sync_ns, destination_sync_ns);
                }
            }
        }

        if (skip_dynamic_hot_split) {
            for (int node_index = 0; node_index < split->graph.n_nodes; ++node_index) {
                skipped_dynamic_tensors.insert(split->graph.nodes[node_index]);
            }
            continue;
        }

        if (dynamic_split_kind != nullptr) {
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"split_submit\",\"split\":%d,\"kind\":\"%s\","
                "\"backend\":\"%s\",\"nodes\":%d",
                split_id, dynamic_split_kind, ggml_backend_name(split_backend), split->graph.n_nodes);
        }
        const bool profile_dynamic_compute =
            sched->expert_cache != nullptr && sched->expert_cache->profile &&
            getenv("GGML_MOE_DYNAMIC_PROFILE_COMPUTE") != nullptr &&
            (dynamic_hot_compute_split || dynamic_cold_compute_split) &&
            !sched->callback_eval;
        std::chrono::steady_clock::time_point dynamic_compute_start;
        if (profile_dynamic_compute) {
            // Profiling-only serialization: establish a clean boundary so the measured
            // interval contains this split's backend work rather than earlier queued work.
            ggml_backend_synchronize(split_backend);
            dynamic_compute_start = std::chrono::steady_clock::now();
        }

        if (!sched->callback_eval) {
            const auto submit_start = std::chrono::steady_clock::now();
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            const uint64_t submit_ns = (uint64_t)
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - submit_start).count();
            if (sched->expert_cache != nullptr && sched->expert_cache->profile) {
                const enum ggml_backend_dev_type submit_type =
                    ggml_backend_dev_type(ggml_backend_get_device(split_backend));
                if (submit_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    sched->expert_cache->profile_submit_cpu_calls++;
                    sched->expert_cache->profile_submit_cpu_ns += submit_ns;
                } else if (submit_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    sched->expert_cache->profile_submit_gpu_calls++;
                    sched->expert_cache->profile_submit_gpu_ns += submit_ns;
                }
                if (dynamic_hot_compute_split) {
                    sched->expert_cache->profile_submit_dynamic_hot_calls++;
                    sched->expert_cache->profile_submit_dynamic_hot_ns += submit_ns;
                }
                if (dynamic_cold_compute_split) {
                    sched->expert_cache->profile_submit_dynamic_cold_calls++;
                    sched->expert_cache->profile_submit_dynamic_cold_ns += submit_ns;
                }
                ggml_backend_moe_dynamic_tracef(
                    "\"event\":\"split_submit_return\",\"split\":%d,\"layer\":%d,"
                    "\"backend\":\"%s\",\"backend_type\":\"%s\",\"nodes\":%d,"
                    "\"dynamic_hot\":%s,\"dynamic_cold\":%s,\"duration_us\":%.3f",
                    split_id, dynamic_layer, ggml_backend_name(split_backend),
                    backend_type_name(split_backend), split->graph.n_nodes,
                    dynamic_hot_compute_split ? "true" : "false",
                    dynamic_cold_compute_split ? "true" : "false",
                    submit_ns / 1000.0);
            }
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            if (profile_dynamic_compute) {
                ggml_backend_synchronize(split_backend);
                const uint64_t compute_ns = (uint64_t)
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - dynamic_compute_start).count();
                if (dynamic_hot_compute_split) {
                    sched->expert_cache->profile_dynamic_hot_compute_calls++;
                    sched->expert_cache->profile_dynamic_hot_compute_ns += compute_ns;
                }
                if (dynamic_cold_compute_split) {
                    sched->expert_cache->profile_dynamic_cold_compute_calls++;
                    sched->expert_cache->profile_dynamic_cold_compute_ns += compute_ns;
                }
                ggml_backend_moe_dynamic_tracef(
                    "\"event\":\"split_compute_profile\",\"split\":%d,\"layer\":%d,"
                    "\"kind\":\"%s\",\"backend\":\"%s\",\"nodes\":%d,"
                    "\"duration_us\":%.3f",
                    split_id, dynamic_layer,
                    dynamic_hot_compute_split ? "hot" : "cold",
                    ggml_backend_name(split_backend), split->graph.n_nodes,
                    compute_ns / 1000.0);
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;
    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    const char * expert_cache_mib_env = getenv("GGML_EXPERT_CACHE_MIB");
    if (expert_cache_mib_env != nullptr && sched->n_copies == 1) {
        size_t expert_cache_budget = 0;
        const bool auto_budget = strcmp(expert_cache_mib_env, "auto") == 0 ||
            strcmp(expert_cache_mib_env, "max") == 0;
        if (auto_budget) {
            ggml_backend_dev_t cache_device = nullptr;
            for (int b = 0; b < n_backends; ++b) {
                ggml_backend_dev_t device = ggml_backend_get_device(backends[b]);
                if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    cache_device = device;
                    break;
                }
            }
            if (cache_device != nullptr) {
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                ggml_backend_dev_memory(cache_device, &free_bytes, &total_bytes);
                const size_t mib = 1024ULL * 1024ULL;
                size_t reserve_bytes = std::max<size_t>(512ULL * mib, total_bytes / 32);
                const char * reserve_env = getenv("GGML_EXPERT_CACHE_RESERVE_MIB");
                if (reserve_env != nullptr && reserve_env[0] != '\0') {
                    char * reserve_end = nullptr;
                    const unsigned long long reserve_mib = strtoull(reserve_env, &reserve_end, 10);
                    if (reserve_end != reserve_env && *reserve_end == '\0' && reserve_mib <= SIZE_MAX / mib) {
                        reserve_bytes = (size_t) reserve_mib * mib;
                    }
                }
                expert_cache_budget = free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0;
                GGML_LOG_INFO(
                    "expert-cache: auto VRAM budget device=%s total=%.2f MiB free=%.2f MiB reserve=%.2f MiB budget=%.2f MiB\n",
                    ggml_backend_dev_name(cache_device),
                    total_bytes / 1024.0 / 1024.0,
                    free_bytes / 1024.0 / 1024.0,
                    reserve_bytes / 1024.0 / 1024.0,
                    expert_cache_budget / 1024.0 / 1024.0);
            } else {
                GGML_LOG_WARN("expert-cache: auto VRAM budget requested but no GPU backend is available\n");
            }
        } else {
            char * budget_end = nullptr;
            const unsigned long long expert_cache_mib = strtoull(expert_cache_mib_env, &budget_end, 10);
            if (budget_end != expert_cache_mib_env && *budget_end == '\0' &&
                expert_cache_mib > 0 && expert_cache_mib <= SIZE_MAX / (1024ULL * 1024ULL)) {
                expert_cache_budget = (size_t) expert_cache_mib * 1024ULL * 1024ULL;
            }
        }
        if (expert_cache_budget > 0) {
            sched->expert_cache = new ggml_backend_sched_expert_cache();
            sched->expert_cache->budget_bytes = expert_cache_budget;
            sched->expert_cache->vmm = getenv("GGML_EXPERT_CACHE_VMM") != nullptr;
            sched->expert_cache->managed = !sched->expert_cache->vmm && getenv("GGML_EXPERT_CACHE_MANAGED") != nullptr;
            sched->expert_cache->profile =
                ggml_backend_moe_dynamic_env_i32("GGML_EXPERT_CACHE_PROFILE", 0) != 0;
            const char * memory_kind = sched->expert_cache->vmm ? "hybrid-vmm" :
                (sched->expert_cache->managed ? "managed" : "device");
            GGML_LOG_INFO("expert-cache: enabled full-layout persistent cache with %.2f MiB budget (%s memory)\n",
                sched->expert_cache->budget_bytes / 1024.0 / 1024.0, memory_kind);
        } else {
            GGML_LOG_WARN("expert-cache: requested budget resolved to zero; cache disabled\n");
        }
    } else if (expert_cache_mib_env != nullptr && sched->n_copies > 1) {
        GGML_LOG_WARN("expert-cache: disabled because pipeline parallel copies are enabled\n");
    }

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    if (sched->expert_cache != nullptr &&
        getenv("GGML_MOE_DYNAMIC_SPLIT_SLOTS") != nullptr &&
        ggml_backend_moe_async_promotion_enabled()) {
        for (int b = 0; b < n_backends; ++b) {
            if (ggml_backend_dev_type(ggml_backend_get_device(backends[b])) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            auto * worker = new ggml_backend_moe_promotion_worker(
                ggml_backend_get_device(backends[b]), sched->expert_cache);
            if (worker->valid()) {
                sched->expert_cache->promotion_worker = worker;
                GGML_LOG_INFO("moe-promotion-worker: enabled on %s with a dedicated backend stream\n",
                    ggml_backend_name(worker->backend));
            } else {
                GGML_LOG_WARN("moe-promotion-worker: failed to create a dedicated backend; using synchronized fallback\n");
                delete worker;
            }
            break;
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }

    if (sched->expert_cache != nullptr && sched->expert_cache->promotion_worker != nullptr) {
        delete sched->expert_cache->promotion_worker;
        sched->expert_cache->promotion_worker = nullptr;
    }

    if (getenv("GGML_MOE_DYNAMIC_SPLIT_SLOTS") != nullptr) {
        ggml_backend_moe_dynamic_log_summary();
    }

    if (sched->expert_cache != nullptr) {
        // Backends are owned by llama_context and are destroyed before the scheduler.
        // Cache buffers own their CUDA allocations directly; freeing those buffers is
        // therefore the only safe teardown action here.
        GGML_LOG_INFO("expert-cache: entries=%zu allocated=%.2f MiB hits=%llu misses=%llu uploaded=%.2f MiB avoided=%.2f MiB attach_attempts=%llu rejected_host=%llu rejected_budget=%llu\n",
            sched->expert_cache->entries.size(),
            sched->expert_cache->allocated_bytes / 1024.0 / 1024.0,
            (unsigned long long) sched->expert_cache->hits,
            (unsigned long long) sched->expert_cache->misses,
            sched->expert_cache->bytes_uploaded / 1024.0 / 1024.0,
            sched->expert_cache->bytes_avoided / 1024.0 / 1024.0,
            (unsigned long long) sched->expert_cache->attach_attempts,
            (unsigned long long) sched->expert_cache->rejected_host_target,
            (unsigned long long) sched->expert_cache->rejected_budget);
        if (sched->expert_cache->profile) {
            GGML_LOG_INFO(
                "expert-cache-profile: selective-inputs=%" PRIu64 " ids-reads=%" PRIu64
                " requested-experts=%" PRIu64 " missing-experts=%" PRIu64 " copy-groups=%" PRIu64
                " copy-bytes=%.2f MiB prefill-groups=%" PRIu64 " prefill-bytes=%.2f MiB"
                " decode-groups=%" PRIu64 " decode-bytes=%.2f MiB"
                " dynamic-groups=%" PRIu64 " dynamic-bytes=%.2f MiB dynamic-issue=%.3f ms"
                " target-sync-calls=%" PRIu64 " target-sync=%.3f ms"
                " dynamic-target-sync-calls=%" PRIu64 " dynamic-target-sync=%.3f ms"
                " hot-splits=%" PRIu64 " hot-skipped=%" PRIu64 " hot-nodes-skipped=%" PRIu64
                " promotion-inputs=%" PRIu64 " zero-fills=%" PRIu64 " zero-bytes=%.2f MiB"
                " hot-compute-calls=%" PRIu64 " hot-compute=%.3f ms"
                " cold-compute-calls=%" PRIu64 " cold-compute=%.3f ms"
                " decision-sync-calls=%" PRIu64 " decision-sync=%.3f ms"
                " input-sync=%.3f ms ids-sync=%.3f ms attach=%.3f ms copy-issue=%.3f ms\n",
                sched->expert_cache->profile_selective_inputs,
                sched->expert_cache->profile_ids_reads,
                sched->expert_cache->profile_experts_requested,
                sched->expert_cache->profile_missing_experts,
                sched->expert_cache->profile_copy_groups,
                sched->expert_cache->profile_copy_bytes / 1024.0 / 1024.0,
                sched->expert_cache->profile_copy_groups_prefill,
                sched->expert_cache->profile_copy_bytes_prefill / 1024.0 / 1024.0,
                sched->expert_cache->profile_copy_groups_decode,
                sched->expert_cache->profile_copy_bytes_decode / 1024.0 / 1024.0,
                sched->expert_cache->profile_dynamic_copy_groups,
                sched->expert_cache->profile_dynamic_copy_bytes / 1024.0 / 1024.0,
                sched->expert_cache->profile_dynamic_copy_issue_ns / 1.0e6,
                sched->expert_cache->profile_target_sync_calls,
                sched->expert_cache->profile_target_sync_ns / 1.0e6,
                sched->expert_cache->profile_dynamic_target_sync_calls,
                sched->expert_cache->profile_dynamic_target_sync_ns / 1.0e6,
                sched->expert_cache->profile_dynamic_hot_splits,
                sched->expert_cache->profile_dynamic_hot_skipped,
                sched->expert_cache->profile_dynamic_hot_nodes_skipped,
                sched->expert_cache->profile_dynamic_hot_promotion_inputs,
                sched->expert_cache->profile_dynamic_zero_fills,
                sched->expert_cache->profile_dynamic_zero_bytes / 1024.0 / 1024.0,
                sched->expert_cache->profile_dynamic_hot_compute_calls,
                sched->expert_cache->profile_dynamic_hot_compute_ns / 1.0e6,
                sched->expert_cache->profile_dynamic_cold_compute_calls,
                sched->expert_cache->profile_dynamic_cold_compute_ns / 1.0e6,
                sched->expert_cache->profile_dynamic_decision_sync_calls,
                sched->expert_cache->profile_dynamic_decision_sync_ns / 1.0e6,
                sched->expert_cache->profile_input_sync_ns / 1.0e6,
                sched->expert_cache->profile_ids_sync_ns / 1.0e6,
                sched->expert_cache->profile_attach_ns / 1.0e6,
                sched->expert_cache->profile_copy_issue_ns / 1.0e6);
            GGML_LOG_INFO(
                "expert-cache-transfer-profile: h2d-groups=%" PRIu64 " h2d-bytes=%.2f MiB"
                " d2h-groups=%" PRIu64 " d2h-bytes=%.2f MiB"
                " d2d-groups=%" PRIu64 " d2d-bytes=%.2f MiB"
                " h2h-groups=%" PRIu64 " h2h-bytes=%.2f MiB"
                " async-groups=%" PRIu64 " fallback-groups=%" PRIu64
                " issue=%.3f ms source-sync=%.3f ms destination-sync=%.3f ms\n",
                sched->expert_cache->profile_transfer_h2d_groups,
                sched->expert_cache->profile_transfer_h2d_bytes / 1024.0 / 1024.0,
                sched->expert_cache->profile_transfer_d2h_groups,
                sched->expert_cache->profile_transfer_d2h_bytes / 1024.0 / 1024.0,
                sched->expert_cache->profile_transfer_d2d_groups,
                sched->expert_cache->profile_transfer_d2d_bytes / 1024.0 / 1024.0,
                sched->expert_cache->profile_transfer_h2h_groups,
                sched->expert_cache->profile_transfer_h2h_bytes / 1024.0 / 1024.0,
                sched->expert_cache->profile_transfer_async_groups,
                sched->expert_cache->profile_transfer_fallback_groups,
                sched->expert_cache->profile_transfer_issue_ns / 1.0e6,
                sched->expert_cache->profile_transfer_source_sync_ns / 1.0e6,
                sched->expert_cache->profile_transfer_destination_sync_ns / 1.0e6);
            GGML_LOG_INFO(
                "expert-cache-submit-profile: cpu-calls=%" PRIu64 " cpu-return=%.3f ms"
                " gpu-calls=%" PRIu64 " gpu-return=%.3f ms"
                " hot-calls=%" PRIu64 " hot-return=%.3f ms"
                " cold-calls=%" PRIu64 " cold-return=%.3f ms\n",
                sched->expert_cache->profile_submit_cpu_calls,
                sched->expert_cache->profile_submit_cpu_ns / 1.0e6,
                sched->expert_cache->profile_submit_gpu_calls,
                sched->expert_cache->profile_submit_gpu_ns / 1.0e6,
                sched->expert_cache->profile_submit_dynamic_hot_calls,
                sched->expert_cache->profile_submit_dynamic_hot_ns / 1.0e6,
                sched->expert_cache->profile_submit_dynamic_cold_calls,
                sched->expert_cache->profile_submit_dynamic_cold_ns / 1.0e6);
        }
        for (auto & entry : sched->expert_cache->entries) {
            ggml_backend_buffer_free(entry.buffer);
        }
        delete sched->expert_cache;
        sched->expert_cache = nullptr;
    }

    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    const auto sched_compute_start = std::chrono::steady_clock::now();
    const bool was_allocated = sched->is_alloc;
    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"sched_compute_begin\",\"nodes\":%d,\"leafs\":%d,"
        "\"splits\":%d,\"allocated\":%s,\"copy\":%d",
        graph->n_nodes, graph->n_leafs, sched->n_splits,
        was_allocated ? "true" : "false", sched->cur_copy);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    if (ggml_backend_moe_dynamic_env_i32("GGML_MOE_DYNAMIC_TRACE_SCHED_SPLITS", 0) != 0) {
        for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
            const auto & split = sched->splits[split_id];
            const char * first_name = split.graph.n_nodes > 0 && split.graph.nodes[0] != nullptr
                ? split.graph.nodes[0]->name : "";
            const char * last_name = split.graph.n_nodes > 0 && split.graph.nodes[split.graph.n_nodes - 1] != nullptr
                ? split.graph.nodes[split.graph.n_nodes - 1]->name : "";
            ggml_backend_moe_dynamic_tracef(
                "\"event\":\"sched_split_layout\",\"split\":%d,"
                "\"backend\":\"%s\",\"nodes\":%d,\"inputs\":%d,"
                "\"i_start\":%d,\"i_end\":%d,\"first\":\"%s\",\"last\":\"%s\"",
                split_id,
                ggml_backend_name(sched->backends[split.backend_id]),
                split.graph.n_nodes,
                split.n_inputs,
                split.i_start,
                split.i_end,
                first_name,
                last_name);
        }
    }

    const enum ggml_status status = ggml_backend_sched_compute_splits(sched);
    const double duration_us = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - sched_compute_start).count() / 1000.0;
    ggml_backend_moe_dynamic_tracef(
        "\"event\":\"sched_compute_end\",\"nodes\":%d,\"splits\":%d,"
        "\"allocated_before\":%s,\"allocated_after\":%s,"
        "\"status\":%d,\"duration_us\":%.3f",
        graph->n_nodes, sched->n_splits,
        was_allocated ? "true" : "false",
        sched->is_alloc ? "true" : "false",
        (int) status, duration_us);
    return status;
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

size_t ggml_backend_sched_get_expert_cache_budget(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->expert_cache != nullptr ? sched->expert_cache->budget_bytes : 0;
}

void ggml_backend_sched_reset_expert_cache_profile(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (sched->expert_cache == nullptr) {
        return;
    }

    auto & cache = *sched->expert_cache;
    cache.hits = 0;
    cache.misses = 0;
    cache.bytes_uploaded = 0;
    cache.bytes_avoided = 0;
    cache.attach_attempts = 0;
    cache.rejected_host_target = 0;
    cache.rejected_budget = 0;
    cache.profile_selective_inputs = 0;
    cache.profile_ids_reads = 0;
    cache.profile_copy_groups = 0;
    cache.profile_copy_bytes = 0;
    cache.profile_copy_groups_prefill = 0;
    cache.profile_copy_bytes_prefill = 0;
    cache.profile_copy_groups_decode = 0;
    cache.profile_copy_bytes_decode = 0;
    cache.profile_dynamic_copy_groups = 0;
    cache.profile_dynamic_copy_bytes = 0;
    cache.profile_dynamic_copy_issue_ns = 0;
    cache.profile_experts_requested = 0;
    cache.profile_missing_experts = 0;
    cache.profile_input_sync_ns = 0;
    cache.profile_target_sync_calls = 0;
    cache.profile_target_sync_ns = 0;
    cache.profile_dynamic_target_sync_calls = 0;
    cache.profile_dynamic_target_sync_ns = 0;
    cache.profile_dynamic_hot_splits = 0;
    cache.profile_dynamic_hot_skipped = 0;
    cache.profile_dynamic_hot_nodes_skipped = 0;
    cache.profile_dynamic_hot_promotion_inputs = 0;
    cache.profile_dynamic_zero_fills = 0;
    cache.profile_dynamic_zero_bytes = 0;
    cache.profile_dynamic_hot_compute_calls = 0;
    cache.profile_dynamic_hot_compute_ns = 0;
    cache.profile_dynamic_cold_compute_calls = 0;
    cache.profile_dynamic_cold_compute_ns = 0;
    cache.profile_dynamic_decision_sync_calls = 0;
    cache.profile_dynamic_decision_sync_ns = 0;
    cache.profile_ids_sync_ns = 0;
    cache.profile_attach_ns = 0;
    cache.profile_copy_issue_ns = 0;
    cache.profile_transfer_h2d_groups = 0;
    cache.profile_transfer_h2d_bytes = 0;
    cache.profile_transfer_d2h_groups = 0;
    cache.profile_transfer_d2h_bytes = 0;
    cache.profile_transfer_d2d_groups = 0;
    cache.profile_transfer_d2d_bytes = 0;
    cache.profile_transfer_h2h_groups = 0;
    cache.profile_transfer_h2h_bytes = 0;
    cache.profile_transfer_async_groups = 0;
    cache.profile_transfer_fallback_groups = 0;
    cache.profile_transfer_issue_ns = 0;
    cache.profile_transfer_source_sync_ns = 0;
    cache.profile_transfer_destination_sync_ns = 0;
    cache.profile_submit_cpu_calls = 0;
    cache.profile_submit_cpu_ns = 0;
    cache.profile_submit_gpu_calls = 0;
    cache.profile_submit_gpu_ns = 0;
    cache.profile_submit_dynamic_hot_calls = 0;
    cache.profile_submit_dynamic_hot_ns = 0;
    cache.profile_submit_dynamic_cold_calls = 0;
    cache.profile_submit_dynamic_cold_ns = 0;
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}

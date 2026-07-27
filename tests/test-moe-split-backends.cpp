#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct backend_pair {
    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;
};

struct weight_store {
    ggml_context * ctx_cpu = nullptr;
    ggml_context * ctx_gpu = nullptr;
    ggml_backend_buffer_t buf_cpu = nullptr;
    ggml_backend_buffer_t buf_gpu = nullptr;
    ggml_tensor * full = nullptr;
    ggml_tensor * hot = nullptr;
};

struct ffn_weight_store {
    ggml_context * ctx_cpu = nullptr;
    ggml_context * ctx_gpu = nullptr;
    ggml_backend_buffer_t buf_cpu = nullptr;
    ggml_backend_buffer_t buf_gpu = nullptr;
    ggml_tensor * full_gate_up = nullptr;
    ggml_tensor * hot_gate_up = nullptr;
    ggml_tensor * full_down = nullptr;
    ggml_tensor * hot_down = nullptr;
};

struct run_result {
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_abs = 0.0;
    double nmse = 0.0;
    int splits = 0;
    int copies = 0;
    std::vector<float> output;
};

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "error: %s\n", message);
    std::exit(1);
}

void check_status(enum ggml_status status, const char * operation) {
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "error: %s failed with status %d\n", operation, static_cast<int>(status));
        std::exit(1);
    }
}

backend_pair init_backends(int cpu_threads) {
    ggml_backend_load_all();
    backend_pair result;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU && result.gpu == nullptr) {
            result.gpu = ggml_backend_dev_init(device, nullptr);
        } else if (type == GGML_BACKEND_DEVICE_TYPE_CPU && result.cpu == nullptr) {
            result.cpu = ggml_backend_dev_init(device, nullptr);
        }
    }
    if (result.gpu == nullptr || result.cpu == nullptr) {
        fail("both a GPU and CPU backend are required");
    }

    ggml_backend_dev_t cpu_device = ggml_backend_get_device(result.cpu);
    ggml_backend_reg_t cpu_registry = ggml_backend_dev_backend_reg(cpu_device);
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(cpu_registry, "ggml_backend_set_n_threads"));
    if (set_threads != nullptr) {
        set_threads(result.cpu, cpu_threads);
    }

    std::printf("gpu=%s cpu=%s cpu_threads=%d\n",
        ggml_backend_name(result.gpu), ggml_backend_name(result.cpu), cpu_threads);
    return result;
}

ggml_context * make_context(size_t tensors) {
    ggml_init_params params = {
        /* .mem_size   = */ std::max<size_t>(1 << 20, tensors * ggml_tensor_overhead() * 4),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        fail("ggml_init failed");
    }
    return ctx;
}

std::vector<uint8_t> quantize_expert(
        ggml_type type,
        int64_t k,
        int64_t m,
        int expert) {
    const size_t elements = static_cast<size_t>(k) * static_cast<size_t>(m);
    std::vector<float> source(elements);
    uint32_t state = 0x9e3779b9u ^ static_cast<uint32_t>(expert * 0x85ebca6bu);
    for (size_t index = 0; index < elements; ++index) {
        state = state * 1664525u + 1013904223u;
        const int32_t centered = static_cast<int32_t>((state >> 8) & 0xffffu) - 32768;
        source[index] = static_cast<float>(centered) / 32768.0f * 0.05f;
    }

    const size_t bytes = ggml_row_size(type, k) * static_cast<size_t>(m);
    std::vector<uint8_t> quantized(bytes);
    const size_t block = ggml_blck_size(type);
    if (elements % block != 0) {
        fail("expert element count is not divisible by quantization block size");
    }
    std::vector<float> importance(static_cast<size_t>(k), 1.0f);
    const float * imatrix = ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr;
    const size_t written = ggml_quantize_chunk(
        type,
        source.data(),
        quantized.data(),
        0,
        elements / block,
        block,
        imatrix);
    if (written != bytes) {
        std::fprintf(stderr, "error: quantized bytes mismatch: expected %zu got %zu\n", bytes, written);
        std::exit(1);
    }
    return quantized;
}

weight_store build_weights(
        const backend_pair & backends,
        ggml_type type,
        int64_t k,
        int64_t m,
        int n_experts,
        int hot_slots) {
    weight_store store;
    store.ctx_cpu = make_context(4);
    store.ctx_gpu = make_context(4);
    store.full = ggml_new_tensor_3d(store.ctx_cpu, type, k, m, n_experts);
    store.hot = ggml_new_tensor_3d(store.ctx_gpu, type, k, m, hot_slots);
    ggml_set_name(store.full, "full_weights");
    ggml_set_name(store.hot, "hot_weights");

    store.buf_cpu = ggml_backend_alloc_ctx_tensors(store.ctx_cpu, backends.cpu);
    store.buf_gpu = ggml_backend_alloc_ctx_tensors(store.ctx_gpu, backends.gpu);
    if (store.buf_cpu == nullptr || store.buf_gpu == nullptr) {
        fail("weight buffer allocation failed");
    }
    ggml_backend_buffer_set_usage(store.buf_cpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(store.buf_gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (int expert = 0; expert < n_experts; ++expert) {
        std::vector<uint8_t> data = quantize_expert(type, k, m, expert);
        if (data.size() != store.full->nb[2]) {
            std::fprintf(stderr, "error: full expert stride mismatch: data=%zu stride=%zu\n",
                data.size(), store.full->nb[2]);
            std::exit(1);
        }
        ggml_backend_tensor_set(
            store.full,
            data.data(),
            static_cast<size_t>(expert) * store.full->nb[2],
            data.size());
        if (expert < hot_slots) {
            if (data.size() != store.hot->nb[2]) {
                fail("hot expert stride mismatch");
            }
            ggml_backend_tensor_set(
                store.hot,
                data.data(),
                static_cast<size_t>(expert) * store.hot->nb[2],
                data.size());
        }
    }
    ggml_backend_synchronize(backends.gpu);
    ggml_backend_synchronize(backends.cpu);
    return store;
}

void free_weights(weight_store & store) {
    if (store.buf_gpu != nullptr) {
        ggml_backend_buffer_free(store.buf_gpu);
    }
    if (store.buf_cpu != nullptr) {
        ggml_backend_buffer_free(store.buf_cpu);
    }
    if (store.ctx_gpu != nullptr) {
        ggml_free(store.ctx_gpu);
    }
    if (store.ctx_cpu != nullptr) {
        ggml_free(store.ctx_cpu);
    }
    store = {};
}

ffn_weight_store build_ffn_weights(
        const backend_pair & backends,
        int n_experts,
        int hot_slots) {
    constexpr int64_t n_embd = 3072;
    constexpr int64_t n_ff = 1024;
    ffn_weight_store store;
    store.ctx_cpu = make_context(8);
    store.ctx_gpu = make_context(8);
    store.full_gate_up = ggml_new_tensor_3d(
        store.ctx_cpu, GGML_TYPE_IQ3_XXS, n_embd, n_ff, n_experts);
    store.full_down = ggml_new_tensor_3d(
        store.ctx_cpu, GGML_TYPE_IQ4_XS, n_ff, n_embd, n_experts);
    store.hot_gate_up = ggml_new_tensor_3d(
        store.ctx_gpu, GGML_TYPE_IQ3_XXS, n_embd, n_ff, hot_slots);
    store.hot_down = ggml_new_tensor_3d(
        store.ctx_gpu, GGML_TYPE_IQ4_XS, n_ff, n_embd, hot_slots);
    ggml_set_name(store.full_gate_up, "full_gate_up");
    ggml_set_name(store.full_down, "full_down");
    ggml_set_name(store.hot_gate_up, "hot_gate_up");
    ggml_set_name(store.hot_down, "hot_down");

    store.buf_cpu = ggml_backend_alloc_ctx_tensors(store.ctx_cpu, backends.cpu);
    store.buf_gpu = ggml_backend_alloc_ctx_tensors(store.ctx_gpu, backends.gpu);
    if (store.buf_cpu == nullptr || store.buf_gpu == nullptr) {
        fail("FFN weight buffer allocation failed");
    }
    ggml_backend_buffer_set_usage(store.buf_cpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(store.buf_gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (int expert = 0; expert < n_experts; ++expert) {
        std::vector<uint8_t> gate_up = quantize_expert(
            GGML_TYPE_IQ3_XXS, n_embd, n_ff, expert);
        std::vector<uint8_t> down = quantize_expert(
            GGML_TYPE_IQ4_XS, n_ff, n_embd, expert + 1000);
        if (gate_up.size() != store.full_gate_up->nb[2] || down.size() != store.full_down->nb[2]) {
            fail("FFN expert stride mismatch");
        }
        ggml_backend_tensor_set(
            store.full_gate_up,
            gate_up.data(),
            static_cast<size_t>(expert) * store.full_gate_up->nb[2],
            gate_up.size());
        ggml_backend_tensor_set(
            store.full_down,
            down.data(),
            static_cast<size_t>(expert) * store.full_down->nb[2],
            down.size());
        if (expert < hot_slots) {
            ggml_backend_tensor_set(
                store.hot_gate_up,
                gate_up.data(),
                static_cast<size_t>(expert) * store.hot_gate_up->nb[2],
                gate_up.size());
            ggml_backend_tensor_set(
                store.hot_down,
                down.data(),
                static_cast<size_t>(expert) * store.hot_down->nb[2],
                down.size());
        }
    }
    ggml_backend_synchronize(backends.gpu);
    ggml_backend_synchronize(backends.cpu);
    return store;
}

void free_ffn_weights(ffn_weight_store & store) {
    if (store.buf_gpu != nullptr) {
        ggml_backend_buffer_free(store.buf_gpu);
    }
    if (store.buf_cpu != nullptr) {
        ggml_backend_buffer_free(store.buf_cpu);
    }
    if (store.ctx_gpu != nullptr) {
        ggml_free(store.ctx_gpu);
    }
    if (store.ctx_cpu != nullptr) {
        ggml_free(store.ctx_cpu);
    }
    store = {};
}

std::vector<float> make_input(int64_t k, int n_used, int n_tokens) {
    std::vector<float> input(static_cast<size_t>(k) * n_used * n_tokens);
    uint32_t state = 0x12345678u;
    for (float & value : input) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<float>(static_cast<int32_t>((state >> 8) & 0xffffu) - 32768) /
            32768.0f * 0.2f;
    }
    return input;
}

std::vector<int32_t> make_full_ids(int n_used, int n_tokens) {
    std::vector<int32_t> ids(static_cast<size_t>(n_used) * n_tokens);
    for (int token = 0; token < n_tokens; ++token) {
        for (int route = 0; route < n_used; ++route) {
            ids[static_cast<size_t>(token) * n_used + route] = route;
        }
    }
    return ids;
}

void calculate_error(
        const std::vector<float> & reference,
        const std::vector<float> & candidate,
        double * max_abs,
        double * nmse) {
    if (reference.size() != candidate.size()) {
        fail("output size mismatch");
    }
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double maximum = 0.0;
    for (size_t index = 0; index < reference.size(); ++index) {
        const double error = static_cast<double>(candidate[index]) - reference[index];
        maximum = std::max(maximum, std::abs(error));
        squared_error += error * error;
        squared_reference += static_cast<double>(reference[index]) * reference[index];
    }
    *max_abs = maximum;
    *nmse = squared_error / std::max(squared_reference, 1e-30);
}

ggml_tensor * aggregate_routes(
        ggml_context * ctx,
        ggml_tensor * experts,
        int64_t n_embd,
        int n_used,
        int n_tokens) {
    std::vector<ggml_tensor *> routes(static_cast<size_t>(n_used));
    for (int route = 0; route < n_used; ++route) {
        routes[static_cast<size_t>(route)] = ggml_view_2d(
            ctx,
            experts,
            n_embd,
            n_tokens,
            experts->nb[2],
            static_cast<size_t>(route) * experts->nb[1]);
    }
    ggml_tensor * sum = routes[0];
    for (int route = 1; route < n_used; ++route) {
        sum = ggml_add(ctx, sum, routes[static_cast<size_t>(route)]);
    }
    return n_used == 1 ? ggml_cont(ctx, sum) : sum;
}

run_result run_ffn_reference(
        const backend_pair & backends,
        const ffn_weight_store & weights,
        int n_used,
        int n_tokens,
        const std::vector<float> & input,
        const std::vector<int32_t> & full_ids,
        int iterations) {
    constexpr int64_t n_embd = 3072;
    ggml_context * ctx = make_context(128);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(b);
    ggml_set_input(ids);

    ggml_tensor * gate = ggml_mul_mat_id(ctx, weights.full_gate_up, b, ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx, weights.full_gate_up, b, ids);
    ggml_tensor * activated = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx, weights.full_down, activated, ids);
    ggml_tensor * out = aggregate_routes(ctx, down, n_embd, n_used, n_tokens);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t only_cpu[] = { backends.cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        only_cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_set_tensor_backend(sched, out, backends.cpu);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        fail("FFN reference graph allocation failed");
    }
    ggml_backend_tensor_set(b, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(ids, full_ids.data(), 0, full_ids.size() * sizeof(int32_t));

    for (int warmup = 0; warmup < 3; ++warmup) {
        check_status(ggml_backend_sched_graph_compute(sched, graph), "FFN reference warmup");
    }
    std::vector<double> timings;
    timings.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = clock_type::now();
        check_status(ggml_backend_sched_graph_compute(sched, graph), "FFN reference compute");
        const auto end = clock_type::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    run_result result;
    result.min_ms = *std::min_element(timings.begin(), timings.end());
    for (double timing : timings) {
        result.mean_ms += timing;
    }
    result.mean_ms /= timings.size();
    result.output.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.output.data(), 0, result.output.size() * sizeof(float));
    result.splits = ggml_backend_sched_get_n_splits(sched);
    result.copies = ggml_backend_sched_get_n_copies(sched);
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return result;
}

run_result run_ffn_split(
        const backend_pair & backends,
        const ffn_weight_store & weights,
        int hot_routes,
        int n_used,
        int n_tokens,
        const std::vector<float> & input,
        const std::vector<int32_t> & full_ids,
        const std::vector<float> & reference,
        int iterations,
        bool parallel) {
    constexpr int64_t n_embd = 3072;
    ggml_context * ctx = make_context(256);
    ggml_tensor * b_hot = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * b_cold = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * ids_hot = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * ids_cold = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(b_hot);
    ggml_set_input(b_cold);
    ggml_set_input(ids_hot);
    ggml_set_input(ids_cold);

    ggml_tensor * hot_gate = ggml_mul_mat_id(ctx, weights.hot_gate_up, b_hot, ids_hot);
    ggml_tensor * hot_up = ggml_mul_mat_id(ctx, weights.hot_gate_up, b_hot, ids_hot);
    ggml_mul_mat_id_set_masked(hot_gate, true);
    ggml_mul_mat_id_set_masked(hot_up, true);
    ggml_tensor * hot_activated = ggml_swiglu_split(ctx, hot_gate, hot_up);
    ggml_tensor * hot_down = ggml_mul_mat_id(ctx, weights.hot_down, hot_activated, ids_hot);
    ggml_mul_mat_id_set_masked(hot_down, true);
    ggml_tensor * hot_sum = aggregate_routes(ctx, hot_down, n_embd, n_used, n_tokens);

    ggml_tensor * cold_gate = ggml_mul_mat_id(ctx, weights.full_gate_up, b_cold, ids_cold);
    ggml_tensor * cold_up = ggml_mul_mat_id(ctx, weights.full_gate_up, b_cold, ids_cold);
    ggml_mul_mat_id_set_masked(cold_gate, true);
    ggml_mul_mat_id_set_masked(cold_up, true);
    ggml_tensor * cold_activated = ggml_swiglu_split(ctx, cold_gate, cold_up);
    ggml_tensor * cold_down = ggml_mul_mat_id(ctx, weights.full_down, cold_activated, ids_cold);
    ggml_mul_mat_id_set_masked(cold_down, true);
    ggml_tensor * cold_sum = aggregate_routes(ctx, cold_down, n_embd, n_used, n_tokens);

    ggml_tensor * merged = ggml_add(ctx, hot_sum, cold_sum);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, merged);

    ggml_backend_t scheduled[] = { backends.gpu, backends.cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        scheduled, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, parallel, true);
    for (ggml_tensor * tensor : { b_hot, ids_hot, hot_gate, hot_up, hot_activated, hot_down, hot_sum }) {
        ggml_backend_sched_set_tensor_backend(sched, tensor, backends.gpu);
    }
    for (ggml_tensor * tensor : { b_cold, ids_cold, cold_gate, cold_up, cold_activated, cold_down, cold_sum, merged }) {
        ggml_backend_sched_set_tensor_backend(sched, tensor, backends.cpu);
    }
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        fail("FFN split graph allocation failed");
    }

    std::vector<int32_t> hot_ids(full_ids.size(), -1);
    std::vector<int32_t> cold_ids(full_ids.size(), -1);
    for (size_t index = 0; index < full_ids.size(); ++index) {
        const int route = static_cast<int>(index % static_cast<size_t>(n_used));
        if (route < hot_routes) {
            hot_ids[index] = full_ids[index];
        } else {
            cold_ids[index] = full_ids[index];
        }
    }
    auto set_inputs = [&] {
        ggml_backend_tensor_set(b_hot, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(b_cold, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(ids_hot, hot_ids.data(), 0, hot_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ids_cold, cold_ids.data(), 0, cold_ids.size() * sizeof(int32_t));
    };

    if (!parallel) {
        set_inputs();
    }
    for (int warmup = 0; warmup < 3; ++warmup) {
        if (parallel) {
            set_inputs();
        }
        check_status(ggml_backend_sched_graph_compute(sched, graph), "FFN split warmup");
    }
    std::vector<double> timings;
    timings.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (parallel) {
            set_inputs();
        }
        const auto start = clock_type::now();
        check_status(ggml_backend_sched_graph_compute(sched, graph), "FFN split compute");
        const auto end = clock_type::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    run_result result;
    result.min_ms = *std::min_element(timings.begin(), timings.end());
    for (double timing : timings) {
        result.mean_ms += timing;
    }
    result.mean_ms /= timings.size();
    result.output.resize(ggml_nelements(merged));
    ggml_backend_tensor_get(merged, result.output.data(), 0, result.output.size() * sizeof(float));
    calculate_error(reference, result.output, &result.max_abs, &result.nmse);
    result.splits = ggml_backend_sched_get_n_splits(sched);
    result.copies = ggml_backend_sched_get_n_copies(sched);
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return result;
}

run_result run_reference(
        const backend_pair & backends,
        const weight_store & weights,
        int64_t k,
        int64_t m,
        int n_used,
        int n_tokens,
        const std::vector<float> & input,
        const std::vector<int32_t> & full_ids,
        int iterations) {
    ggml_context * ctx = make_context(32);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(b);
    ggml_set_input(ids);
    ggml_tensor * out = ggml_mul_mat_id(ctx, weights.full, b, ids);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_t only_cpu[] = { backends.cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        only_cpu, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_set_tensor_backend(sched, out, backends.cpu);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        fail("reference graph allocation failed");
    }
    ggml_backend_tensor_set(b, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(ids, full_ids.data(), 0, full_ids.size() * sizeof(int32_t));

    for (int warmup = 0; warmup < 3; ++warmup) {
        check_status(ggml_backend_sched_graph_compute(sched, graph), "reference warmup");
    }

    std::vector<double> timings;
    timings.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = clock_type::now();
        check_status(ggml_backend_sched_graph_compute(sched, graph), "reference compute");
        const auto end = clock_type::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    run_result result;
    result.mean_ms = 0.0;
    result.min_ms = *std::min_element(timings.begin(), timings.end());
    for (double timing : timings) {
        result.mean_ms += timing;
    }
    result.mean_ms /= timings.size();
    result.output.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.output.data(), 0, result.output.size() * sizeof(float));
    result.splits = ggml_backend_sched_get_n_splits(sched);
    result.copies = ggml_backend_sched_get_n_copies(sched);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return result;
}

run_result run_split(
        const backend_pair & backends,
        const weight_store & weights,
        int64_t k,
        int64_t m,
        int hot_routes,
        int n_used,
        int n_tokens,
        bool merge_on_gpu,
        const std::vector<float> & input,
        const std::vector<int32_t> & full_ids,
        const std::vector<float> & reference,
        int iterations,
        bool parallel) {
    ggml_context * ctx = make_context(64);
    ggml_tensor * b_hot = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n_tokens);
    ggml_tensor * b_cold = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n_tokens);
    ggml_tensor * ids_hot = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * ids_cold = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(b_hot);
    ggml_set_input(b_cold);
    ggml_set_input(ids_hot);
    ggml_set_input(ids_cold);

    ggml_tensor * hot = ggml_mul_mat_id(ctx, weights.hot, b_hot, ids_hot);
    ggml_mul_mat_id_set_masked(hot, true);
    ggml_tensor * cold = ggml_mul_mat_id(ctx, weights.full, b_cold, ids_cold);
    ggml_mul_mat_id_set_masked(cold, true);
    ggml_tensor * merged = ggml_add(ctx, hot, cold);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, merged);

    ggml_backend_t scheduled[] = { backends.gpu, backends.cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        scheduled, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, parallel, true);
    ggml_backend_sched_set_tensor_backend(sched, b_hot, backends.gpu);
    ggml_backend_sched_set_tensor_backend(sched, b_cold, backends.cpu);
    ggml_backend_sched_set_tensor_backend(sched, ids_hot, backends.gpu);
    ggml_backend_sched_set_tensor_backend(sched, ids_cold, backends.cpu);
    ggml_backend_sched_set_tensor_backend(sched, hot, backends.gpu);
    ggml_backend_sched_set_tensor_backend(sched, cold, backends.cpu);
    ggml_backend_sched_set_tensor_backend(sched, merged, merge_on_gpu ? backends.gpu : backends.cpu);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        fail("split graph allocation failed");
    }

    std::vector<int32_t> hot_ids(full_ids.size(), -1);
    std::vector<int32_t> cold_ids(full_ids.size(), -1);
    for (size_t index = 0; index < full_ids.size(); ++index) {
        const int route = static_cast<int>(index % static_cast<size_t>(n_used));
        if (route < hot_routes) {
            hot_ids[index] = full_ids[index]; // compact slot equals selected expert for this test
        } else {
            cold_ids[index] = full_ids[index];
        }
    }
    ggml_backend_tensor_set(b_hot, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(b_cold, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(ids_hot, hot_ids.data(), 0, hot_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ids_cold, cold_ids.data(), 0, cold_ids.size() * sizeof(int32_t));

    for (int warmup = 0; warmup < 3; ++warmup) {
        check_status(ggml_backend_sched_graph_compute(sched, graph), "split warmup");
    }

    std::vector<double> timings;
    timings.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = clock_type::now();
        check_status(ggml_backend_sched_graph_compute(sched, graph), "split compute");
        const auto end = clock_type::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    run_result result;
    result.mean_ms = 0.0;
    result.min_ms = *std::min_element(timings.begin(), timings.end());
    for (double timing : timings) {
        result.mean_ms += timing;
    }
    result.mean_ms /= timings.size();
    result.output.resize(ggml_nelements(merged));
    ggml_backend_tensor_get(merged, result.output.data(), 0, result.output.size() * sizeof(float));
    calculate_error(reference, result.output, &result.max_abs, &result.nmse);
    result.splits = ggml_backend_sched_get_n_splits(sched);
    result.copies = ggml_backend_sched_get_n_copies(sched);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return result;
}

ggml_type parse_type(const std::string & value) {
    if (value == "iq3_xxs") {
        return GGML_TYPE_IQ3_XXS;
    }
    if (value == "iq4_xs") {
        return GGML_TYPE_IQ4_XS;
    }
    if (value == "q4_k") {
        return GGML_TYPE_Q4_K;
    }
    if (value == "f16") {
        return GGML_TYPE_F16;
    }
    fail("unsupported type; use iq3_xxs, iq4_xs, q4_k, or f16");
}

} // namespace

int main(int argc, char ** argv) {
    const std::string type_name = argc > 1 ? argv[1] : "iq3_xxs";
    const int cpu_threads = argc > 2 ? std::atoi(argv[2]) : 24;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 20;
    const int n_tokens = argc > 4 ? std::atoi(argv[4]) : 1;
    const bool test_gpu_merge = argc > 5 && std::strcmp(argv[5], "gpu-merge") == 0;
    const int n_experts = 32;
    const int n_used = 8;
    const int hot_slots = 8;

    backend_pair backends = init_backends(cpu_threads);
    if (type_name == "ffn") {
        std::printf(
            "mode=ffn gate_up=iq3_xxs down=iq4_xs experts=%d topk=%d tokens=%d iterations=%d\n",
            n_experts,
            n_used,
            n_tokens,
            iterations);
        ffn_weight_store weights = build_ffn_weights(backends, n_experts, hot_slots);
        std::vector<float> input = make_input(3072, 1, n_tokens);
        std::vector<int32_t> full_ids = make_full_ids(n_used, n_tokens);
        run_result reference = run_ffn_reference(
            backends, weights, n_used, n_tokens, input, full_ids, iterations);
        std::printf(
            "mode=ffn_cpu_full hot=0 mean_ms=%.3f min_ms=%.3f splits=%d copies=%d\n",
            reference.mean_ms, reference.min_ms, reference.splits, reference.copies);
        for (bool parallel : { false, true }) {
            for (int hot_routes : { 0, 2, 4, 6, 8 }) {
                run_result split = run_ffn_split(
                    backends,
                    weights,
                    hot_routes,
                    n_used,
                    n_tokens,
                    input,
                    full_ids,
                    reference.output,
                    iterations,
                    parallel);
                std::printf(
                    "mode=ffn_split merge=cpu parallel=%d hot=%d mean_ms=%.3f min_ms=%.3f "
                    "speedup=%.3f max_abs=%.8g nmse=%.8g splits=%d copies=%d\n",
                    parallel ? 1 : 0,
                    hot_routes,
                    split.mean_ms,
                    split.min_ms,
                    reference.mean_ms / split.mean_ms,
                    split.max_abs,
                    split.nmse,
                    split.splits,
                    split.copies);
                if (!std::isfinite(split.nmse) || split.nmse > 5e-4) {
                    fail("FFN split output exceeds error tolerance");
                }
            }
        }
        free_ffn_weights(weights);
        ggml_backend_free(backends.gpu);
        ggml_backend_free(backends.cpu);
        return 0;
    }

    const ggml_type type = parse_type(type_name);
    const bool down_shape = type == GGML_TYPE_IQ4_XS;
    const int64_t k = down_shape ? 1024 : 3072;
    const int64_t m = down_shape ? 3072 : 1024;

    std::printf(
        "type=%s k=%lld m=%lld experts=%d topk=%d tokens=%d iterations=%d\n",
        type_name.c_str(),
        static_cast<long long>(k),
        static_cast<long long>(m),
        n_experts,
        n_used,
        n_tokens,
        iterations);

    weight_store weights = build_weights(backends, type, k, m, n_experts, hot_slots);
    std::vector<float> input = make_input(k, n_used, n_tokens);
    std::vector<int32_t> full_ids = make_full_ids(n_used, n_tokens);

    run_result reference = run_reference(
        backends, weights, k, m, n_used, n_tokens, input, full_ids, iterations);
    std::printf(
        "mode=cpu_full hot=0 mean_ms=%.3f min_ms=%.3f splits=%d copies=%d\n",
        reference.mean_ms, reference.min_ms, reference.splits, reference.copies);

    std::vector<bool> merge_backends = { false };
    if (test_gpu_merge) {
        merge_backends.push_back(true);
    }
    for (bool merge_on_gpu : merge_backends) {
        for (bool parallel : { false, true }) {
            for (int hot_routes : { 0, 2, 4, 6, 8 }) {
                run_result split = run_split(
                    backends,
                    weights,
                    k,
                    m,
                    hot_routes,
                    n_used,
                    n_tokens,
                    merge_on_gpu,
                    input,
                    full_ids,
                    reference.output,
                    iterations,
                    parallel);
                std::printf(
                    "mode=split merge=%s parallel=%d hot=%d mean_ms=%.3f min_ms=%.3f speedup=%.3f "
                    "max_abs=%.8g nmse=%.8g splits=%d copies=%d\n",
                    merge_on_gpu ? "gpu" : "cpu",
                    parallel ? 1 : 0,
                    hot_routes,
                    split.mean_ms,
                    split.min_ms,
                    reference.mean_ms / split.mean_ms,
                    split.max_abs,
                    split.nmse,
                    split.splits,
                    split.copies);
                if (!std::isfinite(split.nmse) || split.nmse > 5e-4) {
                    fail("split output exceeds error tolerance");
                }
            }
        }
    }

    free_weights(weights);
    ggml_backend_free(backends.gpu);
    ggml_backend_free(backends.cpu);
    return 0;
}

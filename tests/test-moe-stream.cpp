// Real-GGUF operator probe: compare host-offloaded MUL_MAT_ID with a resident
// reference, using identical inputs and expert IDs. No full model is loaded.
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static int positive(const char * s) {
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    check(end != s && *end == 0 && v > 0 && v <= 16384, "invalid positive integer");
    return int(v);
}
using ctx_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buf_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using sched_ptr = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
static ctx_ptr context() {
    auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    check(ctx != nullptr, "context allocation failed");
    return ctx_ptr(ctx, ggml_free);
}
static buf_ptr alloc(ggml_context * ctx, ggml_backend_buffer_type_t type) {
    auto * buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, type);
    check(buf != nullptr, "tensor buffer allocation failed");
    return buf_ptr(buf, ggml_backend_buffer_free);
}

template<class F> static double measure(F f, int repeats) {
    std::vector<double> times;
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        f();
        const auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main(int argc, char ** argv) {
    try {
        check(argc >= 2, "usage: test-moe-stream model.gguf [tensor_name tokens top_k dense|skew|sparse pinned|pageable repeats]");
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_from_file(argv[1], {true, nullptr}), gguf_free);
        check(bool(file), "cannot read GGUF metadata");
        if (argc == 2) {
            for (int64_t i = 0; i < gguf_get_n_tensors(file.get()); ++i) {
                const std::string name = gguf_get_tensor_name(file.get(), i);
                if (name.find("exps") == std::string::npos) { continue; }
                const int64_t * ne = gguf_get_tensor_ne(file.get(), i);
                std::printf("%s %s [%lld,%lld,%lld,%lld] bytes=%zu\n", name.c_str(),
                    ggml_type_name(gguf_get_tensor_type(file.get(), i)),
                    (long long)ne[0], (long long)ne[1], (long long)ne[2], (long long)ne[3],
                    gguf_get_tensor_size(file.get(), i));
            }
            return 0;
        }
        check(argc >= 5, "tensor name, token count, and top_k are required");
        const char * tensor_name = argv[2];
        const int tokens = positive(argv[3]);
        const int top_k = positive(argv[4]);
        const std::string route = argc > 5 ? argv[5] : "dense";
        const std::string host_kind = argc > 6 ? argv[6] : "pinned";
        const int repeats = argc > 7 ? positive(argv[7]) : 7;
        check(route == "dense" || route == "skew" || route == "sparse", "unknown routing pattern");
        check(host_kind == "pinned" || host_kind == "pageable", "unknown host memory kind");
        const int64_t id = gguf_find_tensor(file.get(), tensor_name);
        check(id >= 0, "tensor not found in this GGUF shard");
        const auto type = gguf_get_tensor_type(file.get(), id);
        const int64_t * ne = gguf_get_tensor_ne(file.get(), id);
        const int experts = int(ne[2]);
        check(experts > 1 && top_k <= experts && ne[3] == 1, "not a compatible expert tensor");
        const bool down = std::string(tensor_name).find("down") != std::string::npos;
        const int channels = down ? top_k : 1;

        // Backend lifetime encloses every allocation, graph, and scheduler.
        backend_ptr gpu(ggml_backend_cuda_init(0), ggml_backend_free);
        backend_ptr cpu(ggml_backend_cpu_init(), ggml_backend_free);
        check(bool(gpu) && bool(cpu), "backend initialization failed");
        auto host_ctx = context();
        auto resident_ctx = context();
        auto input_ctx = context();
        auto reference_ctx = context();
        auto streamed_ctx = context();
        auto * host_w = ggml_new_tensor_3d(host_ctx.get(), type, ne[0], ne[1], ne[2]);
        ggml_set_name(host_w, tensor_name);
        auto host_buffer = alloc(host_ctx.get(), host_kind == "pinned" ?
            ggml_backend_cuda_host_buffer_type() : ggml_backend_cpu_buffer_type());
        ggml_backend_buffer_set_usage(host_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::ifstream data(argv[1], std::ios::binary);
        check(bool(data), "cannot open model data");
        data.seekg(gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), id));
        data.read(static_cast<char *>(host_w->data), ggml_nbytes(host_w));
        check(bool(data), "incomplete tensor data read");
        auto * resident_w = ggml_dup_tensor(resident_ctx.get(), host_w);
        auto resident_buffer = alloc(resident_ctx.get(), ggml_backend_cuda_buffer_type(0));
        ggml_backend_buffer_set_usage(resident_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(resident_w, host_w->data, 0, ggml_nbytes(host_w));
        auto * x = ggml_new_tensor_3d(input_ctx.get(), GGML_TYPE_F32, ne[0], channels, tokens);
        auto * ids = ggml_new_tensor_2d(input_ctx.get(), GGML_TYPE_I32, top_k, tokens);
        auto input_buffer = alloc(input_ctx.get(), ggml_backend_cuda_buffer_type(0));
        std::mt19937 rng(20260914);
        std::uniform_real_distribution<float> uniform(-0.5f, 0.5f);
        std::vector<float> inputs(ggml_nelements(x));
        for (auto & value : inputs) { value = uniform(rng); }
        std::vector<int32_t> routing(size_t(tokens) * top_k);
        for (int t = 0; t < tokens; ++t) {
            for (int k = 0; k < top_k; ++k) {
                int expert;
                if (route == "sparse") {
                    const int active = std::min(experts, std::max(17, top_k));
                    expert = (((t + k) % active) * 31) % experts;
                } else if (route == "skew") {
                    expert = k == 0 ? 0 : 1 + ((t + (k - 1) * 17) % (experts - 1));
                } else {
                    expert = (t * 7 + k * 17) % experts;
                }
                routing[size_t(t) * top_k + k] = expert;
                for (int j = 0; j < k; ++j) {
                    check(routing[size_t(t) * top_k + j] != expert, "test generated duplicate expert IDs");
                }
            }
        }
        ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size() * sizeof(float));
        ggml_backend_tensor_set(ids, routing.data(), 0, routing.size() * sizeof(int32_t));
        auto * reference = ggml_mul_mat_id(reference_ctx.get(), resident_w, x, ids);
        auto * ref_graph = ggml_new_graph_custom(reference_ctx.get(), 64, false);
        ggml_build_forward_expand(ref_graph, reference);
        auto reference_buffer = alloc(reference_ctx.get(), ggml_backend_cuda_buffer_type(0));
        auto * result = ggml_mul_mat_id(streamed_ctx.get(), host_w, x, ids);
        auto * graph = ggml_new_graph_custom(streamed_ctx.get(), 64, false);
        ggml_build_forward_expand(graph, result);
        ggml_backend_t backends[] = {gpu.get(), cpu.get()};
        sched_ptr scheduler(ggml_backend_sched_new(backends, nullptr, 2, 64, false, true), ggml_backend_sched_free);
        ggml_backend_sched_set_tensor_backend(scheduler.get(), result, gpu.get());
        check(ggml_backend_sched_alloc_graph(scheduler.get(), graph), "scheduler allocation failed");
        auto ref_run = [&] {
            check(ggml_backend_graph_compute(gpu.get(), ref_graph) == GGML_STATUS_SUCCESS, "reference compute failed");
            ggml_backend_synchronize(gpu.get());
        };
        auto host_run = [&] {
            check(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS, "host compute failed");
            ggml_backend_sched_synchronize(scheduler.get());
        };
        ref_run();
        host_run();
        const double ref_ms = measure(ref_run, repeats);
        const double host_ms = measure(host_run, repeats);
        std::vector<float> expected(ggml_nelements(reference)), actual(ggml_nelements(result));
        ggml_backend_tensor_get(reference, expected.data(), 0, expected.size() * sizeof(float));
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
        double max_abs = 0, sum_diff2 = 0, sum_ref2 = 0;
        size_t nonfinite = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) { ++nonfinite; continue; }
            const double diff = double(actual[i]) - expected[i];
            max_abs = std::max(max_abs, std::abs(diff));
            sum_diff2 += diff * diff;
            sum_ref2 += double(expected[i]) * expected[i];
        }
        const double relative_l2 = std::sqrt(sum_diff2 / std::max(sum_ref2, 1e-30));
        const bool pass = nonfinite == 0 && max_abs <= 5e-4 && relative_l2 <= 1e-5;
        char description[256];
        ggml_backend_cuda_get_device_description(0, description, sizeof(description));
        const char * enabled = std::getenv("GGML_CUDA_MOE_STREAM");
        const char * group = std::getenv("GGML_CUDA_MOE_STREAM_GROUP");
        std::printf("{\"device\":\"%s\",\"tensor\":\"%s\",\"type\":\"%s\",\"tokens\":%d,\"top_k\":%d,\"route\":\"%s\",\"host_kind\":\"%s\",\"stream\":%s,\"group\":%s,\"weight_mib\":%.3f,\"sched_gpu_mib\":%.3f,\"resident_ms\":%.6f,\"host_ms\":%.6f,\"max_abs\":%.9g,\"relative_l2\":%.9g,\"nonfinite\":%zu,\"pass\":%s}\n",
            description, tensor_name, ggml_type_name(type), tokens, top_k, route.c_str(), host_kind.c_str(),
            enabled && std::string(enabled) == "1" ? "true" : "false", group ? group : "16",
            ggml_nbytes(host_w) / double(1024 * 1024), ggml_backend_sched_get_buffer_size(scheduler.get(), gpu.get()) / double(1024 * 1024),
            ref_ms, host_ms, max_abs, relative_l2, nonfinite, pass ? "true" : "false");
        return pass ? 0 : 2;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-moe-stream: %s\n", error.what());
        return 1;
    }
}

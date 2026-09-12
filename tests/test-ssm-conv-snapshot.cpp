// SSM_CONV consumes the materialized CONCAT snapshot, not its mutable ancestors.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool run_case(ggml_backend_t backend, int nt, int ns, int activation) {
    constexpr int channels = 256;
    ggml_context * ctx = ggml_init({2*1024*1024, nullptr, true});
    if (!ctx) {
        return false;
    }
    auto * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, channels, ns);
    auto * tokens = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, nt, ns);
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
    auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    auto * joined = ggml_concat(ctx, state, ggml_transpose(ctx, tokens), 0);
    auto * conv = ggml_ssm_conv(ctx, joined, weights);
    auto * output = conv;
    auto * producer = ggml_new_graph_custom(ctx, 16, false);
    auto * consumer = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(producer, joined);
    // Run only the consumer after the already-materialized producer. This also
    // models a scheduler split: only joined and weights are declared inputs.
    ggml_graph_add_node(consumer, conv);
    if (activation == 2) {
        output = ggml_add(ctx, output, bias);
        ggml_graph_add_node(consumer, output);
    }
    if (activation != 0) {
        output = ggml_silu(ctx, output);
        ggml_graph_add_node(consumer, output);
    }
    for (int i = 0; i < ggml_graph_n_nodes(consumer); ++i) {
        ggml_graph_node(consumer, i)->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    std::vector<float> s(3*channels*ns), x(channels*nt*ns), w(4*channels), b(channels);
    for (size_t i = 0; i < s.size(); ++i) s[i] = (int(i % 17) - 8)*0.0625f;
    for (size_t i = 0; i < x.size(); ++i) x[i] = (int(i % 23) - 11)*0.03125f;
    for (size_t i = 0; i < w.size(); ++i) w[i] = (int(i % 7) - 3)*0.125f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = (int(i % 5) - 2)*0.0625f;
    ggml_backend_tensor_set(state, s.data(), 0, s.size()*sizeof(float));
    ggml_backend_tensor_set(tokens, x.data(), 0, x.size()*sizeof(float));
    ggml_backend_tensor_set(weights, w.data(), 0, w.size()*sizeof(float));
    ggml_backend_tensor_set(bias, b.data(), 0, b.size()*sizeof(float));
    bool ok = ggml_backend_graph_compute(backend, producer) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    // An ancestor may be updated or its storage reused after CONCAT. Neither
    // change may alter the value of the materialized CONCAT tensor.
    std::vector<float> poison_s(s.size(), -31.0f), poison_x(x.size(), 17.0f);
    ggml_backend_tensor_set(state, poison_s.data(), 0, poison_s.size()*sizeof(float));
    ggml_backend_tensor_set(tokens, poison_x.data(), 0, poison_x.size()*sizeof(float));
    ok = ok && ggml_backend_graph_compute(backend, consumer) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    std::vector<float> actual(channels*nt*ns);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float));
    float max_error = 0;
    for (int seq = 0; seq < ns; ++seq) {
        for (int t = 0; t < nt; ++t) {
            for (int c = 0; c < channels; ++c) {
                float expected = 0;
                for (int k = 0; k < 4; ++k) {
                    const int pos = t + k;
                    const float value = pos < 3 ? s[(seq*channels+c)*3+pos] : x[(seq*nt+pos-3)*channels+c];
                    expected += value*w[c*4+k];
                }
                if (activation == 2) expected += b[c];
                if (activation != 0) expected /= 1 + std::exp(-expected);
                const float value = actual[(seq*nt+t)*channels+c];
                if (!std::isfinite(value)) ok = false;
                max_error = std::max(max_error, std::abs(value-expected));
            }
        }
    }
    ok = ok && max_error < 1e-5f;
    std::printf("%s nt=%d ns=%d activation=%d max_error=%g %s\n", ggml_backend_name(backend), nt, ns, activation, max_error, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [backend-name]\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    int passed = 0, total = 0, devices = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (argc == 2 && std::strcmp(argv[1], ggml_backend_dev_name(dev)) != 0) continue;
        // This regression targets CUDA and its CPU reference; avoid unrelated
        // accelerators which may not implement this operation yet.
        const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (std::strcmp(reg, "CUDA") != 0 && std::strcmp(reg, "CPU") != 0) continue;
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) return 2;
        ++devices;
        for (int ns : {1, 2}) for (int nt : {1, 2, 31, 32, 33, 128}) for (int activation : {0, 1, 2}) {
            ++total;
            if (run_case(backend, nt, ns, activation)) ++passed;
        }
        ggml_backend_free(backend);
    }
    std::printf("%d/%d cases passed on %d device(s)\n", passed, total, devices);
    return devices > 0 && passed == total ? 0 : 1;
}

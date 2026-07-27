// Experimental, opt-in MoE route tracing. Included from ggml-backend.cpp.
// Enable with GGML_MOE_TRACE=/path/to/trace.jsonl.
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

struct ggml_moe_trace_state {
    std::once_flag once;
    std::mutex mutex;
    FILE * stream = nullptr;
    bool flush = false;
    int64_t max_calls = -1;
    std::atomic<uint64_t> next_call { 0 };
    std::unordered_set<std::string> weights;
};

static ggml_moe_trace_state & ggml_moe_trace_state_get() {
    static ggml_moe_trace_state state;
    return state;
}

static bool ggml_moe_trace_true(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static void ggml_moe_trace_json_string(FILE * out, const char * value) {
    std::fputc('"', out);
    for (const unsigned char * p = (const unsigned char *) (value ? value : ""); *p; ++p) {
        switch (*p) {
            case '"': std::fputs("\\\"", out); break;
            case '\\': std::fputs("\\\\", out); break;
            case '\n': std::fputs("\\n", out); break;
            case '\r': std::fputs("\\r", out); break;
            case '\t': std::fputs("\\t", out); break;
            default:
                if (*p < 0x20) {
                    std::fprintf(out, "\\u%04x", (unsigned) *p);
                } else {
                    std::fputc(*p, out);
                }
        }
    }
    std::fputc('"', out);
}

static void ggml_moe_trace_init(ggml_moe_trace_state & state) {
    const char * path = std::getenv("GGML_MOE_TRACE");
    if (!path || !path[0]) {
        return;
    }
    state.stream = std::fopen(path, "wb");
    if (!state.stream) {
        return;
    }
    state.flush = ggml_moe_trace_true("GGML_MOE_TRACE_FLUSH");
    if (const char * value = std::getenv("GGML_MOE_TRACE_MAX_CALLS")) {
        char * end = nullptr;
        const long long parsed = std::strtoll(value, &end, 10);
        if (end != value && *end == '\0' && parsed >= 0) {
            state.max_calls = parsed;
        }
    }
    std::fputs("{\"type\":\"header\",\"version\":2}\n", state.stream);
}

static uint64_t ggml_moe_trace_begin_call() {
    auto & state = ggml_moe_trace_state_get();
    std::call_once(state.once, [&state] { ggml_moe_trace_init(state); });
    if (!state.stream) {
        return UINT64_MAX;
    }
    const uint64_t call = state.next_call.fetch_add(1, std::memory_order_relaxed);
    return state.max_calls >= 0 && call >= (uint64_t) state.max_calls ? UINT64_MAX : call;
}

static void ggml_moe_trace_weight(uint64_t call, const ggml_tensor * tensor, int64_t n_expert, size_t expert_bytes) {
    if (call == UINT64_MAX) {
        return;
    }
    auto & state = ggml_moe_trace_state_get();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::string key = std::string(tensor->name) + '\0' + std::to_string(expert_bytes);
    if (!state.weights.insert(key).second) {
        return;
    }
    std::fputs("{\"type\":\"weight\",\"tensor\":", state.stream);
    ggml_moe_trace_json_string(state.stream, tensor->name);
    std::fprintf(state.stream, ",\"ggml_type\":%d,\"n_expert\":%" PRId64 ",\"expert_bytes\":%zu}\n",
        (int) tensor->type, n_expert, expert_bytes);
    if (state.flush) {
        std::fflush(state.stream);
    }
}

static void ggml_moe_trace_route(
        uint64_t call,
        ggml_backend_t backend,
        const ggml_tensor * weight,
        const ggml_tensor * ids_tensor,
        const std::vector<int32_t> & ids,
        int64_t n_expert) {
    if (call == UINT64_MAX) {
        return;
    }
    auto & state = ggml_moe_trace_state_get();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::fprintf(state.stream, "{\"type\":\"route\",\"call\":%" PRIu64 ",\"backend\":", call);
    ggml_moe_trace_json_string(state.stream, ggml_backend_name(backend));
    std::fputs(",\"tensor\":", state.stream);
    ggml_moe_trace_json_string(state.stream, weight->name);
    std::fputs(",\"ids_tensor\":", state.stream);
    ggml_moe_trace_json_string(state.stream, ids_tensor->name);
    std::fprintf(state.stream,
        ",\"n_expert\":%" PRId64 ",\"top_k\":%" PRId64 ",\"tokens\":%" PRId64 ",\"ids\":[",
        n_expert, ids_tensor->ne[0], ids_tensor->ne[1]);
    for (int64_t token = 0; token < ids_tensor->ne[1]; ++token) {
        if (token) {
            std::fputc(',', state.stream);
        }
        std::fputc('[', state.stream);
        for (int64_t rank = 0; rank < ids_tensor->ne[0]; ++rank) {
            if (rank) {
                std::fputc(',', state.stream);
            }
            const size_t index = token * ids_tensor->nb[1] / sizeof(int32_t) +
                                 rank * ids_tensor->nb[0] / sizeof(int32_t);
            std::fprintf(state.stream, "%d", ids[index]);
        }
        std::fputc(']', state.stream);
    }
    std::fputs("]}\n", state.stream);
    if (state.flush) {
        std::fflush(state.stream);
    }
}

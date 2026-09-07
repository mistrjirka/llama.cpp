#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

// Experimental controllers. Disabled unless the explicit LLAMA_EXPERIMENT_* flags
// are set. No accepted tokens, KV precision, or target verification is changed.
struct server_serving_experiment {
    struct arm {
        int samples = 0;
        double us_per_token = 0.0;
    };
    struct load_bucket {
        std::array<arm, 4> arms{};
        int decisions = 0;
    };
    std::array<load_bucket, 17> buckets{};
    bool adaptive_mtp = false;
    double prefill_budget_ms = 0.0;
    double prefill_ms_per_token = 0.0;
    int prefill_quantum = 0;

    int choose_depth(int active, int limit) {
        limit = std::clamp(limit, 0, 3);
        auto & b = buckets[std::clamp(active, 1, 16)];
        for (int i = 0; i <= limit; ++i) {
            if (b.arms[i].samples < 2) {
                return i;
            }
        }
        // Occasional remeasurement follows changing context/acceptance, without
        // repeatedly paying an exploration round at every new request.
        if (++b.decisions % 64 == 0) {
            return (b.decisions / 64) % (limit + 1);
        }
        int best = 0;
        for (int i = 1; i <= limit; ++i) {
            if (b.arms[i].us_per_token < b.arms[best].us_per_token * 0.98) {
                best = i;
            }
        }
        return best;
    }

    void observe_depth(int active, int depth, uint64_t generated, double elapsed_us) {
        if (depth < 0 || depth > 3 || generated == 0 || elapsed_us <= 0 || !std::isfinite(elapsed_us)) {
            return;
        }
        auto & a = buckets[std::clamp(active, 1, 16)].arms[depth];
        const double cost = elapsed_us / generated;
        a.us_per_token = a.samples++ ? 0.8 * a.us_per_token + 0.2 * cost : cost;
    }

    int prompt_limit(int maximum, int decode_tokens) const {
        if (prefill_budget_ms <= 0 || decode_tokens == 0 || maximum <= decode_tokens) {
            return maximum;
        }
        const double cost = prefill_ms_per_token > 0 ? prefill_ms_per_token : 1.0;
        const int spare = maximum - decode_tokens;
        double proposed = prefill_budget_ms / cost;
        if (prefill_quantum > 0 && proposed >= prefill_quantum) {
            proposed = std::floor(proposed / prefill_quantum) * prefill_quantum;
        }
        const int tokens = (int) std::clamp(proposed, (double) std::min(32, spare), (double) spare);
        return decode_tokens + tokens;
    }

    void observe_prefill(int prompt_tokens, double elapsed_us) {
        if (prompt_tokens <= 0 || elapsed_us <= 0 || !std::isfinite(elapsed_us)) {
            return;
        }
        const double cost = elapsed_us / (1000.0 * prompt_tokens);
        prefill_ms_per_token = prefill_ms_per_token > 0
            ? 0.8 * prefill_ms_per_token + 0.2 * cost : cost;
    }
};

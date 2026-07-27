#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {
constexpr int kLayers = 72;
constexpr int kLags = 3;
constexpr int kExperts = 256;
constexpr int kSelected = 8;
constexpr int kTopK = 16;
constexpr int kIterations = 2000;

size_t table_index(int layer, int lag, int source, int target) {
    return (((size_t) layer * kLags + lag) * kExperts + source) * kExperts + target;
}

size_t source_index(int layer, int lag, int source) {
    return ((size_t) layer * kLags + lag) * kExperts + source;
}

template <class Function>
double benchmark_us_per_token(Function && function, uint64_t & checksum) {
    const auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        checksum += function(iteration);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - begin).count() * 1e6 / kIterations;
}

std::array<int, kTopK> select_top16(const std::array<float, kExperts> & scores) {
    std::array<int, kExperts> order{};
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + kTopK, order.end(), [&](int lhs, int rhs) {
        if (scores[lhs] != scores[rhs]) {
            return scores[lhs] > scores[rhs];
        }
        return lhs < rhs;
    });
    std::array<int, kTopK> result{};
    std::copy_n(order.begin(), kTopK, result.begin());
    return result;
}
}

int main() {
    const size_t weight_count = (size_t) kLayers * kLags * kExperts * kExperts;
    const size_t source_count = (size_t) kLayers * kLags * kExperts;
    std::vector<int8_t> perceptron(weight_count);
    std::vector<uint16_t> transitions(weight_count);
    std::vector<uint16_t> observations(source_count);
    std::vector<uint8_t> features((size_t) kLayers * kLags * kSelected);

    std::mt19937 rng(1);
    std::uniform_int_distribution<int> signed_dist(-8, 8);
    std::uniform_int_distribution<int> count_dist(0, 31);
    std::uniform_int_distribution<int> obs_dist(32, 255);
    std::uniform_int_distribution<int> expert_dist(0, kExperts - 1);
    for (auto & value : perceptron) value = static_cast<int8_t>(signed_dist(rng));
    for (auto & value : transitions) value = static_cast<uint16_t>(count_dist(rng));
    for (auto & value : observations) value = static_cast<uint16_t>(obs_dist(rng));
    for (auto & value : features) value = static_cast<uint8_t>(expert_dist(rng));

    uint64_t checksum = 0;
    const double perceptron_us = benchmark_us_per_token([&](int iteration) {
        uint64_t local = 0;
        for (int layer = 0; layer < kLayers; ++layer) {
            std::array<int32_t, kExperts> scores{};
            for (int lag = 0; lag < kLags; ++lag) {
                for (int chosen = 0; chosen < kSelected; ++chosen) {
                    const int source = features[((layer * kLags + lag) * kSelected) + chosen];
                    const int8_t * row = &perceptron[table_index(layer, lag, source, 0)];
                    for (int target = 0; target < kExperts; ++target) scores[target] += row[target];
                }
            }
            local += static_cast<uint64_t>(scores[(iteration + layer) & 255] + 1024);
        }
        return local;
    }, checksum);

    const double markov_us = benchmark_us_per_token([&](int iteration) {
        uint64_t local = 0;
        for (int layer = 0; layer < kLayers; ++layer) {
            std::array<float, kExperts> scores{};
            for (int lag = 0; lag < kLags; ++lag) {
                const float lag_weight = lag == 0 ? 1.0f : lag == 1 ? 0.72f : 0.5184f;
                for (int chosen = 0; chosen < kSelected; ++chosen) {
                    const int source = features[((layer * kLags + lag) * kSelected) + chosen];
                    const uint16_t * row = &transitions[table_index(layer, lag, source, 0)];
                    const float scale = lag_weight / observations[source_index(layer, lag, source)];
                    for (int target = 0; target < kExperts; ++target) scores[target] += row[target] * scale;
                }
            }
            local += static_cast<uint64_t>(scores[(iteration + layer) & 255] * 1024.0f);
        }
        return local;
    }, checksum);

    const double hybrid_top16_us = benchmark_us_per_token([&](int iteration) {
        uint64_t local = 0;
        for (int layer = 0; layer < kLayers; ++layer) {
            std::array<float, kExperts> scores{};
            for (int lag = 0; lag < kLags; ++lag) {
                const float lag_weight = lag == 0 ? 1.0f : lag == 1 ? 0.72f : 0.5184f;
                for (int chosen = 0; chosen < kSelected; ++chosen) {
                    const int source = features[((layer * kLags + lag) * kSelected) + chosen];
                    const uint16_t * markov_row = &transitions[table_index(layer, lag, source, 0)];
                    const int8_t * neural_row = &perceptron[table_index(layer, lag, source, 0)];
                    const float markov_scale = lag_weight / observations[source_index(layer, lag, source)];
                    for (int target = 0; target < kExperts; ++target) {
                        scores[target] += markov_row[target] * markov_scale + 0.02f * neural_row[target];
                    }
                }
            }
            const auto top = select_top16(scores);
            local += static_cast<uint64_t>(top[(iteration + layer) & (kTopK - 1)]);
        }
        return local;
    }, checksum);

    std::printf(
        "layers=%d experts=%d state_mib=%.2f perceptron_us_per_token=%.3f "
        "markov_us_per_token=%.3f hybrid_top16_us_per_token=%.3f checksum=%llu\n",
        kLayers,
        kExperts,
        (perceptron.size() + transitions.size() * sizeof(uint16_t) + observations.size() * sizeof(uint16_t)) / 1048576.0,
        perceptron_us,
        markov_us,
        hybrid_top16_us,
        static_cast<unsigned long long>(checksum));
    return 0;
}

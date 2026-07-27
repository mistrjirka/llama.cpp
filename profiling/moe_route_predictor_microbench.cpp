#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {
constexpr int kLayers = 72;
constexpr int kLags = 3;
constexpr int kExperts = 256;
constexpr int kSelected = 8;
constexpr int kIterations = 4000;

size_t weight_index(int layer, int lag, int source, int target) {
    return (((size_t) layer * kLags + lag) * kExperts + source) * kExperts + target;
}
}

int main() {
    const size_t weight_count = (size_t) kLayers * kLags * kExperts * kExperts;
    std::vector<int8_t> weights(weight_count);
    std::vector<uint8_t> features((size_t) kLayers * kLags * kSelected);
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> weight_dist(-8, 8);
    std::uniform_int_distribution<int> expert_dist(0, kExperts - 1);
    for (auto & value : weights) value = static_cast<int8_t>(weight_dist(rng));
    for (auto & value : features) value = static_cast<uint8_t>(expert_dist(rng));

    uint64_t checksum = 0;
    const auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        for (int layer = 0; layer < kLayers; ++layer) {
            int32_t scores[kExperts] = {};
            for (int lag = 0; lag < kLags; ++lag) {
                for (int chosen = 0; chosen < kSelected; ++chosen) {
                    const int source = features[((layer * kLags + lag) * kSelected) + chosen];
                    const int8_t * row = &weights[weight_index(layer, lag, source, 0)];
                    for (int target = 0; target < kExperts; ++target) {
                        scores[target] += row[target];
                    }
                }
            }
            int best = 0;
            for (int target = 1; target < kExperts; ++target) {
                if (scores[target] > scores[best]) best = target;
            }
            checksum += static_cast<uint64_t>(best + scores[best]);
            features[(layer * kLags * kSelected) + (iteration % (kLags * kSelected))] =
                static_cast<uint8_t>((best + iteration) & 255);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_s = std::chrono::duration<double>(end - begin).count();
    const double us_per_token = elapsed_s * 1e6 / kIterations;
    const uint64_t additions_per_token = (uint64_t) kLayers * kLags * kSelected * kExperts;
    std::printf("layers=%d weights_mib=%.2f additions_per_token=%llu us_per_token=%.3f checksum=%llu\n",
                kLayers,
                weight_count / 1048576.0,
                static_cast<unsigned long long>(additions_per_token),
                us_per_token,
                static_cast<unsigned long long>(checksum));
    return 0;
}

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

struct Case {
    const char * name;
    std::vector<size_t> pieces;
    int bundle_count;
};

static double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    size_t index = (size_t) ((values.size() - 1) * q);
    return values[index];
}

static void run_case(const Case & test, bool pinned, int iterations) {
    const size_t bundle_bytes = std::accumulate(test.pieces.begin(), test.pieces.end(), size_t{0});
    const size_t bytes = bundle_bytes * (size_t) test.bundle_count;
    void * host = nullptr;
    void * device = nullptr;
    if (pinned) {
        CUDA_CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocPortable));
    } else {
        host = std::malloc(bytes);
        if (!host) {
            std::fprintf(stderr, "malloc failed\n");
            std::exit(1);
        }
    }
    std::memset(host, 0x5a, bytes);
    CUDA_CHECK(cudaMalloc(&device, bytes));

    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    auto issue = [&]() {
        size_t offset = 0;
        for (int bundle = 0; bundle < test.bundle_count; ++bundle) {
            for (size_t piece : test.pieces) {
                CUDA_CHECK(cudaMemcpyAsync(
                    (char *) device + offset,
                    (char *) host + offset,
                    piece,
                    cudaMemcpyHostToDevice,
                    stream));
                offset += piece;
            }
        }
    };

    for (int warmup = 0; warmup < 50; ++warmup) {
        issue();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<double> event_ms;
    std::vector<double> wall_ms;
    event_ms.reserve(iterations);
    wall_ms.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto wall_start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        issue();
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        auto wall_stop = std::chrono::steady_clock::now();
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));
        event_ms.push_back((double) elapsed);
        wall_ms.push_back(std::chrono::duration<double, std::milli>(wall_stop - wall_start).count());
    }

    const double event_mean = std::accumulate(event_ms.begin(), event_ms.end(), 0.0) / event_ms.size();
    const double wall_mean = std::accumulate(wall_ms.begin(), wall_ms.end(), 0.0) / wall_ms.size();
    const double gib = (double) bytes / (double) (1ull << 30);
    std::printf(
        "%-24s host=%-8s bundles=%d pieces=%zu bytes=%zu "
        "event_ms mean=%.4f p50=%.4f p95=%.4f wall_ms mean=%.4f p50=%.4f p95=%.4f "
        "event_GiB_s=%.2f wall_GiB_s=%.2f\n",
        test.name,
        pinned ? "pinned" : "pageable",
        test.bundle_count,
        test.pieces.size(),
        bytes,
        event_mean,
        percentile(event_ms, 0.50),
        percentile(event_ms, 0.95),
        wall_mean,
        percentile(wall_ms, 0.50),
        percentile(wall_ms, 0.95),
        gib / (event_mean / 1000.0),
        gib / (wall_mean / 1000.0));

    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device));
    if (pinned) {
        CUDA_CHECK(cudaFreeHost(host));
    } else {
        std::free(host);
    }
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device=%s asyncEngineCount=%d concurrentKernels=%d\n",
                prop.name, prop.asyncEngineCount, prop.concurrentKernels);

    const std::vector<size_t> normal = {1671168, 1204224, 1204224};
    const std::vector<size_t> large = {2162688, 1671168, 1671168};
    const size_t normal_total = std::accumulate(normal.begin(), normal.end(), size_t{0});
    const size_t large_total = std::accumulate(large.begin(), large.end(), size_t{0});
    const std::vector<Case> tests = {
        {"normal-3-components", normal, 1},
        {"normal-contiguous", {normal_total}, 1},
        {"large-3-components", large, 1},
        {"large-contiguous", {large_total}, 1},
        {"normal-x2", normal, 2},
        {"normal-x4", normal, 4},
        {"normal-x8", normal, 8},
    };
    for (const Case & test : tests) {
        run_case(test, true, 500);
    }
    for (const Case & test : tests) {
        run_case(test, false, 200);
    }
    return 0;
}

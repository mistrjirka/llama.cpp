// Standalone synchronous state-transfer benchmark. No model is loaded.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr size_t mib = 1024 * 1024;
static constexpr size_t max_size = 64 * mib;
static constexpr uint64_t min_bytes = 512 * mib;
static constexpr double min_seconds = 0.5;
static constexpr int repeats = 3;

static void check(cudaError_t result, const char * operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

struct device_buffer {
    void * data = nullptr;
    device_buffer() { check(cudaMalloc(&data, max_size), "cudaMalloc"); }
    ~device_buffer() { if (data) { cudaFree(data); } }
    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;
};

struct host_buffer {
    void * data = nullptr;
    bool pinned;
    host_buffer(size_t size, bool pinned) : pinned(pinned) {
        if (pinned) {
            check(cudaMallocHost(&data, size), "cudaMallocHost");
        } else {
            data = std::malloc(size);
            if (!data) { throw std::runtime_error("malloc failed"); }
        }
    }
    ~host_buffer() {
        if (pinned) { cudaFreeHost(data); } else { std::free(data); }
    }
    host_buffer(const host_buffer &) = delete;
    host_buffer & operator=(const host_buffer &) = delete;
};

static std::string json_string(const std::string & text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c < 32) { out += ' '; }
        else { out += char(c); }
    }
    return out + '"';
}

static std::string read_line(const std::string & path) {
    std::ifstream stream(path);
    std::string value;
    std::getline(stream, value);
    return value;
}

struct result {
    size_t size;
    bool pinned;
    bool upload;
    int repeat;
    uint64_t iterations;
    double seconds;
    std::string link_speed;
    std::string link_width;
};

static void transfer(void * device, void * host, size_t size, bool upload) {
    // Match the production backend's synchronous wrapper around cudaMemcpyAsync.
    check(cudaMemcpyAsync(upload ? device : host, upload ? host : device, size,
        upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, cudaStreamPerThread), "cudaMemcpyAsync");
    check(cudaStreamSynchronize(cudaStreamPerThread), "cudaStreamSynchronize");
}

static void verify(void * device, void * host, const std::vector<uint8_t> & reference, size_t size) {
    std::memset(host, 0, size);
    transfer(device, host, size, false);
    if (std::memcmp(host, reference.data(), size) != 0) {
        throw std::runtime_error("readback bytes differ from the deterministic input");
    }
}

static result measure(void * device, host_buffer & host, const std::vector<uint8_t> & reference,
        size_t size, bool upload, int repeat, const std::string & pci_path) {
    std::memcpy(host.data, reference.data(), size);
    transfer(device, host.data, size, true);
    verify(device, host.data, reference, size);
    for (int i = 0; i < 8; ++i) { transfer(device, host.data, size, upload); }

    uint64_t iterations = 0;
    double seconds = 0;
    const auto start = std::chrono::steady_clock::now();
    do {
        transfer(device, host.data, size, upload);
        ++iterations;
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    } while (seconds < min_seconds || iterations * size < min_bytes);

    result value { size, host.pinned, upload, repeat, iterations, seconds,
        read_line(pci_path + "/current_link_speed"), read_line(pci_path + "/current_link_width") };
    if (!upload && std::memcmp(host.data, reference.data(), size) != 0) {
        throw std::runtime_error("timed readback bytes differ from the deterministic input");
    }
    verify(device, host.data, reference, size);
    return value;
}

int main() {
    try {
        check(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp props {};
        check(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
        int runtime_version = 0;
        int driver_version = 0;
        check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
        check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
        char pci_id[32] = {};
        check(cudaDeviceGetPCIBusId(pci_id, sizeof(pci_id), 0), "cudaDeviceGetPCIBusId");
        std::string pci_address(pci_id);
        std::transform(pci_address.begin(), pci_address.end(), pci_address.begin(),
            [](unsigned char c) { return char(std::tolower(c)); });
        const std::string pci_path = "/sys/bus/pci/devices/" + pci_address;

        device_buffer device;
        std::vector<uint8_t> reference(max_size);
        for (size_t i = 0; i < reference.size(); ++i) {
            const uint32_t value = uint32_t(i) * 1664525u + 1013904223u;
            reference[i] = uint8_t((value >> 24) ^ (value >> 8) ^ uint32_t(i >> 17));
        }
        std::vector<result> results;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (size_t size : { 4 * mib, max_size }) {
                // Alternate the mode order between repeats to expose warmup or clock bias.
                for (int mode = 0; mode < 2; ++mode) {
                    const bool pinned = bool(mode ^ (repeat & 1));
                    host_buffer host(size, pinned);
                    results.push_back(measure(device.data, host, reference, size, true, repeat, pci_path));
                    results.push_back(measure(device.data, host, reference, size, false, repeat, pci_path));
                }
            }
        }
        std::cout << std::setprecision(9);
        std::cout << "{\n  \"device\": " << json_string(props.name)
                  << ",\n  \"compute_capability\": " << json_string(std::to_string(props.major) + "." + std::to_string(props.minor))
                  << ",\n  \"cuda_runtime_version\": " << runtime_version
                  << ",\n  \"cuda_driver_version\": " << driver_version
                  << ",\n  \"pci_bus_id\": " << json_string(pci_address)
                  << ",\n  \"pci_max_link_speed\": " << json_string(read_line(pci_path + "/max_link_speed"))
                  << ",\n  \"pci_max_link_width\": " << json_string(read_line(pci_path + "/max_link_width"))
                  << ",\n  \"gpu_allocation_bytes\": " << max_size
                  << ",\n  \"max_pinned_allocation_bytes\": " << max_size
                  << ",\n  \"min_bytes_per_measurement\": " << min_bytes
                  << ",\n  \"min_seconds_per_measurement\": " << min_seconds
                  << ",\n  \"verification\": \"all bytes equal\",\n  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto & r = results[i];
            const uint64_t bytes = r.iterations * r.size;
            std::cout << "    {\"size_bytes\": " << r.size
                      << ", \"host_memory\": " << json_string(r.pinned ? "pinned" : "pageable")
                      << ", \"direction\": " << json_string(r.upload ? "H2D" : "D2H")
                      << ", \"repeat\": " << r.repeat
                      << ", \"iterations\": " << r.iterations
                      << ", \"total_bytes\": " << bytes
                      << ", \"seconds\": " << r.seconds
                      << ", \"GB_per_second\": " << double(bytes) / 1e9 / r.seconds
                      << ", \"ms_per_transfer\": " << 1000.0 * r.seconds / double(r.iterations)
                      << ", \"pci_current_link_speed\": " << json_string(r.link_speed)
                      << ", \"pci_current_link_width\": " << json_string(r.link_width)
                      << "}" << (i + 1 == results.size() ? "\n" : ",\n");
        }
        std::cout << "  ]\n}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "transfer_bench: " << error.what() << '\n';
        return 1;
    }
}

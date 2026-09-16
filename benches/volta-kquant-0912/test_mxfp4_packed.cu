// Standalone diagnostic: compare packed MXFP4 FP16 conversion bit-for-bit with stock.
#include "../../ggml/src/ggml-cuda/convert.cu"
#include <cstdio>
#include <vector>

static __global__ void compare_bits_mxfp4(const half * a, const half * b, int64_t n, unsigned * errors) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
            i += int64_t(gridDim.x) * blockDim.x) {
        if (__half_as_ushort(a[i]) != __half_as_ushort(b[i])) {
            atomicAdd(errors, 1u);
        }
    }
}

int main() {
    uint32_t rng = 0x7192a4c3U;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };

    for (int nb : {1, 2, 3, 4, 5, 7, 8, 9, 97, 128, 256, 512, 4096, 65536}) {
        const int blocks32 = nb * (QK_K / QK_MXFP4);
        std::vector<block_mxfp4> h(blocks32);
        for (auto & b : h) {
            b.e = uint8_t(next() % 255); // avoid reserved E8M0 NaN code
            for (auto & q : b.qs) q = uint8_t(next());
        }

        block_mxfp4 * x;
        half * ref;
        half * out;
        unsigned * errors;
        const int64_t n = int64_t(nb) * QK_K;
        CUDA_CHECK(cudaMalloc(&x, h.size() * sizeof(*x)));
        CUDA_CHECK(cudaMalloc(&ref, n * sizeof(*ref)));
        CUDA_CHECK(cudaMalloc(&out, n * sizeof(*out)));
        CUDA_CHECK(cudaMalloc(&errors, sizeof(*errors)));
        CUDA_CHECK(cudaMemcpy(x, h.data(), h.size() * sizeof(*x), cudaMemcpyHostToDevice));

        dequantize_block_mxfp4<<<nb, 32>>>(x, ref);
        for (int threads : {32, 128, 256}) {
            CUDA_CHECK(cudaMemset(errors, 0, sizeof(*errors)));
            const int blocks_per_cta = threads / 32;
            dequantize_block_mxfp4_packed<<<(nb + blocks_per_cta - 1) / blocks_per_cta, threads>>>(x, out, nb);
            compare_bits_mxfp4<<<128, 256>>>(ref, out, n, errors);
            unsigned count;
            CUDA_CHECK(cudaMemcpy(&count, errors, sizeof(count), cudaMemcpyDeviceToHost));
            std::printf("superblocks=%d threads=%d compared=%lld bit_mismatches=%u\n",
                    nb, threads, (long long) n, count);
            if (count) return 1;
        }

        CUDA_CHECK(cudaFree(errors));
        CUDA_CHECK(cudaFree(out));
        CUDA_CHECK(cudaFree(ref));
        CUDA_CHECK(cudaFree(x));
    }
    return 0;
}

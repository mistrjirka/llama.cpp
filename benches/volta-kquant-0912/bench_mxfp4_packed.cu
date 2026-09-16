#include "../../ggml/src/ggml-cuda/convert.cu"
#include <cstdio>

static float time_stock(const block_mxfp4 * x, half * y, int nb, int reps) {
    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 10; ++i) dequantize_block_mxfp4<<<nb, 32>>>(x, y);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) dequantize_block_mxfp4<<<nb, 32>>>(x, y);
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); cudaEventDestroy(a); cudaEventDestroy(b);
    return 1000.0f * ms / reps;
}

static float time_packed(const block_mxfp4 * x, half * y, int nb, int threads, int reps) {
    const int bpc = threads / 32;
    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 10; ++i) dequantize_block_mxfp4_packed<<<(nb + bpc - 1) / bpc, threads>>>(x, y, nb);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) dequantize_block_mxfp4_packed<<<(nb + bpc - 1) / bpc, threads>>>(x, y, nb);
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); cudaEventDestroy(a); cudaEventDestroy(b);
    return 1000.0f * ms / reps;
}

int main() {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device=%s cc=%d.%d\n", prop.name, prop.major, prop.minor);
    for (int nb : {32, 64, 128, 256, 512, 2048, 4096, 32768, 65536}) {
        const int blocks32 = nb * (QK_K / QK_MXFP4);
        block_mxfp4 * x; half * y;
        CUDA_CHECK(cudaMalloc(&x, size_t(blocks32) * sizeof(*x)));
        CUDA_CHECK(cudaMalloc(&y, size_t(nb) * QK_K * sizeof(*y)));
        CUDA_CHECK(cudaMemset(x, 1, size_t(blocks32) * sizeof(*x)));
        const int reps = nb >= 32768 ? 200 : 500;
        const float s = time_stock(x, y, nb, reps);
        const float p32 = time_packed(x, y, nb, 32, reps);
        const float p128 = time_packed(x, y, nb, 128, reps);
        const float p256 = time_packed(x, y, nb, 256, reps);
        std::printf("superblocks=%d stock=%.3f p32=%.3f (%+.2f%%) p128=%.3f (%+.2f%%) p256=%.3f (%+.2f%%) us\n",
                nb, s,
                p32, 100.0f * (s / p32 - 1),
                p128, 100.0f * (s / p128 - 1),
                p256, 100.0f * (s / p256 - 1));
        cudaFree(y); cudaFree(x);
    }
    return 0;
}

#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ============== 错误检查宏 ==============
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                    \
                    __FILE__, __LINE__, status);                               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============== 数据结构 ==============
struct BenchResult {
    const char *name;
    double gflops;
    double time_ms;
    float max_abs_error;
    float max_rel_error;
};

// ============== 工具函数 ==============
inline void init_matrix(float *data, int size, unsigned int seed = 42) {
    srand(seed);
    for (int i = 0; i < size; ++i)
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

inline void compute_errors(const float *ref, const float *test, int size,
                           float &max_abs, float &max_rel) {
    max_abs = 0.0f;
    max_rel = 0.0f;
    for (int i = 0; i < size; ++i) {
        float e = fabsf(ref[i] - test[i]);
        if (e > max_abs) max_abs = e;
        float r = fabsf(ref[i]);
        if (r > 1e-6f) {
            float rel = e / r;
            if (rel > max_rel) max_rel = rel;
        }
    }
}

inline double benchmark_kernel(void (*kernel)(const float*, const float*, float*, int, int, int),
                               const float *dA, const float *dB, float *dC,
                               int M, int N, int K,
                               dim3 grid, dim3 block,
                               int warmup, int iters) {
    // warmup
    for (int i = 0; i < warmup; ++i)
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return (double)ms;
}

inline double benchmark_cublas(cublasHandle_t handle,
                               const float *dA, const float *dB, float *dC,
                               int M, int N, int K,
                               int warmup, int iters) {
    const float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < warmup; ++i)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return (double)ms;
}

// ============== 打印结果 ==============
inline void print_result(const BenchResult &r) {
    printf("| %-20s | %8.3f ms | %9.2f GFLOPS | %.6e | %.6e |\n",
           r.name, r.time_ms, r.gflops, r.max_abs_error, r.max_rel_error);
}

#endif // GEMM_COMMON_H

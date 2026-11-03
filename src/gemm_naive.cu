#include "gemm_common.h"

__global__ void gemm_naive(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void run_gemm_naive(const float *dA, const float *dB, float *dC,
                    int M, int N, int K,
                    int warmup, int iters, BenchResult &result) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    result.name = "naive";
    result.time_ms = benchmark_kernel(gemm_naive, dA, dB, dC, M, N, K,
                                      grid, block, warmup, iters);
    result.gflops = 2.0 * M * N * K / 1e9 / (result.time_ms / 1000.0);
}

#include "gemm_common.h"

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void gemm_tiled(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum = 0.0f;

    int numTiles = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; ++t) {
        // 协作加载 A tile
        if (row < M && t * TILE_K + tx < K)
            As[ty][tx] = A[row * K + t * TILE_K + tx];
        else
            As[ty][tx] = 0.0f;

        // 协作加载 B tile
        if (col < N && t * TILE_K + ty < K)
            Bs[ty][tx] = B[(t * TILE_K + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void run_gemm_tiled(const float *dA, const float *dB, float *dC,
                    int M, int N, int K,
                    int warmup, int iters, BenchResult &result) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    result.name = "tiled (32x32)";
    result.time_ms = benchmark_kernel(gemm_tiled, dA, dB, dC, M, N, K,
                                      grid, block, warmup, iters);
    result.gflops = 2.0 * M * N * K / 1e9 / (result.time_ms / 1000.0);
}

#include "gemm_common.h"

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 8
#define THREAD_N 8

__global__ void gemm_double_buffer(const float *A, const float *B, float *C,
                                   int M, int N, int K) {
    __shared__ float As[2][TILE_M][TILE_K];
    __shared__ float Bs[2][TILE_K][TILE_N];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i)
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j)
            sum[i][j] = 0.0f;

    int numTiles = (K + TILE_K - 1) / TILE_K;

    // 预加载第一个 tile
    for (int i = 0; i < THREAD_M; ++i) {
        int r = row + i * (TILE_M / THREAD_M);
        if (r < M && tx < K)
            As[0][ty + i * (TILE_M / THREAD_M)][tx] = A[r * K + tx];
        else
            As[0][ty + i * (TILE_M / THREAD_M)][tx] = 0.0f;
    }
    for (int j = 0; j < THREAD_N; ++j) {
        int c = col + j * (TILE_N / THREAD_N);
        if (c < N && ty < K)
            Bs[0][ty][tx + j * (TILE_N / THREAD_N)] = B[ty * N + c];
        else
            Bs[0][ty][tx + j * (TILE_N / THREAD_N)] = 0.0f;
    }
    __syncthreads();

    int cur_buf = 0;
    for (int t = 0; t < numTiles; ++t) {
        int next_buf = 1 - cur_buf;
        int next_k = (t + 1) * TILE_K;

        // 异步加载下一个 tile
        if (next_k < K) {
            for (int i = 0; i < THREAD_M; ++i) {
                int r = row + i * (TILE_M / THREAD_M);
                if (r < M && next_k + tx < K)
                    As[next_buf][ty + i * (TILE_M / THREAD_M)][tx] = A[r * K + next_k + tx];
                else
                    As[next_buf][ty + i * (TILE_M / THREAD_M)][tx] = 0.0f;
            }
            for (int j = 0; j < THREAD_N; ++j) {
                int c = col + j * (TILE_N / THREAD_N);
                if (c < N && next_k + ty < K)
                    Bs[next_buf][ty][tx + j * (TILE_N / THREAD_N)] = B[(next_k + ty) * N + c];
                else
                    Bs[next_buf][ty][tx + j * (TILE_N / THREAD_N)] = 0.0f;
            }
        }

        // 计算当前 tile
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                float a = As[cur_buf][ty + i * (TILE_M / THREAD_M)][k];
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    sum[i][j] += a * Bs[cur_buf][k][tx + j * (TILE_N / THREAD_N)];
                }
            }
        }

        __syncthreads();
        cur_buf = next_buf;
    }

    // 写回结果
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        int r = row + i * (TILE_M / THREAD_M);
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            int c = col + j * (TILE_N / THREAD_N);
            if (r < M && c < N)
                C[r * N + c] = sum[i][j];
        }
    }
}

void run_gemm_double_buffer(const float *dA, const float *dB, float *dC,
                            int M, int N, int K,
                            int warmup, int iters, BenchResult &result) {
    dim3 block(TILE_N, TILE_M / THREAD_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    result.name = "double-buffer";
    result.time_ms = benchmark_kernel(gemm_double_buffer, dA, dB, dC, M, N, K,
                                      grid, block, warmup, iters);
    result.gflops = 2.0 * M * N * K / 1e9 / (result.time_ms / 1000.0);
}

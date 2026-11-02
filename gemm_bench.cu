#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                   \
                    __FILE__, __LINE__, status);                              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ==================== Naive (FP32) ====================
__global__ void gemm_naive(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ==================== Tiled (FP32) ====================
#define TILE 32
__global__ void gemm_tiled(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        int bRow = t * TILE + ty;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ==================== WMMA (FP16 in, FP32 acc) ====================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_ROWS 128
#define BLOCK_COLS 128
#define WARPS_PER_BLOCK ((BLOCK_ROWS / WMMA_M) * (BLOCK_COLS / WMMA_N))

__global__ void gemm_wmma(const half *A, const half *B, float *C,
                          int M, int N, int K) {
    int warpId = threadIdx.x / 32;
    int warpRow = warpId / (BLOCK_COLS / WMMA_N);
    int warpCol = warpId % (BLOCK_COLS / WMMA_N);

    int rowBase = blockIdx.y * BLOCK_ROWS + warpRow * WMMA_M;
    int colBase = blockIdx.x * BLOCK_COLS + warpCol * WMMA_N;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    int numTiles = K / WMMA_K;
    for (int t = 0; t < numTiles; ++t) {
        int kOff = t * WMMA_K;
        load_matrix_sync(a_frag, A + rowBase * K + kOff, K);
        load_matrix_sync(b_frag, B + kOff * N + colBase, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(C + rowBase * N + colBase, c_frag, N, mem_row_major);
}

// ==================== Benchmark ====================
int main() {
    const int M = 1024, N = 1024, K = 1024;
    const size_t bytes_A  = M * K * sizeof(float);
    const size_t bytes_B  = K * N * sizeof(float);
    const size_t bytes_Ch = M * N * sizeof(float);
    const size_t bytes_Ah = M * K * sizeof(half);
    const size_t bytes_Bh = K * N * sizeof(half);

    float *h_A  = (float *)malloc(bytes_A);
    float *h_B  = (float *)malloc(bytes_B);
    float *h_C1 = (float *)malloc(bytes_Ch);
    float *h_C2 = (float *)malloc(bytes_Ch);
    float *h_C3 = (float *)malloc(bytes_Ch);
    float *h_C4 = (float *)malloc(bytes_Ch);

    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX;

    // FP32 buffers
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_Ch));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // FP16 buffers for WMMA
    half *d_Ah, *d_Bh;
    CHECK_CUDA(cudaMalloc(&d_Ah, bytes_Ah));
    CHECK_CUDA(cudaMalloc(&d_Bh, bytes_Bh));
    {
        half *h_Ah = (half *)malloc(bytes_Ah);
        half *h_Bh = (half *)malloc(bytes_Bh);
        for (int i = 0; i < M * K; ++i) h_Ah[i] = __float2half(h_A[i]);
        for (int i = 0; i < K * N; ++i) h_Bh[i] = __float2half(h_B[i]);
        CHECK_CUDA(cudaMemcpy(d_Ah, h_Ah, bytes_Ah, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_Bh, h_Bh, bytes_Bh, cudaMemcpyHostToDevice));
        free(h_Ah); free(h_Bh);
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // naive
    {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i)
            gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        CHECK_CUDA(cudaMemcpy(h_C1, d_C, bytes_Ch, cudaMemcpyDeviceToHost));
        printf("[naive]   %.3f ms  |  %.2f GFLOPS\n", ms, 2.0*M*N*K/1e9/(ms/1000.0));
    }

    // tiled
    {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i)
            gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        CHECK_CUDA(cudaMemcpy(h_C2, d_C, bytes_Ch, cudaMemcpyDeviceToHost));
        printf("[tiled]   %.3f ms  |  %.2f GFLOPS\n", ms, 2.0*M*N*K/1e9/(ms/1000.0));
    }

    // wmma
    {
        dim3 block(WARPS_PER_BLOCK * 32);
        dim3 grid((N + BLOCK_COLS - 1) / BLOCK_COLS,
                  (M + BLOCK_ROWS - 1) / BLOCK_ROWS);
        gemm_wmma<<<grid, block>>>(d_Ah, d_Bh, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i)
            gemm_wmma<<<grid, block>>>(d_Ah, d_Bh, d_C, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        CHECK_CUDA(cudaMemcpy(h_C3, d_C, bytes_Ch, cudaMemcpyDeviceToHost));
        printf("[wmma]    %.3f ms  |  %.2f GFLOPS\n", ms, 2.0*M*N*K/1e9/(ms/1000.0));
    }

    // cuBLAS
    {
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 20; ++i)
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20.0f;
        CHECK_CUDA(cudaMemcpy(h_C4, d_C, bytes_Ch, cudaMemcpyDeviceToHost));
        printf("[cublas]  %.3f ms  |  %.2f GFLOPS\n", ms, 2.0*M*N*K/1e9/(ms/1000.0));
        cublasDestroy(handle);
    }

    // Correctness
    double maxErr2 = 0, maxErr3 = 0, maxErr4 = 0;
    for (int i = 0; i < M * N; ++i) {
        double e2 = fabs(h_C2[i] - h_C1[i]);
        double e3 = fabs(h_C3[i] - h_C1[i]);
        double e4 = fabs(h_C4[i] - h_C1[i]);
        if (e2 > maxErr2) maxErr2 = e2;
        if (e3 > maxErr3) maxErr3 = e3;
        if (e4 > maxErr4) maxErr4 = e4;
    }
    printf("\nmax error: tiled=%.2e  wmma=%.2e  cublas=%.2e\n", maxErr2, maxErr3, maxErr4);

    free(h_A); free(h_B); free(h_C1); free(h_C2); free(h_C3); free(h_C4);
    CHECK_CUDA(cudaFree(d_A));  CHECK_CUDA(cudaFree(d_B));  CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_Ah)); CHECK_CUDA(cudaFree(d_Bh));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}

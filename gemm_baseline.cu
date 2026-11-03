#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } \
} while(0)
#define CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { fprintf(stderr, "Launch error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } \
} while(0)
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); exit(1); } \
} while(0)

constexpr int TILE_SIZE = 16;

__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE], Bs[TILE_SIZE][TILE_SIZE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty, col = blockIdx.x * TILE_SIZE + tx;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + tx, bRow = t * TILE_SIZE + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

bool validate(float* C, int size, const char* name) {
    double sum = 0;
    for (int i = 0; i < size; ++i) sum += fabs(C[i]);
    if (sum < 1e-3) {
        fprintf(stderr, "[FATAL] %s output is all zeros! Kernel did not execute.\n", name);
        return false;
    }
    return true;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const int warmup = 5, iters = 20;
    const size_t bytes = M * N * sizeof(float);

    float *h_A = (float*)malloc(M*K*sizeof(float));
    float *h_B = (float*)malloc(K*N*sizeof(float));
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < M*K; ++i) h_A[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < K*N; ++i) h_B[i] = (float)rand()/RAND_MAX;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms;

    // === Naive GEMM ===
    {
        dim3 block(16,16), grid((N+15)/16, (M+15)/16);
        for(int i=0;i<warmup;++i) gemm_naive<<<grid,block>>>(d_A,d_B,d_C,M,N,K);
        CHECK_KERNEL(); CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        if (!validate(h_C, M*N, "naive")) return 1;

        CHECK_CUDA(cudaEventRecord(start));
        for(int i=0;i<iters;++i) gemm_naive<<<grid,block>>>(d_A,d_B,d_C,M,N,K);
        CHECK_KERNEL(); CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
        printf("[naive ] %.4f ms | %.2f GFLOPS\n", ms/iters, 2.0*M*N*K/1e9/(ms/iters/1000.0));
    }

    // === Tiled GEMM ===
    {
        dim3 block(TILE_SIZE,TILE_SIZE), grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
        for(int i=0;i<warmup;++i) gemm_tiled<<<grid,block>>>(d_A,d_B,d_C,M,N,K);
        CHECK_KERNEL(); CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        if (!validate(h_C, M*N, "tiled")) return 1;

        CHECK_CUDA(cudaEventRecord(start));
        for(int i=0;i<iters;++i) gemm_tiled<<<grid,block>>>(d_A,d_B,d_C,M,N,K);
        CHECK_KERNEL(); CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
        printf("[tiled ] %.4f ms | %.2f GFLOPS\n", ms/iters, 2.0*M*N*K/1e9/(ms/iters/1000.0));
    }

    // === cuBLAS FP32 ===
    {
        cublasHandle_t handle; CHECK_CUBLAS(cublasCreate(&handle));
        float alpha=1, beta=0;
        for(int i=0;i<warmup;++i) CHECK_CUBLAS(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,d_B,N,d_A,K,&beta,d_C,N));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        for(int i=0;i<iters;++i) CHECK_CUBLAS(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,d_B,N,d_A,K,&beta,d_C,N));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
        printf("[cublas] %.4f ms | %.2f GFLOPS (FP32)\n", ms/iters, 2.0*M*N*K/1e9/(ms/iters/1000.0));
        cublasDestroy(handle);
    }

    printf("\n✅ Baseline benchmark complete.\n");

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}

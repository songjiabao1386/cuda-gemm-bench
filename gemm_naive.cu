#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* 朴素 SGEMM: C = A * B, A[M,K], B[K,N], C[M,N], row-major */
__global__ void sgemm_naive(const float *A, const float *B, float *C,
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

/* CPU FP64 参考实现，用于校验正确性 */
static void sgemm_ref_fp64(const float *A, const float *B, double *C,
                           int M, int N, int K) {
    for (int i = 0; i < M * N; ++i) C[i] = 0.0;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                C[m * N + n] += (double)A[m * K + k] * (double)B[k * N + n];
}

static void run_test(int M, int N, int K) {
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float *)malloc(bytesA);
    float *hB = (float *)malloc(bytesB);
    float *hC = (float *)malloc(bytesC);
    double *hRef = (double *)malloc((size_t)M * N * sizeof(double));

    if (!hA || !hB || !hC || !hRef) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    /* 固定种子，方便复现 */
    for (int i = 0; i < M * K; ++i) hA[i] = (float)(i % 7 - 3);
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(i % 5 - 2);

    /* CPU 参考 */
    sgemm_ref_fp64(hA, hB, hRef, M, N, K);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    int warmup = 3;
    int iters = 20;

    for (int i = 0; i < warmup; ++i)
        sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    /* 校验正确性 */
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double err = fabs((double)hC[i] - hRef[i]);
        if (err > max_abs_err) max_abs_err = err;
        double denom = fabs(hRef[i]);
        if (denom > 1e-12) {
            double rel = err / denom;
            if (rel > max_rel_err) max_rel_err = rel;
        }
    }

    double gflops = 2.0 * M * N * K / 1e9 / (ms / 1000.0);

    printf("naive | %5d %5d %5d | %8.3f ms | %9.2f GFLOPS | "
           "abs_err=%.3e rel_err=%.3e\n",
           M, N, K, ms, gflops, max_abs_err, max_rel_err);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    free(hRef);
}

int main() {
    printf("kernel |   M     N     K |     time |     GFLOPS | correctness\n");
    printf("--------------------------------------------------------------"
           "----------------\n");

    int sizes[] = {128, 256, 512, 1024, 2048};
    int n = sizeof(sizes) / sizeof(sizes[0]);
    for (int i = 0; i < n; ++i)
        run_test(sizes[i], sizes[i], sizes[i]);

    return 0;
}

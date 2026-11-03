
// cuBLAS benchmarks
double benchmark_cublas_fp32(cublasHandle_t handle,
                             const float *dA, const float *dB, float *dC,
                             int M, int N, int K,
                             int warmup, int iters);

double benchmark_cublas_tf32(cublasHandle_t handle,
                             const float *dA, const float *dB, float *dC,
                             int M, int N, int K,
                             int warmup, int iters);

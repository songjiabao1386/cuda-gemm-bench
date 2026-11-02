# CUDA GEMM Benchmark

对比不同 CUDA 矩阵乘法实现的性能，包括 naive、shared memory tiled、WMMA (Tensor Core) 和 cuBLAS。

## 编译运行

nvcc -O3 -std=c++11 -arch=sm_75 gemm_bench.cu -lcublas -o gemm_bench && ./gemm_bench

## 实现说明

| Kernel | 精度 | 说明 |
|---|---|---|
| naive | FP32 | 全局内存直接读写 |
| tiled | FP32 | 32x32 shared memory 分块 |
| wmma | FP16 in / FP32 acc | WMMA API |
| cuBLAS | FP32 | cublasSgemm 参考基线 |

## License

MIT

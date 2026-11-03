# ⚡️ CUDA GEMM Benchmark ⚡️

对比不同 CUDA 矩阵乘法（General Matrix Multiplication, GEMM）实现的性能。涵盖了从基础的 Naive 实现到利用 Tensor Core 的高阶优化方案。

---

## 🚀 性能测试结果 (Benchmark)

测试环境：`Google Colab (NVIDIA T4 GPU)`  
矩阵尺寸：`1024 x 1024 x 1024 (FP32 / WMMA 混合精度)`

| 实施方案 (Kernel) | 描述 | 运行时间 (ms) | 算力 (GFLOPS) | 加速比 (vs Naive) |
| :--- | :--- | :---: | :---: | :---: |
| `[naive]` | 朴素 FP32 实现 (Baseline) | `4.063` | `528.52` | 1.00x |
| `[tiled]` | 共享内存分块 FP32 实现 | `2.586` | `830.54` | `~1.57x` |
| `[wmma]` | Tensor Core FP16 (混合精度) | `0.690` | `3114.40` | `~5.90x` |
| `[cublas]` | NVIDIA cuBLAS 官方库 (FP32) | `0.377` | `5692.69` | `~10.77x` |

---

## 🛠️ 编译与运行

### 编译并运行
```bash
nvcc -O3 -std=c++11 -arch=sm_75 gemm_bench.cu -lcublas -o gemm_bench && ./gemm_bench
```

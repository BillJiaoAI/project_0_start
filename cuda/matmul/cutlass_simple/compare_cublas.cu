/*
 * Compare our simplified CUTLASS GEMM with cuBLAS
 * M=N=K=1024
 * 
 * 这个文件实现了一个简化版的 CUTLASS GEMM，使用 WMMA API 调用 Tensor Core
 * 并与 cuBLAS 官方库进行性能和正确性对比
 */

#include <cuda_runtime.h>   // CUDA运行时API，提供内存分配、拷贝等功能
#include <cublas_v2.h>      // cuBLAS库，提供高度优化的BLAS操作
#include <mma.h>            // WMMA API，用于调用Tensor Core进行矩阵乘累加
#include <stdio.h>          // 标准输入输出库

using namespace nvcuda;     // 使用nvcuda命名空间，简化wmma相关代码

/* ========== 块级参数定义 ========== */
// BLOCK_M/N/K: CUDA Block处理的矩阵块大小
#define BLOCK_M 128         // Block在M维度(输出矩阵行)的大小
#define BLOCK_N 128         // Block在N维度(输出矩阵列)的大小
#define BLOCK_K 16          // Block在K维度(公共维度)的大小，必须是WMMA_K的倍数

/* ========== Warp级参数定义 ========== */
// WARPS_M/N: 每个Block中在M/N维度上的Warp数量
#define WARPS_M 4           // M维度上4个Warp，每个处理16行 -> 4*16=64行
#define WARPS_N 4           // N维度上4个Warp，每个处理16列 -> 4*16=64列

/* ========== 线程级参数定义 ========== */
#define THREADS_PER_WARP 32 // 每个Warp固定32个线程(NVIDIA GPU架构规定)
// 每个Block的线程总数 = M方向Warp数 * N方向Warp数 * 每个Warp线程数
#define THREADS_PER_BLOCK (WARPS_M * WARPS_N * THREADS_PER_WARP)  // 4*4*32=512线程

/* ========== WMMA Tensor Core参数定义 ========== */
// WMMA_M/N/K: Tensor Core一次处理的矩阵片段(fragment)大小
// RTX 5070(B Blackwell)支持的WMMA tile大小: 16x16x16
#define WMMA_M 16           // Tensor Core处理的A矩阵行数/C矩阵行数
#define WMMA_N 16           // Tensor Core处理的B矩阵列数/C矩阵列数
#define WMMA_K 16           // Tensor Core处理的A/B矩阵公共维度大小

/* ========== 核心GEMM Kernel ========== */
// __global__: CUDA kernel标志，从主机调用，在设备上执行
// half: FP16数据类型，16位浮点数，Tensor Core原生支持
// float: FP32数据类型，用于累加结果(c_frag)
__global__ void gemm_wmma(const half *A, const half *B, float *C, int M, int K, int N) {
    /* ========== Shared Memory声明 ========== */
    // __shared__: Shared Memory，Block内所有线程共享，速度接近寄存器
    // smem_A: 存储A矩阵的一个tile，大小 BLOCK_M x BLOCK_K
    __shared__ half smem_A[BLOCK_M][BLOCK_K];
    // smem_B: 存储B矩阵的一个tile，大小 BLOCK_K x BLOCK_N
    __shared__ half smem_B[BLOCK_K][BLOCK_N];

    /* ========== 线程到Warp的映射 ========== */
    // threadIdx.x: 当前线程在Block内的全局索引(0~511)
    int warp_id = threadIdx.x / THREADS_PER_WARP;      // 当前线程所属的Warp ID(0~15)
    int lane_id = threadIdx.x % THREADS_PER_WARP;      // 当前线程在Warp内的Lane ID(0~31)
    
    /* ========== Warp到矩阵位置的映射 ========== */
    // 将Warp ID映射到Block内的二维位置
    int warp_m = warp_id / WARPS_N;                    // Warp在M方向的索引(0~3)
    int warp_n = warp_id % WARPS_N;                    // Warp在N方向的索引(0~3)

    /* ========== Warp处理的全局矩阵位置 ========== */
    // 每个Warp处理一个 WMMA_M x WMMA_N 的输出tile
    int warp_row = blockIdx.y * BLOCK_M + warp_m * WMMA_M;  // 当前Warp处理的C矩阵起始行
    int warp_col = blockIdx.x * BLOCK_N + warp_n * WMMA_N;  // 当前Warp处理的C矩阵起始列

    /* ========== WMMA Fragment声明 ========== */
    // wmma::fragment: Tensor Core操作的矩阵片段，存储在寄存器中
    // matrix_a: A矩阵的片段，16x16x16，FP16，行主序(row_major)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    // matrix_b: B矩阵的片段，16x16x16，FP16，行主序(row_major)
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    // accumulator: 累加结果片段，16x16x16，FP32(更高精度防止溢出)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    /* ========== 初始化累加器 ========== */
    // 将累加器片段所有元素初始化为0
    wmma::fill_fragment(c_frag, 0.0f);

    /* ========== 计算K方向需要处理的tile数量 ========== */
    // num_tiles = ceil(K / BLOCK_K)，即K方向需要多少个16x16的tile
    int num_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    /* ========== K方向循环: 主计算循环 ========== */
    // 对每个K方向的tile进行迭代，逐步累加
    for (int t = 0; t < num_tiles; t++) {
        // 当前tile在K方向的起始位置
        int k_start = t * BLOCK_K;

        /* ========== 计算当前线程负责的Shared Memory位置 ========== */
        // 每个线程负责加载Shared Memory中的多个元素
        // smem_row: Shared Memory中的行索引
        int smem_row = (threadIdx.x / 64) * 4 + (lane_id / 8);
        // smem_col_a: 加载A矩阵时的列索引
        int smem_col_a = (threadIdx.x % 64) / 32 * BLOCK_K + (lane_id % 8);
        // smem_col_b: 加载B矩阵时的列索引
        int smem_col_b = (threadIdx.x % 64) / 32 * BLOCK_N + (lane_id % 8);

        /* ========== 加载A矩阵tile到Shared Memory ========== */
        // 边界检查: 确保线程加载的位置在矩阵范围内
        if (smem_row < BLOCK_M && smem_col_a < BLOCK_K) {
            // 计算全局内存中的行和列
            int g_row = blockIdx.y * BLOCK_M + smem_row;
            int g_col = k_start + smem_col_a;
            // 加载数据，如果超出矩阵范围则填0
            smem_A[smem_row][smem_col_a] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : __float2half(0.0f);
        }

        /* ========== 加载B矩阵tile到Shared Memory ========== */
        // 边界检查: 确保线程加载的位置在矩阵范围内
        if (smem_row < BLOCK_K && smem_col_b < BLOCK_N) {
            // 计算全局内存中的行和列
            int g_row = k_start + smem_row;
            int g_col = blockIdx.x * BLOCK_N + smem_col_b;
            // 加载数据，如果超出矩阵范围则填0
            smem_B[smem_row][smem_col_b] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : __float2half(0.0f);
        }

        /* ========== 同步: 等待所有线程完成Shared Memory加载 ========== */
        // __syncthreads(): Block内所有线程在此处同步，确保数据已完全加载
        __syncthreads();

        /* ========== 获取当前Warp要处理的Shared Memory指针 ========== */
        // 每个Warp从Shared Memory中提取自己的 WMMA_M x WMMA_K 和 WMMA_K x WMMA_N 片段
        half *a_ptr = &smem_A[warp_m * WMMA_M][0];       // A矩阵片段起始地址
        half *b_ptr = &smem_B[0][warp_n * WMMA_N];       // B矩阵片段起始地址

        /* ========== WMMA Load: 从Shared Memory加载到Fragment ========== */
        // wmma::load_matrix_sync: 同步加载，确保数据已就绪
        // 参数: fragment, 内存指针, leading dimension(行步长)
        wmma::load_matrix_sync(a_frag, a_ptr, BLOCK_K);
        wmma::load_matrix_sync(b_frag, b_ptr, BLOCK_N);

        /* ========== WMMA MMA: 矩阵乘累加核心运算 ========== */
        // wmma::mma_sync: 调用Tensor Core执行乘累加
        // c_frag = c_frag + a_frag * b_frag
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        /* ========== 同步: 等待所有Warp完成当前K tile的计算 ========== */
        // 准备加载下一个K tile的数据
        __syncthreads();
    }

    /* ========== WMMA Store: 将结果写回全局内存 ========== */
    // 边界检查: 确保输出位置在矩阵范围内
    if (warp_row + WMMA_M <= M && warp_col + WMMA_N <= N) {
        // wmma::store_matrix_sync: 同步存储
        // 参数: 输出指针, fragment, leading dimension, 内存布局
        wmma::store_matrix_sync(&C[warp_row * N + warp_col], c_frag, N, wmma::mem_row_major);
    }
}

/* ========== 主函数: 主机端代码 ========== */
int main() {
    /* ========== 矩阵维度定义 ========== */
    int M = 1024, K = 1024, N = 1024;  // 矩阵大小: A(MxK) x B(KxN) = C(MxN)

    /* ========== 主机内存分配(FP32) ========== */
    // 用于初始化数据和存储计算结果
    float *hA_f32 = new float[M * K];         // A矩阵主机内存(FP32)
    float *hB_f32 = new float[K * N];         // B矩阵主机内存(FP32)
    float *hC_our = new float[M * N];         // 我们的GEMM结果(FP32)
    float *hC_cublas = new float[M * N];      // cuBLAS结果(FP32)

    /* ========== 初始化矩阵数据 ========== */
    // A矩阵: 填充 0.00, 0.01, 0.02, ..., 0.99, 0.00, 0.01, ...
    for (int i = 0; i < M * K; i++) hA_f32[i] = (float)(i % 100) / 100.0f;
    // B矩阵: 填充 0.00, 0.02, 0.04, ..., 0.98, 0.00, 0.02, ...
    for (int i = 0; i < K * N; i++) hB_f32[i] = (float)(i % 50) / 50.0f;

    /* ========== FP32转FP16: 分配主机FP16内存 ========== */
    // 使用malloc而非new，因为half不是POD类型
    half *hA_f16 = reinterpret_cast<half*>(malloc(M * K * sizeof(half)));
    half *hB_f16 = reinterpret_cast<half*>(malloc(K * N * sizeof(half)));

    /* ========== FP32转FP16: 转换数据 ========== */
    // __float2half: CUDA提供的FP32到FP16转换函数
    for (int i = 0; i < M * K; i++) hA_f16[i] = __float2half(hA_f32[i]);
    for (int i = 0; i < K * N; i++) hB_f16[i] = __float2half(hB_f32[i]);

    /* ========== 设备内存分配 ========== */
    // cudaMalloc: 在GPU显存中分配内存
    half *dA, *dB;                            // 设备端A、B矩阵(FP16)
    float *dC_our, *dC_cublas;                // 设备端C矩阵(FP32)
    cudaMalloc(&dA, M * K * sizeof(half));    // 分配A矩阵显存
    cudaMalloc(&dB, K * N * sizeof(half));    // 分配B矩阵显存
    cudaMalloc(&dC_our, M * N * sizeof(float));    // 分配结果显存(我们的实现)
    cudaMalloc(&dC_cublas, M * N * sizeof(float)); // 分配结果显存(cuBLAS)

    /* ========== 主机到设备数据拷贝 ========== */
    // cudaMemcpy: 内存拷贝函数
    // cudaMemcpyHostToDevice: 从主机内存拷贝到设备显存
    cudaMemcpy(dA, hA_f16, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB_f16, K * N * sizeof(half), cudaMemcpyHostToDevice);

    /* ========== CUDA Event初始化: 用于计时 ========== */
    // cudaEvent: CUDA事件，用于高精度计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);      // 创建开始事件
    cudaEventCreate(&stop);       // 创建结束事件

    /* ========== 计算Grid和Block大小 ========== */
    // dim3: CUDA三维向量类型，用于定义Grid和Block的维度
    dim3 blockSize(THREADS_PER_BLOCK, 1);   // Block大小: 512x1(1D)
    // Grid大小: 按BLOCK_N和BLOCK_M划分，向上取整
    dim3 gridSize((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    /* ========== 预热运行: 确保GPU初始化完成 ========== */
    // 第一次运行可能包含初始化开销，不计入时间
    gemm_wmma<<<gridSize, blockSize>>>(dA, dB, dC_our, M, K, N);
    cudaDeviceSynchronize();      // 等待GPU执行完成

    /* ========== 计时运行: 我们的WMMA实现 ========== */
    cudaEventRecord(start);                          // 记录开始时间
    gemm_wmma<<<gridSize, blockSize>>>(dA, dB, dC_our, M, K, N);  // 启动Kernel
    cudaEventRecord(stop);                           // 记录结束时间
    cudaEventSynchronize(stop);                      // 等待GPU执行完成
    float ms_our;                                    // 存储运行时间(毫秒)
    cudaEventElapsedTime(&ms_our, start, stop);      // 计算时间差

    /* ========== cuBLAS初始化 ========== */
    // cublasHandle_t: cuBLAS句柄，用于管理cuBLAS库状态
    cublasHandle_t handle;
    cublasCreate(&handle);       // 创建cuBLAS句柄

    /* ========== GEMM参数 ========== */
    // alpha: C = alpha * A * B + beta * C 中的alpha系数
    // beta: 如果beta=0，则直接覆盖C；如果beta!=0，则累加
    float alpha = 1.0f;
    float beta = 0.0f;

    /* ========== cuBLAS预热运行 ========== */
    // cublasGemmEx: 扩展版GEMM，支持多种数据类型和Tensor Core
    // 参数顺序(注意): cuBLAS使用列主序，需要调整参数
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,      // A和B都不转置
                 N, M, K,                       // cuBLAS参数: N, M, K(注意顺序！)
                 &alpha,
                 dB, CUDA_R_16F, N,             // B矩阵，FP16，leading dimension=N
                 dA, CUDA_R_16F, K,             // A矩阵，FP16，leading dimension=K
                 &beta,
                 dC_cublas, CUDA_R_32F, N,      // 结果矩阵，FP32，leading dimension=N
                 CUDA_R_32F,                    // 计算精度: FP32
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);// 使用Tensor Core加速
    cudaDeviceSynchronize();      // 等待cuBLAS执行完成

    /* ========== 计时运行: cuBLAS实现 ========== */
    cudaEventRecord(start);
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 dB, CUDA_R_16F, N,
                 dA, CUDA_R_16F, K,
                 &beta,
                 dC_cublas, CUDA_R_32F, N,
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_cublas;
    cudaEventElapsedTime(&ms_cublas, start, stop);

    /* ========== cuBLAS清理 ========== */
    cublasDestroy(handle);        // 销毁cuBLAS句柄

    /* ========== 设备到主机数据拷贝 ========== */
    // 将GPU计算结果拷贝回主机内存，用于对比
    cudaMemcpy(hC_our, dC_our, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_cublas, dC_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    /* ========== 性能计算 ========== */
    // GEMM浮点运算数 = 2 * M * K * N (每个元素需要K次乘加)
    float flops = 2.0 * M * K * N;
    // GFLOPS = 浮点运算数 / 时间(秒) / 1e9
    float gflops_our = flops / (1e9 * ms_our / 1000.0);
    float gflops_cublas = flops / (1e9 * ms_cublas / 1000.0);

    /* ========== 效率计算 ========== */
    // RTX 5070(B Blackwell) FP16峰值算力: 480 TFLOPS
    float peak_fp16_tflops = 480.0f;
    // 效率 = 实际GFLOPS / 峰值GFLOPS * 100%
    float efficiency_our = gflops_our / (peak_fp16_tflops * 1000.0f) * 100.0f;
    float efficiency_cublas = gflops_cublas / (peak_fp16_tflops * 1000.0f) * 100.0f;

    /* ========== 误差计算 ========== */
    // 计算我们的实现与cuBLAS结果的绝对误差之和
    float err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        err += abs(hC_our[i] - hC_cublas[i]);
    }

    /* ========== 输出结果 ========== */
    printf("=== GEMM Performance Comparison ===\n");
    printf("Matrix: %dx%d x %dx%d = %dx%d\n\n", M, K, K, N, M, N);

    printf("Implementation | Time (ms) | GFLOPS    | Efficiency\n");
    printf("---------------|-----------|-----------|-----------\n");
    printf("Our WMMA       | %9.2f | %9.2f | %9.2f%%\n", ms_our, gflops_our, efficiency_our);
    printf("cuBLAS         | %9.2f | %9.2f | %9.2f%%\n", ms_cublas, gflops_cublas, efficiency_cublas);
    printf("---------------|-----------|-----------|-----------\n");
    printf("\n");
    printf("cuBLAS Speedup: %.2fx\n", ms_our / ms_cublas);
    printf("Our vs cuBLAS: %.2f%%\n", gflops_our / gflops_cublas * 100.0f);
    printf("\n");
    printf("Peak FP16: %.1f TFLOPS\n", peak_fp16_tflops);
    printf("Our vs cuBLAS Error: %.6f\n", err);

    /* ========== 清理内存 ========== */
    // 释放主机内存
    delete[] hA_f32; delete[] hB_f32; delete[] hC_our; delete[] hC_cublas;
    free(hA_f16); free(hB_f16);
    // 释放设备内存
    cudaFree(dA); cudaFree(dB); cudaFree(dC_our); cudaFree(dC_cublas);
    // 销毁CUDA事件
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
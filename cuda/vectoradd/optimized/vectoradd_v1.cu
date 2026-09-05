#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// RTX5070 sm_120 最优块大小 256（适配Blackwell SM warp调度）
#define BLOCK_SIZE 256

// 可自定义总向量长度 4M
constexpr int N = 1 << 22;
constexpr size_t BYTES = N * sizeof(float);

// 拆分分块数，用于流水线传输+计算并行
constexpr int STREAM_CNT = 2;
constexpr int CHUNK_N = N / STREAM_CNT;
constexpr size_t CHUNK_BYTES = CHUNK_N * sizeof(float);

__global__ void vectorAdd(const float* A, const float* B, float* C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main()
{
    // ========== 1. 锁页主机内存（Pinned Memory，消除Pageable拷贝开销） ==========
    float *hA, *hB, *hC;
    cudaMallocHost(&hA, BYTES);
    cudaMallocHost(&hB, BYTES);
    cudaMallocHost(&hC, BYTES);

    // 初始化CPU数据
    for (int i = 0; i < N; i++)
    {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    // ========== 2. GPU显存一次性分配（移出热路径，无运行时分配阻塞） ==========
    float *dA, *dB, *dC;
    cudaMalloc(&dA, BYTES);
    cudaMalloc(&dB, BYTES);
    cudaMalloc(&dC, BYTES);

    // ========== 3. 创建多流，实现流水线重叠 ==========
    cudaStream_t streams[STREAM_CNT];
    for (int i = 0; i < STREAM_CNT; i++)
        cudaStreamCreate(&streams[i]);

    int gridSize = (CHUNK_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ====================== 核心流水线执行 ======================
    // 分两块：流0处理前半段、流1处理后半段，传输与计算并行
    for (int s = 0; s < STREAM_CNT; s++)
    {
        size_t hostOff = s * CHUNK_BYTES;
        size_t devOff = s * CHUNK_BYTES;
        int chunkLen = CHUNK_N;

        // 异步H2D上传，不阻塞CPU，立刻下发Kernel任务
        cudaMemcpyAsync(dA + devOff / sizeof(float), hA + hostOff / sizeof(float),
                        CHUNK_BYTES, cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(dB + devOff / sizeof(float), hB + hostOff / sizeof(float),
                        CHUNK_BYTES, cudaMemcpyHostToDevice, streams[s]);

        // 同流内：等当前块上传完成再计算，不同流之间硬件并行
        vectorAdd<<<gridSize, BLOCK_SIZE, 0, streams[s]>>>(
            dA + devOff / sizeof(float),
            dB + devOff / sizeof(float),
            dC + devOff / sizeof(float),
            chunkLen
        );

        // 异步D2H回传，计算和下一流传输重叠
        cudaMemcpyAsync(hC + hostOff / sizeof(float), dC + devOff / sizeof(float),
                        CHUNK_BYTES, cudaMemcpyDeviceToHost, streams[s]);
    }

    // 仅等待所有流全部完成，全局同步一次
    for (int s = 0; s < STREAM_CNT; s++)
        cudaStreamSynchronize(streams[s]);

    // ========== 结果校验 ==========
    float totalErr = 0.0f;
    for (int i = 0; i < N; i++)
        totalErr += abs(hC[i] - 3.0f);
    printf("Total error: %f\n", totalErr);

    // ========== 资源释放 ==========
    for (int i = 0; i < STREAM_CNT; i++)
        cudaStreamDestroy(streams[i]);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);

    return 0;
}
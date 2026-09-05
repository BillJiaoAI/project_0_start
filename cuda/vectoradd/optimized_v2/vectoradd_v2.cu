#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

#define BLOCK_SIZE 256
constexpr int N = 1 << 22;
constexpr size_t BYTES = N * sizeof(float);
constexpr int CHUNK_NUM = 2;
constexpr int CHUNK_N = N / CHUNK_NUM;
constexpr size_t CHUNK_BYTES = CHUNK_N * sizeof(float);

__global__ void vectorAdd(const float* A, const float* B, float* C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main()
{
    // ========== 1. 锁页主机内存（消除Pageable拷贝开销） ==========
    float *hA, *hB, *hC;
    cudaMallocHost(&hA, BYTES);
    cudaMallocHost(&hB, BYTES);
    cudaMallocHost(&hC, BYTES);
    for (int i = 0; i < N; i++)
    {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    // ========== 2. GPU显存一次性分配 ==========
    float *dA, *dB, *dC;
    cudaMalloc(&dA, BYTES);
    cudaMalloc(&dB, BYTES);
    cudaMalloc(&dC, BYTES);

    // ========== 3. 创建2条独立计算流 + 事件做细粒度依赖 ==========
    cudaStream_t stream[2];
    cudaEvent_t uploadDone[2];
    for (int i = 0; i < 2; i++)
    {
        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&uploadDone[i]);
    }
    int gridSize = (CHUNK_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ========== 核心交错流水线：双流完全并行，无互相阻塞 ==========
    // 流0：第0块
    int off0 = 0;
    cudaMemcpyAsync(dA + off0, hA + off0, CHUNK_BYTES, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(dB + off0, hB + off0, CHUNK_BYTES, cudaMemcpyHostToDevice, stream[0]);
    cudaEventRecord(uploadDone[0], stream[0]);
    // 等待本块上传完成后启动Kernel
    cudaStreamWaitEvent(stream[0], uploadDone[0], 0);
    vectorAdd<<<gridSize, BLOCK_SIZE, 0, stream[0]>>>(dA+off0, dB+off0, dC+off0, CHUNK_N);

    // 流1：第1块 【立刻下发，和流0上传/计算硬件并行】
    int off1 = CHUNK_N;
    cudaMemcpyAsync(dA + off1, hA + off1, CHUNK_BYTES, cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(dB + off1, hB + off1, CHUNK_BYTES, cudaMemcpyHostToDevice, stream[1]);
    cudaEventRecord(uploadDone[1], stream[1]);
    cudaStreamWaitEvent(stream[1], uploadDone[1], 0);
    vectorAdd<<<gridSize, BLOCK_SIZE, 0, stream[1]>>>(dA+off1, dB+off1, dC+off1, CHUNK_N);

    // ========== 回传D2H：和另一路上传/Kernel完全并行（利用PCIe全双工） ==========
    // 流0计算完成后异步回传，下行通道不抢占上行HtoD
    cudaMemcpyAsync(hC + off0, dC + off0, CHUNK_BYTES, cudaMemcpyDeviceToHost, stream[0]);
    // 流1计算完成后异步回传，上下行同时跑无阻塞
    cudaMemcpyAsync(hC + off1, dC + off1, CHUNK_BYTES, cudaMemcpyDeviceToHost, stream[1]);

    // 仅全局同步一次，等待所有流全部结束
    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    // 校验结果
    float totalErr = 0.0f;
    for (int i = 0; i < N; i++)
        totalErr += fabs(hC[i] - 3.0f);
    printf("Total error: %f\n", totalErr);

    // 资源释放
    for (int i = 0; i < 2; i++)
    {
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(uploadDone[i]);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    return 0;
}
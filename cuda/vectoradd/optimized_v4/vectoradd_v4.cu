#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

#define BLOCK_SIZE 256
constexpr int N = 1 << 22;
constexpr size_t BYTES = N * sizeof(float);

// 调大分块数量、流数量，数值越大并行调度越平滑
constexpr int CHUNK_COUNT = 8;
constexpr int STREAM_COUNT = 4;

constexpr int CHUNK_N = N / CHUNK_COUNT;
constexpr size_t CHUNK_BYTES = CHUNK_N * sizeof(float);

__global__ void vectorAdd(const float* A, const float* B, float* C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main()
{
    // 锁页主机内存
    float *hA, *hB, *hC;
    cudaMallocHost(&hA, BYTES);
    cudaMallocHost(&hB, BYTES);
    cudaMallocHost(&hC, BYTES);
    for (int i = 0; i < N; i++)
    {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    // GPU显存一次性分配
    float *dA, *dB, *dC;
    cudaMalloc(&dA, BYTES);
    cudaMalloc(&dB, BYTES);
    cudaMalloc(&dC, BYTES);

    // 创建多条流、每个块独立事件
    cudaStream_t streams[STREAM_COUNT];
    cudaEvent_t uploadEvents[CHUNK_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++)
        cudaStreamCreate(&streams[i]);
    for (int i = 0; i < CHUNK_COUNT; i++)
        cudaEventCreate(&uploadEvents[i]);

    int gridSize = (CHUNK_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 交错下发：块0→流0，块1→流1，块2→流2，块3→流3，块4→流0……循环复用流
    for (int chunkIdx = 0; chunkIdx < CHUNK_COUNT; chunkIdx++)
    {
        size_t offset = chunkIdx * CHUNK_N;
        int streamId = chunkIdx % STREAM_COUNT;
        cudaStream_t st = streams[streamId];
        cudaEvent_t evt = uploadEvents[chunkIdx];

        // 异步上传
        cudaMemcpyAsync(dA + offset, hA + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dB + offset, hB + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, st);

        cudaEventRecord(evt, st);
        // 仅同流等待自身上传完成，无跨流阻塞
        cudaStreamWaitEvent(st, evt, 0);

        // 下发计算Kernel
        vectorAdd<<<gridSize, BLOCK_SIZE, 0, st>>>(dA + offset, dB + offset, dC + offset, CHUNK_N);
    }

    // 全部计算完成后统一异步回传DtoH（下行和上行可并行）
    for (int chunkIdx = 0; chunkIdx < CHUNK_COUNT; chunkIdx++)
    {
        size_t offset = chunkIdx * CHUNK_N;
        int streamId = chunkIdx % STREAM_COUNT;
        cudaStream_t st = streams[streamId];
        cudaMemcpyAsync(hC + offset, dC + offset, CHUNK_BYTES, cudaMemcpyDeviceToHost, st);
    }

    // 等待所有流执行完毕
    for (int i = 0; i < STREAM_COUNT; i++)
        cudaStreamSynchronize(streams[i]);

    // 校验结果
    float totalErr = 0.0f;
    for (int i = 0; i < N; i++)
        totalErr += fabs(hC[i] - 3.0f);
    printf("Total error: %f\n", totalErr);

    // 资源释放
    for (int i = 0; i < STREAM_COUNT; i++)
        cudaStreamDestroy(streams[i]);
    for (int i = 0; i < CHUNK_COUNT; i++)
        cudaEventDestroy(uploadEvents[i]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    return 0;
}
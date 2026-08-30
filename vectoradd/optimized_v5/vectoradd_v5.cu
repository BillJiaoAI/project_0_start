#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

#define BLOCK_SIZE 256
constexpr int N = 1 << 22;
constexpr size_t BYTES = N * sizeof(float);

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
    // 1. Pinned锁页主机内存
    float *hA, *hB, *hC;
    cudaMallocHost(&hA, BYTES);
    cudaMallocHost(&hB, BYTES);
    cudaMallocHost(&hC, BYTES);
    for (int i = 0; i < N; i++)
    {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    // 2. GPU显存分配
    float *dA, *dB, *dC;
    cudaMalloc(&dA, BYTES);
    cudaMalloc(&dB, BYTES);
    cudaMalloc(&dC, BYTES);

    // 3. 创建流与事件
    cudaStream_t streams[STREAM_COUNT];
    cudaEvent_t uploadEvents[CHUNK_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++)
        cudaStreamCreate(&streams[i]);
    for (int i = 0; i < CHUNK_COUNT; i++)
        cudaEventCreate(&uploadEvents[i]);

    int gridSize = (CHUNK_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ===================== CUDA Graph 标准捕获（兼容全版本CUDA） =====================
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // 替换报错API：单流捕获，全局多流任务同样会被捕获
    cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal);

    // 捕获上传任务拓扑
    for (int chunkIdx = 0; chunkIdx < CHUNK_COUNT; chunkIdx++)
    {
        size_t offset = chunkIdx * CHUNK_N;
        int streamId = chunkIdx % STREAM_COUNT;
        cudaStream_t st = streams[streamId];
        cudaEvent_t evt = uploadEvents[chunkIdx];

        cudaMemcpyAsync(dA + offset, hA + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dB + offset, hB + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, st);

        cudaEventRecord(evt, st);
        cudaStreamWaitEvent(st, evt, 0);

        vectorAdd<<<gridSize, BLOCK_SIZE, 0, st>>>(dA + offset, dB + offset, dC + offset, CHUNK_N);
    }

    // 捕获D2H回传拓扑
    for (int chunkIdx = 0; chunkIdx < CHUNK_COUNT; chunkIdx++)
    {
        size_t offset = chunkIdx * CHUNK_N;
        int streamId = chunkIdx % STREAM_COUNT;
        cudaStream_t st = streams[streamId];
        cudaMemcpyAsync(hC + offset, dC + offset, CHUNK_BYTES, cudaMemcpyDeviceToHost, st);
    }

    // 结束捕获
    cudaStreamEndCapture(streams[0], &graph);
    // 实例化为可执行图
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    // ==============================================================================

    // 执行Graph
    cudaGraphLaunch(graphExec, streams[0]);

    // 等待所有流全部完成
    for (int i = 0; i < STREAM_COUNT; i++)
        cudaStreamSynchronize(streams[i]);

    // 校验结果
    float totalErr = 0.0f;
    for (int i = 0; i < N; i++)
        totalErr += fabs(hC[i] - 3.0f);
    printf("Total error: %f\n", totalErr);

    // 资源释放
    cudaGraphExecDestroy(graphExec);
    for (int i = 0; i < STREAM_COUNT; i++)
        cudaStreamDestroy(streams[i]);
    for (int i = 0; i < CHUNK_COUNT; i++)
        cudaEventDestroy(uploadEvents[i]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    return 0;
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

// RTX5070 sm_120 最优块大小
#define BLOCK_SIZE 256
// 总向量长度 4M
constexpr int N = 1 << 22;
constexpr size_t BYTES = N * sizeof(float);

// 分块数量，可自行调大，分越多流水线并行度越高
constexpr int CHUNK_CNT = 4;
constexpr int CHUNK_N = N / CHUNK_CNT;
constexpr size_t CHUNK_BYTES = CHUNK_N * sizeof(float);

// 固定两条硬件流，多块交替分发到流0/流1
constexpr int STREAM_CNT = 2;

__global__ void vectorAdd(const float* A, const float* B, float* C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main()
{
    // ====================== 1. Pinned锁页主机内存 ======================
    float *hA, *hB, *hC;
    cudaMallocHost(&hA, BYTES);
    cudaMallocHost(&hB, BYTES);
    cudaMallocHost(&hC, BYTES);
    for (int i = 0; i < N; i++)
    {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    // ====================== 2. GPU显存一次性分配 ======================
    float *dA, *dB, *dC;
    cudaMalloc(&dA, BYTES);
    cudaMalloc(&dB, BYTES);
    cudaMalloc(&dC, BYTES);

    // ====================== 3. 创建2条流 + 每个块独立Event ======================
    cudaStream_t streams[STREAM_CNT];
    cudaEvent_t uploadEvents[CHUNK_CNT];
    for (int i = 0; i < STREAM_CNT; i++)
        cudaStreamCreate(&streams[i]);
    for (int i = 0; i < CHUNK_CNT; i++)
        cudaEventCreate(&uploadEvents[i]);

    int gridSize = (CHUNK_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ====================== 4. 核心：交错下发，两块流交替填充任务 ======================
    // 关键：不要先灌满一条流再处理另一条，块0进流0，块1进流1，块2进流0，块3进流1……
    for (int chunkIdx = 0; chunkIdx < CHUNK_CNT; chunkIdx++)
    {
        // 块偏移
        size_t hostOff = chunkIdx * CHUNK_N;
        size_t devOff = chunkIdx * CHUNK_N;
        // 交替分配到流0 / 流1
        int curStreamId = chunkIdx % STREAM_CNT;
        cudaStream_t curStream = streams[curStreamId];
        cudaEvent_t curEvent = uploadEvents[chunkIdx];

        // 异步H2D上传
        cudaMemcpyAsync(
            dA + devOff, hA + hostOff,
            CHUNK_BYTES, cudaMemcpyHostToDevice, curStream
        );
        cudaMemcpyAsync(
            dB + devOff, hB + hostOff,
            CHUNK_BYTES, cudaMemcpyHostToDevice, curStream
        );

        // 记录上传完成标记
        cudaEventRecord(curEvent, curStream);
        // 同流Kernel等待本块上传完成，仅本地依赖，无跨流阻塞
        cudaStreamWaitEvent(curStream, curEvent, 0);

        // 下发计算Kernel
        vectorAdd<<<gridSize, BLOCK_SIZE, 0, curStream>>>(
            dA + devOff, dB + devOff, dC + devOff, CHUNK_N
        );
    }

    // ====================== 5. 所有计算完成后统一异步D2H回传 ======================
    for (int chunkIdx = 0; chunkIdx < CHUNK_CNT; chunkIdx++)
    {
        size_t hostOff = chunkIdx * CHUNK_N;
        size_t devOff = chunkIdx * CHUNK_N;
        int curStreamId = chunkIdx % STREAM_CNT;
        cudaStream_t curStream = streams[curStreamId];

        cudaMemcpyAsync(
            hC + hostOff, dC + devOff,
            CHUNK_BYTES, cudaMemcpyDeviceToHost, curStream
        );
    }

    // ====================== 6. 等待两条流全部完成 ======================
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    // 结果校验
    float totalErr = 0.0f;
    for (int i = 0; i < N; i++)
        totalErr += fabs(hC[i] - 3.0f);
    printf("Total error: %f\n", totalErr);

    // ====================== 7. 资源释放 ======================
    for (int i = 0; i < STREAM_CNT; i++)
        cudaStreamDestroy(streams[i]);
    for (int i = 0; i < CHUNK_CNT; i++)
        cudaEventDestroy(uploadEvents[i]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);

    return 0;
}

/*
 * Matmul V1: Shared Memory Tiling
 * 
 * NEW CONCEPT: Shared Memory + Tiling + Coalesced Memory Access
 * 
 * WHY: Baseline accesses B matrix with stride=N (1024), which is terrible.
 * Shared Memory acts as a software-managed cache, enabling:
 * 1. Coalesced global memory reads (contiguous access from same warp)
 * 2. Data reuse: Each element in shared tile is reused TILE_SIZE times
 * 
 * AIE ANALOGY:
 * - Shared Memory ~= AIE Local Buffer (32KB/64KB per tile)
 * - Tiling ~= Software-managed data partitioning (AIE does this natively)
 * - Coalesced Access ~= Vectorized DMA transfers in AIE
 * 
 * KEY INSIGHT:
 * Without tiling: Global memory bandwidth is the bottleneck
 * With tiling: Compute becomes the bottleneck (much better!)
 * 
 * PERFORMANCE: ~30-50x vs baseline
 * 
 * COMPILE: nvcc -g -lineinfo -arch=sm_120 -o matmul_v1.exe matmul_v1.cu
 * NSYS: nsys profile --stats=true -o matmul_v1 ./matmul_v1.exe
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

__global__ void matmul_v1(const float *A, const float *B, float *C, int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_cpu(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int M = 1024, K = 1024, N = 1024;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC = new float[M * N];
    float *hC_cpu = new float[M * N];

    for (int i = 0; i < M * K; i++) hA[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)(i % 50) / 50.0f;

    matmul_cpu(hA, hB, hC_cpu, M, K, N);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes_A);
    cudaMalloc(&dB, bytes_B);
    cudaMalloc(&dC, bytes_C);

    cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes_B, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_v1<<<gridSize, blockSize>>>(dA, dB, dC, M, K, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hC, dC, bytes_C, cudaMemcpyDeviceToHost);

    float err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        err += abs(hC[i] - hC_cpu[i]);
    }

    float flops = 2.0 * M * K * N;
    float gflops = flops / (1e9 * ms / 1000.0);

    printf("=== Matmul V1: Shared Memory Tiling ===\n");
    printf("Matrix: %dx%d x %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("Block: %dx%d, Grid: %dx%d\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);
    printf("Tile Size: %dx%d\n", TILE_SIZE, TILE_SIZE);
    printf("Time: %.2f ms\n", ms);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Error: %.6f\n", err);
    printf("Test %s\n", err < 100.0 ? "PASSED" : "FAILED");

    delete[] hA; delete[] hB; delete[] hC; delete[] hC_cpu;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
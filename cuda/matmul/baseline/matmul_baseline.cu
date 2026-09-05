/*
 * Matmul Baseline Version
 * 
 * NEW CONCEPT: CUDA Programming Model Basics
 * - Grid/Block/Thread hierarchy
 * - Global Memory access
 * 
 * AIE ANALOGY:
 * - Grid ~= Array of AIE Tiles
 * - Block ~= AIE Tile
 * - Thread ~= AIE Core (or lane within vector unit)
 * - Global Memory ~= External Memory (DRAM)
 * 
 * PERFORMANCE: ~1x baseline
 * 
 * COMPILE: nvcc -g -lineinfo -arch=sm_120 -o matmul_baseline.exe matmul_baseline.cu
 * NSYS: nsys profile --stats=true -o matmul_baseline ./matmul_baseline.exe
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_baseline(const float *A, const float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
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

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_baseline<<<gridSize, blockSize>>>(dA, dB, dC, M, K, N);
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

    printf("=== Matmul Baseline ===\n");
    printf("Matrix: %dx%d x %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("Block: %dx%d, Grid: %dx%d\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);
    printf("Time: %.2f ms\n", ms);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Error: %.6f\n", err);
    printf("Test %s\n", err < 100.0 ? "PASSED" : "FAILED");

    delete[] hA; delete[] hB; delete[] hC; delete[] hC_cpu;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
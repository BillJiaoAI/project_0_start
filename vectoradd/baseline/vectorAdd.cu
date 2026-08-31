#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1 << 22; // 4,194,304
    size_t bytes = N * sizeof(float);

    float *hA = new float[N];
    float *hB = new float[N];
    float *hC = new float[N];
    for (int i = 0; i < N; i++) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(dA, dB, dC, N);

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    float err = 0.0f;
    for (int i = 0; i < N; i++) {
        err += abs(hC[i] - 3.0f);
    }
    printf("Total error: %f\n", err);

    delete[] hA; delete[] hB; delete[] hC;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
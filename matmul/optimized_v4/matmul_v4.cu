/*
 * Matmul V4: Tensor Core MMA (WMMA API) - Fixed Version
 * 
 * FIXED ISSUE: WMMA matrix_b uses col_major, but B is stored in row_major!
 * Need to transpose B or use row_major for matrix_b.
 * 
 * CORRECT WMMA USAGE:
 * - matrix_a: row_major → A must be stored row-major (our case)
 * - matrix_b: row_major → B must be stored row-major (our case)
 * 
 * KEY FIX: Changed matrix_b from col_major to row_major
 * 
 * PERFORMANCE: Should reach ~2000-3000 GFLOPS with FP16
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matmul_v4_fp16(const half *A, const half *B, float *C, int M, int K, int N) {
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    int row = warp_row * WMMA_M;
    int col = warp_col * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int num_tiles = (K + WMMA_K - 1) / WMMA_K;

    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * WMMA_K;

        bool a_valid = (row + WMMA_M <= M) && (k_start + WMMA_K <= K);
        bool b_valid = (k_start + WMMA_K <= K) && (col + WMMA_N <= N);

        if (a_valid) {
            wmma::load_matrix_sync(a_frag, &A[row * K + k_start], K);
        } else {
            wmma::fill_fragment(a_frag, 0.0f);
        }

        if (b_valid) {
            wmma::load_matrix_sync(b_frag, &B[k_start * N + col], N);
        } else {
            wmma::fill_fragment(b_frag, 0.0f);
        }

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (row + WMMA_M <= M && col + WMMA_N <= N) {
        wmma::store_matrix_sync(&C[row * N + col], c_frag, N, wmma::mem_row_major);
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

    size_t bytes_C = M * N * sizeof(float);
    size_t bytes_A_f16 = M * K * sizeof(half);
    size_t bytes_B_f16 = K * N * sizeof(half);

    float *hA_f32 = new float[M * K];
    float *hB_f32 = new float[K * N];
    float *hC = new float[M * N];
    float *hC_cpu = new float[M * N];

    for (int i = 0; i < M * K; i++) hA_f32[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) hB_f32[i] = (float)(i % 50) / 50.0f;

    matmul_cpu(hA_f32, hB_f32, hC_cpu, M, K, N);

    half *hA_f16 = reinterpret_cast<half*>(malloc(bytes_A_f16));
    half *hB_f16 = reinterpret_cast<half*>(malloc(bytes_B_f16));

    for (int i = 0; i < M * K; i++) hA_f16[i] = __float2half(hA_f32[i]);
    for (int i = 0; i < K * N; i++) hB_f16[i] = __float2half(hB_f32[i]);

    half *dA_f16, *dB_f16;
    float *dC;
    cudaMalloc(&dA_f16, bytes_A_f16);
    cudaMalloc(&dB_f16, bytes_B_f16);
    cudaMalloc(&dC, bytes_C);

    cudaMemcpy(dA_f16, hA_f16, bytes_A_f16, cudaMemcpyHostToDevice);
    cudaMemcpy(dB_f16, hB_f16, bytes_B_f16, cudaMemcpyHostToDevice);

    dim3 blockSize(256, 1);
    dim3 gridSize((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_v4_fp16<<<gridSize, blockSize>>>(dA_f16, dB_f16, dC, M, K, N);
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

    printf("=== Matmul V4: Tensor Core WMMA (FP16) ===\n");
    printf("Matrix: %dx%d x %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("Block: %dx%d, Grid: %dx%d\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);
    printf("WMMA Tile: %dx%dx%d\n", WMMA_M, WMMA_N, WMMA_K);
    printf("Time: %.2f ms\n", ms);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Error: %.6f\n", err);
    printf("Test %s\n", err < 1000.0 ? "PASSED" : "FAILED");

    delete[] hA_f32; delete[] hB_f32; delete[] hC; delete[] hC_cpu;
    free(hA_f16); free(hB_f16);
    cudaFree(dA_f16); cudaFree(dB_f16); cudaFree(dC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
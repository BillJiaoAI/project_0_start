// ============================================================================
// v1_naive: MHA 的朴素 GPU 实现 (教学基线, llama.cpp 中没有这个 kernel)
// ----------------------------------------------------------------------------
// 每个 block 计算 1 个 (head, query) 的输出行, 把 S = QK^T 整行 materialize
// 到 shared memory, 再做 block 级 softmax, 最后加权求和 V。
//
// 这是 FlashAttention 出现前的标准写法, 目的:
//   1. 建立 correct baseline, 后续版本和它对拍;
//   2. 暴露问题: S 整行 O(S) smem / 全局往返, KV 反复从 global 读;
//   3. llama.cpp 的三个 kernel (vec/tile/mma) 就是为了解决这些问题,
//      同时把 online softmax 融进 KV 的分块遍历里 (见 v2/v3/v4)。
//
// llama.cpp 对照:
//   - Q 是 f32, K/V 是 f16:            fattn-common.cuh launch_fattn() 的假设
//   - causal mask 加在 score 上:        fattn-vec.cuh L280  sum += slope*maskh[...]
//   - 本版本没有 online softmax, 也不分块 KV, 是 "整行 S" 的极端形式。
//
// 编译: nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -Xcompiler="/utf-8" -o mha_naive.exe mha_naive.cu
// 运行: ./mha_naive.exe [S] [H]    (默认 S=512, H=8, 要求 S<=4096)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#define D_HEAD 128
#define NTHREADS 128

// ---------------------------------------------------------------------------
// kernel: grid(H, NQ), block(128)。一个 block 负责一个输出行 O[h][i][:]。
// ---------------------------------------------------------------------------
__global__ void mha_naive(const float* __restrict__ Q,
                          const half*  __restrict__ K,
                          const half*  __restrict__ V,
                          const half*  __restrict__ mask,
                          float*       __restrict__ O,
                          const int NQ, const int NKV, const float scale) {
    const int h   = blockIdx.x;
    const int i   = blockIdx.y;         // query 行号
    const int tid = threadIdx.x;
    const int D   = D_HEAD;

    const float* q_row = Q + ((size_t)h * NQ + i) * D;
    const half*  k_mat = K + (size_t)h * NKV * D;   // [NKV][D]
    const half*  v_mat = V + (size_t)h * NKV * D;
    const half*  m_row = mask + (size_t)i * NKV;

    // ---- 第 1 步: 整行 S = scale*QK^T + mask, 写入 shared memory ----
    // 这是 naive 的关键代价: S 的一整行必须活着 (flash 版本只保留 max/sum 两个标量)
    extern __shared__ float smem[];
    float* S = smem;               // [NKV]
    float* red = smem + NKV;       // [NTHREADS] 归约用临时区

    for (int j = tid; j < NKV; j += NTHREADS) {
        const half* k_j = k_mat + (size_t)j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += q_row[d] * __half2float(k_j[d]);
        S[j] = dot * scale + __half2float(m_row[j]);   // llama.cpp 把 mask 加在 score 上
    }
    __syncthreads();

    // ---- 第 2 步: block 内求 max (数值稳定的 softmax) ----
    float m = -INFINITY;
    for (int j = tid; j < NKV; j += NTHREADS) m = fmaxf(m, S[j]);
    red[tid] = m;
    __syncthreads();
    if (tid == 0) {
        float mx = red[0];
        for (int t = 1; t < NTHREADS; ++t) mx = fmaxf(mx, red[t]);
        red[0] = mx;
    }
    __syncthreads();
    m = red[0];

    // ---- 第 3 步: exp 并求和 ----
    float sum = 0.0f;
    for (int j = tid; j < NKV; j += NTHREADS) { S[j] = expf(S[j] - m); sum += S[j]; }
    red[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float s = red[0];
        for (int t = 1; t < NTHREADS; ++t) s += red[t];
        red[0] = s;
    }
    __syncthreads();
    const float inv_sum = 1.0f / red[0];

    // ---- 第 4 步: O[d] = Σ_j P[j] * V[j][d] ----
    // 每个 thread 负责一个 d, 重新线性扫描全部 V (第二个 S^2 级访存开销)
    {
        const int d = tid;   // NTHREADS == D == 128
        float acc = 0.0f;
        for (int j = 0; j < NKV; ++j) {
            acc += (S[j] * inv_sum) * __half2float(v_mat[(size_t)j * D + d]);
        }
        O[((size_t)h * NQ + i) * D + d] = acc;
    }
}

// ---------------------------------------------------------------------------
// host 侧: 数据生成 + CPU 参考 + 对拍 (与 v0_ref 相同的部分直接内联, 保持单文件)
// ---------------------------------------------------------------------------
static void gen_data(std::vector<float>& Q, std::vector<half>& K, std::vector<half>& V,
                     std::vector<half>& mask, int H, int S, int D) {
    srand(42);
    Q.resize((size_t)H * S * D);
    K.resize((size_t)H * S * D);
    V.resize((size_t)H * S * D);
    mask.resize((size_t)S * S);
    auto urand = []() { return (rand() / (float)RAND_MAX) * 2.0f - 1.0f; };
    for (auto& x : Q) x = urand();
    for (auto& x : K) x = __float2half(urand());
    for (auto& x : V) x = __float2half(urand());
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j)
            mask[(size_t)i * S + j] = __float2half(j <= i ? 0.0f : -INFINITY);
}

static void mha_cpu(const std::vector<float>& Q, const std::vector<half>& K,
                    const std::vector<half>& V, const std::vector<half>& mask,
                    std::vector<float>& O, int H, int S, int D, float scale) {
    O.assign((size_t)H * S * D, 0.0f);
    std::vector<float> Sv(S);
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < S; ++i) {
            const float* q = &Q[(size_t)(h * S + i) * D];
            const half*  k = &K[(size_t)h * S * D];
            const half*  v = &V[(size_t)h * S * D];
            const half*  m = &mask[(size_t)i * S];
            float*       o = &O[(size_t)(h * S + i) * D];
            float smax = -INFINITY, ssum = 0.0f;
            for (int j = 0; j < S; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += q[d] * __half2float(k[(size_t)j * D + d]);
                Sv[j] = dot * scale + __half2float(m[j]);
                smax = fmaxf(smax, Sv[j]);
            }
            for (int j = 0; j < S; ++j) { Sv[j] = expf(Sv[j] - smax); ssum += Sv[j]; }
            for (int j = 0; j < S; ++j) {
                const float p = Sv[j] / ssum;
                for (int d = 0; d < D; ++d) o[d] += p * __half2float(v[(size_t)j * D + d]);
            }
        }
}

int main(int argc, char** argv) {
    int S = argc > 1 ? atoi(argv[1]) : 512;
    int H = argc > 2 ? atoi(argv[2]) : 8;
    const int D = D_HEAD;
    if (S > 4096) { printf("S<=4096 (shared memory 限制)\n"); return 1; }
    const float scale = 1.0f / sqrtf((float)D);
    printf("[v1_naive] H=%d NQ=NKV=%d D=%d\n", H, S, D);

    std::vector<float> Qh, O_ref, O_gpu((size_t)H * S * D);
    std::vector<half> K, V, mask;
    gen_data(Qh, K, V, mask, H, S, D);
    mha_cpu(Qh, K, V, mask, O_ref, H, S, D, scale);

    // 上传显存
    float *dQ, *dO; half *dK, *dV, *dM;
    cudaMalloc(&dQ, Qh.size() * 4);
    cudaMalloc(&dK, K.size() * 2);
    cudaMalloc(&dV, V.size() * 2);
    cudaMalloc(&dM, mask.size() * 2);
    cudaMalloc(&dO, O_gpu.size() * 4);
    cudaMemcpy(dQ, Qh.data(), Qh.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, mask.data(), mask.size() * 2, cudaMemcpyHostToDevice);

    const size_t smem = (S + NTHREADS) * sizeof(float);
    dim3 grid(H, S), block(NTHREADS);

    // 预热 + 计时
    for (int w = 0; w < 3; ++w)
        mha_naive<<<grid, block, smem>>>(dQ, dK, dV, dM, dO, S, S, scale);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int reps = 10;
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r)
        mha_naive<<<grid, block, smem>>>(dQ, dK, dV, dM, dO, S, S, scale);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= reps;
    printf("kernel time: %.3f ms  (~%.2f TFLOPS, causal)\n", ms,
           2.0 * H * S * (double)S * D / (ms * 1e-3) / 1e12);

    cudaMemcpy(O_gpu.data(), dO, O_gpu.size() * 4, cudaMemcpyDeviceToHost);

    // 对拍
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t idx = 0; idx < O_gpu.size(); ++idx) {
        const double diff = fabs((double)O_gpu[idx] - O_ref[idx]);
        max_abs = fmax(max_abs, diff);
        max_rel = fmax(max_rel, diff / (fabs(O_ref[idx]) + 1e-3));
    }
    printf("max_abs_err=%.3e max_rel_err=%.3e  %s\n", max_abs, max_rel,
           max_rel < 1e-3 ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dM); cudaFree(dO);
    return max_rel < 1e-3 ? 0 : 1;
}

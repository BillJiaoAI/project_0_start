// ============================================================================
// v0_ref: MHA (Multi-Head Attention) CPU 参考实现
// ----------------------------------------------------------------------------
// 对应 llama.cpp 中的算子: GGML_OP_FLASH_ATTN_EXT (ggml.c 中 ggml_compute_forward_flash_attn_ext_f16)
//
// 数学定义 (单头, causal mask):
//   S[i][j] = scale * <Q[i,:], K[j,:]> + mask[i][j],   scale = 1/sqrt(D)
//   P[i][j] = softmax_j(S[i,:])[j] = exp(S[i][j] - max_j S) / Σ_j' exp(S[i][j'] - max_j S)
//   O[i,:]  = Σ_j P[i][j] * V[j,:]
//
// 数据布局 (模仿 ggml 的逻辑形状 ne0=D, ne1=seq, ne2=head):
//   Q: f32 [H][NQ][D]   (llama.cpp 中 Q 是 f32)
//   K: f16 [H][NKV][D]  (KV cache 是 f16, 本系列只做 f16 路径)
//   V: f16 [H][NKV][D]
//   mask: f16 [NQ][NKV], causal: j<=i 为 0, j>i 为 -INF
//   O:  f32 [H][NQ][D]
//
// 本文件是后面所有 GPU 版本的正确性基准, 同时引出 FlashAttention 的核心动机:
//
//   朴素做法要把 S[NQ][NKV] 整个算出来再过一遍 softmax, 显存 O(S^2)。
//   FlashAttention 的做法是: 把 KV 按块遍历, 对每一块:
//       m_new = max(m_old, max(S_tile))          # running max
//       O     = O * exp(m_old - m_new) + exp(S_tile - m_new) @ V_tile
//       l     = l * exp(m_old - m_new) + sum(exp(S_tile - m_new))   # running 分母
//   最后 O /= l。这就是 "online softmax" (Milakov & Gimelshein, 2018),
//   llama.cpp 的 vec/tile/mma 三种 CUDA kernel 全部基于它, 只是分块方式不同。
//
// 编译: 见 ../build_all.bat
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#ifndef D_HEAD
#define D_HEAD 128   // head dim, llama.cpp 常见为 64/128
#endif

// 确定性数据生成 (所有版本使用相同种子, 保证可互相比较)
static void gen_data(std::vector<float>& Q, std::vector<half>& K, std::vector<half>& V,
                     std::vector<half>& mask, int H, int NQ, int NKV, int D) {
    srand(42);
    Q.resize((size_t)H * NQ * D);
    K.resize((size_t)H * NKV * D);
    V.resize((size_t)H * NKV * D);
    mask.resize((size_t)NQ * NKV);
    auto urand = []() { return (rand() / (float)RAND_MAX) * 2.0f - 1.0f; };
    for (auto& x : Q) x = urand();
    for (auto& x : K) x = __float2half(urand());
    for (auto& x : V) x = __float2half(urand());
    // causal mask: query i 只能看 key 0..i
    for (int i = 0; i < NQ; ++i)
        for (int j = 0; j < NKV; ++j)
            mask[(size_t)i * NKV + j] = __float2half(j <= i ? 0.0f : -INFINITY);
}

// CPU 参考实现, fp32 全精度
static void mha_cpu(const std::vector<float>& Q, const std::vector<half>& K,
                    const std::vector<half>& V, const std::vector<half>& mask,
                    std::vector<float>& O, int H, int NQ, int NKV, int D, float scale) {
    O.assign((size_t)H * NQ * D, 0.0f);
    std::vector<float> S(NKV);
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < NQ; ++i) {
            const float*  q = &Q[(size_t)(h * NQ + i) * D];
            const half*   k = &K[(size_t)h * NKV * D];
            const half*   v = &V[(size_t)h * NKV * D];
            const half*   m = &mask[(size_t)i * NKV];
            float*        o = &O[(size_t)(h * NQ + i) * D];

            // 1) S = scale * QK^T + mask
            float smax = -INFINITY;
            for (int j = 0; j < NKV; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += q[d] * __half2float(k[(size_t)j * D + d]);
                S[j] = dot * scale + __half2float(m[j]);
                smax = fmaxf(smax, S[j]);
            }
            // 2) softmax
            float ssum = 0.0f;
            for (int j = 0; j < NKV; ++j) { S[j] = expf(S[j] - smax); ssum += S[j]; }
            // 3) O = P @ V
            for (int j = 0; j < NKV; ++j) {
                const float p = S[j] / ssum;
                for (int d = 0; d < D; ++d) o[d] += p * __half2float(v[(size_t)j * D + d]);
            }
        }
    }
}

int main(int argc, char** argv) {
    int S = argc > 1 ? atoi(argv[1]) : 512;   // NQ = NKV = S (prefill, causal)
    int H = argc > 2 ? atoi(argv[2]) : 8;
    const int D = D_HEAD;
    const float scale = 1.0f / sqrtf((float)D);
    printf("[v0_ref] H=%d NQ=NKV=%d D=%d scale=%.5f\n", H, S, D, scale);

    std::vector<float> Q, O;
    std::vector<half> K, V, mask;
    gen_data(Q, K, V, mask, H, S, S, D);
    mha_cpu(Q, K, V, mask, O, H, S, S, D, scale);

    // 打印几个样本值, 供其他版本对比
    printf("O[0][0][0..3]   = %.6f %.6f %.6f %.6f\n",
           O[0], O[1], O[2], O[3]);
    printf("O[H/2][S-1][0..3] = %.6f %.6f %.6f %.6f\n",
           O[(size_t)((H / 2) * S + (S - 1)) * D + 0], O[(size_t)((H / 2) * S + (S - 1)) * D + 1],
           O[(size_t)((H / 2) * S + (S - 1)) * D + 2], O[(size_t)((H / 2) * S + (S - 1)) * D + 3]);
    printf("done\n");
    return 0;
}

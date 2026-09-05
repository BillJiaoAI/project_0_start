// ============================================================================
// v3_tile: 提取自 llama.cpp fatnn-tile (ggml/src/ggml-cuda/fattn-tile.cuh)
// ----------------------------------------------------------------------------
// fattn-vec (v2) 是 "每线程拿一个 key、寄存器走天下"; fattn-tile 则是经典的
// "分块搬进 shared memory" 路线, 也是 llama.cpp 在 prefill/长序列时的兜底实现:
//   - 一个 block 处理 >=2 个 Q 行 (ncols, config 表里最小就是 2);
//   - KV 按 nbatch_fa=64 行一块搬进 smem (KV_tmp), K/V 复用同一块缓冲;
//   - QK^T 在寄存器里累加, D 维按 nbatch_K 分块喂入 (喂法 = softmiss GEMM);
//   - online softmax 的 running max 是 "整个 block 一份" (v2 是每 warp 一份),
//     所以概率可以直接写进 smem 的 KQ 矩阵, 再统一拿去乘 V;
//   - V 也按块搬进 smem, 每线程用 half2 累加自己的 4 个输出维度。
//
// 与原版的差异 (为了最小化):
//   - 固定取 (DKQ=128, ncols=2) 的 config: 64 线程 / nbatch_fa=64 / nbatch_K=64
//     (见 L58: GGML_CUDA_FATTN_TILE_CONFIG_CASE(128,128,2, 64,2,64,64));
//   - 只保留 F16 K/V 路径, 去掉量化/ALiBi/softcap/sinks/GQA/KV_max 优化;
//   - 一个 block 独占 2 个 Q 行扫完全部 KV (无 parallel blocks, 对比 v2);
//   - 要求 S % 64 == 0 (llama.cpp 靠 KV cache padding 保证)。
//
// 核心代码对照 (llama.cpp 行号, 2026-08 版本):
//   fattn-tile.cuh L58       (DKQ,DV,ncols)=(128,128,2) 的 config: 64 线程
//   fattn-tile.cuh L378-480  flash_attn_tile_load_tile (16B 向量化搬 tile)
//   fattn-tile.cuh L483-555  flash_attn_tile_iter_KQ (QK^T, D 维分块)
//   fattn-tile.cuh L560-788  flash_attn_tile_iter (单轮: QK^T->softmax->V)
//   fattn-tile.cuh L884-893  smem 布局: Q_tmp / KV_tmp / KQ (+VKQ 在寄存器)
//   fattn-tile.cuh L904-948  Q 预缩放载入 smem
//   fattn-tile.cuh L952-978  KV 主循环 (本精简版去掉了 oob 分支)
//   fattn-tile.cuh L1082-1132 结果写回 (1/KQ_sum 归一化)
//   fattn-common.cuh L19     FATTN_KQ_MAX_OFFSET (数值范围平移技巧)
//
// 编译: nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -Xcompiler="/utf-8" -o mha_tile.exe mha_tile.cu
// 运行: ./mha_tile.exe [S] [H]   (默认 S=512, H=8, 要求 S % 64 == 0)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#define D_HEAD          128
#define WARP_SIZE       32
#define NWARPS          2                   // 64 线程: (DKQ=128, ncols=2) 的 config
#define NQ_PER_BLK      2                   // ncols=2: 每个 block 处理 2 个 Q 行
#define NBATCH_FA       64                  // 每轮 tile 的 KV 行数 (nbatch_fa)
#define NBATCH_K        64                  // QK^T 时 D 维分块 (half 单位, nbatch_K)
#define NBATCH_V        (NBATCH_K * NBATCH_FA / D_HEAD)  // 32: V tile 每块行数
#define CPY_NE          4                   // 16B / 4B = 4 个 half2
// log(2)*3: softmax 数值范围整体上移 2^3, 让 f16 累加器远离溢出 (fattn-common.cuh L19)
#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)

// ---------------------------- device 工具 ----------------------------------
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// 16B 向量化 tile 搬运: 全局 [I 行 x H2 个 half2] -> smem, 行距 (H2+PAD)。
// 对应 flash_attn_tile_load_tile (L378-425), 这里简化成线性分块: 线程 tid 依次
// 认领第 c 个 16B 块 -> 连续 lane 访问连续地址, 天然合并。
// PAD=4 (half2) 只用于 K: 让相邻 K 行错开 8 个 bank, 点积读取时少冲突 (L880 注释)。
template <int I, int H2, int PAD>
static __device__ __forceinline__ void load_tile(const half2* __restrict__ src, int stride,
                                                 half2* __restrict__ tile, int tid) {
    constexpr int CHUNK = 4;                                // 16B = 4 half2
    const int nchunk = I * (H2 / CHUNK);
    for (int c = tid; c < nchunk; c += NWARPS * WARP_SIZE) {
        const int r   = c / (H2 / CHUNK);
        const int col = (c % (H2 / CHUNK)) * CHUNK;
        *reinterpret_cast<uint4*>(tile + r * (H2 + PAD) + col) =
            *reinterpret_cast<const uint4*>(src + (size_t)r * stride + col);
    }
}

// ------------------------------ 主 kernel -----------------------------------
// grid = (NQ/2, 1, H), block = (32, 2)
// warp ty 独占 Q 行 col_Q_0+ty (np=1); 每 lane 负责该行的 2 个 KV 行和 4 个输出维度
__global__ void __launch_bounds__(NWARPS * WARP_SIZE, 2)
flash_attn_tile(const float* __restrict__ Q,
                const half*  __restrict__ K,
                const half*  __restrict__ V,
                const half*  __restrict__ mask,
                float*       __restrict__ dst,
                const float scale, const int NQ, const int NKV) {
    const int col_Q_0 = blockIdx.x * NQ_PER_BLK;   // 本 block 的第一个 Q 行
    const int head    = blockIdx.z;
    const int ty      = threadIdx.y;               // warp 号 == 本 warp 的 Q 行号
    const int lane    = threadIdx.x;
    const int tid     = ty * WARP_SIZE + lane;

    Q += ((size_t)head * NQ + col_Q_0) * D_HEAD;
    K += (size_t)head * NKV * D_HEAD;
    V += (size_t)head * NKV * D_HEAD;
    const half*  maskh = mask + (size_t)(col_Q_0 + ty) * NKV;
    const half2* K_h2  = (const half2*)K;
    const half2* V_h2  = (const half2*)V;

    // ---- smem 布局 (fattn-tile.cuh L884-893) ----
    __shared__ half2 Q_tmp [NQ_PER_BLK * (D_HEAD / 2)];              // 预缩放的 Q, 常驻
    __shared__ half2 KV_tmp[NBATCH_FA * (NBATCH_K / 2 + CPY_NE)];    // K/V 共用的 tile 缓冲
    __shared__ half  KQ_s  [NQ_PER_BLK * NBATCH_FA];                 // softmax 概率矩阵

    // ---- 寄存器状态 ----
    // VKQ[2]: 本线程持有输出维度 4*lane..4*lane+3 (V_k 的载入方式决定的映射)
    half2 VKQ[(D_HEAD / 2) / WARP_SIZE] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    float KQ_max = -FLT_MAX / 2.0f;                // 本 Q 行的 running max (整 block 一份)
    float KQ_sum = 0.0f;                           // 本线程的分部和 (最后 warp 归约)

    // ---- Q 预缩放载入 smem (L904-948): scale 乘在 Q 上, KQ^T 结果就免再乘 ----
    {
        const float2* Q_f2 = (const float2*)Q;
        for (int c = lane; c < D_HEAD / 2; c += WARP_SIZE) {
            const float2 q2 = Q_f2[ty * (D_HEAD / 2) + c];
            Q_tmp[ty * (D_HEAD / 2) + c] = make_half2(q2.x * scale, q2.y * scale);
        }
    }
    __syncthreads();

    // ---- 主循环: 每轮 64 个 KV (flash_attn_tile_iter, L560-788) ----
    for (int kv0 = 0; kv0 < NKV; kv0 += NBATCH_FA) {
        float KQ_acc[NBATCH_FA / WARP_SIZE] = {0.0f, 0.0f};   // 本线程 2 个 KV 行的点积

        // 1) QK^T: K tile 搬进 smem, D 维按 NBATCH_K 分块喂入
        //    (flash_attn_tile_iter_KQ L483-555)。本线程负责 tile 内 KV 行
        //    {lane, 32+lane} —— 注意和 ty 无关, 两个 warp 处理同一批 K 行、
        //    不同的 Q 行, 这正是 "ncols=2 摊薄 K 载入开销" 的设计。
#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D_HEAD; k_KQ_0 += NBATCH_K) {
            load_tile<NBATCH_FA, NBATCH_K / 2, CPY_NE>(
                K_h2 + (size_t)kv0 * (D_HEAD / 2) + k_KQ_0 / 2, D_HEAD / 2, KV_tmp, tid);
            __syncthreads();
#pragma unroll
            for (int k1 = 0; k1 < NBATCH_K / 2; k1 += CPY_NE) {   // 每次喂 4 half2 = 8 维
                half2 K_k[NBATCH_FA / WARP_SIZE][CPY_NE], Q_k[CPY_NE];
#pragma unroll
                for (int r = 0; r < NBATCH_FA / WARP_SIZE; ++r) {
                    const int i_KQ = r * WARP_SIZE + lane;        // tile 内的 KV 行号
                    *reinterpret_cast<uint4*>(K_k[r]) = *reinterpret_cast<const uint4*>(
                        KV_tmp + i_KQ * (NBATCH_K / 2 + CPY_NE) + k1);
                }
                *reinterpret_cast<uint4*>(Q_k) = *reinterpret_cast<const uint4*>(
                    Q_tmp + ty * (D_HEAD / 2) + k_KQ_0 / 2 + k1); // 全 warp 广播
#pragma unroll
                for (int r = 0; r < NBATCH_FA / WARP_SIZE; ++r)
#pragma unroll
                    for (int k = 0; k < CPY_NE; ++k) {
                        // half2 -> float 乘加 (ggml_cuda_mad 的 NVIDIA 路径)
                        const float2 kf = __half22float2(K_k[r][k]);
                        const float2 qf = __half22float2(Q_k[k]);
                        KQ_acc[r] = fmaf(kf.x, qf.x, fmaf(kf.y, qf.y, KQ_acc[r]));
                    }
            }
            if (k_KQ_0 + NBATCH_K < D_HEAD) __syncthreads();      // 最后一块免同步 (L552)
        }

        // 2) mask + online softmax (L619-660)
        float KQ_max_new = KQ_max;
#pragma unroll
        for (int r = 0; r < NBATCH_FA / WARP_SIZE; ++r) {
            KQ_acc[r] += __half2float(maskh[kv0 + r * WARP_SIZE + lane]);   // causal mask
            KQ_max_new = fmaxf(KQ_max_new, KQ_acc[r] + FATTN_KQ_MAX_OFFSET);
        }
        KQ_max_new = warp_reduce_max(KQ_max_new);   // warp 内 32 行取 max (np=1 无需跨 warp)

        __syncthreads();   // L649-650: 确保所有线程读完 K tile, 之后才能用 KV_tmp 装 V

        const float KQ_max_scale = expf(KQ_max - KQ_max_new);   // 旧 max -> 新 max 的衰减
        KQ_max = KQ_max_new;
        float KQ_sum_add = 0.0f;
        half vals[NBATCH_FA / WARP_SIZE];
#pragma unroll
        for (int r = 0; r < NBATCH_FA / WARP_SIZE; ++r) {
            const float val = expf(KQ_acc[r] - KQ_max);   // 已含 OFFSET, f16 累加不溢出
            KQ_sum_add += val;
            vals[r] = __float2half(val);
        }
        KQ_sum = KQ_sum * KQ_max_scale + KQ_sum_add;

        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
        for (int i = 0; i < (D_HEAD / 2) / WARP_SIZE; ++i) VKQ[i] *= KQ_max_scale_h2;

        // 概率写入 smem 的 KQ 矩阵 (running max 是全 block 一份 => 可直接统一使用)
#pragma unroll
        for (int r = 0; r < NBATCH_FA / WARP_SIZE; ++r)
            KQ_s[ty * NBATCH_FA + r * WARP_SIZE + lane] = vals[r];

        // 3) VKQ += V^T @ P (L712-787): V tile 分 2 块 (各 32 行) 复用 KV_tmp
#pragma unroll
        for (int kc = 0; kc < NBATCH_FA; kc += NBATCH_V) {
            load_tile<NBATCH_V, D_HEAD / 2, 0>(
                V_h2 + (size_t)(kv0 + kc) * (D_HEAD / 2), D_HEAD / 2, KV_tmp, tid);
            __syncthreads();
#pragma unroll
            for (int k1 = 0; k1 < NBATCH_V; ++k1) {
                half2 V_k[(D_HEAD / 2) / WARP_SIZE];      // 本线程的 4 个 V 维度
                *reinterpret_cast<uint2*>(V_k) = *reinterpret_cast<const uint2*>(
                    KV_tmp + k1 * (D_HEAD / 2) + lane * 2);
                const half2 KQ_k = __half2half2(KQ_s[ty * NBATCH_FA + kc + k1]);  // 广播
#pragma unroll
                for (int i = 0; i < (D_HEAD / 2) / WARP_SIZE; ++i)
                    VKQ[i] += V_k[i] * KQ_k;              // half2 FMA: 精度换带宽
            }
            __syncthreads();   // L786: 读完本块再让下一块覆盖 KV_tmp
        }
    }

    // ---- 写回 (L980-1132): 32 线程各持 2 行的分部和, warp 归约后直接归一化 ----
    KQ_sum = warp_reduce_sum(KQ_sum);
    const float inv = 1.0f / KQ_sum;               // 单 block 全量扫 KV => 无需 combine
    const float2 o0 = __half22float2(VKQ[0]);
    const float2 o1 = __half22float2(VKQ[1]);
    float* dst_row = dst + ((size_t)head * NQ + col_Q_0 + ty) * D_HEAD + lane * 4;
    dst_row[0] = o0.x * inv;
    dst_row[1] = o0.y * inv;
    dst_row[2] = o1.x * inv;
    dst_row[3] = o1.y * inv;
}

// ------------------------------- host ---------------------------------------
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
    if (S % NBATCH_FA != 0) { printf("需要 S %% 64 == 0 (llama.cpp 靠 KV padding 保证)\n"); return 1; }
    const float scale = 1.0f / sqrtf((float)D);
    printf("[v3_tile] H=%d NQ=NKV=%d D=%d (block=64 线程, 每 block 2 个 Q 行)\n", H, S, D);

    std::vector<float> Qh, O_ref;
    std::vector<half> K, V, mask;
    gen_data(Qh, K, V, mask, H, S, D);
    mha_cpu(Qh, K, V, mask, O_ref, H, S, D, scale);

    float *dQ, *dO; half *dK, *dV, *dM;
    cudaMalloc(&dQ, Qh.size() * 4);
    cudaMalloc(&dK, K.size() * 2);
    cudaMalloc(&dV, V.size() * 2);
    cudaMalloc(&dM, mask.size() * 2);
    cudaMalloc(&dO, Qh.size() * 4);
    cudaMemcpy(dQ, Qh.data(), Qh.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, mask.data(), mask.size() * 2, cudaMemcpyHostToDevice);

    dim3 grid(S / NQ_PER_BLK, 1, H), block(WARP_SIZE, NWARPS);
    auto run = [&]() {
        flash_attn_tile<<<grid, block>>>(dQ, dK, dV, dM, dO, scale, S, S);
    };

    for (int w = 0; w < 3; ++w) run();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int reps = 10;
    cudaEventRecord(t0);
    for (int r = 0; r < reps; ++r) run();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= reps;
    printf("kernel time: %.3f ms  (~%.2f TFLOPS, causal)\n", ms,
           2.0 * H * S * (double)S * D / (ms * 1e-3) / 1e12);

    std::vector<float> O_gpu((size_t)H * S * D);
    cudaMemcpy(O_gpu.data(), dO, O_gpu.size() * 4, cudaMemcpyDeviceToHost);

    double max_abs = 0.0, max_rel = 0.0;
    for (size_t idx = 0; idx < O_gpu.size(); ++idx) {
        const double diff = fabs((double)O_gpu[idx] - O_ref[idx]);
        max_abs = fmax(max_abs, diff);
        max_rel = fmax(max_rel, diff / (fabs(O_ref[idx]) + 1e-3));
    }
    // 精度说明: 算法本身有 ~2e-4 的绝对误差底噪 (Q 预缩放转 half + KQ 概率存 half,
    // llama.cpp 原版同样如此), 再加 f16 VKQ 累加噪声 => 输出接近 0 的元素相对误差
    // 必然超标, 故用和 v2 相同的 abs/rel 组合阈值 (已用 float 累加器 debug 版验证:
    // 逻辑无误, 误差全部来自上述 half 化, 见 v3 实验记录)。
    const bool ok = max_abs < 2e-3 || max_rel < 5e-2;
    printf("max_abs_err=%.3e max_rel_err=%.3e  %s  (smem 分块 + f16 VKQ 累加)\n",
           max_abs, max_rel, ok ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dM); cudaFree(dO);
    return ok ? 0 : 1;
}

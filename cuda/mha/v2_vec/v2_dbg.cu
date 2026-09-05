// ============================================================================
// v2_vec: 提取自 llama.cpp fatnn-vec (ggml/src/ggml-cuda/fattn-vec.cuh)
// ----------------------------------------------------------------------------
// 这是 llama.cpp 三条 FlashAttention CUDA 路径中最 "纯 CUDA core" 的一条,
// 也是解码 (batch=1) 时的主力。特点:
//   - 128 threads (4 warps) 协作处理 1 个 query token (ncols=1);
//   - Q 驻留寄存器 (half2), K 以 16B 向量化载入做点积;
//   - online softmax 在 warp 内完成 (每轮 128 个 key);
//   - V 用 half2 累加进寄存器 VKQ (省寄存器, 代价是精度, llama.cpp 有意为之);
//   - KV 维度上用 gridDim.y 个 parallel block 分摊 + 一个 combine kernel 合并
//     (fattn-common.cuh launch_fattn() + flash_attn_combine_results())。
//
// 与原版的差异 (为了最小化):
//   - 只保留 ncols=1 / F16 K / F16 V 路径; 去掉量化类型/ALiBi/softcap/sinks/GQA;
//   - 去掉 KV_max 优化 (扫描 mask 跳过全 -INF 的 KV 块);
//   - NKV 不是 128 倍数时 llama.cpp 靠 KV cache padding, 这里直接 assert。
//
// 核心代码对照 (llama.cpp 行号, 2026-08 版本):
//   fattn-vec.cuh  L19-528   flash_attn_ext_vec kernel 本体
//   fattn-vec.cuh  L125-145  VKQ/KQ 寄存器与 shared 声明
//   fattn-vec.cuh  L205-247  Q 载入寄存器 (half2, 16B 向量化)
//   fattn-vec.cuh  L254-378  主循环: KQ 点积 -> online softmax -> V 累加
//   fattn-vec.cuh  L414-505  block 内跨 warp 合并 (max 对齐 + VKQ 求和)
//   fattn-common.cuh L87-115 vec_dot_fattn_vec_KQ_f16 (8 线程协作点积)
//   fattn-common.cuh L19     FATTN_KQ_MAX_OFFSET (数值范围平移技巧)
//   fattn-common.cuh L914-970 flash_attn_combine_results (parallel blocks 合并)
//
// 编译: nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -Xcompiler="/utf-8" -o mha_vec.exe mha_vec.cu
// 运行: ./mha_vec.exe [S] [H]   (默认 S=512, H=8, 要求 NKV % 128 == 0)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#define D_HEAD          128
#define WARP_SIZE       32
#define NTHREADS        128                 // 4 warps
#define NTHREADS_KQ     8                   // 每个 key 的点积由 8 个线程协作
#define NTHREADS_V      8                   // V 累加时 8 线程一组处理 4 个 key
#define CPY_NE          4                   // 16B / 4B = 4 个 (float2 或 half2)
#define V_COLS_PER_ITER (WARP_SIZE / NTHREADS_V)      // 4
#define V_ROWS_PER_TH   (2 * CPY_NE)                  // 8 个 half = 4 个 half2
// log(2)*3: 把 softmax 的数值范围整体上移 2^3, 防止 half2 累加器溢出
// (llama.cpp issue #18606 的修复, 见 fattn-common.cuh L13-19)
#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)

// ---------------------------- device 工具 ----------------------------------
template <int width>
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset, width);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}

// 8 线程协作的 QK 点积 (fattn-common.cuh L87-115 的 F16 路径):
// 组内 lane v 负责自己的 8 个 half2 分量, 再用 shuffle 在 8 线程内归约。
__device__ __forceinline__ float vec_dot_KQ_f16(const half2* __restrict__ K_h2,
                                                const half2* __restrict__ Q_reg) {
    float sum = 0.0f;
#pragma unroll
    for (int k0 = 0; k0 < D_HEAD / 2; k0 += NTHREADS_KQ * CPY_NE) {   // 步长 32
        __align__(16) half2 tmp[CPY_NE];
        *reinterpret_cast<uint4*>(tmp) =
            *reinterpret_cast<const uint4*>(K_h2 + k0 + (threadIdx.x % NTHREADS_KQ) * CPY_NE);
#pragma unroll
        for (int i = 0; i < CPY_NE; ++i) {
            const float2 kf = __half22float2(tmp[i]);
            const float2 qf = __half22float2(Q_reg[k0 / NTHREADS_KQ + i]);
            sum = fmaf(kf.x, qf.x, fmaf(kf.y, qf.y, sum));
        }
    }
    return sum;
}

// ------------------------------ 主 kernel -----------------------------------
// grid = (NQ, parallel_blocks, H), block = (32, 4)
// 一个 block 负责 1 个 query (ncols=1); gridDim.y 把 KV 轴切成 parallel blocks
__global__ void __launch_bounds__(NTHREADS, 1)
flash_attn_vec(const float* __restrict__ Q,
               const half*  __restrict__ K,
               const half*  __restrict__ V,
               const half*  __restrict__ mask,
               float*       __restrict__ dst,
               float2*      __restrict__ dst_meta,
               const float scale, const int NQ, const int NKV) {
    const int ic0  = blockIdx.x;        // 本 block 负责的 query 行 (ncols=1)
    const int head = blockIdx.z;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp * WARP_SIZE + lane;
    const int v    = lane % NTHREADS_KQ;   // 8 线程组内的位置 0..7
    const int g    = lane / NTHREADS_KQ;   // 组号 0..3

    Q += (size_t)head * NQ * D_HEAD;
    K += (size_t)head * NKV * D_HEAD;
    V += (size_t)head * NKV * D_HEAD;
    const half*  maskh = mask + (size_t)ic0 * NKV;
    const half2* K_h2  = (const half2*)K;
    const half2* V_h2  = (const half2*)V;

    // ---- 寄存器状态 (fattn-vec.cuh L125-145) ----
    // Q 载入寄存器: 组内 lane v 持有 half2 下标 {4v..4v+3} 和 {32+4v..32+4v+3}
    half2 Q_reg[(D_HEAD / 2) / NTHREADS_KQ];           // [8]
    half2 VKQ[(D_HEAD / 2) / NTHREADS_V];              // [8] 输出累加器 (f16!)
    float KQ_max = -FLT_MAX / 2.0f;                    // 本 warp 已见 key 的 running max
    float KQ_sum = 0.0f;                               // 本 warp 的 running sum
    float KQ_reg;                                      // 本线程持有的那个 key 的 score

    // ---- Q -> 寄存器 (fattn-vec.cuh L205-247) ----
    {
        const float2* Q_f2 = (const float2*)(Q + (size_t)ic0 * D_HEAD);
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int i0 = 0; i0 < D_HEAD / 2; i0 += NTHREADS_KQ * CPY_NE) {  // 步长 32
            __align__(16) float2 tmp[CPY_NE] = {{0.0f, 0.0f}};
            // 4 个 float2 = 32B, 需要两次 16B 载入 (llama.cpp 里 Q 寄存器是 half2,
            // 一个 uint4 就够; 这里是 float2, 一个 uint4 只有 2 个 float2)
            *reinterpret_cast<uint4*>(tmp) =
                *reinterpret_cast<const uint4*>(Q_f2 + i0 + v * CPY_NE);
            *reinterpret_cast<uint4*>(tmp + 2) =
                *reinterpret_cast<const uint4*>(Q_f2 + i0 + v * CPY_NE + 2);
#pragma unroll
            for (int i1 = 0; i1 < CPY_NE; ++i1)
                Q_reg[i0 / NTHREADS_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
        }
#pragma unroll
        for (int k = 0; k < (D_HEAD / 2) / NTHREADS_KQ; ++k) Q_reg[k] *= scale_h2;
    }

    // ---- shared: 先存每轮 KQ score, 最后复用存 VKQ 片段 (fattn-vec.cuh L126) ----
    // 大小取 max(128 个 score, 4 warp * 4 组 * 128 half 的 VKQ 缓冲) = 2048 half
    __shared__ half KQ_buf[(NTHREADS / WARP_SIZE) * V_COLS_PER_ITER * D_HEAD];

    // ---- 主循环: 每轮 128 个 key (fattn-vec.cuh L254-378) ----
    for (int k0 = blockIdx.y * NTHREADS; k0 < NKV; k0 += gridDim.y * NTHREADS,
             K_h2 += gridDim.y * NTHREADS * (D_HEAD / 2),
             V_h2 += gridDim.y * NTHREADS * (D_HEAD / 2)) {

        // 1) KQ 点积: i8 = 组内 key 序号, i_KQ = 本轮全局 key
        //    (fattn-vec.cuh L268-289)
        float KQ_max_new = KQ_max;   // 先保存 old max, llama.cpp L261-265
#pragma unroll
        for (int i8 = 0; i8 < NTHREADS_KQ; ++i8) {
            const int i_KQ = warp * WARP_SIZE + g * NTHREADS_KQ + i8;   // 全局 key
            float sum = vec_dot_KQ_f16(K_h2 + i_KQ * (D_HEAD / 2), Q_reg);
            sum = warp_reduce_sum<NTHREADS_KQ>(sum);                    // 8 线程内归约
            sum += __half2float(maskh[k0 + i_KQ]);                      // causal mask (-INF)
            KQ_max_new = fmaxf(KQ_max_new, sum + FATTN_KQ_MAX_OFFSET);  // L284
            if (v == i8) KQ_reg = sum;   // 组内第 i8 号线程持有该 key 的 score
        }

        // 2) online softmax (warp 内, 只覆盖本 warp 的 32 个 key;
        //    跨 warp 的 max 差异留到 kernel 尾部合并时修正) (L292-318)
#pragma unroll
        for (int offset = NTHREADS_KQ; offset < WARP_SIZE; offset <<= 1)
            KQ_max_new = fmaxf(KQ_max_new,
                               __shfl_xor_sync(0xffffffff, KQ_max_new, offset, WARP_SIZE));
        const float KQ_max_scale = expf(KQ_max - KQ_max_new);   // old->new 的衰减因子
        KQ_max = KQ_max_new;
        KQ_reg = expf(KQ_reg - KQ_max);                         // 本线程 key 的 P 值
        KQ_sum = KQ_sum * KQ_max_scale + KQ_reg;

        KQ_buf[tid] = __float2half(KQ_reg);
        __syncwarp();   // 关键: 下面的 V 累加要读同 warp 其他 lane 写入的 KQ_buf
        const half2 scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
        for (int i = 0; i < (D_HEAD / 2) / NTHREADS_V; ++i) VKQ[i] *= scale_h2;

        // 3) V 累加 (fattn-vec.cuh L324-378):
        //    组 g 负责的 key k = warp*32 + k0i + g, lane v 负责维度 8v..8v+7 等。
        //    读写都在同一 warp 内 => 只需 __syncwarp(), 不用 __syncthreads()。
#pragma unroll
        for (int k0i = 0; k0i < WARP_SIZE; k0i += V_COLS_PER_ITER) {
            const int k = warp * WARP_SIZE + k0i + g;
            const half2 KQ_k = __half2half2(KQ_buf[k]);   // P[key], 广播给组内 8 线程
#pragma unroll
            for (int i0 = 0; i0 < D_HEAD / 2; i0 += NTHREADS_V * V_ROWS_PER_TH / 2) { // 步长 32
                __align__(16) half2 tmp[V_ROWS_PER_TH / 2];   // 4 half2 = 8 half, 16B 载入
                *reinterpret_cast<uint4*>(tmp) = *reinterpret_cast<const uint4*>(
                    V_h2 + k * (D_HEAD / 2) + i0 + v * (V_ROWS_PER_TH / 2));
#pragma unroll
                for (int i1 = 0; i1 < V_ROWS_PER_TH / 2; ++i1)
                    VKQ[i0 / NTHREADS_V + i1] += tmp[i1] * KQ_k;
            }
        }
        __syncwarp();
    }

    // ---- 跨 warp 合并 (fattn-vec.cuh L414-505) ----
    // 4 个 warp 各自维护了自己的 max/sum/VKQ(部分 key), 现在对齐 max 后相加。
    __shared__ float KQ_max_shared[WARP_SIZE];
    __shared__ float KQ_sum_shared[WARP_SIZE];
    if (warp == 0) { KQ_max_shared[lane] = -FLT_MAX / 2.0f; KQ_sum_shared[lane] = 0.0f; }
    __syncthreads();
    if (lane == 0) KQ_max_shared[warp] = KQ_max;
    __syncthreads();
    float kqmax_new = KQ_max_shared[lane];
    kqmax_new = warp_reduce_max(kqmax_new);                    // block 级 max
    const float kqmax_scale = expf(KQ_max - kqmax_new);        // 本 warp 向 block max 对齐
    KQ_max = kqmax_new;
    const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
    for (int i = 0; i < (D_HEAD / 2) / NTHREADS_V; ++i) VKQ[i] *= kqmax_scale_h2;

    // VKQ 写入 shared (复用 KQ_buf), half2 布局 [warp][group][D/2]
    half2* VKQ_tmp = (half2*)KQ_buf + warp * (V_COLS_PER_ITER * D_HEAD / 2) + g * (D_HEAD / 2);
#pragma unroll
    for (int i0 = 0; i0 < D_HEAD / 2; i0 += NTHREADS_V * V_ROWS_PER_TH / 2) {
#pragma unroll
        for (int i1 = 0; i1 < V_ROWS_PER_TH / 2; ++i1)
            VKQ_tmp[i0 + v * (V_ROWS_PER_TH / 2) + i1] = VKQ[i0 / NTHREADS_V + i1];
    }
    KQ_sum *= kqmax_scale;
    KQ_sum = warp_reduce_sum<WARP_SIZE>(KQ_sum);
    if (lane == 0) KQ_sum_shared[warp] = KQ_sum;
    __syncthreads();

    // 输出: 128 个线程各负责 1 个 d; KQ_sum 用 "只有前 4 个 lane 有值" 的
    // warp_reduce 技巧完成跨 warp 求和 (fattn-vec.cuh L486-488)
    KQ_sum = KQ_sum_shared[lane];
    KQ_sum = warp_reduce_sum<WARP_SIZE>(KQ_sum);
    float dst_val = 0.0f;
#pragma unroll
    for (int w = 0; w < NTHREADS / WARP_SIZE; ++w) {
#pragma unroll
        for (int gg = 0; gg < V_COLS_PER_ITER; ++gg)
            dst_val += float(KQ_buf[w * V_COLS_PER_ITER * D_HEAD + gg * D_HEAD + tid]);
    }
    if (gridDim.y == 1) dst_val /= KQ_sum;
    dst[(((size_t)head * NQ + ic0) * gridDim.y + blockIdx.y) * D_HEAD + tid] = dst_val;

    if (gridDim.y > 1 && tid == 0)
        dst_meta[((size_t)head * NQ + ic0) * gridDim.y + blockIdx.y] =
            make_float2(KQ_max, KQ_sum);
}

// --------------------- combine kernel (fattn-common.cuh L914-970) ------------
// 把 parallel blocks 的部分结果按 max 对齐后加权合并
__global__ void flash_attn_combine(const float*  __restrict__ parts,
                                   const float2* __restrict__ meta,
                                   float*       __restrict__ dst,
                                   const int D, const int pb) {
    const size_t row = blockIdx.x;    // (head*NQ + q) 展开后的行
    const int tid = threadIdx.x;
    float kqmax = meta[row * pb + 0].x;
    for (int l = 1; l < pb; ++l) kqmax = fmaxf(kqmax, meta[row * pb + l].x);
    float num = 0.0f, den = 0.0f;
    for (int l = 0; l < pb; ++l) {
        const float2 m = meta[row * pb + l];
        const float s = expf(m.x - kqmax);
        num += s * parts[(row * pb + l) * D + tid];
        den += s * m.y;
    }
    dst[row * D + tid] = num / den;
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
    if (S % NTHREADS != 0) { printf("需要 NKV %% 128 == 0 (llama.cpp 靠 KV padding 保证)\n"); return 1; }
    const float scale = 1.0f / sqrtf((float)D);
    printf("[v2_vec] H=%d NQ=NKV=%d D=%d\n", H, S, D);

    std::vector<float> Qh, O_ref;
    std::vector<half> K, V, mask;
    gen_data(Qh, K, V, mask, H, S, D);
    mha_cpu(Qh, K, V, mask, O_ref, H, S, D, scale);

    // parallel_blocks: llama.cpp 用 occupancy API + 效率搜索 (launch_fattn L1112-1175),
    // 这里简化为 occupancy 与 KV 分块数的最小值。
    int max_blocks_per_sm = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, flash_attn_vec,
                                                  NTHREADS, 0);
    const int ntiles_KV = (S + NTHREADS - 1) / NTHREADS;
    const int pb = max(1, min(max_blocks_per_sm, ntiles_KV));
    printf("parallel_blocks=%d (max_blocks_per_sm=%d)\n", pb, max_blocks_per_sm);

    float *dQ, *dO, *dPart; half *dK, *dV, *dM; float2* dMeta;
    cudaMalloc(&dQ, Qh.size() * 4);
    cudaMalloc(&dK, K.size() * 2);
    cudaMalloc(&dV, V.size() * 2);
    cudaMalloc(&dM, mask.size() * 2);
    cudaMalloc(&dO, Qh.size() * 4);
    cudaMalloc(&dPart, (size_t)H * S * pb * D * 4);
    cudaMalloc(&dMeta, (size_t)H * S * pb * sizeof(float2));
    cudaMemcpy(dQ, Qh.data(), Qh.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, mask.data(), mask.size() * 2, cudaMemcpyHostToDevice);

    dim3 grid(S, pb, H), block(WARP_SIZE, NTHREADS / WARP_SIZE);
    auto run = [&]() {
        // llama.cpp launch_fattn: parallel_blocks == 1 时直接写 KQV, 无需 combine
        flash_attn_vec<<<grid, block>>>(dQ, dK, dV, dM, pb > 1 ? dPart : dO, dMeta, scale, S, S);
        if (pb > 1)
            flash_attn_combine<<<dim3((unsigned)H * S), D>>>(dPart, dMeta, dO, D, pb);
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
    if (pb > 1) {   // dump 一个坏行的 parts/meta, host 端重做 combine
        const int qd = 130;
        std::vector<float2> meta((size_t)H * S * pb);
        std::vector<float> part((size_t)H * S * pb * D);
        cudaMemcpy(meta.data(), dMeta, meta.size() * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(part.data(), dPart, part.size() * 4, cudaMemcpyDeviceToHost);
        const size_t row0 = (size_t)0 * S + qd;   // h=0, q=130
        printf("meta[h0 q%d]: ", qd);
        for (int l = 0; l < pb; ++l)
            printf("(max=%g sum=%g) ", meta[row0 * pb + l].x, meta[row0 * pb + l].y);
        printf("\n");
        float kqmax = -1e30f;
        for (int l = 0; l < pb; ++l) kqmax = fmaxf(kqmax, meta[row0 * pb + l].x);
        float num = 0, den = 0;
        for (int l = 0; l < pb; ++l) {
            const float2 m = meta[row0 * pb + l];
            const float s = expf(m.x - kqmax);
            num += s * part[(row0 * pb + l) * D + 0];
            den += s * m.y;
        }
        printf("host combine dim0 = %g   (O_gpu=%g, O_ref=%g)\n", num / den,
               O_gpu[(size_t)qd * D + 0], O_ref[(size_t)qd * D + 0]);
        printf("parts dim0: ");
        for (int l = 0; l < pb; ++l) printf("%g ", part[(row0 * pb + l) * D + 0]);
        printf("\n");
    }

    double max_abs = 0.0, max_rel = 0.0;
    size_t worst = 0;
    for (size_t idx = 0; idx < O_gpu.size(); ++idx) {
        const double diff = fabs((double)O_gpu[idx] - O_ref[idx]);
        if (diff > max_abs) { max_abs = diff; worst = idx; }
        max_rel = fmax(max_rel, diff / (fabs(O_ref[idx]) + 1e-3));
    }
    printf("max_abs_err=%.3e max_rel_err=%.3e  %s  (VKQ 为 f16 累加, 精度换寄存器)\n",
           max_abs, max_rel, max_rel < 5e-2 ? "PASS" : "FAIL");
    printf("worst idx: h=%zu q=%zu d=%zu  gpu=%.5f ref=%.5f\n",
           worst / (S * D), (worst / D) % S, worst % D, O_gpu[worst], O_ref[worst]);
    {   // 每个 (h,q) 行的误差, 看行分布
        int cnt = 0;
        for (int q = 0; q < S && cnt < 15; ++q) {
            double dmax = 0;
            for (int d = 0; d < D; ++d)
                dmax = fmax(dmax, fabs((double)O_gpu[(size_t)q * D + d] - O_ref[(size_t)q * D + d]));
            if (dmax > 1e-2) { printf("  bad h0 q=%d err=%.4f\n", q, dmax); cnt++; }
        }
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dM); cudaFree(dO);
    cudaFree(dPart); cudaFree(dMeta);
    return max_rel < 5e-2 ? 0 : 1;
}

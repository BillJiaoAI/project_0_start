// ============================================================================
// v4_mma: 提取自 llama.cpp fattn-mma-f16 (ggml/src/ggml-cuda/fattn-mma-f16.cuh)
// ----------------------------------------------------------------------------
// llama.cpp 的第三条 FlashAttention 路径, 也是性能最高的一条: 用 Tensor Core
// (mma.sync m16n8k16) 同时做 QK^T 和 PV。它和 v2/v3 的本质区别是数据不再按
// "线程自己的行" 组织, 而是按 mma 的 fragment 布局散布在 warp 的 32 个线程里:
//
//   固定取 (DKQ=128, DV=128, ncols=8) 的 Ampere config (L59):
//     128 线程 = 4 warp; 每 block 8 个 query; 每轮 nbatch_fa=128 个 KV 行。
//   - Q 预缩放进 smem, 一次性 ldmatrix 成 8 个 B-fragment 驻留寄存器 (Q_in_reg);
//   - 4 个 warp (np=4) 分摊 128 个 KV 行, 每 warp 独立维护自己的 online softmax
//     (KQ_max/KQ_rowsum), 产出自己的 O 部分 (f16 累加, O^T 布局);
//   - S^T = K_A @ Q_B (C 是 f32, 16 KV 行 x 8 query); P 经 get_half2 + movmatrix
//     转置成 P^T 的 B-fragment; O^T = V^T_A @ P^T_B (C 是 f16!);
//   - 由于 softmax 统计量是 warp 一份, 结尾每个 warp 写出自己的 partial
//     (O 部分 + (max,rowsum)), 由 combine kernel 按 exp(max_w - max) 加权合并
//     —— 这正是 fattn-common.cuh flash_attn_combine_results 的活, 只是这里的
//     "parallel block" 是 warp 级的 (v2 是 block 级的)。
//
// 关键技巧 (与 v2/v3 相同的不再赘述):
//   - ldmatrix: 一条指令让 32 个线程各拿到 fragment 里自己那几个元素;
//     .trans 变体在载入的同时完成转置 (V 在 smem 里是 [KV行][dv], A operand
//     要 [dv][KV行] 的 V^T, 靠 ldmatrix.trans 免去显式转置);
//   - movmatrix: 8x8 f16 寄存器内转置, 把 P (C 布局) 变成 P^T (B 布局);
//   - f16 C 累加 (f16.f16.f16.f16): 寄存器减半, 精度换带宽, llama.cpp 有意为之。
//
// 与原版的差异 (为了最小化):
//   - 只保留 ncols=8 / F16 K/V / Q_in_reg 路径; 去掉量化/ALiBi/softcap/sinks/
//     GQA/KV_max 优化/cp.async 流水 (nstages=0, 同步加载)/MLA 的 V_is_K_view;
//   - 去掉 CUDA graph 的 fixup 机制, warp partial 直接写全局数组;
//   - mask tile 载入和 KV tile 载入写成朴素线性分块 (原版按 16B 块递减粒度);
//   - 要求 S % 128 == 0 (nbatch_fa=128; llama.cpp 靠 KV cache padding 保证)。
//
// 核心代码对照 (llama.cpp 行号, 2026-08 版本):
//   fattn-mma-f16.cuh L59      (128,128,8): 128 线程 / nbatch_fa=128 / Q_in_reg
//   fattn-mma-f16.cuh L1036-43 ncols=8 的 tile 尺寸: A<16,8> B<8,8> C_f32<16,8>
//                              C_f16<16,4> (O^T 布局)
//   fattn-mma-f16.cuh L1198-42 Q 预缩放载入 smem (scale 乘在 Q 上)
//   fattn-mma-f16.cuh L1244-51 Q_B: ldmatrix 载入 B-fragment, 驻留寄存器
//   fattn-mma-f16.cuh L603-40  KQ mma: K_A(ldmatrix) @ Q_B, D 维按 8 half2 分块
//   fattn-mma-f16.cuh L694-707 mask 加进 S^T fragment
//   fattn-mma-f16.cuh L687-755 online softmax: fragment 内 max -> shuffle(16,8,4)
//                              -> exp -> rowsum (注意 KQ_idx = l%2)
//   fattn-mma-f16.cuh L851-63  rescale: exp(max旧-max新), FTZ 阈值 -20
//   fattn-mma-f16.cuh L1321-47 KQ_rowsum 循环外跨 lane 归约 (xor 16/8/4)
//   fattn-mma-f16.cuh L921-33  P -> P^T: get_half2 + movmatrix (B-fragment)
//   fattn-mma-f16.cuh L952-1011 V tile 载入 + O^T = V^T_A(ldmatrix.trans) @ P^T
//   fattn-mma-f16.cuh L1428-71 warp partial 写 dst/dst_meta (fixup 简化版)
//   fattn-common.cuh  L914-70  flash_attn_combine_results (此处简化重写)
//   mma.cuh           L226-72  tile<16,8,float> 的 get_i/get_j (C fragment)
//   mma.cuh           L388-428 tile<.,.,half2> 的 get_i/get_j (A/B/f16-C)
//   mma.cuh           L786-918 ldmatrix / ldmatrix.trans / movmatrix
//   mma.cuh           L977,1163 mma.sync m16n8k16 (f16 累加 / f32 累加)
//   mma.cuh           L712-28  get_half2 (f32 C -> f16) / get_transposed
//
// 编译: nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -Xcompiler="/utf-8" -o mha_mma.exe mha_mma.cu
// 运行: ./mha_mma.exe [S] [H]   (默认 S=512, H=8, 要求 S % 128 == 0)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#define D_HEAD          128
#define WARP_SIZE       32
#define NWARPS          4                   // 128 线程: (DKQ,DV,ncols)=(128,128,8) 的 config
#define NCOLS           8                   // 每 block 处理 8 个 query (ncols=8, "窄" 路径)
#define NP              4                   // np=4: 4 个 warp 分摊每轮 128 个 KV 行
#define NBATCH_FA       128                 // 每轮的 KV 行数 (nbatch_fa)
#define NBATCH_K2       64                  // K tile 每行的 half2 数 (= DKQ/2, nbatch_K2)
#define STRIDE_TQ       (D_HEAD / 2 + 4)    // tile_Q 行距 (half2): 68
#define STRIDE_TKV      (NBATCH_K2 + 4)     // tile_K/V 行距 (half2): 68, 避 bank 冲突
// log(2)*3: softmax 数值范围整体上移 2^3, 让 f16 累加器远离溢出 (fattn-common.cuh L19)
#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)
#define SOFTMAX_FTZ_THRESHOLD -20.0f        // rescale 因子小于 exp(-20) 时直接置 0

// ------------------------ fragment 布局速查 (mma.cuh) ------------------------
// mma.sync.aligned.m16n8k16.row.col: A(16x16 f16, row-major) @ B(16x8 f16, col-major) = D(16x8)
// 每 warp 32 线程, lane = threadIdx.x:
//   A (4 half2/lane): x[l] 的逻辑坐标  行=(l%2)*8 + lane/4,  列=(l/2)*4 + lane%4   [half2 单位]
//   B (2 half2/lane): x[l] 的逻辑坐标  行=lane/4 (N=query),  列=l*4 + lane%4 (K=dim)
//   C_f32 (4 float/lane): x[l]        行=(l/2)*8 + lane/4,  列=(lane%4)*2 + l%2
//   C_f16 (2 half2/lane): x[l]        行=l*8 + lane/4,      列=lane%4 (一个 half2 = 2 个 query)
// v4 的矩阵角色: KQ mma 算 S^T (A=K 的 16 行, B=Q 的 8 列, C=16 KV x 8 query);
//               VKQ mma 算 O^T (A=V^T 的 16 dv, B=P^T 的 16 KV, C=16 dv x 8 query)。

// ---------------------------- mma 原语 (mma.cuh) -----------------------------
struct FA16 { half2 x[4]; };   // A operand: tile<16,8,half2>  = 16x16 f16
struct FB88 { half2 x[2]; };   // B operand: tile<8,8,half2>   = 16x8  f16
struct FC168f { float x[4]; }; // C (f32):   tile<16,8,float>  = 16x8
struct FC164h { half2 x[2]; }; // C (f16):   tile<16,4,half2>  = 16x8

// ldmatrix.x4 非 trans: 载入 A fragment (mma.cuh L829-837)
static __device__ __forceinline__ void load_ldmatrix_a(FA16& t, const half2* p, int stride) {
    int* xi = (int*)t.x;
    const int* xs = (const int*)p + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3]) : "l"(xs));
}
// ldmatrix.x4.trans: 载入的同时转置 —— V 在 smem 是 [KV][dv], A operand 要 V^T,
// 寄存器顺序 {x0,x2,x1,x3} 是 ldmatrix.trans 的矩阵编号和 A 布局的对应关系
// (mma.cuh L884-894, 照抄)
static __device__ __forceinline__ void load_ldmatrix_atrans(FA16& t, const half2* p, int stride) {
    int* xi = (int*)t.x;
    const int* xs = (const int*)p + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3]) : "l"(xs));
}
// ldmatrix.x2 非 trans: 载入 B fragment (mma.cuh L785-798)
static __device__ __forceinline__ void load_ldmatrix_b(FB88& t, const half2* p, int stride) {
    int* xi = (int*)t.x;
    const int* xs = (const int*)p + (threadIdx.x % 8) * stride + ((threadIdx.x / 8) * 4) % 8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(xi[0]), "=r"(xi[1]) : "l"(xs));
}
// S^T 累加: f16 输入, f32 累加 (mma.cuh L1156-1166)
static __device__ __forceinline__ void mma_f32(FC168f& D, const FA16& A, const FB88& B) {
    int* Dxi = (int*)D.x; const int* Ai = (const int*)A.x; const int* Bi = (const int*)B.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Ai[0]), "r"(Ai[1]), "r"(Ai[2]), "r"(Ai[3]), "r"(Bi[0]), "r"(Bi[1]));
}
// O^T 累加: f16 输入, f16 累加 (寄存器减半; mma.cuh L970-990)
static __device__ __forceinline__ void mma_f16(FC164h& D, const FA16& A, const FB88& B) {
    int* Dxi = (int*)D.x; const int* Ai = (const int*)A.x; const int* Bi = (const int*)B.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Ai[0]), "r"(Ai[1]), "r"(Ai[2]), "r"(Ai[3]), "r"(Bi[0]), "r"(Bi[1]));
}
// movmatrix: 8x8 f16 寄存器内转置 (mma.cuh L28-39)
static __device__ __forceinline__ int movmatrix(int x) {
    int ret; asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(ret) : "r"(x)); return ret;
}
// f32 C fragment -> f16 (相邻两个 float 拼成 half2; mma.cuh L712-720)
static __device__ __forceinline__ FC164h get_half2(const FC168f& f) {
    FC164h r;
#pragma unroll
    for (int l0 = 0; l0 < 4; l0 += 2) r.x[l0 / 2] = make_half2(f.x[l0], f.x[l0 + 1]);
    return r;
}
// P (C 布局, 16 KV x 8 query) -> P^T (B 布局, 16 KV x 8 query 的转置视角)
// get_transposed: movmatrix 两次 (mma.cuh L722-728)
static __device__ __forceinline__ FB88 get_transposed(const FC164h& t) {
    FB88 r; *((int*)&r.x[0]) = movmatrix(*((const int*)&t.x[0]));
            *((int*)&r.x[1]) = movmatrix(*((const int*)&t.x[1])); return r;
}

// ------------------------------ 主 kernel -----------------------------------
// grid = (NQ/8, 1, H), block = (32, 4)
// 4 个 warp 处理同 8 个 query, 每轮各分摊 32 个 KV 行 (行号 {16w..16w+16} 和
// {64+16w..64+16w+16}, 由 i_KQ_0 = i_KQ_00 + warp*16 决定); softmax 统计量
// warp 一份, 结尾写 partial, combine kernel 合并。
__global__ void __launch_bounds__(NWARPS * WARP_SIZE)
flash_attn_mma_f16(const float* __restrict__ Q,
                   const half*  __restrict__ K,
                   const half*  __restrict__ V,
                   const half*  __restrict__ mask,
                   float*       __restrict__ dst_part,   // warp partial: [(h*4+w)*NQ+q][D]
                   float2*      __restrict__ dst_meta,   // [(h*4+w)*NQ+q] = (max, rowsum)
                   const float scale, const int NQ, const int NKV) {
    const int q0   = blockIdx.x * NCOLS;   // 本 block 的第一个 query 行
    const int head = blockIdx.z;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp * WARP_SIZE + lane;

    Q += ((size_t)head * NQ + q0) * D_HEAD;
    K += (size_t)head * NKV * D_HEAD;
    V += (size_t)head * NKV * D_HEAD;
    const half2* K_h2 = (const half2*)K;
    const half2* V_h2 = (const half2*)V;

    // ---- smem 布局 (L1175-1178): Q_in_reg => tile_Q 载完 Q 后让位给 tile_K 复用 ----
    extern __shared__ half2 smem[];
    half2* tile_Q  = smem;                                    // [8][68], 之后复用为 tile_K
    half2* tile_V  = tile_Q + NBATCH_FA * STRIDE_TKV;         // [128][68]
    half*  tile_mask = (half*)(tile_V + NBATCH_FA * STRIDE_TKV); // [8][136]

    // ---- 寄存器状态 ----
    FB88 Q_B[D_HEAD / 16];                 // Q 的 8 个 B-fragment (各 16 维 x 8 query), 常驻
    FC164h VKQ_C[D_HEAD / 16];             // O^T 累加器: 8 个 tile, 各 16 dv x 8 query (f16!)
    float KQ_max[2]    = {-FLT_MAX / 2, -FLT_MAX / 2};  // 每 query 列一份; -FLT_MAX/2 防 -INF 减出 NaN
    float KQ_rowsum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int t = 0; t < D_HEAD / 16; ++t) VKQ_C[t] = {};

    // ---- Q 预缩放载入 smem (L1198-1242): scale 乘在 Q 上, KQ^T 就免再乘 ----
    {
        const float2* Q_f2 = (const float2*)Q;
        const half2 scale_h2 = make_half2(scale, scale);
        for (int c = tid; c < NCOLS * (D_HEAD / 2); c += NWARPS * WARP_SIZE) {
            const int jc = c / (D_HEAD / 2), k = c % (D_HEAD / 2);
            const float2 tmp = Q_f2[jc * (D_HEAD / 2) + k];
            tile_Q[jc * STRIDE_TQ + k] = scale_h2 * make_half2(tmp.x, tmp.y);
        }
    }
    __syncthreads();

    // ---- Q_B: ldmatrix 载入 B-fragment, 驻留寄存器 (L1244-1251) ----
    // Q_B[k/8] 覆盖维度 [16k, 16k+16); 全部 4 个 warp 载相同内容 (它们处理同 8 query)
#pragma unroll
    for (int k0 = 0; k0 < D_HEAD / 2; k0 += 8)
        load_ldmatrix_b(Q_B[k0 / 8], tile_Q + k0, STRIDE_TQ);
    __syncthreads();   // Q_B 全部读完后才允许 KV 主循环把 tile_Q 缓冲覆写成 tile_K

    // ---- 主循环: 每轮 nbatch_fa=128 个 KV 行 ----
    for (int kv0 = 0; kv0 < NKV; kv0 += NBATCH_FA) {
        // 1) K tile + mask tile 载入 smem (原版 flash_attn_ext_f16_load_tile/mask,
        //    按 16B 块递减粒度; 这里简化成线性分块: 128 行 x 16 块)
        for (int c = tid; c < NBATCH_FA * (NBATCH_K2 / 4); c += NWARPS * WARP_SIZE) {
            const int r = c / (NBATCH_K2 / 4), k = (c % (NBATCH_K2 / 4)) * 4;
            *reinterpret_cast<uint4*>(tile_Q + r * STRIDE_TKV + k) =
                *reinterpret_cast<const uint4*>(K_h2 + (size_t)(kv0 + r) * (D_HEAD / 2) + k);
        }
        for (int c = tid; c < NCOLS * NBATCH_FA; c += NWARPS * WARP_SIZE) {
            const int j = c / NBATCH_FA, i = c % NBATCH_FA;
            tile_mask[j * (NBATCH_FA + 8) + i] = mask[(size_t)(q0 + j) * NKV + kv0 + i];
        }
        __syncthreads();

        // 2) KQ mma: S^T = K_A @ Q_B (L603-640)。每 warp 2 个 C tile (各 16 KV 行),
        //    D 维按 8 half2 = 16 维分块喂入 (Q_B 常驻, 只 ldmatrix K)。
        FC168f KQ_C[2] = {};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int i_KQ_0 = c * 64 + warp * 16;    // 本 warp 在本 C tile 的 KV 行起点
#pragma unroll
            for (int k = 0; k < NBATCH_K2; k += 8) {  // 每步 16 维
                FA16 K_A;
                load_ldmatrix_a(K_A, tile_Q + i_KQ_0 * STRIDE_TKV + k, STRIDE_TKV);
                mma_f32(KQ_C[c], K_A, Q_B[k / 8]);
            }
        }

        // 3) mask 加进 S^T (L694-707): fragment 坐标 -> (KV 行 i, query 列 j)
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = c * 64 + warp * 16 + (l / 2) * 8 + lane / 4;  // get_i(l)
                const int j = (lane % 4) * 2 + l % 2;                       // get_j(l)
                KQ_C[c].x[l] += __half2float(tile_mask[j * (NBATCH_FA + 8) + i]);
            }

        // 4) online softmax (L687-755)。一个 query 列的 32 个 KV 值散布在 8 个线程
        //    (同 lane%4) 里: 先 fragment 内 max, 再 shuffle_xor 16/8/4 归约。
        float KQ_max_new[2] = {KQ_max[0], KQ_max[1]}, KQ_rowsum_add[2] = {0.0f, 0.0f};
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int l = 0; l < 4; ++l)
                KQ_max_new[l % 2] = fmaxf(KQ_max_new[l % 2], KQ_C[c].x[l] + FATTN_KQ_MAX_OFFSET);
#pragma unroll
        for (int col = 0; col < 2; ++col)
#pragma unroll
            for (int off = 16; off >= 4; off >>= 1)   // 同列的 8 线程互减归约
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], off));
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                KQ_C[c].x[l] = expf(KQ_C[c].x[l] - KQ_max_new[l % 2]);   // P, 未归一化
                KQ_rowsum_add[l % 2] += KQ_C[c].x[l];
            }

        // 5) rescale (L851-918): 旧 max -> 新 max 的衰减乘进 O 累加器和 rowsum
        float KQ_max_scale[2];
        const half2 KQ_max_scale_h2 = make_half2(1.0f, 1.0f);  // placeholder, 下面逐列算
        (void)KQ_max_scale_h2;
#pragma unroll
        for (int col = 0; col < 2; ++col) {
            const float diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(diff);
            // FTZ: max 几乎没动时 scale≈1 无害; 但 f16 累加器怕极小因子, 阈值以下直接 0
            *((uint32_t*)&KQ_max_scale[col]) *= diff >= SOFTMAX_FTZ_THRESHOLD;
            KQ_max[col] = KQ_max_new[col];
            KQ_rowsum[col] = KQ_max_scale[col] * KQ_rowsum[col] + KQ_rowsum_add[col];
        }
        const half2 mscale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]); // 每 half2 = 2 个相邻 query
#pragma unroll
        for (int t = 0; t < D_HEAD / 16; ++t)
#pragma unroll
            for (int l = 0; l < 2; ++l) VKQ_C[t].x[l] *= mscale_h2;

        // 6) P -> P^T 的 B-fragment (L921-933): f32 C -> f16 C -> movmatrix 转置
        FB88 Bp[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) Bp[c] = get_transposed(get_half2(KQ_C[c]));

        // 7) V tile 载入 (整 DV 一块) + O^T mma (L952-1011)
        for (int c = tid; c < NBATCH_FA * (NBATCH_K2 / 4); c += NWARPS * WARP_SIZE) {
            const int r = c / (NBATCH_K2 / 4), k = (c % (NBATCH_K2 / 4)) * 4;
            *reinterpret_cast<uint4*>(tile_V + r * STRIDE_TKV + k) =
                *reinterpret_cast<const uint4*>(V_h2 + (size_t)(kv0 + r) * (D_HEAD / 2) + k);
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < D_HEAD / 16; ++t) {          // i_VKQ_0 = t*16 (dv 维)
#pragma unroll
            for (int k00 = 0; k00 < NBATCH_FA / 2; k00 += 32) {   // 每 warp 16 个 KV 行
                FA16 A;   // V^T tile: dv [16t,16t+16) x KV [2*(k00+warp*8), +16)
                load_ldmatrix_atrans(A, tile_V + 2 * (k00 + warp * 8) * STRIDE_TKV + t * 8, STRIDE_TKV);
                mma_f16(VKQ_C[t], A, Bp[k00 / 32]);               // 注意行号是 2*k0, 故上限为 nbatch_fa/2
            }
        }
        __syncthreads();   // 读完 tile_Q(K)/tile_V 再进下一轮覆盖
    }

    // ---- KQ_rowsum 跨 lane 归约 (L1321-1347) ----
    // 每个 query 列的 exp 和散布在同 lane%4 的 8 个线程里 (与 KQ_max 的归约同构,
    // llama.cpp 是循环结束后一次性归约)。之所以能循环外再归约: 每轮的
    // KQ_max_scale 对同组线程是同一个值 (KQ_max_new 已归约), 逐线程带 scale 累加
    // 再求和 == 每轮归约; rescale 的 FTZ 也不受影响 (diff 组内一致)。
#pragma unroll
    for (int col = 0; col < 2; ++col)
#pragma unroll
        for (int off = 16; off >= 4; off >>= 1)
            KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], off);

    // ---- 写 warp partial (L1428-1471 的简化: 直接写全局, combine 里合并) ----
    // VKQ_C[t].x[l] = (query 2*(lane%4)+{0,1}, dv 16t + 8l + lane/4) 的 f16 对
    float* part = dst_part + ((size_t)(head * NWARPS + warp) * NQ) * D_HEAD;
#pragma unroll
    for (int t = 0; t < D_HEAD / 16; ++t)
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            const float2 v = __half22float2(VKQ_C[t].x[l]);
            const int d = t * 16 + l * 8 + lane / 4;
            part[(size_t)(q0 + (lane % 4) * 2 + 0) * D_HEAD + d] = v.x;
            part[(size_t)(q0 + (lane % 4) * 2 + 1) * D_HEAD + d] = v.y;
        }
    if (lane < 4) {   // lane/4==0 的 4 个线程恰好覆盖 8 个 query 列的统计量
        float2* meta = dst_meta + (size_t)(head * NWARPS + warp) * NQ + q0;
        meta[lane * 2 + 0] = make_float2(KQ_max[0], KQ_rowsum[0]);
        meta[lane * 2 + 1] = make_float2(KQ_max[1], KQ_rowsum[1]);
    }
}

// ------------------------- combine kernel (简化版) ---------------------------
// fattn-common.cuh flash_attn_combine_results 的数学: 每个 warp partial 是
//   O_w = sum_v exp(s_v - max_w) V_v,  rowsum_w = sum_v exp(s_v - max_w)
// 用 max = max_w 做公共基准加权合并即可无损还原真 softmax:
//   O = sum_w exp(max_w - max) O_w / sum_w exp(max_w - max) rowsum_w
__global__ void combine(const float* __restrict__ part, const float2* __restrict__ meta,
                        float* __restrict__ dst, int total, int NQ, int D, int nparts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;               // total = H*NQ*D; 只判 NQ*D 会漏掉 head>=1!
    const int head = idx / (NQ * D), rem = idx % (NQ * D);
    const int q = rem / D, d = rem % D;

    float m = -FLT_MAX / 2;
    for (int p = 0; p < nparts; ++p) m = fmaxf(m, meta[((size_t)head * nparts + p) * NQ + q].x);
    float num = 0.0f, den = 0.0f;
    for (int p = 0; p < nparts; ++p) {
        const float2 mr = meta[((size_t)head * nparts + p) * NQ + q];
        const float w = expf(mr.x - m);   // 全 masked 的 partial: w=0, num/den 贡献 0
        num += w * part[(((size_t)head * nparts + p) * NQ + q) * D + d];
        den += w * mr.y;
    }
    dst[(size_t)head * NQ * D + rem] = num / den;
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
    if (S % NBATCH_FA != 0) { printf("需要 S %% 128 == 0 (llama.cpp 靠 KV padding 保证)\n"); return 1; }
    const float scale = 1.0f / sqrtf((float)D);
    printf("[v4_mma] H=%d NQ=NKV=%d D=%d (block=128 线程, 每 block 8 个 Q 行, 4 warp 分摊 KV)\n", H, S, D);

    std::vector<float> Qh, O_ref;
    std::vector<half> K, V, mask;
    gen_data(Qh, K, V, mask, H, S, D);
    mha_cpu(Qh, K, V, mask, O_ref, H, S, D, scale);

    float *dQ, *dO, *dO_part; half *dK, *dV, *dM; float2* dMeta;
    cudaMalloc(&dQ, Qh.size() * 4);
    cudaMalloc(&dK, K.size() * 2);
    cudaMalloc(&dV, V.size() * 2);
    cudaMalloc(&dM, mask.size() * 2);
    cudaMalloc(&dO, Qh.size() * 4);
    cudaMalloc(&dO_part, (size_t)H * NWARPS * S * D * 4);      // 每 warp 一份 partial
    cudaMalloc(&dMeta, (size_t)H * NWARPS * S * sizeof(float2));
    cudaMemcpy(dQ, Qh.data(), Qh.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, mask.data(), mask.size() * 2, cudaMemcpyHostToDevice);

    // smem: tile_K(复用 tile_Q 的缓冲) + tile_V + tile_mask, 共 ~70KB, 需 opt-in
    const size_t smem_size = (size_t)NBATCH_FA * STRIDE_TKV * 4 * 2
                           + (size_t)NCOLS * (NBATCH_FA + 8) * 2;
    cudaError_t attr_err = cudaFuncSetAttribute(flash_attn_mma_f16,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
    if (attr_err != cudaSuccess) { printf("cudaFuncSetAttribute 失败: %s\n", cudaGetErrorString(attr_err)); return 1; }

    dim3 grid(S / NCOLS, 1, H), block(WARP_SIZE, NWARPS);
    auto run = [&]() {
        flash_attn_mma_f16<<<grid, block, smem_size>>>(dQ, dK, dV, dM, dO_part, dMeta, scale, S, S);
        combine<<<(H * S * D + 255) / 256, 256>>>(dO_part, dMeta, dO, H * S * D, S, D, NWARPS);
    };

    for (int w = 0; w < 3; ++w) run();
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) { printf("kernel 错误: %s\n", cudaGetErrorString(kerr)); return 1; }
    cudaDeviceSynchronize();
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
    size_t bad = 0; int n_bad = 0;
    for (size_t idx = 0; idx < O_gpu.size(); ++idx) {
        const double diff = fabs((double)O_gpu[idx] - O_ref[idx]);
        if (diff > max_abs) { max_abs = diff; bad = idx; }
        max_rel = fmax(max_rel, diff / (fabs(O_ref[idx]) + 1e-3));
        if (diff > 1e-2) ++n_bad;
    }
    const size_t bh = bad / ((size_t)S * D), bq = (bad / D) % S, bd = bad % D;
    printf("worst: h=%zu q=%zu d=%zu gpu=%.6f ref=%.6f   (>1e-2: %d / %zu)\n",
           bh, bq, bd, O_gpu[bad], O_ref[bad], n_bad, O_gpu.size());
    // 精度说明: 与 v3 同源的误差底噪 (Q 预缩放转 half + P 存 half), 外加 f16 VKQ
    // mma 累加 (llama.cpp 原版同样如此); 用和 v2/v3 相同的 abs/rel 组合阈值。
    const bool ok = max_abs < 2e-3 || max_rel < 5e-2;
    printf("max_abs_err=%.3e max_rel_err=%.3e  %s  (tensor core mma + f16 VKQ 累加)\n",
           max_abs, max_rel, ok ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dM); cudaFree(dO);
    cudaFree(dO_part); cudaFree(dMeta);
    return ok ? 0 : 1;
}

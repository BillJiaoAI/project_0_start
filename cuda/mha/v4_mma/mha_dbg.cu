// 调试版 mha_mma: 对比 warp 级 partial/meta 与 CPU 同粒度计算; dump S^T
// argv: dbg.exe [S] [H] [stage]   stage=1: 对比 partial+meta; stage=2: dump S^T(iter0, head0, blk0)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#define D_HEAD          128
#define WARP_SIZE       32
#define NWARPS          4
#define NCOLS           8
#define NBATCH_FA       128
#define NBATCH_K2       64
#define STRIDE_TQ       (D_HEAD / 2 + 4)
#define STRIDE_TKV      (NBATCH_K2 + 4)
#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)

struct FA16 { half2 x[4]; };
struct FB88 { half2 x[2]; };
struct FC168f { float x[4]; };
struct FC164h { half2 x[2]; };

static __device__ __forceinline__ void load_ldmatrix_a(FA16& t, const half2* p, int stride) {
    int* xi = (int*)t.x;
    const int* xs = (const int*)p + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3]) : "l"(xs));
}
static __device__ __forceinline__ void load_ldmatrix_atrans(FA16& t, const half2* p, int stride) {
    int* xi = (int*)t.x;
    const int* xs = (const int*)p + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3]) : "l"(xs));
}
static __device__ __forceinline__ void load_ldmatrix_b(FB88& t, const half2* p, int stride) {
    int* xi = (int*)t.x;
    const int* xs = (const int*)p + (threadIdx.x % 8) * stride + ((threadIdx.x / 8) * 4) % 8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(xi[0]), "=r"(xi[1]) : "l"(xs));
}
static __device__ __forceinline__ void mma_f32(FC168f& D, const FA16& A, const FB88& B) {
    int* Dxi = (int*)D.x; const int* Ai = (const int*)A.x; const int* Bi = (const int*)B.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Ai[0]), "r"(Ai[1]), "r"(Ai[2]), "r"(Ai[3]), "r"(Bi[0]), "r"(Bi[1]));
}
static __device__ __forceinline__ void mma_f16(FC164h& D, const FA16& A, const FB88& B) {
    int* Dxi = (int*)D.x; const int* Ai = (const int*)A.x; const int* Bi = (const int*)B.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Ai[0]), "r"(Ai[1]), "r"(Ai[2]), "r"(Ai[3]), "r"(Bi[0]), "r"(Bi[1]));
}
static __device__ __forceinline__ int movmatrix(int x) {
    int ret; asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(ret) : "r"(x)); return ret;
}
static __device__ __forceinline__ FC164h get_half2(const FC168f& f) {
    FC164h r;
#pragma unroll
    for (int l0 = 0; l0 < 4; l0 += 2) r.x[l0 / 2] = make_half2(f.x[l0], f.x[l0 + 1]);
    return r;
}
static __device__ __forceinline__ FB88 get_transposed(const FC164h& t) {
    FB88 r; *((int*)&r.x[0]) = movmatrix(*((const int*)&t.x[0]));
            *((int*)&r.x[1]) = movmatrix(*((const int*)&t.x[1])); return r;
}

__global__ void __launch_bounds__(NWARPS * WARP_SIZE)
flash_attn_mma_f16(const float* __restrict__ Q, const half* __restrict__ K,
                   const half* __restrict__ V, const half* __restrict__ mask,
                   float* __restrict__ dst_part, float2* __restrict__ dst_meta,
                   float* __restrict__ dbgS,
                   const float scale, const int NQ, const int NKV) {
    const int q0   = blockIdx.x * NCOLS;
    const int head = blockIdx.z;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp * WARP_SIZE + lane;

    Q += ((size_t)head * NQ + q0) * D_HEAD;
    K += (size_t)head * NKV * D_HEAD;
    V += (size_t)head * NKV * D_HEAD;
    const half2* K_h2 = (const half2*)K;
    const half2* V_h2 = (const half2*)V;

    extern __shared__ half2 smem[];
    half2* tile_Q  = smem;
    half2* tile_V  = tile_Q + NBATCH_FA * STRIDE_TKV;
    half*  tile_mask = (half*)(tile_V + NBATCH_FA * STRIDE_TKV);

    FB88 Q_B[D_HEAD / 16];
    FC164h VKQ_C[D_HEAD / 16];
    float KQ_max[2]    = {-FLT_MAX / 2, -FLT_MAX / 2};
    float KQ_rowsum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int t = 0; t < D_HEAD / 16; ++t) VKQ_C[t] = {};

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

#pragma unroll
    for (int k0 = 0; k0 < D_HEAD / 2; k0 += 8)
        load_ldmatrix_b(Q_B[k0 / 8], tile_Q + k0, STRIDE_TQ);
    __syncthreads();

    for (int kv0 = 0; kv0 < NKV; kv0 += NBATCH_FA) {
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

        FC168f KQ_C[2] = {};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int i_KQ_0 = c * 64 + warp * 16;
#pragma unroll
            for (int k = 0; k < NBATCH_K2; k += 8) {
                FA16 K_A;
                load_ldmatrix_a(K_A, tile_Q + i_KQ_0 * STRIDE_TKV + k, STRIDE_TKV);
                mma_f32(KQ_C[c], K_A, Q_B[k / 8]);
            }
        }

        // dump S^T (iter 0): 仅 head0/blk0, KV 行 kv0+i, query q0+j
        if (dbgS && blockIdx.z == 0 && blockIdx.x == 0 && kv0 == 0)
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int i = c * 64 + warp * 16 + (l / 2) * 8 + lane / 4;
                    const int j = (lane % 4) * 2 + l % 2;
                    dbgS[i * NCOLS + j] = KQ_C[c].x[l];
                }

#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = c * 64 + warp * 16 + (l / 2) * 8 + lane / 4;
                const int j = (lane % 4) * 2 + l % 2;
                KQ_C[c].x[l] += __half2float(tile_mask[j * (NBATCH_FA + 8) + i]);
            }

        float KQ_max_new[2] = {KQ_max[0], KQ_max[1]}, KQ_rowsum_add[2] = {0.0f, 0.0f};
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int l = 0; l < 4; ++l)
                KQ_max_new[l % 2] = fmaxf(KQ_max_new[l % 2], KQ_C[c].x[l] + FATTN_KQ_MAX_OFFSET);
#pragma unroll
        for (int col = 0; col < 2; ++col)
#pragma unroll
            for (int off = 16; off >= 4; off >>= 1)
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], off));
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                KQ_C[c].x[l] = expf(KQ_C[c].x[l] - KQ_max_new[l % 2]);
                KQ_rowsum_add[l % 2] += KQ_C[c].x[l];
            }

        float KQ_max_scale[2];
#pragma unroll
        for (int col = 0; col < 2; ++col) {
            const float diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(diff);
            *((uint32_t*)&KQ_max_scale[col]) *= diff >= -20.0f;
            KQ_max[col] = KQ_max_new[col];
            KQ_rowsum[col] = KQ_max_scale[col] * KQ_rowsum[col] + KQ_rowsum_add[col];
        }
        const half2 mscale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
        for (int t = 0; t < D_HEAD / 16; ++t)
#pragma unroll
            for (int l = 0; l < 2; ++l) VKQ_C[t].x[l] *= mscale_h2;

        FB88 Bp[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) Bp[c] = get_transposed(get_half2(KQ_C[c]));

        for (int c = tid; c < NBATCH_FA * (NBATCH_K2 / 4); c += NWARPS * WARP_SIZE) {
            const int r = c / (NBATCH_K2 / 4), k = (c % (NBATCH_K2 / 4)) * 4;
            *reinterpret_cast<uint4*>(tile_V + r * STRIDE_TKV + k) =
                *reinterpret_cast<const uint4*>(V_h2 + (size_t)(kv0 + r) * (D_HEAD / 2) + k);
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < D_HEAD / 16; ++t) {
#pragma unroll
            for (int k00 = 0; k00 < NBATCH_FA / 2; k00 += 32) {
                FA16 A;
                load_ldmatrix_atrans(A, tile_V + 2 * (k00 + warp * 8) * STRIDE_TKV + t * 8, STRIDE_TKV);
                mma_f16(VKQ_C[t], A, Bp[k00 / 32]);
            }
        }
        __syncthreads();
    }

    // KQ_rowsum 跨 lane 归约 (llama.cpp L1321-47): 同 lane%4 的 8 线程各持部分和
#pragma unroll
    for (int col = 0; col < 2; ++col)
#pragma unroll
        for (int off = 16; off >= 4; off >>= 1)
            KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], off);

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
    if (lane < 4) {
        if (dbgS && blockIdx.z == 0 && blockIdx.x == 0) {
            dbgS[(size_t)NBATCH_FA * NCOLS + warp * 8 + lane * 2 + 0] = KQ_max[0];
            dbgS[(size_t)NBATCH_FA * NCOLS + warp * 8 + lane * 2 + 1] = KQ_max[1];
            // 索引诊断: kernel 侧计算的 meta 下标 (应为 warp*NQ + q0 + 2*lane)
            dbgS[(size_t)NBATCH_FA * NCOLS + 32 + warp * 4 + lane] =
                (float)((head * NWARPS + warp) * NQ + q0 + lane * 2);
        }
        float2* meta = dst_meta + (size_t)(head * NWARPS + warp) * NQ + q0;
        meta[lane * 2 + 0] = make_float2(KQ_max[0], KQ_rowsum[0]);
        meta[lane * 2 + 1] = make_float2(KQ_max[1], KQ_rowsum[1]);
    }
}

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

int main(int argc, char** argv) {
    int S = argc > 1 ? atoi(argv[1]) : 128;
    int H = argc > 2 ? atoi(argv[2]) : 1;
    int stage = argc > 3 ? atoi(argv[3]) : 1;
    const int D = D_HEAD;
    const float scale = 1.0f / sqrtf((float)D);

    std::vector<float> Qh, O_ref;
    std::vector<half> K, V, mask;
    gen_data(Qh, K, V, mask, H, S, D);

    float *dQ, *dO_part, *dS = nullptr; half *dK, *dV, *dM; float2* dMeta;
    cudaMalloc(&dQ, Qh.size() * 4);
    cudaMalloc(&dK, K.size() * 2);
    cudaMalloc(&dV, V.size() * 2);
    cudaMalloc(&dM, mask.size() * 2);
    cudaMalloc(&dO_part, (size_t)H * NWARPS * S * D * 4);
    cudaMalloc(&dMeta, (size_t)H * NWARPS * S * sizeof(float2));
    if (stage >= 2) cudaMalloc(&dS, ((size_t)NBATCH_FA * NCOLS + 64) * 4);
    cudaMemcpy(dQ, Qh.data(), Qh.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, mask.data(), mask.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(dO_part, 0, (size_t)H * NWARPS * S * D * 4);
    if (stage == 3) cudaMemset(dMeta, 0xFF, (size_t)H * NWARPS * S * sizeof(float2));   // NaN 预填
    else cudaMemset(dMeta, 0, (size_t)H * NWARPS * S * sizeof(float2));

    const size_t smem_size = (size_t)NBATCH_FA * STRIDE_TKV * 4 * 2
                           + (size_t)NCOLS * (NBATCH_FA + 8) * 2;
    cudaFuncSetAttribute(flash_attn_mma_f16, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
    dim3 grid(S / NCOLS, 1, H), block(WARP_SIZE, NWARPS);
    flash_attn_mma_f16<<<grid, block, smem_size>>>(dQ, dK, dV, dM, dO_part, dMeta, dS, scale, S, S);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("kernel: %s\n", cudaGetErrorString(err)); return 1; }

    std::vector<float> part((size_t)H * NWARPS * S * D);
    std::vector<float2> meta((size_t)H * NWARPS * S);
    cudaMemcpy(part.data(), dO_part, part.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(meta.data(), dMeta, meta.size() * sizeof(float2), cudaMemcpyDeviceToHost);

    if (stage == 2) {
        std::vector<float> Sg((size_t)NBATCH_FA * NCOLS + 64);
        cudaMemcpy(Sg.data(), dS, Sg.size() * 4, cudaMemcpyDeviceToHost);
        // CPU S^T: head0, query q(0..7), KV 行 i(0..127); S^T[i][q] = dot(K[i],Q[q])*scale + mask
        double max_err = 0; int bi = -1, bq = -1;
        for (int i = 0; i < 128; ++i)
            for (int q = 0; q < 8; ++q) {
                float dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += Qh[q * D + d] * __half2float(K[i * D + d]);
                float ref = dot * scale + (i <= q ? 0.0f : -INFINITY);
                double e = fabs(Sg[i * NCOLS + q] - ref);
                if (e > max_err) { max_err = e; bi = i; bq = q; }
            }
        printf("stage2: S^T max_err=%.3e at (i=%d,q=%d) gpu=%.4f cpu=%.4f\n",
               max_err, bi, bq, bi >= 0 ? Sg[bi * NCOLS + bq] : 0.0f,
               bi >= 0 ? (bi <= bq ? 0.0f : -INFINITY) : 0.0f);
        int nbad = 0;
        for (int i = 0; i < 128; ++i)
            for (int q = 0; q < 8; ++q) {
                float dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += Qh[q * D + d] * __half2float(K[i * D + d]);
                if (i > q) continue;   // 跳过 -INF 的掩码位
                const double e = fabs(Sg[i * NCOLS + q] - dot * scale);
                if (e > 1e-2) { if (nbad < 8) printf("  BAD i=%d q=%d gpu=%.4f cpu=%.4f\n", i, q, Sg[i * NCOLS + q], dot * scale); ++nbad; }
            }
        printf("  unmasked mismatch count: %d\n", nbad);
        return 0;
    }
    if (stage == 3) {
        std::vector<float> Sg((size_t)NBATCH_FA * NCOLS + 64);
        cudaMemcpy(Sg.data(), dS, Sg.size() * 4, cudaMemcpyDeviceToHost);
        // dbgS 尾部 32 float = kernel 内 lane<4 写出的 KQ_max[0..1] (head0/blk0)
        for (int w = 0; w < NWARPS; ++w) {
            printf("warp %d in-kernel KQ_max (lane0..3): ", w);
            for (int l = 0; l < 4; ++l)
                printf("(%.3e,%.3e) ", Sg[(size_t)NBATCH_FA * NCOLS + w * 8 + l * 2],
                       Sg[(size_t)NBATCH_FA * NCOLS + w * 8 + l * 2 + 1]);
            printf("\n");
        }
        // kernel 侧 meta 下标诊断 (head0/blk0, 应为 warp*128 + 2*lane)
        for (int w = 0; w < NWARPS; ++w) {
            printf("warp %d meta idx (lane0..3): ", w);
            for (int l = 0; l < 4; ++l)
                printf("%d ", (int)Sg[(size_t)NBATCH_FA * NCOLS + 32 + w * 4 + l]);
            printf("\n");
        }
        // 全量 meta dump: 找 w2/w3 的 (-1.7e38,0) 到底写到了哪
        int n_nan = 0, n_mmax = 0, n_other = 0;
        for (size_t i = 0; i < meta.size(); ++i) {
            if (isnan(meta[i].x)) ++n_nan;
            else if (meta[i].x < -1e38) ++n_mmax;
            else ++n_other;
        }
        printf("meta: NaN=%d (-1.7e38)=%d other=%d total=%zu\n", n_nan, n_mmax, n_other, meta.size());
        for (int w = 0; w < NWARPS; ++w) {
            int cnt = 0;
            for (int q = 0; q < S; ++q) if (meta[(size_t)w * S + q].x < -1e38) ++cnt;
            printf("warp %d: -1.7e38 count=%d (期望 w0:0 w1:16 w2:32 w3:48)\n", w, cnt);
        }
        printf("warp2 q=28..40: ");
        for (int q = 28; q <= 40; ++q) printf("(%.2e,%.2e) ", meta[(size_t)2 * S + q].x, meta[(size_t)2 * S + q].y);
        printf("\nwarp3 q=44..56: ");
        for (int q = 44; q <= 56; ++q) printf("(%.2e,%.2e) ", meta[(size_t)3 * S + q].x, meta[(size_t)3 * S + q].y);
        printf("\n");
        return 0;
    }

    // stage 1: CPU warp partial 对比 (head 0, queries 0..7)
    // warp w 处理 KV 行 {16w..16w+16} ∪ {64+16w..64+16w+16}
    for (int q = 0; q < 8; ++q) {
        // 全局 softmax 统计
        float smax = -INFINITY;
        std::vector<float> sv(S);
        for (int j = 0; j < S; ++j) {
            float dot = 0;
            for (int d = 0; d < D; ++d) dot += Qh[q * D + d] * __half2float(K[j * D + d]);
            sv[j] = dot * scale + (j <= q ? 0.0f : -INFINITY);
            smax = fmaxf(smax, sv[j]);
        }
        for (int w = 0; w < NWARPS; ++w) {
            // CPU warp partial: O_w = sum exp(sv-offset - max_w) V,  max_w = max(sv+offset) over warp rows
            float m_w = -FLT_MAX / 2;
            for (int r = 0; r < 32; ++r) {
                const int j = (r / 16) * 64 + w * 16 + (r % 16);
                if (j <= q) m_w = fmaxf(m_w, sv[j] + FATTN_KQ_MAX_OFFSET);
            }
            float rs_w = 0;
            std::vector<float> ow(D, 0.0f);
            for (int r = 0; r < 32; ++r) {
                const int j = (r / 16) * 64 + w * 16 + (r % 16);
                if (j > q) continue;
                const float p = expf(sv[j] - m_w);
                rs_w += p;
                for (int d = 0; d < D; ++d) ow[d] += p * __half2float(V[j * D + d]);
            }
            const float2 mg = meta[(size_t)w * S + q];
            double e1 = 0, e2 = fabs(mg.x - m_w) / (fabs(m_w) + 1.0f), e3 = fabs(mg.y - rs_w) / (rs_w + 1e-3);
            for (int d = 0; d < D; ++d) e1 = fmax(e1, fabs(part[((size_t)w * S + q) * D + d] - ow[d]));
            printf("q=%d w=%d: max gpu=%.4f cpu=%.4f (rel %.2e) rowsum gpu=%.4f cpu=%.4f (rel %.2e) Oerr=%.3e\n",
                   q, w, mg.x, m_w, e2, mg.y, rs_w, e3, e1);
        }
    }
    return 0;
}

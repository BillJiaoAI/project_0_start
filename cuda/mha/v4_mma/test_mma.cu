// 最小 mma 原语验证: A@B (m16n8k16) + ldmatrix.trans + movmatrix 的布局测试
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#define WARP_SIZE 32

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
static __device__ __forceinline__ int movmatrix(int x) {
    int ret; asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(ret) : "r"(x)); return ret;
}
// (get_half2 + get_transposed 的 P->P^T 链条见 mha_mma.cu, 由主 kernel 端到端验证)

// ---------------- 测试 1: mma_f32 D=A@B, C fragment 重建 16x8 float ----------------
// 测试 2: ldmatrix.trans 重建 V^T (16x8)
// 测试 3: movmatrix 重建 8x8 转置
__global__ void test(const half* A, const half* Bm, const half* V, const half* M8,
                     float* outC, half* outVt, half* outM8t) {
    __shared__ half2 sA[16 * 8];   // A: 16 行 x 8 half2 (16 f16 列)
    __shared__ half2 sB[8 * 8];    // B: 8 行(query) x 8 half2 (16 f16 列)  (B^T 视角: [n][k])
    __shared__ half2 sV[16 * 8];   // V: 16 行(KV) x 8 half2 (16 dv)
    __shared__ half2 sM[8 * 4];    // M8: 8x8 f16 = 每行 4 half2 (movmatrix 的 8x8 b16 tile)
    const int lane = threadIdx.x;
    for (int i = lane; i < 16 * 8; i += 32) sA[i] = ((const half2*)A)[i];
    for (int i = lane; i < 8 * 8; i += 32) sB[i] = ((const half2*)Bm)[i];
    for (int i = lane; i < 16 * 8; i += 32) sV[i] = ((const half2*)V)[i];
    for (int i = lane; i < 8 * 4; i += 32) sM[i] = ((const half2*)M8)[i];
    __syncthreads();

    // T1: C = A @ B  (A: 16x16 f16, B: 16x8 f16 = Bm 的转置视角: Bm 是 [n=8][k=16])
    FA16 A_f; load_ldmatrix_a(A_f, sA, 8);
    FB88 B_f; load_ldmatrix_b(B_f, sB, 8);
    FC168f C = {};
    mma_f32(C, A_f, B_f);
    // C f32 fragment: x[l] -> (row (l/2)*8 + lane/4, col 2*(lane%4) + l%2)
    for (int l = 0; l < 4; ++l)
        outC[((l / 2) * 8 + lane / 4) * 8 + 2 * (lane % 4) + l % 2] = C.x[l];

    // T2: A2 = V^T fragment (V smem [KV][dv]) -> 重建 16x8: row=dv, col=KV
    FA16 A2; load_ldmatrix_atrans(A2, sV, 8);
    for (int l = 0; l < 4; ++l) {
        const int r = (l % 2) * 8 + lane / 4;         // A fragment get_i (half2 tile<16,8>)
        const int c = (l / 2) * 8 + (lane % 4) * 2;   // get_j (half2 单位 -> f16 列)
        outVt[r * 16 + c]     = __low2half(A2.x[l]);
        outVt[r * 16 + c + 1] = __high2half(A2.x[l]);
    }

    // T3: movmatrix: M8 (8x8, thread T 持 row T/4, f16 列 2*(T%4)+{0,1}) -> 转置后同分布
    half2 m = sM[(lane / 4) * 4 + lane % 4];
    half2 mt; *((int*)&mt) = movmatrix(*((int*)&m));
    outM8t[(lane / 4) * 8 + 2 * (lane % 4)]     = __low2half(mt);
    outM8t[(lane / 4) * 8 + 2 * (lane % 4) + 1] = __high2half(mt);
}

int main() {
    // A: 16x16, B^T: 8x16 (Bm[n][k]), C = A@B 16x8
    std::vector<half> Ah(16 * 16), Bth(8 * 16), Vh(16 * 16), M8h(8 * 8);
    std::vector<float> C_ref(16 * 8, 0.0f);
    srand(7);
    auto uh = []() { return __float2half((rand() / (float)RAND_MAX) * 2 - 1); };
    for (auto& x : Ah) x = uh();
    for (auto& x : Bth) x = uh();
    for (auto& x : Vh) x = uh();
    for (auto& x : M8h) x = uh();
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 16; ++k)
                C_ref[i * 8 + j] += __half2float(Ah[i * 16 + k]) * __half2float(Bth[j * 16 + k]);

    half *dA, *dB, *dV, *dM, *oVt, *oM8t; float* oC;
    cudaMalloc(&dA, Ah.size() * 2); cudaMalloc(&dB, Bth.size() * 2);
    cudaMalloc(&dV, Vh.size() * 2); cudaMalloc(&dM, M8h.size() * 2);
    cudaMalloc(&oC, 16 * 8 * 4); cudaMalloc(&oVt, 16 * 16 * 2); cudaMalloc(&oM8t, 8 * 8 * 2);
    cudaMemcpy(dA, Ah.data(), Ah.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bth.data(), Bth.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, Vh.data(), Vh.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, M8h.data(), M8h.size() * 2, cudaMemcpyHostToDevice);
    test<<<1, 32>>>(dA, dB, dV, dM, oC, oVt, oM8t);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("kernel: %s\n", cudaGetErrorString(err)); return 1; }

    std::vector<float> Cg(16 * 8); std::vector<half> Vtg(16 * 16), M8tg(8 * 8);
    cudaMemcpy(Cg.data(), oC, 16 * 8 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(Vtg.data(), oVt, 16 * 16 * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(M8tg.data(), oM8t, 8 * 8 * 2, cudaMemcpyDeviceToHost);

    double e1 = 0; for (int i = 0; i < 16 * 8; ++i) e1 = fmax(e1, fabs(Cg[i] - C_ref[i]));
    double e2 = 0; for (int i = 0; i < 16 * 16; ++i)
        e2 = fmax(e2, fabs(__half2float(Vtg[i]) - __half2float(Vh[(i % 16) * 16 + i / 16]))); // Vt[i][j]=V[j][i]
    double e3 = 0; for (int i = 0; i < 8 * 8; ++i)
        e3 = fmax(e3, fabs(__half2float(M8tg[i]) - __half2float(M8h[(i % 8) * 8 + i / 8])));
    printf("T1 mma_f32 A@B : %.3e %s\nT2 ldmatrix.trans: %.3e %s\nT3 movmatrix   : %.3e %s\n",
           e1, e1 < 1e-3 ? "OK" : "BAD", e2, e2 < 1e-6 ? "OK" : "BAD", e3, e3 < 1e-6 ? "OK" : "BAD");
    return (e1 < 1e-3 && e2 < 1e-6 && e3 < 1e-6) ? 0 : 1;
}

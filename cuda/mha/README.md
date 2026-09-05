# MHA 的 5 种实现：从数学到 llama.cpp 的 Tensor Core

从 llama.cpp 的 `GGML_OP_FLASH_ATTN_EXT` 三条 CUDA 路径（vec / tile / mma）里，
逐条提取出**最小可编译运行**的独立版本，加上两个基线（CPU 参考 / 朴素 GPU），
按学习顺序排列。每个版本单文件自带 CPU 对拍，跑完打印 PASS/FAIL。

## 编译运行

```bat
build_all.bat          # 编译全部
build_all.bat run      # 编译并依次运行（应全打 PASS）
```

要求：nvcc + MSVC（脚本会尝试用 vswhere 自动加载 vcvars64）。
单独跑某个版本：`v4_mma\mha_mma.exe [S] [H]`（S 默认 512，v3 要求 S%64==0，v4 要求 S%128==0，
对应 llama.cpp 靠 KV cache padding 保证）。

## 统一约定（所有版本一致，方便对拍）

| 张量 | 布局 | 类型 | 说明 |
|------|------|------|------|
| Q | [H][NQ][D] | f32 | llama.cpp 中 Q 是 f32 |
| K / V | [H][NKV][D] | f16 | KV cache 是 f16，全系列只做 f16 路径 |
| mask | [NQ][NKV] | f16 | causal：j<=i 为 0，j>i 为 -INF |
| O | [H][NQ][D] | f32 | 输出 |

D=128 固定，scale = 1/sqrt(D)，随机数据 seed=42，CPU 参考逐元素对拍。

## 学习路线

| 版本 | 文件 | 核心思想 | llama.cpp 对应 |
|------|------|----------|----------------|
| v0_ref | `v0_ref/mha_ref.cu` | 数学定义本身 + online softmax 伪码引子（CPU） | `ggml.c` ggml_compute_forward_flash_attn_ext_f16 |
| v1_naive | `v1_naive/mha_naive.cu` | 每 block 一行 Q：S 整行 materialize 进 smem，block 级 softmax（无 flash，教学基线） | 无（FA 出现前的标准写法） |
| v2_vec | `v2_vec/mha_vec.cu` | 每线程拿一个 key，**寄存器走天下**；online softmax 每 warp 一份；parallel blocks 切 KV 维再合并 | `fattn-vec.cuh`（ fatsn 路径 1） |
| v3_tile | `v3_tile/mha_tile.cu` | KV 按 64 行一块搬进 smem，QK^T 在寄存器分块累加；running max 全 block 一份，概率写回 smem 的 KQ 矩阵再乘 V | `fattn-tile.cuh`（路径 2，prefill/长序列兜底） |
| v4_mma | `v4_mma/mha_mma.cu` | Tensor Core `mma.sync.m16n8k16` 同时做 QK^T 和 PV；数据按 fragment 布局散布在 warp 里；f16 累加 O | `fattn-mma-f16.cuh`（路径 3，性能最高） |

实测（RTX 5070 Ti, S=512, H=8, D=128, causal）：

| 版本 | 耗时 | 吞吐 | 精度 (max_abs_err) |
|------|------|------|--------------------|
| v1_naive | 2.27 ms | 0.24 TFLOPS | 2e-7 |
| v2_vec | 0.247 ms | 2.17 TFLOPS | 3.4e-4 |
| v3_tile | 0.262 ms | 2.05 TFLOPS | 1.1e-3 |
| v4_mma | 0.141 ms | 3.9 TFLOPS | 4.2e-4 |

v2/v3/v4 的误差底噪来自同一组取舍：Q 预缩放转 half（scale 乘在 Q 上）+ P 存 half +
VKQ 用 f16 累加 —— llama.cpp 原版有意为之（寄存器/带宽换精度）。

## v4_mma 的 fragment 布局速查

`mma.sync.aligned.m16n8k16.row.col`（A 16x16 f16 行主序 @ B 16x8 f16 列主序 = C 16x8），
lane = threadIdx.x，每线程持有元素逻辑坐标：

```
A (4 half2/lane):  行=(l%2)*8 + lane/4,  列=(l/2)*4 + lane%4      [half2 单位]
B (2 half2/lane):  行=lane/4 (N=query),   列=l*4 + lane%4 (K=dim)
C_f32 (4 float):   行=(l/2)*8 + lane/4,   列=(lane%4)*2 + l%2
C_f16 (2 half2):   行=l*8 + lane/4,       列=lane%4 (一个 half2 = 2 个 query)
```

v4 的矩阵角色：KQ mma 算 **S^T**（A=K 的 16 行，B=Q 的 8 列，C=16 KV x 8 query）；
VKQ mma 算 **O^T**（A=V^T 的 16 dv，B=P^T 的 16 KV，C=16 dv x 8 query）。
一个 query 列的 32 个 KV 值散布在同 `lane%4` 的 8 个线程里 —— KQ_max 用
`shfl_xor 16/8/4` 归约，KQ_rowsum 在循环结束后用同样的归约合并。
P（C_f16 布局）到 P^T（B 布局）靠 `get_half2` + `movmatrix`（8x8 寄存器内转置）。

固定取 llama.cpp 的 Ampere config `(DKQ,DV,ncols)=(128,128,8)`：128 线程 = 4 warp，
每 block 8 个 query，每轮 nbatch_fa=128 个 KV 行，4 个 warp 各分摊 32 行、各自维护
online softmax，结尾各写一份 partial（O 部分 + (max,rowsum)），由 combine kernel 按
`exp(max_w - max)` 加权合并 —— 即 warp 级的 parallel blocks（v2 是 block 级）。

每个文件头部有更完整的"与原版差异"和"核心代码对照（llama.cpp 行号）"注释，建议对照阅读。

## 调试工具

- `v4_mma/test_mma.cu`：mma/ldmatrix(.trans)/movmatrix 原语的最小布局自测。
  写 fragment 代码前先跑通它，能省掉大量"坐标映射猜谜"时间。
- `v4_mma/mha_dbg.cu`：v4 的 warp 粒度调试版（dump S^T、逐 warp 对拍 partial+meta）。
  `mha_dbg.exe [S] [H] [stage]`，stage 1=partial/meta 对拍，2=dump S^T，3=meta 诊断。

## v4 调试实录（三个 bug，都有教学价值）

1. **KQ_rowsum 漏了跨 lane 归约**（真 kernel bug）：每个 query 列的 exp 和散布在 8 个
   同 `lane%4` 线程里，逐线程带 scale 累加后必须 `shfl_xor 16/8/4` 归约一次（对应
   fattn-mma-f16.cuh L1321-47）。漏掉 → combine 分母只有真值的 1/4 左右 → 输出整体偏大。
   注意它能在循环外归约的前提：每轮的 KQ_max_scale 对同组线程是同一个值。
2. **combine 的越界保护写成 `idx >= NQ*D`**：只是单个 head 的元素数，H=1 时恰好相等
   侥幸通过，H>=2 时 head>=1 的输出整段没被写（新分配页恰好是 0）。多 head 一跑就炸，
   单 head 测试永远测不出来 —— 测试维度要覆盖所有并行维度。
3. **调试工具自己的 bug 造成的假象**：`cudaMemcpy(meta.data(), dMeta, meta.size()*4)`
   —— float2 是 8 字节，只拷了前一半，于是"warp 2/3 的 meta 全是 (0,0)"追了一晚上，
   其实 kernel 写得完全正确。调试工具本身也要用 compute-sanitizer / 最小化验证。

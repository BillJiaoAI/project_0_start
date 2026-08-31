# MLIR 从零到一学习指南

## 1. MLIR 体系概览

### 什么是 MLIR?

MLIR (Multi-Level Intermediate Representation) 是 LLVM 项目的一部分，提供了一个可扩展的、多层次的中间表示框架。

**核心设计理念:**
- **分层 Dialect**: 不同抽象层次有不同的语言（Dialect）
- **可组合 Pass**: 独立的优化转换
- **统一基础设施**: 共享的类型系统、属性系统、变换框架

### 为什么需要多层 IR?

| 层次 | 抽象级别 | 用途 | 典型操作 |
|------|----------|------|----------|
| **高层** | 数学/算法 | 算法表达、自动微分 | `linalg.generic`, `tensor` |
| **中层** | 循环/控制流 | 循环优化、并行化 | `affine.for`, `scf.for` |
| **低层** | 内存/SIMD | 内存优化、向量化 | `memref`, `vector` |
| **硬件层** | 特定架构 | GPU/CPU 代码生成 | `gpu`, `nvvm`, `arm` |

---

## 2. MLIR 基础概念

### 2.1 Operation (操作)

MLIR 的基本构建块，类似于 LLVM 的 Instruction。

```mlir
%result = arith.addf %arg0, %arg1 : f32
```

**组成部分:**
- **名称**: `arith.addf`（Dialect名.操作名）
- **操作数**: `%arg0`, `%arg1`
- **结果**: `%result`
- **类型**: `f32`
- **属性**: 可选的元数据

### 2.2 Attribute (属性)

编译时已知的常量值，附加在 Operation 上。

```mlir
%c0 = arith.constant 0 : i32           // 整数属性
%float = arith.constant 3.14 : f64      // 浮点属性
%bool = arith.constant true             // 布尔属性
```

### 2.3 Type (类型)

MLIR 有丰富的类型系统：

| 类型 | 示例 | 说明 |
|------|------|------|
| **标量类型** | `i32`, `f32`, `i1` | 基本整数/浮点/布尔 |
| **MemRef** | `memref<4x8xf32>` | 内存引用（多维数组） |
| **Tensor** | `tensor<16x16xf32>` | 不可变张量 |
| **Vector** | `vector<4xf32>` | SIMD 向量 |
| **Function** | `(memref<10xf32>) -> ()` | 函数类型 |

### 2.4 Pass (转换)

对 IR 进行优化或转换的独立单元。

**常见 Pass:**
- `--canonicalize`: 规范化 IR
- `--cse`: 公共子表达式消除
- `--affine-loop-tile`: 循环分块
- `--lower-affine`: Affine → SCF 转换

### 2.5 Module (模块)

MLIR 的顶级容器，包含函数、全局变量等。

```mlir
module {
  func.func @main() -> () {
    return
  }
}
```

---

## 3. MemRef Dialect（核心重点）

### 3.1 什么是 MemRef?

MemRef 是 MLIR 中表示**内存引用**的类型，用于建模多维数组。

```mlir
memref<4x8xf32>           // 4行8列的float32矩阵
memref<1024xf32>          // 1024元素的float32向量
memref<?x?xf32>           // 动态形状的矩阵
```

### 3.2 Layout 布局

MemRef 支持灵活的内存布局描述：

**行优先 (Row-major):**
```mlir
memref<4x8xf32, strided<[8, 1], offset: 0>>
// strides[0] = 8: 每行8个元素
// strides[1] = 1: 每列1个元素
```

**列优先 (Column-major):**
```mlir
memref<4x8xf32, strided<[1, 4], offset: 0>>
// strides[0] = 1: 每行1个元素
// strides[1] = 4: 每列4个元素
```

### 3.3 基本操作

```mlir
%A = memref.alloc() : memref<4x8xf32>    // 分配内存
memref.store %val, %A[%i, %j]            // 存储
%elem = memref.load %A[%i, %j]           // 加载
memref.dealloc %A                        // 释放内存
```

### 3.4 Subview（分块视图）

创建原 MemRef 的子视图，实现分块：

```mlir
// 从16x16矩阵中提取左上角8x8子块
%tile = memref.subview %A[0, 0] [8, 8] [1, 1] 
  : memref<16x16xf32> to memref<8x8xf32, strided<[16, 1], offset: 0>>
```

**参数说明:**
- `[0, 0]`: 起始偏移
- `[8, 8]`: 子块大小
- `[1, 1]`: 步长
- 返回的 memref 保留原始 stride 信息

---

## 4. Affine Dialect

### 4.1 Affine 循环

Affine 循环是 MLIR 中表达规则嵌套循环的方式：

```mlir
affine.for %i = 0 to 10 {
  affine.for %j = 0 to 10 {
    // 循环体
  }
}
```

**特点:**
- 边界和步长必须是仿射表达式（线性表达式）
- 支持自动依赖分析
- 便于进行循环优化（tiling, fusion, unrolling）

### 4.2 Affine Map

仿射映射，用于计算循环索引：

```mlir
%result = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)> (%i, %j)
```

### 4.3 循环 Tiling

将大循环分解为小的 tile，提升缓存局部性：

```mlir
// 原始循环
affine.for %i = 0 to 64 {
  affine.for %j = 0 to 64 {
    // computation
  }
}

// Tiled (16x16)
affine.for %i = 0 to 64 step 16 {
  affine.for %j = 0 to 64 step 16 {
    affine.for %ii = %i to min(%i + 16, 64) {
      affine.for %jj = %j to min(%j + 16, 64) {
        // computation on tile
      }
    }
  }
}
```

### 4.4 Pass 优化

```bash
mlir-opt input.mlir \
  --affine-loop-tile="tile-sizes=16,16" \
  --affine-loop-unroll="unroll-factor=4" \
  --affine-loop-fusion
```

---

## 5. Vector Dialect

### 5.1 向量类型

```mlir
vector<4xf32>          // 4个float32
vector<4x4xf32>        // 4x4矩阵向量
vector<8xi32>          // 8个int32
```

### 5.2 向量操作

```mlir
// 从内存加载向量
%vec = vector.transfer_read %A[%i], %c0 
  : memref<256xf32>, vector<4xf32>

// 向量运算
%result = arith.addf %vec_a, %vec_b : vector<4xf32>
%result = arith.mulf %vec_a, %vec_b : vector<4xf32>

// 存储向量到内存
vector.transfer_write %result, %C[%i] 
  : vector<4xf32>, memref<256xf32>
```

### 5.3 向量收缩 (Contraction)

用于矩阵乘法等操作：

```mlir
%result = vector.contract %a_tile, %b_tile, %acc 
  : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
```

---

## 6. GPU Dialect

### 6.1 GPU 模块和函数

```mlir
module attributes {gpu.container_module} {
  gpu.module @kernel_module {
    gpu.func @my_kernel(%A: memref<1024xf32>) kernel {
      // kernel body
      gpu.return
    }
  }
}
```

### 6.2 线程层次

```mlir
%block_id_x = gpu.block_id x      // block索引
%block_id_y = gpu.block_id y
%thread_id_x = gpu.thread_id x    // thread索引
%thread_id_y = gpu.thread_id y
%block_dim_x = gpu.block_dim x    // block大小
%grid_dim_x = gpu.grid_dim x      // grid大小
```

### 6.3 共享内存

```mlir
%shared_buf = memref.alloc() {gpu.shared_memory} : memref<16x16xf32>
// 使用共享内存
gpu.barrier  // 线程同步
memref.dealloc %shared_buf
```

### 6.4 启动 GPU Kernel

```mlir
gpu.launch_func @kernel_module::@my_kernel
  blocks in (%num_blocks_x, %num_blocks_y, %c1)
  threads in (%threads_per_block_x, %threads_per_block_y, %c1)
  args(%A, %B, %C)
```

---

## 7. 完整降级链路

以 GEMM 为例，展示完整的 MLIR 降级流程：

```
Level 1: High-level (Linalg)
┌─────────────────────────────────────────────────────┐
│ linalg.matmul(%A, %B, %C)                           │
│ // 数学级别的矩阵乘法表达                            │
└─────────────────────────────────────────────────────┘
                         │
                         ▼ --linalg-expand
Level 2: Affine
┌─────────────────────────────────────────────────────┐
│ affine.for %i = 0 to M                             │
│   affine.for %j = 0 to N                           │
│     affine.for %k = 0 to K                         │
│       // scalar computation                         │
└─────────────────────────────────────────────────────┘
                         │
                         ▼ --affine-loop-tile
Level 3: Tiled Affine
┌─────────────────────────────────────────────────────┐
│ affine.for %i = 0 to M step 16                     │
│   affine.for %j = 0 to N step 16                   │
│     affine.for %k = 0 to K step 16                 │
│       // tile computation                          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼ --lower-affine
Level 4: SCF
┌─────────────────────────────────────────────────────┐
│ scf.for %i = %c0 to %M step %c16                   │
│   scf.for %j = %c0 to %N step %c16                 │
│     // scf-based loops                             │
└─────────────────────────────────────────────────────┘
                         │
                         ▼ --vectorize
Level 5: Vector
┌─────────────────────────────────────────────────────┐
│ %a_tile = vector.transfer_read %A[...]             │
│ %b_tile = vector.transfer_read %B[...]             │
│ %result = arith.mulf %a_tile, %b_tile              │
│ vector.transfer_write %result, %C[...]             │
└─────────────────────────────────────────────────────┘
                         │
                         ▼ --lower-to-gpu
Level 6: GPU
┌─────────────────────────────────────────────────────┐
│ gpu.func @gemm_kernel(...) kernel                  │
│   %shared_A = memref.alloc() {gpu.shared_memory}   │
│   // thread/block mapping                          │
│   gpu.barrier                                      │
└─────────────────────────────────────────────────────┘
                         │
                         ▼ --lower-to-nvvm
Level 7: NVVM/PTX
┌─────────────────────────────────────────────────────┐
│ .reg .f32 %f<4>;                                   │
│ ld.global.f32 %f0, [%A];                           │
│ st.shared.f32 [%shared_A], %f0;                    │
│ // PTX instructions                                │
└─────────────────────────────────────────────────────┘
```

---

## 8. 实践路线图

### 阶段 1: MLIR 基础

1. 学习 Operation、Attribute、Type 概念
2. 理解 Module 和 Function 结构
3. 运行基础 .mlir 文件

### 阶段 2: MemRef 核心

1. 理解 MemRef 类型和布局
2. 掌握 memref.alloc/load/store/dealloc
3. 理解 subview 的分块机制

### 阶段 3: Affine 优化

1. 学习 affine.for 和 affine.map
2. 理解循环 tiling、unrolling、fusion
3. 使用 mlir-opt 运行 Pass

### 阶段 4: GPU 编程

1. 理解 GPU Dialect 的线程层次
2. 掌握共享内存使用
3. 编写完整的 GPU kernel

### 阶段 5: 完整 GEMM 实现

1. 手写 Affine GEMM
2. 应用 tiling 和向量化
3. 降级到 GPU Dialect
4. 生成 PTX 并在 GPU 上运行

---

## 9. 常用命令

```bash
# 查看 IR
mlir-opt input.mlir

# 应用 Pass
mlir-opt input.mlir --canonicalize --cse -o output.mlir

# 逐级打印 IR
mlir-opt input.mlir --print-ir-after-all

# 生成 LLVM IR
mlir-translate --mlir-to-llvmir input.mlir -o output.ll

# 生成 PTX（需要 GPU Dialect）
mlir-translate --mlir-to-nvvmir input.mlir -o output.ptx

# 运行 CPU 代码
mlir-cpu-runner input.mlir -e main

# 运行 GPU 代码
mlir-gpu-runner input.mlir -e main
```

---

## 10. 目录结构说明

```
MLIR/
├── 01_basics/           # MLIR 基础概念
│   ├── 01_hello_world.mlir
│   ├── 02_operation_structure.mlir
│   ├── 03_attributes_types.mlir
│   ├── 04_pass_intro.mlir
│   └── 05_control_flow.mlir
├── 02_memref/           # MemRef Dialect
│   ├── 01_memref_basic.mlir
│   ├── 02_memref_layout.mlir
│   ├── 03_memref_subview.mlir
│   └── 04_memref_dynamic_shape.mlir
├── 03_affine/           # Affine Dialect
│   ├── 01_affine_loop.mlir
│   ├── 02_affine_tiling.mlir
│   └── 03_affine_map.mlir
├── 04_gpu/              # GPU Dialect
│   ├── 01_gpu_kernel.mlir
│   └── 02_gpu_shared_memory.mlir
├── 05_gemm_full/        # 完整 GEMM 实现
│   ├── 01_gemm_affine.mlir
│   ├── 02_gemm_tiled.mlir
│   └── 03_gemm_gpu.mlir
├── 06_vector/           # Vector Dialect
│   ├── 01_vector_add.mlir
│   └── 02_vector_matmul.mlir
├── 07_lowering/         # 降级流程演示
│   ├── 01_lowering_pipeline.sh
│   └── 02_lowering_demo.py
└── 08_run_scripts/      # 运行脚本
    ├── run_basics.bat
    ├── run_memref.bat
    ├── run_affine.bat
    ├── run_gpu.bat
    └── run_gemm.bat
```

---

## 11. 关键概念对比

| 概念 | MLIR | CUDA | 说明 |
|------|------|------|------|
| **内存抽象** | `memref<MxNxf32>` | `float *` + pitch | MemRef 包含形状和布局信息 |
| **循环** | `affine.for` | `for` | Affine 支持自动依赖分析 |
| **线程** | `gpu.thread_id` | `threadIdx` | 语义相同 |
| **共享内存** | `{gpu.shared_memory}` | `__shared__` | 属性标记 vs 关键字 |
| **同步** | `gpu.barrier` | `__syncthreads()` | 语义相同 |
| **Tile** | `memref.subview` | 手动指针计算 | MLIR 自动管理 stride |

---

## 12. 学习建议

1. **从基础开始**: 先理解 Operation、Type、Attribute 的概念
2. **动手实践**: 手写 .mlir 文件，用 mlir-opt 查看效果
3. **逐级深入**: 从 Affine → SCF → Vector → GPU 逐步学习
4. **关注降级**: 理解每一步 Pass 如何转换 IR
5. **联系 CUDA**: 将 MLIR 概念与 CUDA 概念对应起来理解

祝你学习顺利！🚀

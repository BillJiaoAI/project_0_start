import re

class MLIRCode:
    def __init__(self, code):
        self.code = code
    
    def __str__(self):
        return self.code
    
    def apply_pass(self, pass_name, description):
        print(f"\n{'='*60}")
        print(f"Pass: {pass_name}")
        print(f"Description: {description}")
        print(f"{'='*60}")
        print(self.code)
        print(f"\n{'='*60}")
        return self

def gemm_affine():
    return MLIRCode("""module {
  func.func @gemm(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) -> () {
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        affine.for %k = 0 to 64 {
          %a_val = memref.load %A[%i, %k] : memref<64x64xf32>
          %b_val = memref.load %B[%k, %j] : memref<64x64xf32>
          %c_val = memref.load %C[%i, %j] : memref<64x64xf32>
          %mul = arith.mulf %a_val, %b_val : f32
          %add = arith.addf %c_val, %mul : f32
          memref.store %add, %C[%i, %j] : memref<64x64xf32>
        }
      }
    }
    return
  }
}""")

def gemm_tiled():
    return MLIRCode("""module {
  func.func @gemm(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) -> () {
    affine.for %i = 0 to 64 step 16 {
      affine.for %j = 0 to 64 step 16 {
        affine.for %k = 0 to 64 step 16 {
          affine.for %ii = %i to min(%i + 16, 64) {
            affine.for %jj = %j to min(%j + 16, 64) {
              affine.for %kk = %k to min(%k + 16, 64) {
                %a_val = memref.load %A[%ii, %kk] : memref<64x64xf32>
                %b_val = memref.load %B[%kk, %jj] : memref<64x64xf32>
                %c_val = memref.load %C[%ii, %jj] : memref<64x64xf32>
                %mul = arith.mulf %a_val, %b_val : f32
                %add = arith.addf %c_val, %mul : f32
                memref.store %add, %C[%ii, %jj] : memref<64x64xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}""")

def gemm_scf():
    return MLIRCode("""module {
  func.func @gemm(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) -> () {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    scf.for %i = %c0 to %c64 step %c16 {
      scf.for %j = %c0 to %c64 step %c16 {
        scf.for %k = %c0 to %c64 step %c16 {
          scf.for %ii = %i to %c64 step %c1 {
            scf.for %jj = %j to %c64 step %c1 {
              scf.for %kk = %k to %c64 step %c1 {
                %a_val = memref.load %A[%ii, %kk] : memref<64x64xf32>
                %b_val = memref.load %B[%kk, %jj] : memref<64x64xf32>
                %c_val = memref.load %C[%ii, %jj] : memref<64x64xf32>
                %mul = arith.mulf %a_val, %b_val : f32
                %add = arith.addf %c_val, %mul : f32
                memref.store %add, %C[%ii, %jj] : memref<64x64xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}""")

def gemm_vectorized():
    return MLIRCode("""module {
  func.func @gemm(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) -> () {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    
    scf.for %i = %c0 to %c64 step %c4 {
      scf.for %j = %c0 to %c64 step %c4 {
        %acc = arith.constant 0.0 : vector<4x4xf32>
        %result = scf.for %k = %c0 to %c64 step %c4 iter_args(%acc = %acc) -> (vector<4x4xf32>) {
          %a_tile = vector.transfer_read %A[%i, %k], %c0 {in_bounds = [true, true]} : memref<64x64xf32>, vector<4x4xf32>
          %b_tile = vector.transfer_read %B[%k, %j], %c0 {in_bounds = [true, true]} : memref<64x64xf32>, vector<4x4xf32>
          %mul = arith.mulf %a_tile, %b_tile : vector<4x4xf32>
          %add = arith.addf %acc, %mul : vector<4x4xf32>
          scf.yield %add : vector<4x4xf32>
        }
        vector.transfer_write %result, %C[%i, %j] {in_bounds = [true, true]} : vector<4x4xf32>, memref<64x64xf32>
      }
    }
    return
  }
}""")

def gemm_gpu():
    return MLIRCode("""module attributes {gpu.container_module} {
  gpu.module @gemm_module {
    gpu.func @gemm_kernel(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) kernel {
      %shared_A = memref.alloc() {gpu.shared_memory} : memref<16x16xf32>
      %shared_B = memref.alloc() {gpu.shared_memory} : memref<16x16xf32>
      
      %block_i = gpu.block_id x
      %block_j = gpu.block_id y
      %thread_i = gpu.thread_id x
      %thread_j = gpu.thread_id y
      
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      
      %global_i = arith.addi %thread_i, (arith.muli %block_i, %c16) : index
      %global_j = arith.addi %thread_j, (arith.muli %block_j, %c16) : index
      
      %c0_f = arith.constant 0.0 : f32
      %acc = scf.for %k = %c0 to %c64 step %c16 iter_args(%acc = %c0_f) -> (f32) {
        %a_val = memref.load %A[%global_i, (arith.addi %k, %thread_j)] : memref<64x64xf32>
        %b_val = memref.load %B[(arith.addi %k, %thread_i), %global_j] : memref<64x64xf32>
        
        memref.store %a_val, %shared_A[%thread_i, %thread_j] : memref<16x16xf32>
        memref.store %b_val, %shared_B[%thread_i, %thread_j] : memref<16x16xf32>
        
        gpu.barrier
        
        %inner_acc = scf.for %kk = %c0 to %c16 step %c1 iter_args(%inner_acc = %acc) -> (f32) {
          %a_tile = memref.load %shared_A[%thread_i, %kk] : memref<16x16xf32>
          %b_tile = memref.load %shared_B[%kk, %thread_j] : memref<16x16xf32>
          %mul = arith.mulf %a_tile, %b_tile : f32
          %add = arith.addf %inner_acc, %mul : f32
          scf.yield %add : f32
        }
        
        gpu.barrier
        scf.yield %inner_acc : f32
      }
      
      memref.store %acc, %C[%global_i, %global_j] : memref<64x64xf32>
      
      memref.dealloc %shared_A : memref<16x16xf32>
      memref.dealloc %shared_B : memref<16x16xf32>
      gpu.return
    }
  }
}""")

def main():
    print("="*60)
    print("MLIR GEMM Lowering Pipeline - Step by Step")
    print("="*60)
    
    print("\n" + "="*60)
    print("Level 1: High-Level Affine GEMM")
    print("="*60)
    print("Description: Pure affine loops, no tiling, no vectorization")
    print("Characteristics: Simple but inefficient - no memory locality")
    gemm_affine().apply_pass("affine-loop-tile", "Tile loops with 16x16x16")
    
    print("\n" + "="*60)
    print("Level 2: Tiled Affine GEMM")
    print("="*60)
    print("Description: Outer loops tiled by 16, inner loops handle tile elements")
    print("Characteristics: Improved memory locality, ready for shared memory")
    gemm_tiled().apply_pass("lower-affine", "Convert affine loops to SCF")
    
    print("\n" + "="*60)
    print("Level 3: SCF GEMM")
    print("="*60)
    print("Description: Affine constructs converted to Standard Control Flow")
    print("Characteristics: More flexible control flow, ready for vectorization")
    gemm_scf().apply_pass("vectorize", "Vectorize inner loops with SIMD")
    
    print("\n" + "="*60)
    print("Level 4: Vectorized GEMM")
    print("="*60)
    print("Description: Inner computations use vector operations (SIMD)")
    print("Characteristics: Exploits data-level parallelism, uses vector registers")
    gemm_vectorized().apply_pass("lower-to-gpu", "Lower to GPU dialect")
    
    print("\n" + "="*60)
    print("Level 5: GPU GEMM")
    print("="*60)
    print("Description: GPU kernel with shared memory tiling")
    print("Characteristics: Maps to GPU hardware, uses shared memory for tiles")
    gemm_gpu().apply_pass("lower-to-nvvm", "Lower to NVVM/PTX")
    
    print("\n" + "="*60)
    print("Summary of Lowering Pipeline")
    print("="*60)
    print("""
1. Affine Dialect → High-level mathematical representation
   - affine.for, affine.apply, affine.map
   - Automatic dependency analysis, loop optimization

2. SCF Dialect → Control flow representation
   - scf.for, scf.if, scf.yield
   - More flexible, supports arbitrary control flow

3. Vector Dialect → SIMD computation
   - vector.transfer_read/write, vector.contract
   - Data-level parallelism, register-level optimizations

4. GPU Dialect → GPU-specific operations
   - gpu.func, gpu.block_id, gpu.thread_id, gpu.barrier
   - Grid/block/thread mapping, shared memory

5. NVVM Dialect → NVIDIA PTX generation
   - Direct mapping to PTX instructions
   - Final code generation

Key Concepts:
- Dialect: Domain-specific language within MLIR
- Pass: Transformation applied to IR
- Lowering: Transforming from high-level to low-level dialects
- MemRef: Memory reference type for multi-dimensional arrays
- Tile: Subset of data loaded into shared memory
""")

if __name__ == "__main__":
    main()

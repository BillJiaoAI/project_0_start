module attributes {gpu.container_module} {
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
        memref.store %c0_f, %shared_A[%thread_i, %thread_j] : memref<16x16xf32>
        memref.store %c0_f, %shared_B[%thread_i, %thread_j] : memref<16x16xf32>
        
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
  
  func.func @main() -> () {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    
    %A = memref.alloc() : memref<64x64xf32>
    %B = memref.alloc() : memref<64x64xf32>
    %C = memref.alloc() : memref<64x64xf32>
    
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        %ival = arith.index_cast %i : index to f32
        %jval = arith.index_cast %j : index to f32
        %a_val = arith.addf %ival, %jval : f32
        %b_val = arith.subf %ival, %jval : f32
        %c_val = arith.constant 0.0 : f32
        memref.store %a_val, %A[%i, %j] : memref<64x64xf32>
        memref.store %b_val, %B[%i, %j] : memref<64x64xf32>
        memref.store %c_val, %C[%i, %j] : memref<64x64xf32>
      }
    }
    
    %stream = gpu.create_stream
    gpu.launch_func @gemm_module::@gemm_kernel
      blocks in (%c4, %c4, %c1) threads in (%c16, %c16, %c1)
      args(%A, %B, %C) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>
    gpu.wait %stream
    gpu.destroy_stream %stream
    
    %result = memref.load %C[%c0, %c0] : memref<64x64xf32>
    vector.print %result : f32
    
    memref.dealloc %A : memref<64x64xf32>
    memref.dealloc %B : memref<64x64xf32>
    memref.dealloc %C : memref<64x64xf32>
    return
  }
}

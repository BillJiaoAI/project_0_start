module attributes {gpu.container_module} {
  gpu.module @kernel_module {
    gpu.func @shared_mem_kernel(%A: memref<256xf32>, %B: memref<256xf32>, %C: memref<256xf32>) kernel {
      %shared_A = memref.alloc() {gpu.shared_memory} : memref<32xf32>
      %shared_B = memref.alloc() {gpu.shared_memory} : memref<32xf32>
      
      %thread_idx = gpu.thread_id x
      %block_idx = gpu.block_id x
      %block_dim = gpu.block_dim x
      
      %linear_idx = arith.addi %thread_idx, (arith.muli %block_idx, %block_dim) : index
      
      %a_val = memref.load %A[%linear_idx] : memref<256xf32>
      %b_val = memref.load %B[%linear_idx] : memref<256xf32>
      
      memref.store %a_val, %shared_A[%thread_idx] : memref<32xf32>
      memref.store %b_val, %shared_B[%thread_idx] : memref<32xf32>
      
      gpu.barrier
      
      %shared_a = memref.load %shared_A[%thread_idx] : memref<32xf32>
      %shared_b = memref.load %shared_B[%thread_idx] : memref<32xf32>
      %c_val = arith.addf %shared_a, %shared_b : f32
      
      memref.store %c_val, %C[%linear_idx] : memref<256xf32>
      
      memref.dealloc %shared_A : memref<32xf32>
      memref.dealloc %shared_B : memref<32xf32>
      gpu.return
    }
  }
}

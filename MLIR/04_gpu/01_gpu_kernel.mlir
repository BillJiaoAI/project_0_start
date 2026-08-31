module attributes {gpu.container_module} {
  gpu.module @kernel_module {
    gpu.func @vector_add_kernel(%A: memref<1024xf32>, %B: memref<1024xf32>, %C: memref<1024xf32>) kernel {
      %block_idx = gpu.block_id x
      %thread_idx = gpu.thread_id x
      %block_dim = gpu.block_dim x
      
      %linear_idx = arith.addi %thread_idx, (arith.muli %block_idx, %block_dim) : index
      
      %a_val = memref.load %A[%linear_idx] : memref<1024xf32>
      %b_val = memref.load %B[%linear_idx] : memref<1024xf32>
      %c_val = arith.addf %a_val, %b_val : f32
      memref.store %c_val, %C[%linear_idx] : memref<1024xf32>
      
      gpu.return
    }
  }
  
  func.func @main() -> () {
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    %A = memref.alloc() : memref<1024xf32>
    %B = memref.alloc() : memref<1024xf32>
    %C = memref.alloc() : memref<1024xf32>
    
    scf.for %i = %c0 to %c1024 step %c1 {
      %ival = arith.index_cast %i : index to f32
      memref.store %ival, %A[%i] : memref<1024xf32>
      memref.store %ival, %B[%i] : memref<1024xf32>
    }
    
    %stream = gpu.create_stream
    gpu.launch_func @kernel_module::@vector_add_kernel
      blocks in (%c32, %c1, %c1) threads in (%c32, %c1, %c1)
      args(%A, %B, %C) : memref<1024xf32>, memref<1024xf32>, memref<1024xf32>
    gpu.wait %stream
    gpu.destroy_stream %stream
    
    %result = memref.load %C[%c0] : memref<1024xf32>
    vector.print %result : f32
    
    memref.dealloc %A : memref<1024xf32>
    memref.dealloc %B : memref<1024xf32>
    memref.dealloc %C : memref<1024xf32>
    return
  }
}

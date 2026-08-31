module {
  func.func @gemm_affine(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) -> () {
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
}

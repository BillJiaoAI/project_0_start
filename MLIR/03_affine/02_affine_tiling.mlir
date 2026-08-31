module {
  func.func @affine_tiling() -> () {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    
    %A = memref.alloc() : memref<16x16xf32>
    %B = memref.alloc() : memref<16x16xf32>
    
    affine.for %i = 0 to 16 step 4 {
      affine.for %j = 0 to 16 step 4 {
        affine.for %ii = %i to min(%i + 4, 16) {
          affine.for %jj = %j to min(%j + 4, 16) {
            %ival = arith.index_cast %ii : index to f32
            %jval = arith.index_cast %jj : index to f32
            %val = arith.addf %ival, %jval : f32
            memref.store %val, %A[%ii, %jj] : memref<16x16xf32>
          }
        }
      }
    }
    
    memref.dealloc %A : memref<16x16xf32>
    memref.dealloc %B : memref<16x16xf32>
    return
  }
}

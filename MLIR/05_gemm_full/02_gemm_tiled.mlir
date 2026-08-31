module {
  func.func @gemm_tiled(%A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>) -> () {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    
    %shared_A = memref.alloc() : memref<16x16xf32>
    %shared_B = memref.alloc() : memref<16x16xf32>
    
    affine.for %i = 0 to 64 step 16 {
      affine.for %j = 0 to 64 step 16 {
        affine.for %k = 0 to 64 step 16 {
          affine.for %ii = %i to min(%i + 16, 64) {
            affine.for %kk = %k to min(%k + 16, 64) {
              %a_val = memref.load %A[%ii, %kk] : memref<64x64xf32>
              %tile_i = arith.subi %ii, %i : index
              %tile_k = arith.subi %kk, %k : index
              memref.store %a_val, %shared_A[%tile_i, %tile_k] : memref<16x16xf32>
            }
          }
          
          affine.for %kk = %k to min(%k + 16, 64) {
            affine.for %jj = %j to min(%j + 16, 64) {
              %b_val = memref.load %B[%kk, %jj] : memref<64x64xf32>
              %tile_k = arith.subi %kk, %k : index
              %tile_j = arith.subi %jj, %j : index
              memref.store %b_val, %shared_B[%tile_k, %tile_j] : memref<16x16xf32>
            }
          }
          
          affine.for %ii = %i to min(%i + 16, 64) {
            affine.for %jj = %j to min(%j + 16, 64) {
              affine.for %kk = 0 to 16 {
                %tile_i = arith.subi %ii, %i : index
                %tile_j = arith.subi %jj, %j : index
                %a_tile = memref.load %shared_A[%tile_i, %kk] : memref<16x16xf32>
                %b_tile = memref.load %shared_B[%kk, %tile_j] : memref<16x16xf32>
                %c_val = memref.load %C[%ii, %jj] : memref<64x64xf32>
                %mul = arith.mulf %a_tile, %b_tile : f32
                %add = arith.addf %c_val, %mul : f32
                memref.store %add, %C[%ii, %jj] : memref<64x64xf32>
              }
            }
          }
        }
      }
    }
    
    memref.dealloc %shared_A : memref<16x16xf32>
    memref.dealloc %shared_B : memref<16x16xf32>
    return
  }
}

module {
  func.func @vector_matmul(%A: memref<16x16xf32>, %B: memref<16x16xf32>, %C: memref<16x16xf32>) -> () {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    
    scf.for %i = %c0 to %c16 step %c4 {
      scf.for %j = %c0 to %c16 step %c4 {
        %acc = arith.constant 0.0 : vector<4x4xf32>
        %result = scf.for %k = %c0 to %c16 step %c4 iter_args(%acc = %acc) -> (vector<4x4xf32>) {
          %a_tile = vector.transfer_read %A[%i, %k], %c0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<4x4xf32>
          %b_tile = vector.transfer_read %B[%k, %j], %c0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<4x4xf32>
          %mul = arith.mulf %a_tile, %b_tile : vector<4x4xf32>
          %add = arith.addf %acc, %mul : vector<4x4xf32>
          scf.yield %add : vector<4x4xf32>
        }
        vector.transfer_write %result, %C[%i, %j] {in_bounds = [true, true]} : vector<4x4xf32>, memref<16x16xf32>
      }
    }
    return
  }
}

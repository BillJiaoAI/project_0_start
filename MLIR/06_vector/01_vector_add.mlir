module {
  func.func @vector_add(%A: memref<256xf32>, %B: memref<256xf32>, %C: memref<256xf32>) -> () {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    
    scf.for %i = %c0 to %c256 step %c4 {
      %vec_a = vector.transfer_read %A[%i], %c0 {in_bounds = [true]} : memref<256xf32>, vector<4xf32>
      %vec_b = vector.transfer_read %B[%i], %c0 {in_bounds = [true]} : memref<256xf32>, vector<4xf32>
      %vec_c = arith.addf %vec_a, %vec_b : vector<4xf32>
      vector.transfer_write %vec_c, %C[%i] {in_bounds = [true]} : vector<4xf32>, memref<256xf32>
    }
    return
  }
}

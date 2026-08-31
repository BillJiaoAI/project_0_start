module {
  func.func @dynamic_shape(%m: index, %n: index) -> memref<?x?xf32> {
    %A = memref.alloc(%m, %n) : memref<?x?xf32>
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %val = arith.constant 1.0 : f32
        memref.store %val, %A[%i, %j] : memref<?x?xf32>
      }
    }
    
    return %A : memref<?x?xf32>
  }
}

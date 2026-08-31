module {
  func.func @memref_basic() -> () {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    
    %A = memref.alloc() : memref<4x8xf32>
    %c0 = arith.constant 0 : f32
    
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        %val = arith.addf %c0, %c0 : f32
        memref.store %val, %A[%i, %j] : memref<4x8xf32>
      }
    }
    
    %c1 = arith.constant 1 : index
    %elem = memref.load %A[%c1, %c1] : memref<4x8xf32>
    vector.print %elem : f32
    
    memref.dealloc %A : memref<4x8xf32>
    return
  }
}

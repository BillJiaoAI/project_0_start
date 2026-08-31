module {
  func.func @affine_loop() -> () {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    
    %A = memref.alloc() : memref<10x10xf32>
    
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 10 {
        %ival = arith.index_cast %i : index to f32
        %jval = arith.index_cast %j : index to f32
        %val = arith.addf %ival, %jval : f32
        memref.store %val, %A[%i, %j] : memref<10x10xf32>
      }
    }
    
    %elem = memref.load %A[%c1, %c1] : memref<10x10xf32>
    vector.print %elem : f32
    
    memref.dealloc %A : memref<10x10xf32>
    return
  }
}

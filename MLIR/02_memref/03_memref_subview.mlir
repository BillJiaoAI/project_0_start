module {
  func.func @subview_demo() -> () {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    %A = memref.alloc() : memref<16x16xf32>
    
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %ival = arith.index_cast %i : index to f32
        %jval = arith.index_cast %j : index to f32
        %val = arith.addf %ival, %jval : f32
        memref.store %val, %A[%i, %j] : memref<16x16xf32>
      }
    }
    
    %tile0 = memref.subview %A[%c0, %c0] [%c8, %c8] [%c1, %c1] : memref<16x16xf32> to memref<8x8xf32, strided<[16, 1], offset: 0>>
    %tile1 = memref.subview %A[%c8, %c8] [%c8, %c8] [%c1, %c1] : memref<16x16xf32> to memref<8x8xf32, strided<[16, 1], offset: 128>>
    
    %elem0 = memref.load %tile0[%c0, %c0] : memref<8x8xf32, strided<[16, 1], offset: 0>>
    %elem1 = memref.load %tile1[%c0, %c0] : memref<8x8xf32, strided<[16, 1], offset: 128>>
    
    vector.print %elem0 : f32
    vector.print %elem1 : f32
    
    memref.dealloc %A : memref<16x16xf32>
    return
  }
}

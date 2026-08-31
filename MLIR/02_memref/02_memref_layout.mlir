module {
  func.func @layout_demo() -> () {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    
    %A_row = memref.alloc() : memref<4x8xf32, strided<[8, 1], offset: 0>>
    %A_col = memref.alloc() : memref<4x8xf32, strided<[1, 4], offset: 0>>
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c8 step %c1 {
        %val = arith.constant 1.0 : f32
        memref.store %val, %A_row[%i, %j] : memref<4x8xf32, strided<[8, 1], offset: 0>>
        memref.store %val, %A_col[%i, %j] : memref<4x8xf32, strided<[1, 4], offset: 0>>
      }
    }
    
    memref.dealloc %A_row : memref<4x8xf32, strided<[8, 1], offset: 0>>
    memref.dealloc %A_col : memref<4x8xf32, strided<[1, 4], offset: 0>>
    return
  }
}

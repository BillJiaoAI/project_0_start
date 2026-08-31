module {
  func.func @cond_demo(%x: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %cmp = arith.cmpi sgt, %x, %c10 : i32
    
    %result = scf.if %cmp -> (i32) {
      %then_val = arith.muli %x, %x : i32
      scf.yield %then_val : i32
    } else {
      %else_val = arith.addi %x, %c10 : i32
      scf.yield %else_val : i32
    }
    
    return %result : i32
  }
}

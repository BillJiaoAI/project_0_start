module {
  func.func @main() -> () {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %c0, %c1 : i32
    vector.print %sum : i32
    return
  }
}

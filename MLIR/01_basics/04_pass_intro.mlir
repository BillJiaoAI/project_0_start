module {
  func.func @add_twice(%a: f32, %b: f32) -> f32 {
    %c0 = arith.constant 0.0 : f32
    %add1 = arith.addf %a, %b : f32
    %add2 = arith.addf %add1, %c0 : f32
    return %add2 : f32
  }
}

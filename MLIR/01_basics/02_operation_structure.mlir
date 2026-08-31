module {
  func.func @example(%arg0: f32, %arg1: f32) -> f32 {
    %c42 = arith.constant 42.0 : f32
    %result = arith.addf %arg0, %arg1 : f32
    %final = arith.mulf %result, %c42 : f32
    return %final : f32
  }
}

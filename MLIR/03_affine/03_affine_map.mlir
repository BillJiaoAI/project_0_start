module {
  func.func @affine_map_demo(%arg0: index, %arg1: index) -> (index, index) {
    %result0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)> (%arg0, %arg1)
    %result1 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1 * 3)> (%arg0, %arg1)
    return %result0, %result1 : index, index
  }
}

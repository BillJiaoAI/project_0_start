module {
  func.func @attrs_demo() -> () {
    %int_const = arith.constant 42 : i32
    %float_const = arith.constant 3.14159 : f64
    %str_attr = arith.constant dense<"hello"> : tensor<5xi8>
    
    %bool_true = arith.constant true
    %bool_false = arith.constant false
    
    vector.print %int_const : i32
    vector.print %float_const : f64
    return
  }
}

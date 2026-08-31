#!/bin/bash

MLIR_FILE="../05_gemm_full/01_gemm_affine.mlir"

echo "=== 原始 IR ==="
cat $MLIR_FILE

echo ""
echo "=== Step 1: Affine 循环展开 ==="
mlir-opt $MLIR_FILE --affine-loop-unroll="unroll-factor=4" -o /tmp/step1.mlir
cat /tmp/step1.mlir

echo ""
echo "=== Step 2: Affine 分块 ==="
mlir-opt /tmp/step1.mlir --affine-loop-tile="tile-sizes=16,16,16" -o /tmp/step2.mlir
cat /tmp/step2.mlir

echo ""
echo "=== Step 3: Affine → SCF 转换 ==="
mlir-opt /tmp/step2.mlir --lower-affine -o /tmp/step3.mlir
cat /tmp/step3.mlir

echo ""
echo "=== Step 4: 向量化 ==="
mlir-opt /tmp/step3.mlir --vectorize="vectorize-reduction" -o /tmp/step4.mlir
cat /tmp/step4.mlir

echo ""
echo "=== Step 5: Vector → SCF 展开 ==="
mlir-opt /tmp/step4.mlir --lower-vector="contract-lowering=outerproduct" -o /tmp/step5.mlir
cat /tmp/step5.mlir

echo ""
echo "=== Step 6: 最终优化 ==="
mlir-opt /tmp/step5.mlir --canonicalize --cse -o /tmp/final.mlir
cat /tmp/final.mlir

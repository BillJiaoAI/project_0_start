@echo off
echo Running GEMM Full Pipeline
echo ==========================

set MLIR_OPT=mlir-opt
set MLIR_TRANSLATE=mlir-translate

echo.
echo 1. Affine GEMM
echo --------------
%MLIR_OPT% ..\05_gemm_full\01_gemm_affine.mlir --canonicalize
echo.

echo 2. Tiled GEMM
echo -------------
%MLIR_OPT% ..\05_gemm_full\02_gemm_tiled.mlir --canonicalize
echo.

echo 3. GPU GEMM
echo -----------
%MLIR_OPT% ..\05_gemm_full\03_gemm_gpu.mlir --canonicalize
echo.

echo 4. Full Lowering Pipeline
echo -------------------------
echo Step 1: Tile the loops
%MLIR_OPT% ..\05_gemm_full\01_gemm_affine.mlir --affine-loop-tile="tile-sizes=16,16,16" -o /tmp/gemm_tiled.mlir
echo.

echo Step 2: Lower affine to SCF
%MLIR_OPT% /tmp/gemm_tiled.mlir --lower-affine -o /tmp/gemm_scf.mlir
echo.

echo Step 3: Vectorize
%MLIR_OPT% /tmp/gemm_scf.mlir --vectorize="vectorize-reduction" -o /tmp/gemm_vector.mlir
echo.

echo Step 4: Lower vector
%MLIR_OPT% /tmp/gemm_vector.mlir --lower-vector="contract-lowering=outerproduct" -o /tmp/gemm_lowered.mlir
echo.

echo Step 5: Final optimizations
%MLIR_OPT% /tmp/gemm_lowered.mlir --canonicalize --cse
echo.

pause

@echo off
echo Running Affine Dialect Examples
echo ================================

set MLIR_OPT=mlir-opt

echo.
echo 1. Basic Affine Loop
echo --------------------
%MLIR_OPT% ..\03_affine\01_affine_loop.mlir --canonicalize
echo.

echo 2. Affine Tiling
echo ----------------
%MLIR_OPT% ..\03_affine\02_affine_tiling.mlir --canonicalize
echo.

echo 3. Affine Map
echo -------------
%MLIR_OPT% ..\03_affine\03_affine_map.mlir --canonicalize
echo.

echo 4. Automatic Tiling Pass
echo ------------------------
%MLIR_OPT% ..\03_affine\01_affine_loop.mlir --affine-loop-tile="tile-sizes=4,4" --canonicalize
echo.

echo 5. Loop Fusion
echo --------------
%MLIR_OPT% ..\03_affine\01_affine_loop.mlir --affine-loop-fusion --canonicalize
echo.

pause

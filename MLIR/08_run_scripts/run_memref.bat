@echo off
echo Running MemRef Dialect Examples
echo ================================

set MLIR_OPT=mlir-opt
set MLIR_RUNNER=mlir-cpu-runner

echo.
echo 1. Basic MemRef Operations
echo --------------------------
%MLIR_OPT% ..\02_memref\01_memref_basic.mlir --canonicalize --print-ir-after-all
echo.

echo 2. MemRef Layout (Row-major vs Column-major)
echo ---------------------------------------------
%MLIR_OPT% ..\02_memref\02_memref_layout.mlir --canonicalize
echo.

echo 3. MemRef Subview (Tiling)
echo ---------------------------
%MLIR_OPT% ..\02_memref\03_memref_subview.mlir --canonicalize
echo.

echo 4. Dynamic Shape MemRef
echo -----------------------
%MLIR_OPT% ..\02_memref\04_memref_dynamic_shape.mlir --canonicalize
echo.

pause

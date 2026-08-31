@echo off
echo Running GPU Dialect Examples
echo =============================

set MLIR_OPT=mlir-opt
set MLIR_GPU_RUNNER=mlir-gpu-runner
set MLIR_TRANSLATE=mlir-translate

echo.
echo 1. GPU Vector Add Kernel
echo ------------------------
%MLIR_OPT% ..\04_gpu\01_gpu_kernel.mlir --canonicalize
echo.

echo 2. GPU Shared Memory
echo --------------------
%MLIR_OPT% ..\04_gpu\02_gpu_shared_memory.mlir --canonicalize
echo.

echo 3. Lower GPU to NVVM
echo --------------------
%MLIR_OPT% ..\04_gpu\01_gpu_kernel.mlir --lower-gpu-to-nvvm --canonicalize
echo.

echo 4. Generate PTX
echo ---------------
%MLIR_TRANSLATE% --mlir-to-nvvmir ..\04_gpu\01_gpu_kernel.mlir
echo.

pause

@echo off
echo === Building Simplified CUTLASS GEMM ===
nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -o cutlass_simple.exe cutlass_simple.cu
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)
echo Build SUCCESS
echo.
echo === Running ===
cutlass_simple.exe
echo.
echo === Generating PTX ===
nvcc -ptx -arch=sm_120 -Wno-deprecated-gpu-targets -o cutlass_simple.ptx cutlass_simple.cu
echo.
echo === PTX Analysis (MMA instructions) ===
findstr /i "mma" cutlass_simple.ptx
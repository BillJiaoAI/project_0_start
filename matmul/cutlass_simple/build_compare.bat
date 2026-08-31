@echo off
echo === Building cuBLAS Comparison ===
nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -Xcompiler="/utf-8" -lcublas -o compare_cublas.exe compare_cublas.cu
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)
echo Build SUCCESS
echo.
echo === Running Comparison ===
compare_cublas.exe
echo.
echo === Generating PTX for MMA analysis ===
nvcc -ptx -arch=sm_120 -Wno-deprecated-gpu-targets -Xcompiler="/utf-8" -o cutlass_simple.ptx cutlass_simple.cu
echo.
echo === PTX Analysis (MMA instructions) ===
echo --- MMA instructions found: ---
findstr /i "mma" cutlass_simple.ptx
echo.
echo --- Total instructions count: ---
find /c /v "" cutlass_simple.ptx
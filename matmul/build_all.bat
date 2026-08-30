@echo off
echo ============================================
echo Matmul Build and Profile Script
echo RTX 5070 (Blackwell) - sm_120
echo ============================================

set CUDA_PATH=d:\software2_ln\cuda_128
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo [1/4] Building Baseline...
cd baseline
nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -o matmul_baseline.exe matmul_baseline.cu
if %errorlevel% equ 0 (
    echo   Build SUCCESS
    nsys profile --stats=true -o matmul_baseline ./matmul_baseline.exe
) else (
    echo   Build FAILED
)
cd ..

echo.
echo [2/4] Building V1: Shared Memory Tiling...
cd optimized_v1
nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -o matmul_v1.exe matmul_v1.cu
if %errorlevel% equ 0 (
    echo   Build SUCCESS
    nsys profile --stats=true -o matmul_v1 ./matmul_v1.exe
) else (
    echo   Build FAILED
)
cd ..

echo.
echo [3/4] Building V2: Bank Conflict Avoidance...
cd optimized_v2
nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -o matmul_v2.exe matmul_v2.cu
if %errorlevel% equ 0 (
    echo   Build SUCCESS
    nsys profile --stats=true -o matmul_v2 ./matmul_v2.exe
) else (
    echo   Build FAILED
)
cd ..

echo.
echo [4/4] Building V3: Async Copy...
cd optimized_v3
nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -o matmul_v3.exe matmul_v3.cu
if %errorlevel% equ 0 (
    echo   Build SUCCESS
    nsys profile --stats=true -o matmul_v3 ./matmul_v3.exe
) else (
    echo   Build FAILED
)
cd ..

echo.
echo [5/5] Building V4: Tensor Core WMMA...
cd optimized_v4
nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -o matmul_v4.exe matmul_v4.cu
if %errorlevel% equ 0 (
    echo   Build SUCCESS
    nsys profile --stats=true -o matmul_v4 ./matmul_v4.exe
) else (
    echo   Build FAILED
)
cd ..

echo.
echo ============================================
echo All builds completed!
echo NSYS reports saved in each version directory.
echo ============================================
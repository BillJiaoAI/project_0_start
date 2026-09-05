@echo off
chcp 65001 >nul
echo Generating SASS for V1 and V2...

set CUDA_PATH=d:\software2_ln\cuda_128
set PATH=%CUDA_PATH%\bin;%PATH%

echo [1/2] V1 Shared Memory Tiling...
cd optimized_v1
nvcc -arch=sm_120 -cubin -o matmul_v1.cubin matmul_v1.cu
cuobjdump -sass matmul_v1.cubin > matmul_v1_sass.txt
echo   Done!

echo [2/2] V2 Bank Conflict Padding...
cd ..\optimized_v2
nvcc -arch=sm_120 -cubin -o matmul_v2.cubin matmul_v2.cu
cuobjdump -sass matmul_v2.cubin > matmul_v2_sass.txt
echo   Done!

cd ..
echo.
echo SASS files generated:
echo   - optimized_v1/matmul_v1_sass.txt
echo   - optimized_v2/matmul_v2_sass.txt
pause
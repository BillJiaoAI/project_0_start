@echo off
chcp 65001 >nul
echo ============================================
echo CUDA Performance Analysis Script
echo Comparing V1 vs V2 (Bank Conflict Padding)
echo ============================================
echo.

set CUDA_PATH=d:\software2_ln\cuda_128
set PATH=%CUDA_PATH%\bin;%PATH%

echo [1/5] Generating PTX assembly...
cd baseline
nvcc -arch=sm_120 -ptx -o matmul_baseline.ptx matmul_baseline.cu 2>nul
cd ..\optimized_v1
nvcc -arch=sm_120 -ptx -o matmul_v1.ptx matmul_v1.cu 2>nul
cd ..\optimized_v2
nvcc -arch=sm_120 -ptx -o matmul_v2.ptx matmul_v2.cu 2>nul
cd ..
echo   PTX generated!
echo.

echo [2/5] Comparing PTX instruction counts...
python ptx_compare.py > ptx_comparison.txt
type ptx_comparison.txt
echo.

echo [3/5] Running nsys profile for V1...
nsys profile --stats=true -o matmul_v1_nsys optimized_v1\matmul_v1.exe > nsys_log_v1.txt 2>&1
echo   V1 nsys report generated
echo.

echo [4/5] Running nsys profile for V2...
nsys profile --stats=true -o matmul_v2_nsys optimized_v2\matmul_v2.exe > nsys_log_v2.txt 2>&1
echo   V2 nsys report generated
echo.

echo [5/5] Extracting nsys stats...
python extract_nsys_stats.py > nsys_comparison.txt
type nsys_comparison.txt
echo.

echo ============================================
echo Analysis complete!
echo Files generated:
echo   - ptx_comparison.txt (PTX instruction comparison)
echo   - nsys_comparison.txt (nsys statistics comparison)
echo   - matmul_v1_nsys.nsys-rep (V1 nsys report)
echo   - matmul_v2_nsys.nsys-rep (V2 nsys report)
echo ============================================
pause
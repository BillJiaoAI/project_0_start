@echo off
chcp 65001 >nul
echo ============================================
echo Verifying Hypotheses #2 and #3
echo ============================================

set CUDA_PATH=d:\software2_ln\cuda_128
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\nsight-compute;%PATH%

echo [1/4] V1 Memory Metrics (Hypothesis #2)...
cd optimized_v1
ncu --metrics sm__shared_mem_transactions_per_request.avg,sm__ldst_throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_memory.avg ./matmul_v1.exe > v1_mem_metrics.txt
type v1_mem_metrics.txt
echo.

echo [2/4] V2 Memory Metrics (Hypothesis #2)...
cd ..\optimized_v2
ncu --metrics sm__shared_mem_transactions_per_request.avg,sm__ldst_throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_memory.avg ./matmul_v2.exe > v2_mem_metrics.txt
type v2_mem_metrics.txt
echo.

echo [3/4] V1 Pipeline Metrics (Hypothesis #3)...
cd ..\optimized_v1
ncu --metrics sm__inst_executed_pipe_fma.avg,sm__inst_executed_pipe_int.avg,sm__inst_executed_pipe_tensor.avg,sm__pipe_utilization.avg.pct ./matmul_v1.exe > v1_pipe_metrics.txt
type v1_pipe_metrics.txt
echo.

echo [4/4] V2 Pipeline Metrics (Hypothesis #3)...
cd ..\optimized_v2
ncu --metrics sm__inst_executed_pipe_fma.avg,sm__inst_executed_pipe_int.avg,sm__inst_executed_pipe_tensor.avg,sm__pipe_utilization.avg.pct ./matmul_v2.exe > v2_pipe_metrics.txt
type v2_pipe_metrics.txt
echo.

cd ..
echo ============================================
echo Done! Check the metrics above for:
echo   - sm__shared_mem_transactions_per_request: V2 should be higher
echo   - sm__ldst_throughput: V2 should be lower  
echo   - sm__inst_executed_pipe_int: V2 should be higher
echo   - sm__pipe_utilization: V2 should be lower
echo ============================================
pause
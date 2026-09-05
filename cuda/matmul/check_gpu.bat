@echo off
echo === GPU Info ===
nvidia-smi --query-gpu=gpu_name,compute_cap,memory.total --format=csv
echo.
echo === CUDA Capability ===
nvcc --version
@echo off
REM ============================================================
REM  Build llama.cpp from source with CPU + CUDA backends
REM  - Uses MSVC cl.exe (vcvars64.bat)
REM  - Uses CMake + Ninja (shipped with Visual Studio)
REM  - CUDA toolkit at D:\software2_ln\cuda_128 (nvcc 13.3)
REM ============================================================

echo [1/4] Setting up MSVC environment ...
call "D:\software2_ln\visual_st\community_product\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo [ERROR] vcvars64.bat failed & exit /b 1 )

echo [2/4] Setting up paths (CMake, Ninja, CUDA) ...
set "CMAKE_DIR=D:\software2_ln\visual_st\community_product\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
set "NINJA_DIR=D:\software2_ln\visual_st\community_product\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "PATH=%CMAKE_DIR%;%NINJA_DIR%;%PATH%"
set "CUDA_PATH=D:\software2_ln\cuda_128"
set "PATH=%CUDA_PATH%\bin;%PATH%"

cd /d "D:\software2_ln\project_0_start\llama\llama.cpp"

echo [3/4] CMake configure (CPU + CUDA, Release, Ninja) ...
REM RTX 5070 = Blackwell, compute capability 12.0
cmake -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=120 ^
    -DGGML_CUDA_USE_CUDNN=OFF ^
    -DLLAMA_BUILD_SERVER=OFF ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=OFF ^
    -DBUILD_SHARED_LIBS=OFF
if errorlevel 1 ( echo [ERROR] CMake configure failed & exit /b 1 )

echo [4/4] Building (this may take several minutes for CUDA kernels) ...
cmake --build build --config Release -- -j 8
if errorlevel 1 ( echo [ERROR] Build failed & exit /b 1 )

echo.
echo [OK] llama.cpp build complete!
echo     Static lib: build\src\Release\llama.lib (or build\src\llama.lib)
echo     Headers:    include\llama.h, ggml\include\ggml.h
exit /b 0

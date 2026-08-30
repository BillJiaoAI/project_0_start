@echo off
REM ============================================================
REM  Build the llama.cpp inference demo (v2)
REM  Links against llama.cpp static libs (CPU + CUDA backends)
REM ============================================================

echo [1/3] Setting up MSVC environment ...
call "D:\software2_ln\visual_st\community_product\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo [ERROR] vcvars64.bat failed & exit /b 1 )

echo [2/3] Setting up paths ...
set "LLAMA_ROOT=D:\software2_ln\project_0_start\llama\llama.cpp"
set "BUILD_DIR=%LLAMA_ROOT%\build"
set "CUDA_PATH=D:\software2_ln\cuda_128"
set "PATH=%CUDA_PATH%\bin;%PATH%"

REM --- Include paths ---
set "INCLUDES=/I"%LLAMA_ROOT%\include" /I"%LLAMA_ROOT%\ggml\include""

REM --- Library search paths ---
set "LIBPATHS=/LIBPATH:"%BUILD_DIR%\src" /LIBPATH:"%BUILD_DIR%\ggml\src" /LIBPATH:"%BUILD_DIR%\ggml\src\ggml-cuda" /LIBPATH:"%CUDA_PATH%\lib\x64""

REM --- Libraries to link ---
REM Core: llama -> ggml -> ggml-base + ggml-cpu + ggml-cuda
REM CUDA runtime libs needed by ggml-cuda
set "LIBS=llama.lib ggml.lib ggml-cpu.lib ggml-cuda.lib ggml-base.lib cudart.lib cublas.lib cublasLt.lib cuda.lib advapi32.lib shell32.lib ole32.lib user32.lib"

cd /d "D:\software2_ln\project_0_start\llama\v2_simple_inference"

echo [3/3] Compiling main.cpp ...
cl.exe /nologo /EHsc /std:c++17 /O2 /utf-8 /MD %INCLUDES% main.cpp ^
    /Fe:llama_demo.exe ^
    /link %LIBPATHS% %LIBS%

if errorlevel 1 ( echo [ERROR] Build failed & exit /b 1 )
echo.
echo [OK] llama_demo.exe built successfully!
exit /b 0

@echo off
rem ============================================================================
rem build_all.bat: build (and optionally run) the MHA learning series v0 ~ v4
rem ----------------------------------------------------------------------------
rem usage:
rem   build_all.bat          build only
rem   build_all.bat run      build then run each version (each self-checks
rem                          against a CPU reference and prints PASS)
rem
rem note: nvcc needs the MSVC host compiler (cl.exe). If cl is not on PATH we
rem       try vswhere + vcvars64; if that fails, run this script from the
rem       "x64 Native Tools Command Prompt for VS".
rem ============================================================================
setlocal
cd /d "%~dp0"

where cl >nul 2>&1
if errorlevel 1 (
    for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do (
        call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    )
)
where cl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cl.exe not found. Run this from the x64 Native Tools prompt.
    exit /b 1
)

set NVCC=nvcc -O2 -arch=sm_120 -Wno-deprecated-gpu-targets -Xcompiler="/utf-8"

echo [build] v0_ref  (CPU reference)
%NVCC% -o v0_ref\mha_ref.exe v0_ref\mha_ref.cu || goto :err
echo [build] v1_naive (naive GPU, materialize S)
%NVCC% -o v1_naive\mha_naive.exe v1_naive\mha_naive.cu || goto :err
echo [build] v2_vec  (llama.cpp fattn-vec)
%NVCC% -o v2_vec\mha_vec.exe v2_vec\mha_vec.cu || goto :err
echo [build] v3_tile (llama.cpp fattn-tile)
%NVCC% -o v3_tile\mha_tile.exe v3_tile\mha_tile.cu || goto :err
echo [build] v4_mma  (llama.cpp fattn-mma-f16, tensor core)
%NVCC% -o v4_mma\mha_mma.exe v4_mma\mha_mma.cu || goto :err
%NVCC% -o v4_mma\test_mma.exe v4_mma\test_mma.cu || goto :err
nvcc -g -lineinfo -arch=sm_120 -Wno-deprecated-gpu-targets -Xcompiler="/utf-8" -o v4_mma\mha_dbg.exe v4_mma\mha_dbg.cu || goto :err
echo [build] all done
echo.

if /i not "%1"=="run" exit /b 0

echo ======== v0_ref: CPU reference ========
v0_ref\mha_ref.exe || goto :err
echo ======== v1_naive: naive GPU ========
v1_naive\mha_naive.exe || goto :err
echo ======== v2_vec: fattn-vec (warp-level + parallel blocks) ========
v2_vec\mha_vec.exe || goto :err
echo ======== v3_tile: fattn-tile (smem tiled flash) ========
v3_tile\mha_tile.exe || goto :err
echo ======== v4_mma: fattn-mma-f16 (tensor core) ========
v4_mma\mha_mma.exe || goto :err
echo ======== v4 test_mma: mma/ldmatrix/movmatrix primitive self-test ========
v4_mma\test_mma.exe || goto :err
echo.
echo ======== ALL PASS ========
exit /b 0

:err
echo [ERROR] build or run failed
exit /b 1

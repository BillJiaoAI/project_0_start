@echo off
REM ============================================================
REM build.bat - compile llama_demo
REM Uses MSVC cl.exe, dynamically links Ollama's libllama.dll
REM ============================================================

setlocal

REM ---- 1. Set up MSVC build environment ----
set "VS_DIR=D:\software2_ln\visual_st\community_product"
set "VCVARS=%VS_DIR%\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VCVARS%" (
    echo [ERROR] vcvars64.bat not found: %VCVARS%
    exit /b 1
)

echo [1/2] Setting up MSVC environment ...
call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] vcvars64.bat failed
    exit /b 1
)

REM ---- 2. Compile ----
echo [2/2] Compiling main.cpp ...
cl.exe /nologo /EHsc /std:c++17 /O2 /utf-8 main.cpp /Fe:llama_demo.exe /link

if errorlevel 1 (
    echo.
    echo [ERROR] Compilation failed
    exit /b 1
)

echo.
echo [OK] Build success: llama_demo.exe
endlocal

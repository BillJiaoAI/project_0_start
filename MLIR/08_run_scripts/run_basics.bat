@echo off
echo Running MLIR Basics Examples
echo =============================

set MLIR_OPT=mlir-opt
set MLIR_RUNNER=mlir-cpu-runner
set MLIR_TRANSLATE=mlir-translate

echo.
echo 1. Hello World
echo --------------
%MLIR_OPT% ..\01_basics\01_hello_world.mlir --canonicalize --cse
echo.

echo 2. Operation Structure
echo ----------------------
%MLIR_OPT% ..\01_basics\02_operation_structure.mlir --canonicalize
echo.

echo 3. Attributes and Types
echo -----------------------
%MLIR_OPT% ..\01_basics\03_attributes_types.mlir --canonicalize
echo.

echo 4. Control Flow
echo ---------------
%MLIR_OPT% ..\01_basics\05_control_flow.mlir --canonicalize
echo.

pause

@echo off
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 source.cu
  exit /b 2
)
set "SRC=%~1"
call "%~dp0run_single_nvcc.bat" -std=c++20 -O2 -Xcompiler "/MD /std:c++20" -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 -I"%CD%\csrc" -I"%CD%\csrc\cutlass\include" -I"%CD%\tests" "%SRC%" -o "%CD%\tests\bin_single_test.exe"

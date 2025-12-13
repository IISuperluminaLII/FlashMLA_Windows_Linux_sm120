@echo off
setlocal ENABLEDELAYEDEXPANSION

for %%v in ("%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" ^
            "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" ^
            "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat") do (
  if exist "%%~v" (
    call "%%~v"
    goto :vsok
  )
)
echo [build] Could not find vcvars64.bat. Edit this script.
exit /b 1
:vsok

if "%CUDA_PATH%"=="" set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set NVCC="%CUDA_PATH%\bin\nvcc.exe"
if not exist %NVCC% (
  echo [build] nvcc not found at %NVCC%
  exit /b 2
)

pushd "%~dp0\.."
set ROOT=%CD%
if not exist buildlogs mkdir buildlogs

set NVCCFLAGS=-std=c++20 -O2 -Xcompiler "/MD /std:c++20" -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 -I"%ROOT%\csrc" -I"%ROOT%\csrc\cutlass\include" -I"%ROOT%\tests"

echo [build] Compiling fwd_softmax_views_hypothesis_tests.cu
%NVCC% %NVCCFLAGS% "%ROOT%\tests\sm120\layout\laws\fwd_softmax_views_hypothesis_tests.cu" -o "%ROOT%\tests\bin_fwd_softmax_views_hypothesis_tests.exe"
if errorlevel 1 (
  echo [FAIL] Compilation failed
  popd
  exit /b 3
)

echo [PASS] Compilation succeeded
echo [run] Running hypothesis test...
"%ROOT%\tests\bin_fwd_softmax_views_hypothesis_tests.exe"

popd
exit /b 0

@echo off
setlocal
set VC_VARS=
for %%P in (
  "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) do (
  if exist %%~P set VC_VARS=%%~P
)
if not defined VC_VARS (
  echo [ERROR] vcvars64.bat not found
  exit /b 1
)
call "%VC_VARS%"

set PROJ=%~dp0\..
pushd %PROJ%
for /f "usebackq delims=" %%I in (`py -3.12 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'include'))"`) do set TORCH_INC1=%%I
for /f "usebackq delims=" %%I in (`py -3.12 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'include','torch','csrc','api','include'))"`) do set TORCH_INC2=%%I
set PY_INC1=%LocalAppData%\Programs\Python\Python312\include
set PY_INC2=%LocalAppData%\Programs\Python\Python312\Include

"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -c csrc\sm120\prefill\dense\fmha_cutlass_bwd_sm120.cu -o build\sm120_bwd.obj ^
 -I csrc -I csrc\sm90 -I csrc\cutlass\include -I csrc\cutlass\tools\util\include ^
 -I "%TORCH_INC1%" -I "%TORCH_INC2%" -I "%PY_INC1%" -I "%PY_INC2%" ^
 -include msvc_compat.h -O3 -D_USE_MATH_DEFINES -Wno-deprecated-declarations ^
 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ ^
 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --ptxas-options=-v,--register-usage-level=10 ^
 -Xcompiler /Zc:__cplusplus -Xcompiler /permissive- ^
 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 ^
 -DFLASH_MLA_DISABLE_SM100 -DFLASH_MLA_BUILD_SM120 2>&1 
set ERR=%ERRORLEVEL%
popd
exit /b %ERR%
if not exist buildlogs mkdir buildlogs >nul 2>nul

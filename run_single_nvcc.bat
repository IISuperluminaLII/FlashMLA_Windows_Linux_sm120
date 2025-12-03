@echo off
setlocal
set VCVARSPATH=
for %%v in ("%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" ^
            "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" ^
            "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat") do (
  if exist "%%~v" (
    set "VCVARSPATH=%%~v"
    goto :vcfound
  )
)
echo [single-nvcc] Could not find vcvars64.bat
exit /b 1
:vcfound
call "%VCVARSPATH%" >nul 2>&1
if "%CUDA_PATH%"=="" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
"%NVCC%" %*

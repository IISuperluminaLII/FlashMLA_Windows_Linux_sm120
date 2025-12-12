@echo off
setlocal
set SCRIPT_DIR=%~dp0
pushd %SCRIPT_DIR%\..

rem Select Python executable (allow override via PYTHON_EXE)
if not defined PYTHON_EXE (
  where py >nul 2>nul && (
    for /f "usebackq delims=" %%P in (`py -3.12 -c "import sys;print(sys.executable)" 2^>nul`) do set PYTHON_EXE=%%P
    if not defined PYTHON_EXE (
      for /f "usebackq delims=" %%P in (`py -3.10 -c "import sys;print(sys.executable)" 2^>nul`) do set PYTHON_EXE=%%P
    )
  )
  if not defined PYTHON_EXE set PYTHON_EXE=python
)

set FLASH_MLA_ARCH=sm100
set PYTHONPATH=%CD%

"%PYTHON_EXE%" benchmark\bench_flash_mla.py --target flash_mla %*

popd
endlocal


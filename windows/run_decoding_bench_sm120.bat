@echo off
setlocal
set SCRIPT_DIR=%~dp0
pushd %SCRIPT_DIR%\..

set FLASH_MLA_ARCH=sm120
set PYTHONPATH=%CD%

python benchmark\bench_flash_mla.py --target flash_mla %*

popd
endlocal


@echo off
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\PyCharmProjectsSpaceConflict\150BLLM\external\FlashMLA\build\temp.win-amd64-cpython-312\Release
set FLASH_MLA_BUILD_BWD=1
set FLASH_MLA_DISABLE_SM90=1
set FLASH_MLA_DISABLE_SM100=1
set FLASH_MLA_ARCH=sm120
set DISTUTILS_USE_SDK=1
set MSSdk=1
ninja -v

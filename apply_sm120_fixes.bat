@echo off
echo Applying SM120 Blackwell Workstation Fixes based on community solutions...
echo.

REM Based on successful implementations from:
REM - vLLM PR#17280 (SM120 w8a8 support)
REM - FlashAttention issue#1683 (RTX 50 support)
REM - CUTLASS 3.9 TMEM fixes

echo Step 1: Setting environment variables for SM120 build
set FLASH_MLA_DISABLE_SM100=0
set FLASH_MLA_DISABLE_SM90=1
set FLASH_MLA_SM120_DISABLE_BWD=0
set ENABLE_SCALED_MM_SM120=1
set TORCH_CUDA_ARCH_LIST=12.0

echo.
echo Step 2: Build configuration for SM120
echo - CUDA architecture: 12.0 (SM120)
echo - Cluster shape forced to 1x1x1 (no multicast on GeForce)
echo - TMA/TMEM enabled (Blackwell has 4th gen tensor cores)
echo - Using CUTLASS with SM120 support
echo.

echo Build flags:
echo FLASH_MLA_DISABLE_SM100=%FLASH_MLA_DISABLE_SM100%
echo FLASH_MLA_DISABLE_SM90=%FLASH_MLA_DISABLE_SM90%
echo FLASH_MLA_SM120_DISABLE_BWD=%FLASH_MLA_SM120_DISABLE_BWD%
echo ENABLE_SCALED_MM_SM120=%ENABLE_SCALED_MM_SM120%
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
echo.

echo Step 3: Running build with SM120 configuration
"C:\Users\Shashank Murthy\.conda\envs\150BLLM\python.exe" setup.py clean --all
"C:\Users\Shashank Murthy\.conda\envs\150BLLM\python.exe" setup.py build_ext --inplace

echo.
echo Build complete. Check output for any errors.
#!/usr/bin/env bash
set -e
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="/usr/local/cuda-12.9/bin:$PATH"
export FLASH_MLA_ARCH=sm120
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
SO=flash_mla/cuda_sm120.cpython-312-x86_64-linux-gnu.so
# Rename the (possibly live-mapped) old .so aside BEFORE building. rename(2) keeps the
# inode alive for any running training process; setuptools then writes a fresh file.
if [ -f "$SO" ]; then
  mv -f "$SO" "${SO}.prebuild_bak"
  echo "MOVED_OLD_SO_ASIDE -> ${SO}.prebuild_bak"
fi
echo "BUILD_START $($PY --version) nvcc=$(command -v nvcc)"
"$PY" setup.py build_ext --inplace
echo "WSL_BUILD_DONE"
ls -la "$SO"

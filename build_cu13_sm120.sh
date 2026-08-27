#!/usr/bin/env bash
# Build FlashMLA sm120 with nvcc 13.2 against torch 2.9.1+cu130 in the PARALLEL env
# 150BLLM_cu13, inside an isolated copy at ~/flashmla_cu13 (never touches the live repo's
# 12.9-built .so or the live training env). Battery then runs from the copy.
set -e
SRC=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
DST="$HOME/flashmla_cu13"
PY=/home/shashankm/miniconda3/envs/150BLLM_cu13/bin/python

rsync -a --delete \
  --exclude=build --exclude='*.so' --exclude='*.pyd' --exclude=__pycache__ \
  --exclude=.git --exclude='tests/bin_*' --exclude='tests/*.exe' --exclude='tests/*.lib' \
  --exclude='tests/*.exp' \
  "$SRC/" "$DST/"
echo "RSYNC_DONE"

cd "$DST"
# CUDA 13.0 = EXACT match with torch 2.9.1+cu130's bundled runtime (13.0 == 13.0,
# no version skew at all; 13.2 would compile too but warns on the minor).
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="/usr/local/cuda-13.0/bin:$PATH"
export FLASH_MLA_ARCH=sm120
echo "nvcc: $(command -v nvcc) ($(nvcc --version | tail -1))"
"$PY" setup.py build_ext --inplace 2>&1 | tail -3
echo "CU13_BUILD_DONE"
ls -la flash_mla/*.so

#!/usr/bin/env bash
# Rebuild the live .so, then ptxas-probe the sparse fp8 decode TU: both mma tiers
# (batch-parallel CFG=1 + splitkv CFG=2) must show 0 spill (operator directive).
set -u
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
bash build_wsl_sm120.sh 2>&1 | tail -3
NVCC=/usr/local/cuda-13.0/bin/nvcc
"$NVCC" -std=c++17 -arch=sm_120 -DFLASH_MLA_BUILD_SM120 \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I csrc -I csrc/cutlass/include -I /usr/local/cuda-13.0/include/cccl \
  --ptxas-options=-v -c csrc/sm120/decode/sparse_fp8/splitkv_mla.cu -o /tmp/_sp_probe.o 2>&1 \
  | grep -B1 -A2 "sparse_fp8_decode_mma"
echo "SPARSE_SPLITKV_BUILD_PROBE_DONE"

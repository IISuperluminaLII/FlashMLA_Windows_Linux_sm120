#!/usr/bin/env bash
# Rebuild the live .so (nvcc 13.0 / torch 2.13.0+cu130), then ptxas-probe the
# dense decode TU for the mma kernel's register/spill stats (directive: 0 spill).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
bash build_wsl_sm120.sh 2>&1 | tail -3
NVCC=/usr/local/cuda-13.0/bin/nvcc
"$NVCC" -std=c++17 -arch=sm_120 -DFLASH_MLA_BUILD_SM120 \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I csrc -I csrc/cutlass/include -I /usr/local/cuda-13.0/include/cccl \
  --ptxas-options=-v -c csrc/sm120/decode/dense/splitkv_mla.cu -o /tmp/_dd_probe.o 2>&1 \
  | grep -A2 "dense_decode_mma_kernel"
echo "BUILD_PROBE_DONE"

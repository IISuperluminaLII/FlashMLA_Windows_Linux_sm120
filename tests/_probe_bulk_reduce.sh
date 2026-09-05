#!/usr/bin/env bash
# Build + run the fp32 tile-reduction throughput probe on the sm_120 device (device 0,
# PCI order). Tiny footprint (64MB), a few hundred ms of GPU time.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT/tests"
NVCC="$CUDA_HOME/bin/nvcc"
"$NVCC" -std=c++17 -O3 -arch=sm_120 -o /tmp/_probe_bulk_reduce _probe_bulk_reduce.cu || { echo "[FAILED] nvcc"; exit 1; }
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/_probe_bulk_reduce

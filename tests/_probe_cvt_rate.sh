#!/usr/bin/env bash
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT/tests"
"$CUDA_HOME/bin/nvcc" -std=c++17 -O3 -arch=sm_120 -o /tmp/_probe_cvt_rate _probe_cvt_rate.cu || { echo "[FAILED] nvcc"; exit 1; }
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/_probe_cvt_rate

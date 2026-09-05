#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
"$CUDA_HOME/bin/nvcc" -arch=sm_120 -O2 -std=c++17 -o /tmp/_probe_tma_tensormap tests/_probe_tma_tensormap.cu 2>&1 | grep -E 'error|warning' | head -10
[ -x /tmp/_probe_tma_tensormap ] || { echo "[FAILED] compile"; exit 1; }
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/_probe_tma_tensormap

#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
"$CUDA_HOME/bin/nvcc" -arch=sm_120 -O3 -std=c++17 -o /tmp/_probe_l2_smem_bw tests/_probe_l2_smem_bw.cu 2>&1 | grep -E 'error' | head -5
[ -x /tmp/_probe_l2_smem_bw ] || { echo "[FAILED] compile"; exit 1; }
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /tmp/_probe_l2_smem_bw

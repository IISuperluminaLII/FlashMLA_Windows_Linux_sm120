#!/usr/bin/env bash
# e4m3 mma.sync probe on device 0 (layout map + token-permuted PV construction + peak TFLOPS).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT/tests"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
echo "[GPU-CHECK]"; nvidia-smi --query-gpu=name,utilization.gpu,power.draw,memory.used --format=csv,noheader | head -2
ps -eo pid,etime,cmd | grep -E "python|flash" | grep -v grep | head -10
"$CUDA_HOME/bin/nvcc" -std=c++17 -O3 -arch=sm_120 -o /tmp/_probe_fp8_mma _probe_fp8_mma.cu && /tmp/_probe_fp8_mma
echo "PROBE_RC=$?"

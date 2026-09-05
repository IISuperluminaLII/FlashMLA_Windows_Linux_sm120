#!/usr/bin/env bash
# Backward at the large shape b=8 H=32 (FA2 sm80 reference ~263 TF bwd @4096), BWD_MMA_CFG from env (default 2).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
"$PY" tests/_patch_bwd_bench_shape.py
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_FWD_MMA=1 FLASH_MLA_SM120_BWD_MMA=1
export FLASH_MLA_SM120_FWD_MMA_CFG="${FLASH_MLA_SM120_FWD_MMA_CFG:-7}" FLASH_MLA_SM120_BWD_MMA_CFG="${FLASH_MLA_SM120_BWD_MMA_CFG:-2}"
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()"
echo "----- bwd 128/128 b=8 H=32, BWD_CFG=$FLASH_MLA_SM120_BWD_MMA_CFG -----"
BENCH_B=8 BENCH_H=32 "$PY" tests/bench_bwd_vs_sdpa.py 2>&1 | grep -E "b=.*S="
echo "BWD_BIGSHAPE_DONE"

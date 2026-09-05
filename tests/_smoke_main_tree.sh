#!/usr/bin/env bash
# Smoke-test the MAIN-tree extension at the final tiers (import path + accuracy suite).
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_FWD_MMA=1 FLASH_MLA_SM120_BWD_MMA=1 FLASH_MLA_SM120_FWD_MMA_CFG=11 FLASH_MLA_SM120_BWD_MMA_CFG=8
"$PY" -c "import flash_mla.cuda_sm120 as e; print('MAIN_EXT', e.__file__)" 2>&1 | grep MAIN_EXT
"$PY" tests/test_accuracy_vs_sdpa.py 2>&1 | grep -E '^\[RESULT\]'
FLASH_MLA_BWD_IDENTITY_NEW_CFG=8 "$PY" tests/test_bwd_cfg4_dkdv_identity.py 2>&1 | grep -E '^\[RESULT\]'
echo "SMOKE_MAIN_DONE"

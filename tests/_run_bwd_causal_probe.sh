#!/usr/bin/env bash
# Red/green runner for the backward causal-alignment fix: NaN presence per BWD tier for sq != skv.
# Usage: bash tests/_run_bwd_causal_probe.sh [tiers...]  (default: 0 2 3 4)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_FWD_MMA=1 FLASH_MLA_SM120_BWD_MMA=1
for c in ${*:-0 2 3 4}; do
  FLASH_MLA_SM120_BWD_MMA_CFG=$c "$PY" tests/_probe_bwd_causal_convention.py 2>&1 | grep -E 'CFG=|Error|error' | head -8
done
echo "BWD_CAUSAL_PROBE_DONE"

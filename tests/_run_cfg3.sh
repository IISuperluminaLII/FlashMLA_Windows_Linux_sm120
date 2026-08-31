#!/usr/bin/env bash
# CFG=3 measurement only (decode + CFG 0/1/2 numbers already recorded).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export FLASH_MLA_SM120_SPARSE_FWD_CFG=3

echo "== prefill perf CFG=3 (was: legacy 9.67, CFG1 17.9, CFG2 19.9 TFlops) =="
"$PY" tests/_diag_sparse_perf.py 2>&1 | grep -E "Prefill:|RESULT"
echo "== prefill correctness 24 at CFG=3 =="
"$PY" tests/_diag_sparse_red.py 2>&1 | tail -1
echo "CFG3_DONE"

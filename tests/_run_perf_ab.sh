#!/usr/bin/env bash
# Post-A0/unroll measurement set on the 12.9 build:
#  1. decode 34 regression + decode perf 8 (after the Phase-E unroll fix)
#  2. sparse prefill perf: legacy (env unset) vs A0 (FLASH_MLA_SM120_SPARSE_FWD_CFG=1)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "== decode 34 regression (unroll fix) =="
"$PY" tests/_diag_sparse_decode.py 2>&1 | tail -1
echo "== decode perf 8 (was 27/37 TFLOPS) =="
"$PY" tests/_diag_sparse_decode_perf.py 2>&1 | grep -E "ms,|RESULT"
echo "== prefill perf LEGACY (baseline was ~9.7-9.8 TFlops) =="
unset FLASH_MLA_SM120_SPARSE_FWD_CFG
"$PY" tests/_diag_sparse_perf.py 2>&1 | grep -E "Prefill:|RESULT"
echo "== prefill perf A0 (FLASH_MLA_SM120_SPARSE_FWD_CFG=1) =="
export FLASH_MLA_SM120_SPARSE_FWD_CFG=1
"$PY" tests/_diag_sparse_perf.py 2>&1 | grep -E "Prefill:|RESULT"
echo "== prefill perf A0+A1 (FLASH_MLA_SM120_SPARSE_FWD_CFG=2) =="
export FLASH_MLA_SM120_SPARSE_FWD_CFG=2
"$PY" tests/_diag_sparse_perf.py 2>&1 | grep -E "Prefill:|RESULT"
echo "== prefill perf CFG=3 (A0+A1+A2/A3/A4) =="
export FLASH_MLA_SM120_SPARSE_FWD_CFG=3
"$PY" tests/_diag_sparse_perf.py 2>&1 | grep -E "Prefill:|RESULT"
echo "== prefill correctness 24 at CFG=3 =="
"$PY" tests/_diag_sparse_red.py 2>&1 | tail -1
echo "PERF_AB_DONE"

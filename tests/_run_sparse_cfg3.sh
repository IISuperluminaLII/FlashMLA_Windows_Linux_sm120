#!/usr/bin/env bash
# Sparse fp8 decode cluster-crossover (CFG=3) certification:
#   1) 34-case + ext-5 correctness at CFG=3 (multi-split + 2-CTA DSM crossover)
#   2) perf A/B: bench + serving shapes at CFG=3 (vs the recorded CFG=1/2 rows)
#   3) default-path regression (env unset -> legacy WMMA)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

export FLASH_MLA_SM120_SPARSE_DECODE_CFG=3
echo "== sparse decode 34 at CFG=3 (cluster crossover) =="
"$PY" tests/_diag_sparse_decode.py 2>&1 | tail -1
echo "== sparse decode ext-5 at CFG=3 =="
"$PY" tests/_diag_sparse_decode_ext.py 2>&1 | tail -1
echo "== perf at CFG=3 =="
"$PY" tests/_diag_sparse_splitkv_bench.py 2>&1 | grep -E "CASE|ms,|RESULT"

unset FLASH_MLA_SM120_SPARSE_DECODE_CFG
echo "== default-path regression (env unset, legacy WMMA) =="
"$PY" tests/_diag_sparse_decode.py 2>&1 | tail -1
echo "SPARSE_CFG3_DONE"

#!/usr/bin/env bash
# Sparse fp8 decode split-KV (CFG=2) certification:
#   1) 34-case + ext-5 correctness at CFG=2 (REAL multi-split: 47-64 parts at
#      these shapes -- the oracle discriminates broken split logic directly)
#   2) perf A/B: bench + serving shapes at CFG=1 then CFG=2
#   3) default-path regression (env unset -> legacy WMMA)
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

export FLASH_MLA_SM120_SPARSE_DECODE_CFG=2
echo "== sparse decode 34 at CFG=2 (split-KV) =="
"$PY" tests/_diag_sparse_decode.py 2>&1 | tail -1
echo "== sparse decode ext-5 at CFG=2 =="
"$PY" tests/_diag_sparse_decode_ext.py 2>&1 | tail -1
echo "== perf at CFG=2 =="
"$PY" tests/_diag_sparse_splitkv_bench.py 2>&1 | grep -E "CASE|ms,|RESULT"

export FLASH_MLA_SM120_SPARSE_DECODE_CFG=1
echo "== perf at CFG=1 (A/B reference) =="
"$PY" tests/_diag_sparse_splitkv_bench.py 2>&1 | grep -E "CASE|ms,|RESULT"

unset FLASH_MLA_SM120_SPARSE_DECODE_CFG
echo "== default-path regression (env unset, legacy WMMA) =="
"$PY" tests/_diag_sparse_decode.py 2>&1 | tail -1
echo "SPARSE_CFG2_DONE"

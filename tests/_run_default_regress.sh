#!/usr/bin/env bash
# Default-path regression after fwd.cu dispatch edit: CFG env UNSET -> legacy path must stay 24/24.
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
unset FLASH_MLA_SM120_SPARSE_FWD_CFG
echo "== sparse prefill fwd default path (CFG unset) =="
"$PY" tests/_diag_sparse_red.py 2>&1 | tail -1
echo "DEFAULT_REGRESS_DONE"

#!/usr/bin/env bash
# Bwd CFG=2 (mma + red.global.add.v2.f32 scatter): 6/6 oracle, then A/B vs CFG=1 on the
# SDPA fwd+bwd row (CFG=1 was 7.057 ms total, ~5.83 ms bwd; atomic-bound half ~2.9 ms).
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "== bwd oracle 6 at CFG=2 (vectorized red scatter) =="
FLASH_MLA_SM120_SPARSE_BWD_CFG=2 "$PY" tests/_diag_sparse_bwd_redgreen.py 2>&1 | tail -1
echo "== A/B fwd+bwd: CFG=1 (scalar atomics) =="
FLASH_MLA_SM120_SPARSE_FWD_CFG=4 FLASH_MLA_SM120_SPARSE_BWD_CFG=1 FLASH_MLA_SM120_SPARSE_DECODE_CFG=1 \
  "$PY" tests/bench_sdpa_comparison.py 2>&1 | grep -E "fwd\+bwd s_q"
echo "== A/B fwd+bwd: CFG=2 (red.v2) =="
FLASH_MLA_SM120_SPARSE_FWD_CFG=4 FLASH_MLA_SM120_SPARSE_BWD_CFG=2 FLASH_MLA_SM120_SPARSE_DECODE_CFG=1 \
  "$PY" tests/bench_sdpa_comparison.py 2>&1 | grep -E "fwd\+bwd s_q"
echo "== bwd default regression (env unset) =="
"$PY" tests/_diag_sparse_bwd_redgreen.py 2>&1 | tail -1
echo "BWD_CFG2_DONE"

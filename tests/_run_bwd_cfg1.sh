#!/usr/bin/env bash
# Bwd CFG=1 (mma.sync) verification + bench. Targets: 6/6 red-green oracle (A/A2/B/C
# + new D ragged-topk + E all-invalid-rows), then fwd+bwd beats gather+SDPA (< 20.0 ms
# at s_q=512; WMMA bwd era was 40.4 ms total). Also: default-path (env unset) bwd
# regression stays green on the same 6.
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

export FLASH_MLA_SM120_SPARSE_BWD_CFG=1
echo "== bwd red-green 4 at CFG=1 (mma.sync) =="
"$PY" tests/_diag_sparse_bwd_redgreen.py 2>&1 | tail -1
echo "== SDPA head-to-head fwd+bwd (CFG fwd=4, bwd=1; target < 20.0 ms) =="
FLASH_MLA_SM120_SPARSE_FWD_CFG=4 FLASH_MLA_SM120_SPARSE_DECODE_CFG=1 \
  "$PY" tests/bench_sdpa_comparison.py 2>&1 | grep -E "fwd s_q|fwd\+bwd s_q|decode b=" | head -4

unset FLASH_MLA_SM120_SPARSE_BWD_CFG
echo "== bwd default-path regression (env unset, WMMA) =="
"$PY" tests/_diag_sparse_bwd_redgreen.py 2>&1 | tail -1
echo "BWD_CFG1_DONE"

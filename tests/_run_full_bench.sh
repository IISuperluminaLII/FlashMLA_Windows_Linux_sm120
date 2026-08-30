#!/usr/bin/env bash
# THE full FlashMLA sm120 benchmark: training + inference, every path, at the
# recommended tier set (FWD=4, BWD=2, DENSE=4, SPARSE_DECODE=1 bench / 4 serving;
# dense prefill mma fwd+bwd are default-on). Sequential (max-2-tests rule).
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "===== [1] DENSE DECODE: authors' 16-case sweep @ DENSE_DECODE_CFG=4 ====="
FLASH_MLA_SM120_DENSE_DECODE_CFG=4 "$PY" tests/_diag_dense_decode_perf.py 2>&1 \
  | grep -E "ms,|RESULT"

echo "===== [2] DENSE DECODE small-head (M16/M32/h64/h128 tiers) @ CFG=4 ====="
FLASH_MLA_SM120_DENSE_DECODE_CFG=4 "$PY" tests/_diag_dense_m16_bench.py 2>&1 \
  | grep -E "CASE|ms,|RESULT"

echo "===== [3] SPARSE FP8 DECODE bench+serving @ DECODE_CFG=1 (max-batch) ====="
FLASH_MLA_SM120_SPARSE_DECODE_CFG=1 "$PY" tests/_diag_sparse_splitkv_bench.py 2>&1 \
  | grep -E "CASE|ms,|RESULT"
echo "===== [3b] SPARSE FP8 DECODE bench+serving @ DECODE_CFG=4 (serving) ====="
FLASH_MLA_SM120_SPARSE_DECODE_CFG=4 "$PY" tests/_diag_sparse_splitkv_bench.py 2>&1 \
  | grep -E "CASE|ms,|RESULT"

echo "===== [4] SPARSE PREFILL FWD: authors' 10-case sweep (s_kv 4K..128K) @ FWD_CFG=4 ====="
FLASH_MLA_SM120_SPARSE_FWD_CFG=4 "$PY" tests/_diag_sparse_perf.py 2>&1 \
  | grep -E "TFlops|tflops|ms|RESULT|PASS" | grep -vE "^\s*$" | tail -14

echo "===== [5] FLASHMLA vs TORCH SDPA (3 paths, cross-checked) @ FWD=4 BWD=2 DECODE=1 ====="
FLASH_MLA_SM120_SPARSE_FWD_CFG=4 FLASH_MLA_SM120_SPARSE_BWD_CFG=2 \
FLASH_MLA_SM120_SPARSE_DECODE_CFG=1 "$PY" tests/bench_sdpa_comparison.py 2>&1 \
  | grep -E "fwd|decode|cos|speedup|====" | grep -vE "^\s*$"

echo "===== [6] AUTHORS' E2E TRAINING BENCH (dense fwd+bwd via autograd vs Torch) ====="
# PYTHONSAFEPATH: benchmark/triton/ (the authors' reference kernels) shadows
# the real triton package via the script-dir sys.path entry, breaking dynamo
# ("triton.language has no attribute dtype"). Safe path keeps the script dir
# off sys.path; flash_mla resolves from the editable install.
PYTHONSAFEPATH=1 "$PY" benchmark/bench_flash_mla_training.py 2>&1 | tail -20

echo "FULL_BENCH_DONE"

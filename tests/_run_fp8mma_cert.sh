#!/usr/bin/env bash
# Run the fp8-mma certification test inside the snapshot at one CFG, keep the full log, and print
# the mismatch details (max-error positions from check_is_allclose) + the RESULT line.
#   bash tests/_run_fp8mma_cert.sh [cfg=5] [quick|full] [snap]
set -u
CFG="${1:-5}"; MODE="${2:-quick}"; SNAP="${3:-$HOME/flashmla_snap}"
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
cp "$MAIN/tests/test_sparse_decode_fp8mma.py" "$SNAP/tests/"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_SPARSE_DECODE_CFG="$CFG"
LOG="$SNAP/cert_cfg${CFG}.log"
"$PY" tests/test_sparse_decode_fp8mma.py $([ "$MODE" = quick ] && echo quick) > "$LOG" 2>&1
grep -v FutureWarning "$LOG" | grep -B3 -A10 'mismatch' | grep -vE '^\[OK\]|^\[CASE\]' | cut -c1-220 | head -100
grep -E '^\[RESULT\]|^\[ENV\]' "$LOG"

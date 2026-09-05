#!/usr/bin/env bash
# Sequential certification of the three agent-left rungs in the isolated snapshot:
#   fwd rung 11 (kv_tile 128 on the TMA ring), bwd CFG=4 (FROST structure), sparse fwd CFG=5 (warp-spec ring).
# Launch detached: setsid nohup bash tests/_snap_chain_all3.sh [snap] > chain6.log 2>&1 < /dev/null &
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
echo "=== (1) overlay ($(date +%H:%M:%S))"
cp -v "$MAIN"/csrc/sm120/prefill/dense/fmha_fwd_mma_sm120.cuh "$MAIN"/csrc/sm120/prefill/dense/fmha_cutlass_fwd_sm120.cu \
      "$MAIN"/csrc/sm120/prefill/dense/fmha_bwd_mma_sm120.cuh "$MAIN"/csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu "$SNAP/csrc/sm120/prefill/dense/"
cp -v "$MAIN"/csrc/sm120/prefill/sparse/fwd.cu "$MAIN"/csrc/sm120/prefill/sparse/fwd_ws.cuh "$SNAP/csrc/sm120/prefill/sparse/"
cp -v "$MAIN"/tests/_run_fwd_ladder.sh "$MAIN"/tests/_run_fwd_bigshape.sh "$MAIN"/tests/_run_fwd_diag.sh "$MAIN"/tests/_run_bwd_ladder.sh "$MAIN"/tests/_run_bwd_bigshape.sh "$MAIN"/tests/test_bwd_cfg4_dkdv_identity.py "$SNAP/tests/"
echo "=== (2) rebuild ($(date +%H:%M:%S))"
bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build6.log" 2>&1; RC=$?
grep -E ' error|Error' "$SNAP/build6.log" | grep -v FutureWarning | head -8; tail -n 2 "$SNAP/build6.log"; echo "REBUILD_RC=$RC ($(date +%H:%M:%S))"
if [ "$RC" -ne 0 ]; then echo "CHAIN_ABORT_BUILD_FAILED"; exit 1; fi
"$PY" -c "import flash_mla.cuda_sm120 as e; print('EXT_FROM', e.__file__)" 2>&1 | grep EXT_FROM
echo "=== (3) FWD ladder 10 11 ($(date +%H:%M:%S))"
bash "$SNAP/tests/_run_fwd_ladder.sh" 10 11 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (4) FWD bigshape 11 ($(date +%H:%M:%S))"
RUNGS="11" bash "$SNAP/tests/_run_fwd_bigshape.sh" 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (5) FWD diag ($(date +%H:%M:%S))"
bash "$SNAP/tests/_run_fwd_diag.sh" 2>&1 | grep -vE 'pynvml|FutureWarning' | sed -n '/rung 11, b=8/,$p'
echo "=== (6) BWD ladder 2 vs 4 at FWD 10 ($(date +%H:%M:%S))"
FLASH_MLA_SM120_FWD_MMA_CFG=10 bash "$SNAP/tests/_run_bwd_ladder.sh" 2 4 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (7) BWD bigshape CFG=4 ($(date +%H:%M:%S))"
FLASH_MLA_SM120_FWD_MMA_CFG=10 FLASH_MLA_SM120_BWD_MMA_CFG=4 bash "$SNAP/tests/_run_bwd_bigshape.sh" 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (8) BWD dk/dv identity CFG=4 vs CFG=2 ($(date +%H:%M:%S))"
"$PY" "$SNAP/tests/test_bwd_cfg4_dkdv_identity.py" 2>&1 | grep -vE 'pynvml|FutureWarning' | tail -30
echo "=== (9) SPARSE fwd CFG=5: red suite (24) DIAG 0 / RED probe DIAG 1 (must FAIL) / skew DIAG 3; CFG=4 regression ($(date +%H:%M:%S))"
for combo in "5 0" "5 1" "5 3" "4 0"; do
  set -- $combo
  echo "--- SPARSE_FWD_CFG=$1 DIAG=$2"
  FLASH_MLA_SM120_SPARSE_FWD_CFG=$1 FLASH_MLA_SM120_SPARSE_FWD_DIAG=$2 "$PY" tests/_diag_sparse_red.py 2>&1 | grep -E '^\[RESULT\]|FAIL|Error' | tail -3
done
echo "=== (10) SPARSE fwd perf CFG 4 vs 5 (+ DIAG 2 = softmax skipped) ($(date +%H:%M:%S))"
for combo in "4 0" "5 0" "5 2"; do
  set -- $combo
  echo "--- SPARSE_FWD_CFG=$1 DIAG=$2"
  FLASH_MLA_SM120_SPARSE_FWD_CFG=$1 FLASH_MLA_SM120_SPARSE_FWD_DIAG=$2 "$PY" tests/_diag_sparse_perf.py 2>&1 | grep -vE 'pynvml|FutureWarning' | grep -E 'ms|TF|GB/s|RESULT|FAIL' | head -8
done
echo "CHAIN_DONE ($(date +%H:%M:%S))"

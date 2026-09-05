#!/usr/bin/env bash
# Snapshot chain for the fwd rungs 8/9: overlay the two fwd files from the main tree, incremental
# rebuild, then the certification: fwd ladder (suites 44+126 per rung + benches) for the rungs given,
# bigshape b=8 H=32, and the diag pairs. Launch detached:
#   setsid nohup bash tests/_snap_chain_fwd8.sh [snap] [rungs...] > chain3.log 2>&1 < /dev/null &
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"; shift || true
RUNGS="${*:-7 8 9}"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
echo "=== (1) overlay fwd files + runners ($(date +%H:%M:%S))"
cp -v "$MAIN/csrc/sm120/prefill/dense/fmha_fwd_mma_sm120.cuh" "$SNAP/csrc/sm120/prefill/dense/"
cp -v "$MAIN/csrc/sm120/prefill/dense/fmha_cutlass_fwd_sm120.cu" "$SNAP/csrc/sm120/prefill/dense/"
cp -v "$MAIN/tests/_run_fwd_ladder.sh" "$MAIN/tests/_run_fwd_bigshape.sh" "$MAIN/tests/_run_fwd_diag.sh" "$SNAP/tests/"
echo "=== (2) incremental rebuild ($(date +%H:%M:%S))"
bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build3.log" 2>&1; RC=$?
grep -E ' error|Error' "$SNAP/build3.log" | grep -v FutureWarning | head -8; tail -n 2 "$SNAP/build3.log"; echo "REBUILD_RC=$RC ($(date +%H:%M:%S))"
if [ "$RC" -ne 0 ]; then echo "CHAIN_ABORT_BUILD_FAILED"; exit 1; fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$PY" -c "import flash_mla.cuda_sm120 as e; print('EXT_FROM', e.__file__)" 2>&1 | grep EXT_FROM
echo "=== (3) fwd ladder $RUNGS ($(date +%H:%M:%S))"
bash "$SNAP/tests/_run_fwd_ladder.sh" $RUNGS 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (4) bigshape ($(date +%H:%M:%S))"
RUNGS="$RUNGS" bash "$SNAP/tests/_run_fwd_bigshape.sh" 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (5) diag ($(date +%H:%M:%S))"
bash "$SNAP/tests/_run_fwd_diag.sh" 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "CHAIN_DONE ($(date +%H:%M:%S))"

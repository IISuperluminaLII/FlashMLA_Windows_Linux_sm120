#!/usr/bin/env bash
# Snapshot chain for bwd CFG=4: overlay the two bwd files (+ the new dk/dv identity test) from the main
# tree, incremental rebuild, bwd ladder 2 vs 4 with the fwd at rung 10, the big shape at CFG=4, and the
# new discriminating test. Launch detached:
#   setsid nohup bash tests/_snap_chain_bwd4.sh [snap] > chain5.log 2>&1 < /dev/null &
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"; shift || true
RUNGS="${*:-2 4}"; NEWCFG="${RUNGS##* }"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
echo "=== (1) overlay bwd files ($(date +%H:%M:%S))"
cp -v "$MAIN/csrc/sm120/prefill/dense/fmha_bwd_mma_sm120.cuh" "$MAIN/csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu" "$SNAP/csrc/sm120/prefill/dense/"
[ -f "$MAIN/tests/test_bwd_cfg4_dkdv_identity.py" ] && cp -v "$MAIN/tests/test_bwd_cfg4_dkdv_identity.py" "$SNAP/tests/"
cp -v "$MAIN/tests/_probe_bwd_causal_convention.py" "$MAIN/tests/_run_bwd_causal_probe.sh" "$SNAP/tests/"
cp -v "$MAIN/flash_mla/flash_mla_interface.py" "$SNAP/flash_mla/"   # workspace sizing (CFG>=6 64-row alignment)
cp -v "$MAIN/tests/_run_bwd_ladder.sh" "$MAIN/tests/_run_bwd_bigshape.sh" "$SNAP/tests/"
echo "=== (2) incremental rebuild ($(date +%H:%M:%S))"
bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build5.log" 2>&1; RC=$?
grep -E ' error|Error' "$SNAP/build5.log" | grep -v FutureWarning | head -8; tail -n 2 "$SNAP/build5.log"; echo "REBUILD_RC=$RC ($(date +%H:%M:%S))"
if [ "$RC" -ne 0 ]; then echo "CHAIN_ABORT_BUILD_FAILED"; exit 1; fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$PY" -c "import flash_mla.cuda_sm120 as e; print('EXT_FROM', e.__file__)" 2>&1 | grep EXT_FROM
echo "=== (2b) causal-alignment probe, tiers 0 2 3 4 (expect grad_nan all False) ($(date +%H:%M:%S))"
bash "$SNAP/tests/_run_bwd_causal_probe.sh" 0 2 $RUNGS 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (3) bwd ladder $RUNGS at FWD_CFG=10 ($(date +%H:%M:%S))"
FLASH_MLA_SM120_FWD_MMA_CFG=10 bash "$SNAP/tests/_run_bwd_ladder.sh" $RUNGS 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (4) bigshape bwd CFG=$NEWCFG ($(date +%H:%M:%S))"
FLASH_MLA_SM120_FWD_MMA_CFG=10 FLASH_MLA_SM120_BWD_MMA_CFG=$NEWCFG bash "$SNAP/tests/_run_bwd_bigshape.sh" 2>&1 | grep -vE 'pynvml|FutureWarning'
if [ -f "$SNAP/tests/test_bwd_cfg4_dkdv_identity.py" ]; then
  echo "=== (5) dk/dv identity test CFG=$NEWCFG vs CFG=2 ($(date +%H:%M:%S))"
  FLASH_MLA_BWD_IDENTITY_NEW_CFG=$NEWCFG CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$PY" "$SNAP/tests/test_bwd_cfg4_dkdv_identity.py" 2>&1 | grep -vE 'pynvml|FutureWarning' | tail -25
fi
echo "CHAIN_DONE ($(date +%H:%M:%S))"

#!/usr/bin/env bash
# Chain in the ISOLATED snapshot: (1) verify the snapshot .so is a real fresh compile, (2) fwd
# diag (rung 7 vs softmax-skipped), (3) overlay the finished bwd CFG=3 files from the main tree,
# (4) incremental rebuild, (5) bwd ladder 2 vs 3. Runs detached (nohup); log = $SNAP/chain.log.
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
echo "=== (1) build provenance ($(date +%H:%M:%S))"
grep -c 'nvcc' "$SNAP/build.log"; ls -la --time-style=+%H:%M:%S build/temp.linux-x86_64-cpython-312/csrc/sm120/prefill/dense/*.o
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$PY" -c "import flash_mla, flash_mla.cuda_sm120 as e; print('FLASH_MLA_FROM', flash_mla.__file__); print('EXT_FROM', e.__file__)"
echo "=== (2) fwd diag ($(date +%H:%M:%S))"
bash "$SNAP/tests/_run_fwd_diag.sh" 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (3) overlay bwd CFG=3 files ($(date +%H:%M:%S))"
cp -v "$MAIN/csrc/sm120/prefill/dense/fmha_bwd_mma_sm120.cuh" "$SNAP/csrc/sm120/prefill/dense/fmha_bwd_mma_sm120.cuh"
cp -v "$MAIN/csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu" "$SNAP/csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu"
echo "=== (4) incremental rebuild ($(date +%H:%M:%S))"
bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build2.log" 2>&1; RC=$?
tail -n 3 "$SNAP/build2.log"; echo "REBUILD_RC=$RC ($(date +%H:%M:%S))"
if [ "$RC" -ne 0 ]; then echo "CHAIN_ABORT_BUILD_FAILED"; exit 1; fi
echo "=== (5) bwd ladder 2 vs 3 ($(date +%H:%M:%S))"
bash "$SNAP/tests/_run_bwd_ladder.sh" 2 3 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "CHAIN_DONE ($(date +%H:%M:%S))"

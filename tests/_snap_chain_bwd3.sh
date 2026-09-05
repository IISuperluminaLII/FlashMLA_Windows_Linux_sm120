#!/usr/bin/env bash
# Snapshot chain part 2: incremental rebuild (bwd CFG=3 overlaid) + bwd ladder 2 vs 3. Launch with
# setsid so the WSL session teardown cannot SIGINT ninja: setsid nohup bash <this> > log 2>&1 < /dev/null &
set -u
SNAP="${1:-$HOME/flashmla_snap}"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
echo "=== (4) incremental rebuild ($(date +%H:%M:%S))"
bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build2.log" 2>&1; RC=$?
grep -E 'error|Error' "$SNAP/build2.log" | grep -v FutureWarning | head -5; tail -n 2 "$SNAP/build2.log"; echo "REBUILD_RC=$RC ($(date +%H:%M:%S))"
if [ "$RC" -ne 0 ]; then echo "CHAIN_ABORT_BUILD_FAILED"; exit 1; fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$PY" -c "import flash_mla.cuda_sm120 as e; print('EXT_FROM', e.__file__)" 2>&1 | grep EXT_FROM
echo "=== (5) bwd ladder 2 vs 3 ($(date +%H:%M:%S))"
bash "$SNAP/tests/_run_bwd_ladder.sh" 2 3 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "CHAIN_DONE ($(date +%H:%M:%S))"

#!/usr/bin/env bash
# Snapshot chain for the sparse fp8 decode OPT ladder: overlay the sparse decode dir from the main
# tree, incremental rebuild, then per OPT rung x {CFG=1, CFG=4}: correctness (34 + ext 5) and the
# split-KV bench. Launch detached: setsid nohup bash tests/_snap_chain_sparse_opt.sh [snap] [opt rungs] > chain4.log 2>&1 < /dev/null &
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"; shift || true
OPTS="${*:-0 1 2 3}"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
echo "=== (1) overlay sparse decode files ($(date +%H:%M:%S))"
cp -v "$MAIN"/csrc/sm120/decode/sparse_fp8/* "$SNAP/csrc/sm120/decode/sparse_fp8/"
for f in tests/_diag_sparse_decode.py tests/_diag_sparse_decode_ext.py tests/_diag_sparse_splitkv_bench.py; do [ -f "$MAIN/$f" ] && cp -v "$MAIN/$f" "$SNAP/tests/"; done
echo "=== (2) incremental rebuild ($(date +%H:%M:%S))"
bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build4.log" 2>&1; RC=$?
grep -E ' error|Error' "$SNAP/build4.log" | grep -v FutureWarning | head -8; tail -n 2 "$SNAP/build4.log"; echo "REBUILD_RC=$RC ($(date +%H:%M:%S))"
if [ "$RC" -ne 0 ]; then echo "CHAIN_ABORT_BUILD_FAILED"; exit 1; fi
"$PY" -c "import flash_mla.cuda_sm120 as e; print('EXT_FROM', e.__file__)" 2>&1 | grep EXT_FROM
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()" 2>/dev/null
FAILED=0
for opt in $OPTS; do
  for cfg in 1 4; do
    export FLASH_MLA_SM120_SPARSE_DECODE_OPT="$opt" FLASH_MLA_SM120_SPARSE_DECODE_CFG="$cfg"
    echo "################ OPT=$opt CFG=$cfg ################"
    for suite in tests/_diag_sparse_decode.py tests/_diag_sparse_decode_ext.py; do
      last="$("$PY" "$suite" 2>&1 | grep -E '^\[RESULT\]' | tail -1)"
      if printf "%s" "$last" | grep -q " 0 failed"; then echo "[OK] opt=$opt cfg=$cfg $suite: $last"; else echo "[FAILED] opt=$opt cfg=$cfg $suite: ${last:-<no RESULT line>}"; "$PY" "$suite" 2>&1 | grep -E "FAIL|Error|error" | head -6; FAILED=1; fi
    done
    echo "----- splitkv bench (opt=$opt cfg=$cfg) -----"
    "$PY" tests/_diag_sparse_splitkv_bench.py 2>&1 | grep -vE 'pynvml|FutureWarning' | grep -E 'ms|TF|GB/s|bench|serving' | head -8
  done
done
unset FLASH_MLA_SM120_SPARSE_DECODE_OPT FLASH_MLA_SM120_SPARSE_DECODE_CFG
echo "----- default regression (OPT/CFG unset) -----"
last="$("$PY" tests/_diag_sparse_decode.py 2>&1 | grep -E '^\[RESULT\]' | tail -1)"; echo "default: $last"; printf "%s" "$last" | grep -q " 0 failed" || FAILED=1
echo "SPARSE_OPT_RESULT: $([ $FAILED -eq 0 ] && echo PASS || echo FAILED)"
echo "CHAIN_DONE ($(date +%H:%M:%S))"

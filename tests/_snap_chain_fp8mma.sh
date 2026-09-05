#!/usr/bin/env bash
# Snapshot chain for the FP8 tensor-math sparse decode tiers (CFG=5/6): overlay the edited files
# from the main tree, build WITH the opt-in (FLASH_MLA_SM120_FP8_MMA=1), ptxas resource check of the
# new kernels, certification test at CFG=5 and 6, default + bf16-tier regression (34+5 at CFG unset /
# 1 / 4 -- byte-identical launch paths), then the perf probe at CFG 1/4/5/6.
# Launch detached: setsid nohup bash tests/_snap_chain_fp8mma.sh [snap] > chain_fp8.log 2>&1 < /dev/null &
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"
MODE="${2:-full}"     # full | quick (quick = smaller certification case list)
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export FLASH_MLA_SM120_FP8_MMA=1
echo "=== (1) overlay ($(date +%H:%M:%S))"
cp -v "$MAIN"/csrc/sm120/decode/sparse_fp8/* "$SNAP/csrc/sm120/decode/sparse_fp8/"
cp -v "$MAIN"/csrc/pybind.cpp "$SNAP/csrc/pybind.cpp"
cp -v "$MAIN"/setup.py "$SNAP/setup.py"
cp -v "$MAIN"/build_wsl_sm120.sh "$SNAP/build_wsl_sm120.sh"
for f in tests/test_sparse_decode_fp8mma.py tests/_diag_sparse_fp8mma_bench.py tests/_diag_sparse_decode.py tests/_diag_sparse_decode_ext.py tests/_diag_sparse_splitkv_bench.py tests/_res_usage_filtered.sh; do
  [ -f "$MAIN/$f" ] && cp -v "$MAIN/$f" "$SNAP/tests/"
done
echo "=== (2) build with FLASH_MLA_SM120_FP8_MMA=1 ($(date +%H:%M:%S))"
bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build_fp8.log" 2>&1; RC=$?
grep -E ' error|Error|error:' "$SNAP/build_fp8.log" | grep -v FutureWarning | head -20; tail -n 2 "$SNAP/build_fp8.log"; echo "REBUILD_RC=$RC ($(date +%H:%M:%S))"
if [ "$RC" -ne 0 ]; then echo "CHAIN_ABORT_BUILD_FAILED"; exit 1; fi
"$PY" -c "import flash_mla.cuda_sm120 as e; print('EXT_FROM', e.__file__, 'fp8_mma=', e.__sm120_fp8_mma__)" 2>&1 | grep EXT_FROM
echo "=== (3) ptxas resource usage of the fp8mma kernels"
SO="$(ls "$SNAP"/flash_mla/cuda_sm120*.so | head -1)"
"$CUDA_HOME/bin/cuobjdump" -res-usage "$SO" 2>/dev/null | grep -A1 "fp8mma" | grep -E "Function|REG" | sed 's/^/  /' | head -8
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()" 2>/dev/null
FAILED=0
for cfg in 5 6; do
  export FLASH_MLA_SM120_SPARSE_DECODE_CFG="$cfg"
  echo "################ CERT fp8mma CFG=$cfg ($(date +%H:%M:%S)) ################"
  "$PY" tests/test_sparse_decode_fp8mma.py $([ "$MODE" = quick ] && echo quick) 2>&1 | grep -vE 'pynvml|FutureWarning' | grep -E "^\[|cos_diff|lse max|mismatch|Anomaly" | tail -60
  last="$("$PY" tests/test_sparse_decode_fp8mma.py $([ "$MODE" = quick ] && echo quick) 2>&1 | grep -E '^\[RESULT\]' | tail -1)"
  if printf "%s" "$last" | grep -q " 0 failed"; then echo "[OK] cert cfg=$cfg: $last"; else echo "[FAILED] cert cfg=$cfg: ${last:-<no RESULT line>}"; FAILED=1; fi
done
echo "=== (4) bf16-tier / default regression (authors' oracle) ==="
for cfg in unset 1 4; do
  if [ "$cfg" = unset ]; then unset FLASH_MLA_SM120_SPARSE_DECODE_CFG; else export FLASH_MLA_SM120_SPARSE_DECODE_CFG="$cfg"; fi
  for suite in tests/_diag_sparse_decode.py tests/_diag_sparse_decode_ext.py; do
    last="$("$PY" "$suite" 2>&1 | grep -E '^\[RESULT\]' | tail -1)"
    if printf "%s" "$last" | grep -q " 0 failed"; then echo "[OK] cfg=$cfg $suite: $last"; else echo "[FAILED] cfg=$cfg $suite: ${last:-<no RESULT line>}"; FAILED=1; fi
  done
done
echo "=== (5) authors' oracle AT the fp8 tiers (informational: expected to exceed cos 5e-6 by contract)"
for cfg in 5 6; do
  export FLASH_MLA_SM120_SPARSE_DECODE_CFG="$cfg"
  "$PY" tests/_diag_sparse_decode.py 2>&1 | grep -E '^\[RESULT\]' | tail -1 | sed "s/^/  cfg=$cfg authors-oracle: /"
done
echo "=== (6) perf ($(date +%H:%M:%S))"
for cfg in 1 4 5 6; do
  export FLASH_MLA_SM120_SPARSE_DECODE_CFG="$cfg"
  "$PY" tests/_diag_sparse_fp8mma_bench.py 2>&1 | grep -E "^\[BENCH\]|Error|error"
done
unset FLASH_MLA_SM120_SPARSE_DECODE_CFG
echo "FP8MMA_RESULT: $([ $FAILED -eq 0 ] && echo PASS || echo FAILED)"
echo "CHAIN_DONE ($(date +%H:%M:%S))"

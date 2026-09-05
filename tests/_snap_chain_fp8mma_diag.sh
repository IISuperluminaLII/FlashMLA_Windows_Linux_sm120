#!/usr/bin/env bash
# fp8mma attribution chain: overlay + incremental rebuild (fp8 opt-in), SASS instruction mix of the
# fp8mma kernels, per-phase clock64 timing at CFG=5/6 on the bench + serving shapes, then the quick
# certification at CFG=5 with the current test, then perf at CFG 5/6.
# Launch: setsid nohup bash tests/_snap_chain_fp8mma_diag.sh > chain_fp8diag.log 2>&1 < /dev/null &
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export FLASH_MLA_SM120_FP8_MMA=1
echo "=== (1) overlay + rebuild ($(date +%H:%M:%S))"
cp "$MAIN"/csrc/sm120/decode/sparse_fp8/* "$SNAP/csrc/sm120/decode/sparse_fp8/"
cp "$MAIN"/csrc/pybind.cpp "$SNAP/csrc/pybind.cpp"; cp "$MAIN"/setup.py "$SNAP/setup.py"; cp "$MAIN"/build_wsl_sm120.sh "$SNAP/build_wsl_sm120.sh"
for f in tests/test_sparse_decode_fp8mma.py tests/_diag_sparse_fp8mma_bench.py tests/_sass_loop_stats.py tests/_sass_loop_stats.sh; do cp "$MAIN/$f" "$SNAP/tests/"; done
bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build_fp8d.log" 2>&1; RC=$?
grep -E ' error|error:' "$SNAP/build_fp8d.log" | grep -v FutureWarning | head -20; echo "REBUILD_RC=$RC ($(date +%H:%M:%S))"
if [ "$RC" -ne 0 ]; then echo "CHAIN_ABORT_BUILD_FAILED"; exit 1; fi
SO="$(ls "$SNAP"/flash_mla/cuda_sm120*.so | head -1)"
echo "=== (2) ptxas -res-usage (fp8mma)"
"$CUDA_HOME/bin/cuobjdump" -res-usage "$SO" 2>/dev/null | grep -A1 "fp8mma" | grep -E "REG" | sed 's/^/  /' | head -8
echo "=== (3) SASS stats fp8mma<false,false>"
bash tests/_sass_loop_stats.sh "fp8mma_kernelILb0ELb0E" "$SO" 2>&1 | grep -v FutureWarning | head -60
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()" 2>/dev/null
echo "=== (4) timing diag ($(date +%H:%M:%S))"
for cfg in 5 6; do
  export FLASH_MLA_SM120_SPARSE_DECODE_CFG="$cfg" FLASH_MLA_SM120_SPARSE_DECODE_DIAG_TIMING=1
  "$PY" tests/_diag_sparse_fp8mma_bench.py 2>&1 | grep -E "FP8MMA-TIMING|BENCH|Error|error" | awk '!seen[$0]++' | head -12
done
unset FLASH_MLA_SM120_SPARSE_DECODE_DIAG_TIMING
echo "=== (5) cert quick CFG=5 ($(date +%H:%M:%S))"
export FLASH_MLA_SM120_SPARSE_DECODE_CFG=5
"$PY" tests/test_sparse_decode_fp8mma.py quick 2>&1 | grep -E "^\[RESULT\]|^\[FAILED\]|\(4a\)|\(4b\)|bound|scale dev" | tail -60
echo "=== (6) perf ($(date +%H:%M:%S))"
for cfg in 5 6; do
  export FLASH_MLA_SM120_SPARSE_DECODE_CFG="$cfg"
  "$PY" tests/_diag_sparse_fp8mma_bench.py 2>&1 | grep -E "^\[BENCH\]|Error|error"
done
echo "CHAIN_DONE ($(date +%H:%M:%S))"

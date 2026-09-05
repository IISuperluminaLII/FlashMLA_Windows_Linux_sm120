#!/usr/bin/env bash
# Main-tree rebuild WITH the fp8 tensor-math opt-in (clean rebuild: the flag is part of the
# toolchain stamp), then smoke: import + flag, quick fp8 certification at CFG=6 and the default
# 34-case sparse-decode regression, all against the MAIN tree's .so (no PYTHONPATH override).
# Launch: setsid nohup bash tests/_build_main_fp8.sh > build_main_fp8.log 2>&1 < /dev/null &
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
cd "$MAIN"
source "$MAIN/env_sm120.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export FLASH_MLA_SM120_FP8_MMA=1
echo "=== main-tree build fp8_mma=1 ($(date +%H:%M:%S))"
bash "$MAIN/build_wsl_sm120.sh" > "$MAIN/build_main_fp8_inner.log" 2>&1; RC=$?
grep -E ' error|error:|BUILD_STALE' "$MAIN/build_main_fp8_inner.log" | grep -v FutureWarning | head -10; tail -n 3 "$MAIN/build_main_fp8_inner.log"
echo "MAIN_BUILD_RC=$RC ($(date +%H:%M:%S))"
if [ "$RC" -ne 0 ]; then echo "MAIN_BUILD_FAILED"; exit 1; fi
unset PYTHONPATH
cd "$MAIN/tests"
"$PY" -c "import flash_mla.cuda_sm120 as e; print('EXT_FROM', e.__file__, 'fp8_mma=', e.__sm120_fp8_mma__)" 2>&1 | grep EXT_FROM
export FLASH_MLA_SM120_SPARSE_DECODE_CFG=6
"$PY" test_sparse_decode_fp8mma.py quick 2>&1 | grep -E '^\[RESULT\]|^\[ENV\]' | sed 's/^/  cert cfg=6 (main): /'
unset FLASH_MLA_SM120_SPARSE_DECODE_CFG
"$PY" _diag_sparse_decode.py 2>&1 | grep -E '^\[RESULT\]' | sed 's/^/  default sparse-decode-34 (main): /'
"$PY" _diag_sparse_decode_ext.py 2>&1 | grep -E '^\[RESULT\]' | sed 's/^/  default sparse-decode-ext-5 (main): /'
echo "MAIN_SMOKE_DONE ($(date +%H:%M:%S))"

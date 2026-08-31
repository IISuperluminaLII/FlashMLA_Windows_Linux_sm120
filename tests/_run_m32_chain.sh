#!/usr/bin/env bash
# M32 single-pass tier certification (dense CFG=4):
#   1) small-head bench at CFG=4 -- M32 fires on h22/h32/h16-sq2 (RED-GREEN:
#      these shapes never ran M32), M16 keeps h16-sq1, BM=64 keeps h64
#   2) same bench at CFG=3 -- the A/B reference (same shapes on BM=64 at the
#      SAME raised 188-part schedule, so the delta isolates the M32 kernel)
#   3) dense 16-case at CFG=4 (h_q=128 -> M16/M32 inert; gate regression)
#   4) default regression (env unset -> legacy WMMA)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
FAIL=0

echo "== [0] rebuild =="
if ! bash build_wsl_sm120.sh > /tmp/_m32_build.log 2>&1; then
    echo "BUILD_FAILED"
    tail -60 /tmp/_m32_build.log
    exit 1
fi
echo "WSL_BUILD_DONE"
grep -E "dense_decode_mma_m32|registers|spill" /tmp/_m32_build.log | grep -B1 -A2 "m32" | head -20
ls -l flash_mla/cuda_sm120.cpython-312-x86_64-linux-gnu.so

stage() {
    local name="$1"; shift
    echo "== ${name} =="
    "$@" > /tmp/_m32_stage.log 2>&1
    local rc=$?
    grep -E "\[CASE\]|\[FAILED\]|\[RESULT\]|ms," /tmp/_m32_stage.log
    if [ $rc -ne 0 ]; then
        echo "STAGE_FAILED: ${name} (exit ${rc})"
        FAIL=1
    fi
}

stage "[1] small-head bench at CFG=4 (M32 fires on h22/h32/h16-sq2)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=4 "$PY" tests/_diag_dense_m16_bench.py
stage "[2] small-head bench at CFG=3 (BM=64 reference, same 188-part schedule)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=3 "$PY" tests/_diag_dense_m16_bench.py
stage "[3] dense 16-case at CFG=4 (M16/M32 inert, gate regression)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=4 "$PY" tests/_diag_dense_decode_perf.py
stage "[4] default regression (env unset, legacy WMMA)" \
    "$PY" tests/_diag_dense_decode_perf.py

if [ $FAIL -ne 0 ]; then
    echo "M32_CHAIN_RESULT: FAILED"
    exit 1
fi
echo "M32_CHAIN_RESULT: PASS"

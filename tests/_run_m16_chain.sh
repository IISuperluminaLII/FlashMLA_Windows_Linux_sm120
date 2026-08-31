#!/usr/bin/env bash
# M16 dense-decode tier certification chain (hardened: per-stage exit-code
# propagation + [FAILED] lines KEPT -- the first chain's grep filter dropped
# the exception text and its exit code masked a red suite):
#   0) full rebuild
#   1) CFG=2 small-head A/B (M16 fires on the h16 rows)
#   2) CFG=1 reference on the same shapes (BM=64 tier)
#   3) CFG=2 dense 16-case (h_q=128 -> M16 inert; gate regression vs CFG=1)
#   4) default regression (env unset -> legacy WMMA)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
FAIL=0

echo "== [0] rebuild =="
if ! bash build_wsl_sm120.sh > /tmp/_m16_build.log 2>&1; then
    echo "BUILD_FAILED"
    tail -40 /tmp/_m16_build.log
    exit 1
fi
echo "WSL_BUILD_DONE"
ls -l flash_mla/cuda_sm120.cpython-312-x86_64-linux-gnu.so

stage() {
    local name="$1"; shift
    echo "== ${name} =="
    "$@" > /tmp/_m16_stage.log 2>&1
    local rc=$?
    grep -E "\[CASE\]|\[FAILED\]|\[RESULT\]|ms," /tmp/_m16_stage.log
    if [ $rc -ne 0 ]; then
        echo "STAGE_FAILED: ${name} (exit ${rc})"
        FAIL=1
    fi
}

stage "[1] M16 A/B: CFG=2 (M16 fires on h16)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=2 "$PY" tests/_diag_dense_m16_bench.py
stage "[2] CFG=1 reference (BM=64 on same shapes)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=1 "$PY" tests/_diag_dense_m16_bench.py
stage "[3] dense 16-case at CFG=2 (M16 inert, gate regression)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=2 "$PY" tests/_diag_dense_decode_perf.py
stage "[4] default regression (env unset, legacy WMMA)" \
    "$PY" tests/_diag_dense_decode_perf.py

if [ $FAIL -ne 0 ]; then
    echo "M16_CHAIN_RESULT: FAILED"
    exit 1
fi
echo "M16_CHAIN_RESULT: PASS"

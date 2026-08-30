#!/usr/bin/env bash
# M32 multi-band certification (dense CFG=5: every shape > 16 rows runs as
# single-pass 32-row bands):
#   1) dense 16-case at CFG=5 -- h128 rows decompose to 4/8 bands (RED-GREEN
#      for the row0 generalization: multi-band causal fold, band-partial rows,
#      band outputs through finals AND 47/23-part split schedules)
#   2) small-head bench at CFG=5 -- h64 becomes M32 x 2 bands (A/B vs its
#      CFG=3/4 BM=64 number 0.505 ms); h16/h22/h32 unchanged tiers
#   3) dense 16-case at CFG=4 -- gate regression (must match CFG=3: M32 gate
#      still qsph <= 32 there)
#   4) default regression (env unset)
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
FAIL=0

echo "== [0] rebuild =="
if ! bash build_wsl_sm120.sh > /tmp/_m32mb_build.log 2>&1; then
    echo "BUILD_FAILED"
    tail -60 /tmp/_m32mb_build.log
    exit 1
fi
echo "WSL_BUILD_DONE"
grep -A3 "dense_decode_mma_m32_kernel" /tmp/_m32mb_build.log | grep -E "spill|registers" | head -4
ls -l flash_mla/cuda_sm120.cpython-312-x86_64-linux-gnu.so

stage() {
    local name="$1"; shift
    echo "== ${name} =="
    "$@" > /tmp/_m32mb_stage.log 2>&1
    local rc=$?
    grep -E "\[CASE\]|\[FAILED\]|\[RESULT\]|ms," /tmp/_m32mb_stage.log
    if [ $rc -ne 0 ]; then
        echo "STAGE_FAILED: ${name} (exit ${rc})"
        FAIL=1
    fi
}

stage "[1] dense 16-case at CFG=5 (multi-band M32 on h128 rows)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=5 "$PY" tests/_diag_dense_decode_perf.py
stage "[2] small-head bench at CFG=5 (h64 -> M32 x 2 bands)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=5 "$PY" tests/_diag_dense_m16_bench.py
stage "[3] dense 16-case at CFG=4 (gate regression, BM=64 tier intact)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=4 "$PY" tests/_diag_dense_decode_perf.py
stage "[4] default regression (env unset, legacy WMMA)" \
    "$PY" tests/_diag_dense_decode_perf.py

if [ $FAIL -ne 0 ]; then
    echo "M32MB_CHAIN_RESULT: FAILED"
    exit 1
fi
echo "M32MB_CHAIN_RESULT: PASS"

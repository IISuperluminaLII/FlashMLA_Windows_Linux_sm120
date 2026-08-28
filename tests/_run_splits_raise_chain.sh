#!/usr/bin/env bash
# Split-cap raise certification (dense CFG=3, sparse CFG=4: cap 64 -> 192,
# combine 128/192 tiers). Correctness runs THROUGH the raised schedules
# (94/188-way splits + the new combine instantiations + degenerate rows), then
# perf A/B, then default regressions. Per-stage exit propagation; [FAILED]
# lines kept.
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
FAIL=0

echo "== [0] rebuild =="
if ! bash build_wsl_sm120.sh > /tmp/_splits_build.log 2>&1; then
    echo "BUILD_FAILED"
    tail -40 /tmp/_splits_build.log
    exit 1
fi
echo "WSL_BUILD_DONE"
ls -l flash_mla/cuda_sm120.cpython-312-x86_64-linux-gnu.so

stage() {
    local name="$1"; shift
    echo "== ${name} =="
    "$@" > /tmp/_splits_stage.log 2>&1
    local rc=$?
    grep -E "\[CASE\]|\[FAILED\]|\[RESULT\]|ms," /tmp/_splits_stage.log
    if [ $rc -ne 0 ]; then
        echo "STAGE_FAILED: ${name} (exit ${rc})"
        FAIL=1
    fi
}

stage "[1] dense 16-case at CFG=3 (94-way splits, combine 128-tier)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=3 "$PY" tests/_diag_dense_decode_perf.py
stage "[2] dense small-head bench at CFG=3 (188 parts, combine 192-tier)" \
    env FLASH_MLA_SM120_DENSE_DECODE_CFG=3 "$PY" tests/_diag_dense_m16_bench.py
stage "[3] sparse 34-case at CFG=4 (raised split-KV)" \
    env FLASH_MLA_SM120_SPARSE_DECODE_CFG=4 "$PY" tests/_diag_sparse_decode.py
stage "[4] sparse ext-5 at CFG=4" \
    env FLASH_MLA_SM120_SPARSE_DECODE_CFG=4 "$PY" tests/_diag_sparse_decode_ext.py
stage "[5] sparse bench+serving at CFG=4" \
    env FLASH_MLA_SM120_SPARSE_DECODE_CFG=4 "$PY" tests/_diag_sparse_splitkv_bench.py
stage "[6] dense default regression (env unset)" \
    "$PY" tests/_diag_dense_decode_perf.py
stage "[7] sparse default regression (env unset)" \
    "$PY" tests/_diag_sparse_decode.py

if [ $FAIL -ne 0 ]; then
    echo "SPLITS_RAISE_RESULT: FAILED"
    exit 1
fi
echo "SPLITS_RAISE_RESULT: PASS"

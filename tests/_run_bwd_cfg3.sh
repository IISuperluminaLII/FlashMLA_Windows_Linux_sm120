#!/usr/bin/env bash
# Bwd CFG=3 (lane-paired red.global.add.v4.f32 scatter) certification:
#   0) rebuild (ninja header deps + staleness guard)
#   1) 6/6 torch-autograd oracle at CFG=3 (discriminates any pairing/col error
#      in the fused 16B payloads directly -- wrong lanes -> wrong dK/dV)
#   2) A/B fwd+bwd on the SDPA comparison row: CFG=2 (red.v2) vs CFG=3 (red.v4)
#   3) default regression (env unset)
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
FAIL=0

echo "== [0] rebuild =="
if ! bash build_wsl_sm120.sh > /tmp/_bwd3_build.log 2>&1; then
    echo "BUILD_FAILED"
    tail -60 /tmp/_bwd3_build.log
    exit 1
fi
echo "WSL_BUILD_DONE"
grep -B4 "Used .* registers" /tmp/_bwd3_build.log | grep -A4 "bwd_mma_kernel" | grep -E "spill|registers" | head -6

echo "== [1] bwd oracle 6 at CFG=3 (red.v4 lane-paired scatter) =="
FLASH_MLA_SM120_SPARSE_BWD_CFG=3 "$PY" tests/_diag_sparse_bwd_redgreen.py > /tmp/_bwd3_s1.log 2>&1
RC=$?
tail -1 /tmp/_bwd3_s1.log
if [ $RC -ne 0 ]; then echo "STAGE_FAILED: oracle CFG=3"; FAIL=1; fi

echo "== [2a] A/B fwd+bwd: CFG=2 (red.v2) =="
FLASH_MLA_SM120_SPARSE_FWD_CFG=4 FLASH_MLA_SM120_SPARSE_BWD_CFG=2 FLASH_MLA_SM120_SPARSE_DECODE_CFG=1 \
  "$PY" tests/bench_sdpa_comparison.py 2>&1 | grep -E "fwd\+bwd s_q" || { echo "STAGE_FAILED: bench CFG=2"; FAIL=1; }
echo "== [2b] A/B fwd+bwd: CFG=3 (red.v4) =="
FLASH_MLA_SM120_SPARSE_FWD_CFG=4 FLASH_MLA_SM120_SPARSE_BWD_CFG=3 FLASH_MLA_SM120_SPARSE_DECODE_CFG=1 \
  "$PY" tests/bench_sdpa_comparison.py 2>&1 | grep -E "fwd\+bwd s_q" || { echo "STAGE_FAILED: bench CFG=3"; FAIL=1; }

echo "== [3] bwd default regression (env unset) =="
"$PY" tests/_diag_sparse_bwd_redgreen.py > /tmp/_bwd3_s3.log 2>&1
RC=$?
tail -1 /tmp/_bwd3_s3.log
if [ $RC -ne 0 ]; then echo "STAGE_FAILED: default regression"; FAIL=1; fi

if [ $FAIL -ne 0 ]; then
    echo "BWD_CFG3_RESULT: FAILED"
    exit 1
fi
echo "BWD_CFG3_RESULT: PASS"

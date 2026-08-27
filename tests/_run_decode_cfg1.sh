#!/usr/bin/env bash
# Decode CFG=1 (mma.sync) verification + bench. Targets: 34/34 + 5/5 correctness, then
# beat the gather+SDPA decode baseline (2.11 ms at b=128, s_q=2, topk=2048; WMMA was
# 3.21 ms / 45-46 TFLOPS). Also: default-path (env unset) decode regression stays green.
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

export FLASH_MLA_SM120_SPARSE_DECODE_CFG=1
echo "== decode correctness 34 at CFG=1 (mma.sync) =="
"$PY" tests/_diag_sparse_decode.py 2>&1 | tail -1
echo "== decode extended 5 at CFG=1 =="
"$PY" tests/_diag_sparse_decode_ext.py 2>&1 | tail -1
echo "== decode perf CFG=1 (WMMA was 2.16 ms s_q=1 / 3.21 ms s_q=2) =="
"$PY" tests/_diag_sparse_decode_perf.py 2>&1 | grep -E "Decode|RESULT" | head -12
echo "== SDPA head-to-head decode (target: beat 2.11 ms) =="
"$PY" tests/bench_sdpa_comparison.py 2>&1 | grep -E "decode|==" | head -8

unset FLASH_MLA_SM120_SPARSE_DECODE_CFG
echo "== decode default-path regression (env unset, WMMA) =="
"$PY" tests/_diag_sparse_decode.py 2>&1 | tail -1
echo "DECODE_CFG1_DONE"

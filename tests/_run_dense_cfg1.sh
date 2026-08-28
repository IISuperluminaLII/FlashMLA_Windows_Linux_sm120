#!/usr/bin/env bash
# Dense decode CFG=1 (mma.sync + authors' split-KV): 16/16 correctness at CFG=1,
# perf (GB/s -- legacy era was 15-24 GB/s vs ~1040 GB/s compute-bound effective
# ceiling at h_q=128), then default-path regression (env unset, legacy WMMA).
# NOTE: get_mla_metadata AND the kernel read the same env -- both run inside each
# test process, so the env must be set for the WHOLE process (it is, via export).
set -u
PY=/home/shashankm/miniconda3/envs/150BLLM/bin/python
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

export FLASH_MLA_SM120_DENSE_DECODE_CFG=1
echo "== dense decode correctness+perf 16 at CFG=1 (mma + split-KV) =="
"$PY" tests/_diag_dense_decode_perf.py 2>&1 | grep -E "GB/s|TFLOPS|RESULT" | tail -20

unset FLASH_MLA_SM120_DENSE_DECODE_CFG
echo "== dense decode default-path regression (env unset, legacy) =="
"$PY" tests/_diag_dense_decode_perf.py 2>&1 | tail -1
echo "DENSE_CFG1_DONE"

#!/usr/bin/env bash
# nvcc-13.0-vs-12.9 codegen A/B on IDENTICAL source: re-sync the cu13 copy (picks up the
# full CFG ladder), rebuild with nvcc 13.0 + torch cu130 env, run the same perf sweeps.
set -e
bash /mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA/build_cu13_sm120.sh

PY=/home/shashankm/miniconda3/envs/150BLLM_cu13/bin/python
cd "$HOME/flashmla_cu13"
export PYTHONPATH="$HOME/flashmla_cu13"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "== cu13 decode perf (12.9 was: 34 / 45-46 TFLOPS) =="
"$PY" tests/_diag_sparse_decode_perf.py 2>&1 | grep -E "ms,|RESULT"
echo "== cu13 prefill perf CFG=3 (12.9 was 20.9-21.4 TFlops) =="
export FLASH_MLA_SM120_SPARSE_FWD_CFG=3
"$PY" tests/_diag_sparse_perf.py 2>&1 | grep -E "Prefill:|RESULT"
echo "== cu13 correctness: decode 34 + prefill 24 at CFG=3 =="
"$PY" tests/_diag_sparse_decode.py 2>&1 | tail -1
"$PY" tests/_diag_sparse_red.py 2>&1 | tail -1
echo "CU13_AB_DONE"

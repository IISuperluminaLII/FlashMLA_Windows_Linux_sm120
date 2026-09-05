#!/usr/bin/env bash
# Overlay the sparse-decode dir into the snapshot and ptxas-probe the sparse decode TU with the
# fp8-mma opt-in define (fast: the snapshot lives on the WSL native disk).
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"
cp "$MAIN"/csrc/sm120/decode/sparse_fp8/* "$SNAP/csrc/sm120/decode/sparse_fp8/"
cp "$MAIN"/tests/_ptxas_probe_tu.sh "$SNAP/tests/_ptxas_probe_tu.sh"
cd "$SNAP"
EXTRA_NVCC_FLAGS="-DFLASH_MLA_SM120_FP8_MMA" bash tests/_ptxas_probe_tu.sh csrc/sm120/decode/sparse_fp8/splitkv_mla.cu "fp8mma|sparse_fp8_decode_mma_kernel"

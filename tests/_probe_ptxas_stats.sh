#!/usr/bin/env bash
# ptxas -v stats for the three sm120 sparse kernels (12.9 toolchain, same flags as setup.py).
# MEASURE-FIRST gate from design-prefill-configs.md 2.1 (A0): if the prefill fwd shows lmem
# spill for rO[128], A0 (full unroll) is the single most valuable change.
# Acceptance gate from design-sparse-decode.md 7.5#2: decode kernel must show 0 spills.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
NVCC=/usr/local/cuda-12.9/bin/nvcc
FLAGS="-O3 -std=c++17 -DNDEBUG -DFLASH_MLA_DISABLE_SM90 -DFLASH_MLA_DISABLE_SM100 -DFLASH_MLA_BUILD_SM120 \
  -gencode arch=compute_120,code=sm_120 \
  -Icsrc -Icsrc/sm90 -Icsrc/cutlass/include -Icsrc/cutlass/tools/util/include"

probe() {
    local name="$1" src="$2" kpat="$3"
    echo "===== $name ====="
    $NVCC $FLAGS -Xptxas -v -c "$src" -o /tmp/ptxas_probe.o 2>&1 \
      | grep -A 3 "$kpat" | grep -E "Function|registers|spill|lmem|stack|bytes" | head -8
}

probe "sparse prefill fwd" csrc/sm120/prefill/sparse/fwd.cu       "sparse_prefill_fwd_kernel"
probe "sparse prefill bwd" csrc/sm120/prefill/sparse/bwd.cu       "sparse_prefill_bwd_kernel"
probe "sparse fp8 decode " csrc/sm120/decode/sparse_fp8/splitkv_mla.cu "sparse_fp8_decode_kernel"
echo "PTXAS_PROBE_DONE"

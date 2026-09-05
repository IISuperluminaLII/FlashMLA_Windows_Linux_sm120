#!/usr/bin/env bash
# Print ptxas resource usage (regs/smem/spill) of the BUILT sm120 extension per kernel.
# CPU-only; reads the .so, touches no GPU.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
SO="$(ls flash_mla/cuda_sm120*.so | head -1)"
CUOBJ="${CUDA_HOME}/bin/cuobjdump"
echo "SO=$SO"
"$CUOBJ" -res-usage "$SO" 2>/dev/null | grep -A1 -E "fmha_fwd_mma_kernel|fmha_bwd_mma_dkdv|fmha_bwd_mma_dq|bwd_mma_delta|dense_decode_mma|sparse.*mma|splitkv" | grep -E "Function|REG|SHARED|SPILL" | sed 's/Function \(.*\):/FN \1/' | cut -c1-200
echo "RES_USAGE_DONE"

#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
OUT=/tmp/_probe_warpspec_ptx.log
"$CUDA_HOME/bin/nvcc" -arch=sm_120 -O3 -std=c++17 -Xptxas -v -cubin -o /tmp/_probe_warpspec.cubin tests/_probe_warpspec_ptx.cu > $OUT 2>&1
RC=$?
grep -E 'error|Used|spill' $OUT | head -20
echo "PROBE_RC=$RC (0 = every primitive ACCEPTED on plain sm_120)"

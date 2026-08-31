#!/usr/bin/env bash
# Register/SASS-level tracing of the compiled extension (operator directive:
# "see exactly what regs are being hit"):
#   1) per-kernel REG/SMEM/stack from the shipped cubin (cuobjdump -res-usage)
#   2) spill-instruction scan (LDL/STL in SASS) per new kernel
#   3) the crossover kernel's exchange-loop SASS (st.async/mapa/mbarrier encodings)
#   4) profiler availability (ncu) for stall-level tracing
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
SO=flash_mla/cuda_sm120.cpython-312-x86_64-linux-gnu.so
CUOBJ=/usr/local/cuda-13.0/bin/cuobjdump

echo "== [1] per-kernel resource usage =="
"$CUOBJ" -res-usage "$SO" 2>/dev/null \
  | grep -A6 -E "Function .*(dense_decode_mma|sparse_fp8_decode_mma|sparse_bwd_mma)" \
  | grep -E "Function|REG|SMEM|STACK|LOCAL" | sed 's/Function :/\nFunction:/'

echo ""
echo "== [2] spill-op scan (LDL/STL occurrences inside each kernel body) =="
"$CUOBJ" -sass "$SO" 2>/dev/null > /tmp/_sass_full.txt
for K in dense_decode_mma_kernel dense_decode_mma_m16_kernel \
         sparse_fp8_decode_mma_kernel sparse_fp8_decode_mma_splitkv_kernel \
         sparse_fp8_decode_mma_splitkv_x_kernel; do
    CNT=$(awk "/Function : .*${K}/{f=1} f&&/^[[:space:]]*Function :/&&!/${K}/{f=0} f" /tmp/_sass_full.txt \
          | grep -cE "[[:space:]](LDL|STL)[. ]" || true)
    echo "  $K : $CNT spill ops"
done

echo ""
echo "== [3] crossover exchange-loop SASS (DSM/mbarrier instruction encodings) =="
awk "/Function : .*splitkv_x_kernel/{f=1} f" /tmp/_sass_full.txt \
  | grep -E "MAPA|STAS|ST\.ASYNC|SYNCS|BMOV|MBAR|ELECT|R2UR" | sort | uniq -c | sort -rn | head -15

echo ""
echo "== [4] mma/ldsm density per kernel (issue-mix sanity) =="
for K in dense_decode_mma_kernel dense_decode_mma_m16_kernel sparse_fp8_decode_mma_splitkv_x_kernel; do
    BODY=$(awk "/Function : .*${K}/{f=1} f&&/^[[:space:]]*Function :/&&!/${K}/{f=0} f" /tmp/_sass_full.txt)
    HMMA=$(printf "%s" "$BODY" | grep -c "HMMA" || true)
    LDSM=$(printf "%s" "$BODY" | grep -c "LDSM" || true)
    TOTAL=$(printf "%s" "$BODY" | grep -cE "^[[:space:]]+/\*[0-9a-f]+\*/" || true)
    echo "  $K : HMMA=$HMMA LDSM=$LDSM total-instrs=$TOTAL"
done

echo ""
echo "== [5] profiler availability =="
(command -v ncu && ncu --version 2>&1 | head -2) || echo "NCU_NOT_FOUND_IN_PATH"
ls /usr/local/cuda-13.0/bin/ 2>/dev/null | grep -iE "ncu|nsight" || true
echo "TRACE_ARSENAL_DONE"

#!/usr/bin/env bash
# RED proof for tests/test_sparse_decode_fp8mma.py: two deliberate kernel mutations, applied ONLY to
# the snapshot's copy of splitkv_mla_fp8mma.cuh (the main tree is never touched), each rebuilt and
# run through the quick certification at CFG=5 -- the test MUST report failures for both:
#   m1  odd d-column B fragments built from the even selector (0x7531 -> 0x6420): a wrong-element
#       class -> assertions (2)/(3) (cos class) and (4a) (per-element e4m3 bound).
#   m2  P' scale E inflated by 1% on the row-a half without the matching F: a systematic scale
#       error the contract's cancellation forbids -> assertion (4b) (per-group zero-mean test).
# Then the pristine file is restored, rebuilt, and the quick cert must pass again (GREEN).
# Launch: setsid nohup bash tests/_snap_chain_fp8mma_red.sh > chain_fp8red.log 2>&1 < /dev/null &
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export FLASH_MLA_SM120_FP8_MMA=1 FLASH_MLA_SM120_SPARSE_DECODE_CFG=5
K="$SNAP/csrc/sm120/decode/sparse_fp8/splitkv_mla_fp8mma.cuh"
cp "$MAIN/csrc/sm120/decode/sparse_fp8/splitkv_mla_fp8mma.cuh" "$K"
cp "$MAIN/tests/test_sparse_decode_fp8mma.py" "$SNAP/tests/"
cp "$K" "$K.pristine"
run_cert() {   # $1 = label, $2 = expected (FAIL|PASS)
  bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build_red_$1.log" 2>&1; RC=$?
  if [ "$RC" -ne 0 ]; then echo "[FAILED] build $1 rc=$RC"; grep -E ' error|error:' "$SNAP/build_red_$1.log" | head -5; return 1; fi
  local last; last="$("$PY" tests/test_sparse_decode_fp8mma.py quick 2>&1 | grep -E '^\[RESULT\]' | tail -1)"
  echo "  $1: $last"
  if [ "$2" = FAIL ]; then
    if printf "%s" "$last" | grep -q " 0 failed"; then echo "[FAILED] RED-PROOF $1: the test did NOT catch the mutation"; return 1; fi
    echo "[OK] RED $1: mutation caught"
  else
    if printf "%s" "$last" | grep -q " 0 failed"; then echo "[OK] GREEN $1"; else echo "[FAILED] GREEN $1"; return 1; fi
  fi
}
FAILED=0
echo "=== m1: odd-column selector -> even ($(date +%H:%M:%S))"
cp "$K.pristine" "$K"
grep -c "0x7531u" "$K"
sed -i 's/f_prmt(R\[0\], R\[1\], 0x7531u), f_prmt(R\[2\], R\[3\], 0x7531u)/f_prmt(R[0], R[1], 0x6420u), f_prmt(R[2], R[3], 0x6420u)/' "$K"
grep -q "0x7531u" "$K" && { echo "[FAILED] m1 sed did not apply"; FAILED=1; } || run_cert m1 FAIL || FAILED=1
echo "=== m2: +1% E on row a ($(date +%H:%M:%S))"
cp "$K.pristine" "$K"
sed -i 's/E\[g\]\[0\] = (nm0 == -INFINITY) ? 0.f : F_PSCALE \* __expf(nmax0 - nm0);/E[g][0] = (nm0 == -INFINITY) ? 0.f : F_PSCALE * 1.01f * __expf(nmax0 - nm0);/' "$K"
grep -q "1.01f" "$K" || { echo "[FAILED] m2 sed did not apply"; FAILED=1; }
grep -q "1.01f" "$K" && { run_cert m2 FAIL || FAILED=1; }
echo "=== restore pristine + GREEN ($(date +%H:%M:%S))"
cp "$K.pristine" "$K"
run_cert green PASS || FAILED=1
echo "FP8MMA_RED_RESULT: $([ $FAILED -eq 0 ] && echo PASS || echo FAILED)"
echo "CHAIN_DONE ($(date +%H:%M:%S))"

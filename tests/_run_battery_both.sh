#!/usr/bin/env bash
# Certify BOTH modes in one shot:
#   1) default -- every CFG env unset -> legacy paths, byte-identical contract
#   2) all-mma -- every CFG ladder at its top tier:
#        FLASH_MLA_SM120_SPARSE_FWD_CFG=4    (sparse prefill fwd mma.sync)
#        FLASH_MLA_SM120_SPARSE_DECODE_CFG=4 (fp8 decode splitkv + 192-cap;
#                                             exercises the raised combine)
#        FLASH_MLA_SM120_SPARSE_BWD_CFG=2    (sparse bwd mma + red.v2 scatter)
#        FLASH_MLA_SM120_DENSE_DECODE_CFG=4  (dense splitkv + M16 + M32 +
#                                             192-cap; the dispatch gates fan
#                                             the battery shapes across ALL
#                                             mma tiers)
# Any suite failure propagates (run_battery.sh exits nonzero).
# Usage: _run_battery_both.sh [python-path]
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
PY="${1:-$PY}"
cd "$FMLA_ROOT"
unset FLASH_MLA_SM120_SPARSE_FWD_CFG FLASH_MLA_SM120_SPARSE_DECODE_CFG \
      FLASH_MLA_SM120_SPARSE_BWD_CFG FLASH_MLA_SM120_DENSE_DECODE_CFG 2>/dev/null || true
echo "=== BATTERY default (legacy paths) ==="
bash tests/run_battery.sh "$PY" || exit 1
echo "=== BATTERY all-mma tiers (FWD=4 DECODE=4 BWD=2 DENSE=4 FWD_MMA=11 BWD_MMA=8) ==="
export FLASH_MLA_SM120_SPARSE_FWD_CFG=4
export FLASH_MLA_SM120_SPARSE_DECODE_CFG=4
export FLASH_MLA_SM120_SPARSE_BWD_CFG=2
export FLASH_MLA_SM120_DENSE_DECODE_CFG=4
export FLASH_MLA_SM120_FWD_MMA_CFG=11   # dense prefill fwd ladder recommended rung (2026-09-05: rung 11 kv128 TMA)
export FLASH_MLA_SM120_BWD_MMA_CFG=8    # dense prefill bwd: FROST structure + fragment-order dQ v4 + issue-after-B1 (2026-09-05)
bash tests/run_battery.sh "$PY" || exit 1
# 3) FP8 tensor-math sparse decode tiers (SPARSE_DECODE_CFG=5/6) -- only when the extension was
#    built with FLASH_MLA_SM120_FP8_MMA=1 (their own certification test: the authors' oracle
#    tolerance is bf16-math-class by construction). A build without the flag is a documented
#    [SKIP], not a failure.
HAS_FP8="$("$PY" -c 'import flash_mla.cuda_sm120 as e; print(int(bool(getattr(e, "__sm120_fp8_mma__", False))))' 2>/dev/null | tail -1)"
if [ "$HAS_FP8" = "1" ]; then
  echo "=== FP8 tensor-math tiers (SPARSE_DECODE_CFG=5 / 6, quick certification) ==="
  for cfg in 5 6; do
    export FLASH_MLA_SM120_SPARSE_DECODE_CFG="$cfg"
    last="$(cd tests && "$PY" test_sparse_decode_fp8mma.py quick 2>&1 | grep -E '^\[RESULT\]' | tail -1)"
    if printf "%s" "$last" | grep -q " 0 failed"; then echo "[OK] fp8mma cfg=$cfg: $last"; else echo "[FAILED] fp8mma cfg=$cfg: ${last:-<no RESULT line>}"; exit 1; fi
  done
  unset FLASH_MLA_SM120_SPARSE_DECODE_CFG
else
  echo "[SKIP] FP8 tensor-math tiers: extension built without FLASH_MLA_SM120_FP8_MMA=1"
fi
echo "BATTERY_BOTH_DONE"

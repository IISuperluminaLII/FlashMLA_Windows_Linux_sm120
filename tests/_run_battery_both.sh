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
PY="${1:-/home/shashankm/miniconda3/envs/150BLLM/bin/python}"
cd "$(dirname "$0")/.."
unset FLASH_MLA_SM120_SPARSE_FWD_CFG FLASH_MLA_SM120_SPARSE_DECODE_CFG \
      FLASH_MLA_SM120_SPARSE_BWD_CFG FLASH_MLA_SM120_DENSE_DECODE_CFG 2>/dev/null || true
echo "=== BATTERY default (legacy paths) ==="
bash tests/run_battery.sh "$PY" || exit 1
echo "=== BATTERY all-mma tiers (FWD=4 DECODE=4 BWD=2 DENSE=4) ==="
export FLASH_MLA_SM120_SPARSE_FWD_CFG=4
export FLASH_MLA_SM120_SPARSE_DECODE_CFG=4
export FLASH_MLA_SM120_SPARSE_BWD_CFG=2
export FLASH_MLA_SM120_DENSE_DECODE_CFG=4
bash tests/run_battery.sh "$PY" || exit 1
echo "BATTERY_BOTH_DONE"

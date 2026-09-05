#!/usr/bin/env bash
# Final tier-set certification in the snapshot: dual-mode battery (default + all-mma with the given fwd/bwd
# tiers) and the model-level cert (parity tiers + grad equivalence) from the 150BLLM side.
# Usage: setsid nohup bash tests/_snap_chain_final_cert.sh [snap] [fwd_tier] [bwd_tier] > chain_final.log 2>&1 < /dev/null &
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
REPO=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM
SNAP="${1:-$HOME/flashmla_snap}"; FWD="${2:-11}"; BWD="${3:-6}"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
sed -e "s/^export FLASH_MLA_SM120_FWD_MMA_CFG=[0-9]*/export FLASH_MLA_SM120_FWD_MMA_CFG=$FWD/" -e "s/^export FLASH_MLA_SM120_BWD_MMA_CFG=[0-9]*/export FLASH_MLA_SM120_BWD_MMA_CFG=$BWD/" "$MAIN/tests/_run_battery_both.sh" > "$SNAP/tests/_run_battery_both.sh"
echo "=== (1) battery both modes (all-mma fwd=$FWD bwd=$BWD) ($(date +%H:%M:%S))"
bash "$SNAP/tests/_run_battery_both.sh" 2>&1 | grep -E 'BATTERY|RESULT|FAILED|SUITE'
echo "=== (2) model-level cert fwd=$FWD bwd=$BWD ($(date +%H:%M:%S))"
cd "$REPO" && FLASH_MLA_SM120_FWD_MMA_CFG=$FWD FLASH_MLA_SM120_BWD_MMA_CFG=$BWD bash bench/_run_tier_cert.sh 2>&1 | grep -E 'EXT_FROM|OK|FAILED|PASS|RESULT|G1|G2|G3|TIER|rror'
echo "FINAL_CERT_DONE ($(date +%H:%M:%S))"

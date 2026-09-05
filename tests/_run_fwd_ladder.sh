#!/usr/bin/env bash
# Closed loop for the dense prefill forward CFG ladder (FLASH_MLA_SM120_FWD_MMA_CFG 0..4):
#   per rung: correctness (tests/test_accuracy_vs_sdpa.py 44 + tests/test_adversarial_accuracy.py
#   126, script-style "[RESULT] N passed, M failed" -- any failure aborts) then the fwd A/B
#   (tests/bench_fwd_vs_sdpa.py 128/128; tests/bench_sdpa_comparison.py fwd rows 192/128).
# Usage: bash tests/_run_fwd_ladder.sh [rungs...]   (default: 0 1 2 3 4)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export FLASH_MLA_SM120_FWD_MMA=1
RUNGS="${*:-0 1 2 3 4}"
FAILED=0
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()"
for cfg in $RUNGS; do
    export FLASH_MLA_SM120_FWD_MMA_CFG="$cfg"
    echo "################ FWD_CFG=$cfg ################"
    for suite in tests/test_accuracy_vs_sdpa.py tests/test_adversarial_accuracy.py; do
        last="$("$PY" "$suite" 2>&1 | grep -E "^\[RESULT\]" | tail -1)"
        if printf "%s" "$last" | grep -q " 0 failed"; then
            echo "[OK] cfg=$cfg $suite: $last"
        else
            echo "[FAILED] cfg=$cfg $suite: ${last:-<no RESULT line>}"
            "$PY" "$suite" 2>&1 | grep -E "FAIL|Error|error" | head -8
            FAILED=1
        fi
    done
    echo "----- bench fwd 128/128 vs SDPA (cfg=$cfg) -----"
    "$PY" tests/bench_fwd_vs_sdpa.py 2>&1 | grep -E "b=.*S=" | tail -6
    echo "----- bench fwd-only 192/128 + 128/128 at S=1024/4096/8192 (cfg=$cfg) -----"
    BENCH_S_LIST=1024,4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -E "^\s*\[FUSED\]" | head -6
done
echo "FWD_LADDER_RESULT: $([ $FAILED -eq 0 ] && echo PASS || echo FAILED)"
echo "FWD_LADDER_DONE"

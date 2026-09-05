#!/usr/bin/env bash
# Closed loop for the dense prefill backward ladder (FLASH_MLA_SM120_BWD_MMA_CFG 0..N):
#   per rung: correctness (tests/test_accuracy_vs_sdpa.py 44 incl. bwd, tests/test_adversarial_accuracy.py
#   126, tests/test_mla_192_128_fused_bwd.py varlen+causal at 192/128) then the A/B:
#   tests/bench_bwd_vs_sdpa.py (128/128), tests/bench_mla_192_e2e.py (192/128 fwd+bwd),
#   authors' benchmark/bench_flash_mla_training.py --causal (PYTHONSAFEPATH).
# The forward runs at FLASH_MLA_SM120_FWD_MMA_CFG (default 7 = recommended) unless overridden.
# Usage: bash tests/_run_bwd_ladder.sh [rungs...]   (default: 0 2)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export FLASH_MLA_SM120_FWD_MMA=1
export FLASH_MLA_SM120_BWD_MMA=1
export FLASH_MLA_SM120_FWD_MMA_CFG="${FLASH_MLA_SM120_FWD_MMA_CFG:-7}"
RUNGS="${*:-0 2}"
FAILED=0
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()"
for cfg in $RUNGS; do
    export FLASH_MLA_SM120_BWD_MMA_CFG="$cfg"
    echo "################ BWD_CFG=$cfg (FWD_CFG=$FLASH_MLA_SM120_FWD_MMA_CFG) ################"
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
    out="$("$PY" tests/test_mla_192_128_fused_bwd.py 2>&1)"
    rc=$?
    if [ $rc -eq 0 ] && ! printf "%s" "$out" | grep -qiE "FAIL|Traceback"; then
        echo "[OK] cfg=$cfg tests/test_mla_192_128_fused_bwd.py: $(printf "%s" "$out" | grep -iE "PASS|ALL" | tail -1)"
    else
        echo "[FAILED] cfg=$cfg tests/test_mla_192_128_fused_bwd.py (rc=$rc)"
        printf "%s\n" "$out" | grep -iE "FAIL|Error|rel" | head -8
        FAILED=1
    fi
    echo "----- bench bwd 128/128 vs SDPA (cfg=$cfg) -----"
    "$PY" tests/bench_bwd_vs_sdpa.py 2>&1 | grep -E "b=.*S=" | tail -6
    echo "----- bench 192/128 e2e fwd+bwd (cfg=$cfg) -----"
    "$PY" tests/bench_mla_192_e2e.py 2>&1 | grep -E "b=.*S=" | tail -6
    echo "----- authors' e2e training bench, causal (cfg=$cfg) -----"
    PYTHONSAFEPATH=1 "$PY" benchmark/bench_flash_mla_training.py --causal 2>&1 | grep -E "Benchmarking|FlashMLA|Torch" | tail -12
done
echo "BWD_LADDER_RESULT: $([ $FAILED -eq 0 ] && echo PASS || echo FAILED)"
echo "BWD_LADDER_DONE"

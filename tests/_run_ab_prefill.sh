#!/usr/bin/env bash
# A/B runner for the dense prefill fwd/bwd ladders. Runs, in order:
#   [1] tests/bench_fwd_vs_sdpa.py     (128/128 fwd vs cuDNN SDPA, %peak)
#   [2] tests/bench_bwd_vs_sdpa.py     (128/128 bwd vs cuDNN SDPA, %peak)
#   [3] tests/bench_mla_192_e2e.py     (192/128 model shape, fwd+bwd via autograd)
#   [4] benchmark/bench_flash_mla_training.py --causal (authors' e2e, PYTHONSAFEPATH)
# Env knobs (default = legacy rung 0): FLASH_MLA_SM120_FWD_MMA_CFG, FLASH_MLA_SM120_BWD_MMA_CFG.
#   FLASH_MLA_SM120_FWD_MMA_CFG=2 bash tests/_run_ab_prefill.sh
# Sequential (max-2-tests rule); device 0 (PCI order) = the only sm_120 part.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export FLASH_MLA_SM120_FWD_MMA=1
export FLASH_MLA_SM120_BWD_MMA=1
echo "=== A/B prefill: FWD_CFG=${FLASH_MLA_SM120_FWD_MMA_CFG:-0} BWD_CFG=${FLASH_MLA_SM120_BWD_MMA_CFG:-0} ==="
# clock warm-up (idle P8 180-435 MHz poisons the first sub-ms burst)
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()"
echo "----- [1] fwd 128/128 vs SDPA -----"
"$PY" tests/bench_fwd_vs_sdpa.py 2>&1 | grep -vE "Warning|pynvml|^\s*$" | tail -12
echo "----- [2] bwd 128/128 vs SDPA -----"
"$PY" tests/bench_bwd_vs_sdpa.py 2>&1 | grep -vE "Warning|pynvml|^\s*$" | tail -12
echo "----- [3] 192/128 e2e fwd+bwd (model shape) -----"
"$PY" tests/bench_mla_192_e2e.py 2>&1 | grep -vE "Warning|pynvml|^\s*$" | tail -12
echo "----- [4] authors' e2e training bench, causal -----"
PYTHONSAFEPATH=1 "$PY" benchmark/bench_flash_mla_training.py --causal 2>&1 | grep -E "Benchmarking|FlashMLA|Torch" | tail -12
echo "AB_PREFILL_DONE"

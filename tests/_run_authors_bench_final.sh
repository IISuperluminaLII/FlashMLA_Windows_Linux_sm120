#!/usr/bin/env bash
# The authors' training benchmark (benchmark/bench_flash_mla_training.py, b=16 h=16 head_dim 128, fwd+bwd via
# autograd vs torch SDPA) at the FINAL tiers, both masks, after the certification chain finishes.
set -u
SNAP="${1:-$HOME/flashmla_snap}"
cd "$SNAP"; export PYTHONPATH="$SNAP"; source "$SNAP/env_sm120.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONSAFEPATH=1
export FLASH_MLA_SM120_FWD_MMA=1 FLASH_MLA_SM120_BWD_MMA=1 FLASH_MLA_SM120_FWD_MMA_CFG=11 FLASH_MLA_SM120_BWD_MMA_CFG=8
for i in $(seq 1 120); do grep -q FINAL_CERT_DONE "$SNAP/chain_final8.log" 2>/dev/null && break; sleep 10; done
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()" 2>/dev/null
echo "=== authors' bench, CAUSAL, fwd 11 / bwd 8 ($(date +%H:%M:%S))"
"$PY" benchmark/bench_flash_mla_training.py --causal 2>&1 | grep -vE 'pynvml|FutureWarning' | grep -E 'seqlen|FlashMLA|Torch|ms'
echo "=== authors' bench, NON-CAUSAL, fwd 11 / bwd 8 ($(date +%H:%M:%S))"
"$PY" benchmark/bench_flash_mla_training.py 2>&1 | grep -vE 'pynvml|FutureWarning' | grep -E 'seqlen|FlashMLA|Torch|ms'
echo "=== authors' bench, CAUSAL, LEGACY tiers (fwd 0 / bwd 0) for the session delta"
FLASH_MLA_SM120_FWD_MMA_CFG=0 FLASH_MLA_SM120_BWD_MMA_CFG=0 "$PY" benchmark/bench_flash_mla_training.py --causal 2>&1 | grep -vE 'pynvml|FutureWarning' | grep -E 'seqlen|FlashMLA|Torch|ms'
echo "AUTHORS_BENCH_DONE ($(date +%H:%M:%S))"

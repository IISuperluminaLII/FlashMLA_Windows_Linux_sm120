#!/usr/bin/env bash
# CFG=4 (mma.sync sparse) verification + bench. Targets: correctness 24/24, then beat
# the gather+SDPA baseline (3.23 ms at s_q=512 -> >= ~90 TFlops effective; authors' perf
# shape reference: CFG3 was 21.4 TFlops / 109ms).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export FLASH_MLA_SM120_SPARSE_FWD_CFG=4

echo "== correctness 24 at CFG=4 (mma.sync) =="
"$PY" tests/_diag_sparse_red.py 2>&1 | tail -1
echo "== prefill perf CFG=4 (CFG3 was 20.9-21.4 TFlops) =="
"$PY" tests/_diag_sparse_perf.py 2>&1 | grep -E "Prefill:|RESULT"
echo "== SDPA head-to-head (target: beat 3.23 ms fwd / 20.0 ms fwd+bwd) =="
"$PY" tests/bench_sdpa_comparison.py 2>&1 | grep -E "sparse prefill|fwd s_q|fwd\+bwd s_q|==" | head -6
echo "CFG4_DONE"

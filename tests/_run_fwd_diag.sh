#!/usr/bin/env bash
# Forward diagnostics: (a) rung-7 kernel vs the same loop with the softmax skipped (rung 99,
# wrong numerics, measures the tensor+ldmatrix-only rate); (b) rung 5 at the large shape
# b=8 H=32 (apples-to-apples with FA2 sm80's 351 TF). Device 0, fwd-only.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_FWD_MMA=1
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()"
echo "----- rung 7, b=2 H=8 -----"
FLASH_MLA_SM120_FWD_MMA_CFG=7 BENCH_S_LIST=4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -E "^\s*\[FUSED\]"
echo "----- rung 7 loop with SOFTMAX SKIPPED (diag 99, wrong numerics), b=2 H=8 -----"
FLASH_MLA_SM120_FWD_MMA_DIAG_NOSOFTMAX=1 BENCH_S_LIST=4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -E "^\s*\[FUSED\]"
echo "----- rung 7, b=8 H=32 (FA2 sm80 reference: 351 TF at S=4096, 344 at 8192) -----"
FLASH_MLA_SM120_FWD_MMA_CFG=7 BENCH_B=8 BENCH_H=32 BENCH_S_LIST=4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -E "^\s*\[FUSED\]"
echo "----- rung 0 (legacy), b=8 H=32 -----"
FLASH_MLA_SM120_FWD_MMA_CFG=0 BENCH_B=8 BENCH_H=32 BENCH_S_LIST=4096 "$PY" tests/bench_fwd_only.py 2>&1 | grep -E "^\s*\[FUSED\]"
echo "FWD_DIAG_DONE"
# (b2) softmax-skipped loop at the large shape too (the share of the gap that is NOT tensor work)
echo "----- rung 7 loop with SOFTMAX SKIPPED, b=8 H=32 -----"
FLASH_MLA_SM120_FWD_MMA_DIAG_NOSOFTMAX=1 BENCH_B=8 BENCH_H=32 BENCH_S_LIST=4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -E "^\s*\[FUSED\]"
echo "FWD_DIAG_DONE_B2"
# (c) rung 8 (warp-specialized 128-row CTA) vs its softmax-skipped twin (diag value 2 -> kCfg 98)
echo "----- rung 8, b=8 H=32 -----"
FLASH_MLA_SM120_FWD_MMA_CFG=8 BENCH_B=8 BENCH_H=32 BENCH_S_LIST=4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -E "^\s*\[FUSED\]"
echo "----- rung 8 loop with SOFTMAX SKIPPED (diag 98), b=8 H=32 -----"
FLASH_MLA_SM120_FWD_MMA_DIAG_NOSOFTMAX=2 BENCH_B=8 BENCH_H=32 BENCH_S_LIST=4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -E "^\s*\[FUSED\]"
echo "FWD_DIAG_DONE_C"
# (d) rung 11 (kv_tile 128, single-stage split consumption on the TMA ring) vs its softmax-skipped twin (diag 3 -> 97)
echo "----- rung 11, b=8 H=32 -----"
FLASH_MLA_SM120_FWD_MMA_CFG=11 BENCH_B=8 BENCH_H=32 BENCH_S_LIST=4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -F "[FUSED]"
echo "----- rung 11 loop with SOFTMAX SKIPPED (diag 97), b=8 H=32 -----"
FLASH_MLA_SM120_FWD_MMA_DIAG_NOSOFTMAX=3 BENCH_B=8 BENCH_H=32 BENCH_S_LIST=4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -F "[FUSED]"
echo "FWD_DIAG_DONE_D"

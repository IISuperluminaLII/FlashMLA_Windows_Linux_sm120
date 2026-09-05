#!/usr/bin/env bash
# Per-phase cycle attribution of the CFG=6 backward tile (diag 94) at the three reference shapes.
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"
cd "$SNAP"; export PYTHONPATH="$SNAP"; source "$SNAP/env_sm120.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_FWD_MMA=1 FLASH_MLA_SM120_BWD_MMA=1 FLASH_MLA_SM120_FWD_MMA_CFG=11 FLASH_MLA_SM120_BWD_MMA_CFG=6
cp -v "$MAIN"/csrc/sm120/prefill/dense/fmha_bwd_mma_sm120.cuh "$MAIN"/csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu "$SNAP/csrc/sm120/prefill/dense/"
bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build_t.log" 2>&1; RC=$?; echo "REBUILD_RC=$RC ($(date +%H:%M:%S))"
[ "$RC" -ne 0 ] && { grep -E ' error' "$SNAP/build_t.log" | head; echo CHAIN_ABORT_BUILD_FAILED; exit 1; }
"$PY" tests/_patch_bwd_bench_shape.py >/dev/null 2>&1
echo "--- 128/128 b=2 H=8 (last launch of the S=4096 bench)"
FLASH_MLA_SM120_BWD_MMA_DIAG_TIMING=1 BENCH_B=2 BENCH_H=8 "$PY" tests/bench_bwd_vs_sdpa.py 2>&1 | grep -E 'BWD-TIMING' | tail -1
echo "--- 128/128 b=8 H=32"
FLASH_MLA_SM120_BWD_MMA_DIAG_TIMING=1 BENCH_B=8 BENCH_H=32 "$PY" tests/bench_bwd_vs_sdpa.py 2>&1 | grep -E 'BWD-TIMING' | tail -1
echo "--- 192/128 model shape b=2 H=8 (last launch)"
FLASH_MLA_SM120_BWD_MMA_DIAG_TIMING=1 "$PY" tests/bench_mla_192_e2e.py 2>&1 | grep -E 'BWD-TIMING' | tail -1
echo "TIMING_CHAIN_DONE"

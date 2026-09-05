#!/usr/bin/env bash
# Snapshot chain: (a) fwd rung 11 vs 12 (ladder gates + benches, bigshape 12, diag 96), (b) bwd CFG=5 vs the
# no-reduction diag 95 (bench only; wrong dQ) at b=2 H=8 and b=8 H=32.
# Launch detached: setsid nohup bash tests/_snap_chain_r12_nored.sh [snap] > chain9.log 2>&1 < /dev/null &
set -u
MAIN=/mnt/c/PyCharmProjectsSpaceConflict/150BLLM/external/FlashMLA
SNAP="${1:-$HOME/flashmla_snap}"
cd "$SNAP"
export PYTHONPATH="$SNAP"
source "$SNAP/env_sm120.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_FWD_MMA=1 FLASH_MLA_SM120_BWD_MMA=1
echo "=== (1) overlay ($(date +%H:%M:%S))"
cp -v "$MAIN"/csrc/sm120/prefill/dense/fmha_fwd_mma_sm120.cuh "$MAIN"/csrc/sm120/prefill/dense/fmha_cutlass_fwd_sm120.cu \
      "$MAIN"/csrc/sm120/prefill/dense/fmha_bwd_mma_sm120.cuh "$MAIN"/csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu "$SNAP/csrc/sm120/prefill/dense/"
cp -v "$MAIN"/tests/_run_fwd_ladder.sh "$MAIN"/tests/_run_fwd_bigshape.sh "$MAIN"/tests/_run_fwd_diag.sh "$SNAP/tests/"
echo "=== (2) rebuild ($(date +%H:%M:%S))"
bash "$SNAP/build_wsl_sm120.sh" > "$SNAP/build9.log" 2>&1; RC=$?
grep -E ' error|Error' "$SNAP/build9.log" | grep -v FutureWarning | head -8; tail -n 2 "$SNAP/build9.log"; echo "REBUILD_RC=$RC ($(date +%H:%M:%S))"
if [ "$RC" -ne 0 ]; then echo "CHAIN_ABORT_BUILD_FAILED"; exit 1; fi
"$PY" -c "import flash_mla.cuda_sm120 as e; print('EXT_FROM', e.__file__)" 2>&1 | grep EXT_FROM
echo "=== (3) FWD ladder 11 12 ($(date +%H:%M:%S))"
bash "$SNAP/tests/_run_fwd_ladder.sh" 11 12 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (4) FWD bigshape 12 ($(date +%H:%M:%S))"
RUNGS="12" bash "$SNAP/tests/_run_fwd_bigshape.sh" 2>&1 | grep -vE 'pynvml|FutureWarning'
echo "=== (5) FWD diag 96 (rung 12, softmax skipped) b=8 H=32 ($(date +%H:%M:%S))"
FLASH_MLA_SM120_FWD_MMA_DIAG_NOSOFTMAX=4 BENCH_B=8 BENCH_H=32 BENCH_S_LIST=4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -F "[FUSED]"
echo "=== (6) BWD CFG=5 vs diag NORED (bench only) ($(date +%H:%M:%S))"
export FLASH_MLA_SM120_FWD_MMA_CFG=10 FLASH_MLA_SM120_BWD_MMA_CFG=5
echo "--- CFG=5 b=2 H=8";  "$PY" tests/bench_bwd_vs_sdpa.py 2>&1 | grep -E "b=.*S=" | tail -3
echo "--- NORED b=2 H=8"; FLASH_MLA_SM120_BWD_MMA_DIAG_NORED=1 "$PY" tests/bench_bwd_vs_sdpa.py 2>&1 | grep -E "b=.*S=" | tail -3
"$PY" tests/_patch_bwd_bench_shape.py >/dev/null 2>&1
echo "--- CFG=5 b=8 H=32";  BENCH_B=8 BENCH_H=32 "$PY" tests/bench_bwd_vs_sdpa.py 2>&1 | grep -E "b=.*S=" | tail -3
echo "--- NORED b=8 H=32"; FLASH_MLA_SM120_BWD_MMA_DIAG_NORED=1 BENCH_B=8 BENCH_H=32 "$PY" tests/bench_bwd_vs_sdpa.py 2>&1 | grep -E "b=.*S=" | tail -3
echo "--- CFG=5 192/128 fwd+bwd (model shape)"; "$PY" tests/bench_mla_192_e2e.py 2>&1 | grep -E "fwd\+bwd" | tail -3
echo "--- NORED 192/128 fwd+bwd"; FLASH_MLA_SM120_BWD_MMA_DIAG_NORED=1 "$PY" tests/bench_mla_192_e2e.py 2>&1 | grep -E "fwd\+bwd" | tail -3
echo "CHAIN_DONE ($(date +%H:%M:%S))"

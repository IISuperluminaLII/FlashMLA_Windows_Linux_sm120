#!/usr/bin/env bash
# ncu stall-level profile of the dense decode kernels (operator directive:
# entire arsenal / trace what limits us). Two single-launch profiles:
#   A) BM=64 kernel at CFG=3 (h128 row: what caps ~950 GB/s?)
#   B) M16 kernel at CFG=2 (h16 row: what sits between 1540 and 1792 GB/s?)
# Driver launches the kernel exactly once (correctness pass only, b=32).
# Sections: SOL + memory workload + scheduler/warp stalls + occupancy.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
NCU=/usr/local/cuda-13.0/bin/ncu
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

if [ ! -x "$NCU" ]; then echo "NCU_MISSING"; exit 0; fi

run_prof() {
    local tag="$1" kregex="$2" case="$3" cfg="$4"
    echo "== [$tag] kernel=$kregex case=$case CFG=$cfg =="
    NCU_CASE="$case" FLASH_MLA_SM120_DENSE_DECODE_CFG="$cfg" \
    "$NCU" --kernel-name "regex:$kregex" --launch-count 1 \
        --section SpeedOfLight --section MemoryWorkloadAnalysis \
        --section SchedulerStats --section WarpStateStats --section Occupancy \
        --target-processes all --print-summary per-kernel \
        "$PY" tests/_diag_ncu_case.py 2>&1 \
      | grep -vE "^\s*$" | sed -n '1,200p'
    echo "-- [$tag] done --"
}

OUT=$(run_prof probeA "dense_decode_mma_kernel" h128 3)
if printf "%s" "$OUT" | grep -qE "ERR_NVGPUCTRPERM|insufficient permission"; then
    echo "NCU_PERM_DENIED (WSL perf counters restricted; enable in NVIDIA control panel dev settings)"
    printf "%s\n" "$OUT" | tail -5
    exit 0
fi
printf "%s\n" "$OUT"
run_prof probeB "dense_decode_mma_m16_kernel" h16 2
echo "NCU_PROBE_DONE"

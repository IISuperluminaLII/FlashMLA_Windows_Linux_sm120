#!/usr/bin/env bash
# SM clock + power while the BACKWARD bench runs (compare with tests/_clock_probe.sh for the forward).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_FWD_MMA=1 FLASH_MLA_SM120_BWD_MMA=1
export FLASH_MLA_SM120_FWD_MMA_CFG="${FLASH_MLA_SM120_FWD_MMA_CFG:-11}" FLASH_MLA_SM120_BWD_MMA_CFG="${FLASH_MLA_SM120_BWD_MMA_CFG:-8}"
SAMP=/tmp/_clock_samples_bwd.csv
"$PY" tests/_patch_bwd_bench_shape.py >/dev/null 2>&1
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,nounits -lms 100 > "$SAMP" 2>/dev/null &
SPID=$!
BENCH_B=8 BENCH_H=32 "$PY" tests/bench_bwd_vs_sdpa.py 2>&1 | grep -E "b=.*S=" | tail -1
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
"$PY" - "$SAMP" <<'PY'
import sys, csv
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) >= 4 and r[1].strip().isdigit()]
clk = sorted(int(r[1]) for r in rows); pw = sorted(float(r[3]) for r in rows if r[3].strip().replace('.','',1).isdigit())
busy = [c for c in clk if c > 1500]
if busy: print(f"BWD under_load(>1500MHz): n={len(busy)} min={busy[0]} median={busy[len(busy)//2]} max={busy[-1]} MHz -> mma.sync peak at median = {1024*188*busy[len(busy)//2]/1e6:.0f} TF")
if pw: print(f"power_max={pw[-1]:.0f} W power_p90={pw[int(len(pw)*0.9)]:.0f} W")
PY
echo "CLOCK_PROBE_BWD_DONE"

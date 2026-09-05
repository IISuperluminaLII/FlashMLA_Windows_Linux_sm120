#!/usr/bin/env bash
# Sample the SM clock + power while the fwd bench runs, so "% of peak" is judged at the clock the
# board actually sustains under attention load (peak 545 TF was a short burst at 2865 MHz).
# Usage: bash tests/_clock_probe.sh [cfg]   (device 0; PID-scoped stop of the sampler THIS script started)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
CFG="${1:-7}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_FWD_MMA=1
SAMP=/tmp/_clock_samples.csv
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,nounits -lms 100 > "$SAMP" 2>/dev/null &
SPID=$!
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()"
FLASH_MLA_SM120_FWD_MMA_CFG=$CFG BENCH_B=8 BENCH_H=32 BENCH_S_LIST=4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -F "[FUSED]"
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
"$PY" - "$SAMP" <<'PY'
import sys, csv
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) >= 4 and r[1].strip().isdigit()]
clk = sorted(int(r[1]) for r in rows); pw = sorted(float(r[3]) for r in rows if r[3].strip().replace('.','',1).isdigit())
n = len(clk)
if n == 0: print("NO_SAMPLES"); sys.exit(0)
busy = [c for c in clk if c > 1500]
print(f"samples={n} clk_min={clk[0]} clk_median={clk[n//2]} clk_p90={clk[int(n*0.9)]} clk_max={clk[-1]} MHz")
if busy: print(f"under_load(>1500MHz): n={len(busy)} min={busy[0]} median={busy[len(busy)//2]} max={busy[-1]} MHz -> mma.sync peak at median = {1024*188*busy[len(busy)//2]/1e6:.0f} TF")
if pw: print(f"power_max={pw[-1]:.0f} W power_p90={pw[int(len(pw)*0.9)]:.0f} W")
PY
echo "CLOCK_PROBE_DONE"

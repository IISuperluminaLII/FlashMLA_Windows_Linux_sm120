#!/usr/bin/env bash
# Restore the pre-build .so if a failed build left only the backup, then bench the forward at the
# large shape b=8 H=32 (FA2 sm80 reference: 351 TF at S=4096, 344 at 8192) at RUNGS (env, default 7) and rung 0.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT"
SO="flash_mla/cuda_sm120$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
if [ ! -f "$SO" ] && [ -f "$SO.prebuild_bak" ]; then mv "$SO.prebuild_bak" "$SO"; echo "[WARN] restored $SO from .prebuild_bak (previous build failed)"; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_FWD_MMA=1
"$PY" -c "import torch; x=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); [x@x for _ in range(30)]; torch.cuda.synchronize()"
RUNGS="${RUNGS:-7}"
for cfg in $RUNGS; do
echo "----- rung $cfg, b=8 H=32 -----"
FLASH_MLA_SM120_FWD_MMA_CFG=$cfg BENCH_B=8 BENCH_H=32 BENCH_S_LIST=2048,4096,8192 "$PY" tests/bench_fwd_only.py 2>&1 | grep -F "[FUSED]"
done
echo "----- rung 0 (legacy), b=8 H=32 -----"
FLASH_MLA_SM120_FWD_MMA_CFG=0 BENCH_B=8 BENCH_H=32 BENCH_S_LIST=4096 "$PY" tests/bench_fwd_only.py 2>&1 | grep -E "^\s*\[FUSED\]"
echo "FWD_BIGSHAPE_DONE"

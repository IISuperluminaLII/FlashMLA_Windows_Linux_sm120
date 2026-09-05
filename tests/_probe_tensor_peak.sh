#!/usr/bin/env bash
# Settle the sm_120 tensor peak on THIS board: cuBLAS bf16 GEMM (torch.matmul) + a pure
# mma.sync microbench with 8 independent accumulators/warp and 64 warps/SM. Device 0.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env_sm120.sh"
cd "$FMLA_ROOT/tests"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
"$PY" - <<'PYEOF'
import torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
for n in (4096, 8192, 12288):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16); b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3): a @ b
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); a @ b; e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e))
    print(f"[cuBLAS] bf16 GEMM {n}^3: {best:8.3f} ms  {2*n**3/best/1e9:8.1f} TFLOPS")
    del a, b
PYEOF
"$CUDA_HOME/bin/nvcc" -std=c++17 -O3 -arch=sm_120 -o /tmp/_probe_tensor_peak _probe_tensor_peak.cu && /tmp/_probe_tensor_peak

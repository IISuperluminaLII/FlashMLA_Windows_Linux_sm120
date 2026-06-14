"""192/128 MLA (the model config) A/B: raw mma.sync kernels vs the default WMMA kernels.
SDPA can't run 192/128 (asymmetric dims), so we compare against the kernels the model uses
today. Toggle env between runs:
  mma : FLASH_MLA_SM120_FWD_MMA=1 FLASH_MLA_SM120_BWD_MMA=1
  wmma: (unset)  -> default fused WMMA fwd + WMMA MLA bwd
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/bench_mla_192_mma_vs_wmma.py
"""
import os, torch
torch.manual_seed(0)
DEV = "cuda"
import flash_mla
from flash_mla.flash_mla_interface import flash_attn_varlen_func, _flash_attn_varlen_backward
PEAK = 251.9
FWD = os.environ.get("FLASH_MLA_SM120_FWD_MMA") == "1"
BWD = os.environ.get("FLASH_MLA_SM120_BWD_MMA") == "1"
print(f"[INFO] dev={torch.cuda.get_device_name(0)} fwd={'MMA' if FWD else 'WMMA'} bwd={'MMA' if BWD else 'WMMA'}")


def min_ms(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def run(batch, S, H, Dqk=192, Dvo=128, causal=True):
    scale = Dqk ** -0.5
    # fwd flops: QK over Dqk + PV over Dvo ; bwd ~ 2.5x (use Dqk for both as a nominal scale)
    f_flops = 2.0 * batch * H * S * S * (Dqk + Dvo) * (0.5 if causal else 1.0)
    b_flops = 2.5 * f_flops
    total = batch * S
    cu = torch.arange(0, total + 1, S, device=DEV, dtype=torch.int32)
    q = torch.randn(total, H, Dqk, device=DEV, dtype=torch.bfloat16).mul_(0.5)
    k = torch.randn(total, H, Dqk, device=DEV, dtype=torch.bfloat16).mul_(0.5)
    v = torch.randn(total, H, Dvo, device=DEV, dtype=torch.bfloat16).mul_(0.5)
    fwd = min_ms(lambda: flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=causal, is_varlen=True))
    with torch.no_grad():
        out, lse = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=causal, is_varlen=True)
    do = torch.randn_like(out)
    bwd = min_ms(lambda: _flash_attn_varlen_backward(do, q, k, v, out, lse, cu, cu, S, S, causal=causal, softmax_scale=scale, is_varlen=True))
    ftf, btf = f_flops / (fwd / 1e3) / 1e12, b_flops / (bwd / 1e3) / 1e12
    print(f"  b={batch} S={S} H={H}: FWD {fwd:7.3f}ms {ftf:7.2f}TF ({ftf/PEAK*100:4.1f}%) | "
          f"BWD {bwd:7.3f}ms {btf:7.2f}TF ({btf/PEAK*100:4.1f}%)")


if __name__ == "__main__":
    for S in [1024, 2048, 4096]:
        run(2, S, 8)

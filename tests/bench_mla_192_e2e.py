"""TRUE end-to-end fwd+bwd through autograd at the model's 192/128 config -- exactly what
one training step's attention costs (forward builds the graph, autograd.grad runs the
backward, including the LSE save / .contiguous() / fp32->bf16 cast the real path does).
A/B by env (run twice):
  mma : FLASH_MLA_SM120_FWD_MMA=1 FLASH_MLA_SM120_BWD_MMA=1
  wmma: (unset)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/bench_mla_192_e2e.py
"""
import os, torch
torch.manual_seed(0)
DEV = "cuda"
import flash_mla
from flash_mla.flash_mla_interface import flash_attn_varlen_func
PEAK = 251.9
FWD = os.environ.get("FLASH_MLA_SM120_FWD_MMA") == "1"
BWD = os.environ.get("FLASH_MLA_SM120_BWD_MMA") == "1"
print(f"[INFO] dev={torch.cuda.get_device_name(0)} fwd={'MMA' if FWD else 'WMMA'} bwd={'MMA' if BWD else 'WMMA'}")


def min_ms(fn, iters=30, warmup=8):
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
    # total attention FLOPs for fwd+bwd ~= 3.5x fwd (fwd 1x + bwd 2.5x)
    f_flops = 2.0 * batch * H * S * S * (Dqk + Dvo) * (0.5 if causal else 1.0)
    e2e_flops = 3.5 * f_flops
    total = batch * S
    cu = torch.arange(0, total + 1, S, device=DEV, dtype=torch.int32)
    q = torch.randn(total, H, Dqk, device=DEV, dtype=torch.bfloat16).mul_(0.5).requires_grad_(True)
    k = torch.randn(total, H, Dqk, device=DEV, dtype=torch.bfloat16).mul_(0.5).requires_grad_(True)
    v = torch.randn(total, H, Dvo, device=DEV, dtype=torch.bfloat16).mul_(0.5).requires_grad_(True)
    gout = torch.randn(total, H, Dvo, device=DEV, dtype=torch.bfloat16)

    def fwd_bwd():
        out, _ = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=causal, is_varlen=True)
        torch.autograd.grad(out, [q, k, v], gout, retain_graph=False)

    ms = min_ms(fwd_bwd)
    tf = e2e_flops / (ms / 1e3) / 1e12
    print(f"  b={batch} S={S} H={H}: fwd+bwd {ms:8.3f} ms  {tf:7.2f} TF ({tf/PEAK*100:4.1f}% peak)")


if __name__ == "__main__":
    for S in [1024, 2048, 4096]:
        run(2, S, 8)

"""
Backward FLOPS vs the ceiling, contention-robust (mirrors bench_fwd_vs_sdpa.py).

Times OUR backward (_flash_attn_varlen_backward, isolated -- forward run once up front)
against torch SDPA's backward (autograd.grad) on the SAME GPU, so contention cancels in
the RATIO. 128/128 only (SDPA needs symmetric head dims). Backward FLOPs ~= 2.5x forward
(5 matmuls: S recompute, dP, dV, dQ, dK), each 2*S*S*D.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/bench_bwd_vs_sdpa.py
"""
import os
import torch
import torch.nn.functional as F
torch.manual_seed(0)
DEV = "cuda"
import flash_mla
from flash_mla.flash_mla_interface import flash_attn_varlen_func, _flash_attn_varlen_backward, FLASH_MLA_LOADED_VARIANT
PEAK = 251.9          # RTX PRO 6000 Blackwell dense bf16 TFLOPS
print(f"[INFO] variant={FLASH_MLA_LOADED_VARIANT} dev={torch.cuda.get_device_name(0)}")


def min_ms(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def run(batch, S, H, D=128, causal=True):
    scale = D ** -0.5
    flops = 10.0 * batch * H * S * S * D * (0.5 if causal else 1.0)  # 5 matmuls * 2
    total = batch * S
    cu = torch.arange(0, total + 1, S, device=DEV, dtype=torch.int32)
    q = torch.randn(total, H, D, device=DEV, dtype=torch.bfloat16).mul_(0.5)
    k = torch.randn(total, H, D, device=DEV, dtype=torch.bfloat16).mul_(0.5)
    v = torch.randn(total, H, D, device=DEV, dtype=torch.bfloat16).mul_(0.5)

    # OURS: run forward ONCE, then time only the backward kernel.
    with torch.no_grad():
        out, lse = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=causal, is_varlen=True)
    do = torch.randn_like(out)
    ours = min_ms(lambda: _flash_attn_varlen_backward(do, q, k, v, out, lse, cu, cu, S, S,
                  causal=causal, softmax_scale=scale, is_varlen=True))

    # SDPA: build graph once, time autograd.grad repeatedly (cuDNN flash backward).
    qb = q.view(batch, S, H, D).transpose(1, 2).contiguous().requires_grad_(True)
    kb = k.view(batch, S, H, D).transpose(1, 2).contiguous().requires_grad_(True)
    vb = v.view(batch, S, H, D).transpose(1, 2).contiguous().requires_grad_(True)
    ob = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal)
    gb = torch.randn_like(ob)
    sdpa = min_ms(lambda: torch.autograd.grad([ob], [qb, kb, vb], [gb], retain_graph=True))

    our_tf = flops / (ours / 1e3) / 1e12
    sdpa_tf = flops / (sdpa / 1e3) / 1e12
    ratio = our_tf / sdpa_tf
    print(f"  b={batch} S={S} H={H}: OURS {ours:7.3f}ms {our_tf:7.2f}TF ({our_tf/PEAK*100:4.1f}%peak) | "
          f"SDPA {sdpa:7.3f}ms {sdpa_tf:7.2f}TF ({sdpa_tf/PEAK*100:4.1f}%peak) | ours/SDPA={ratio:5.2f}x")


if __name__ == "__main__":
    for S in [1024, 2048, 4096]:
        run(2, S, 8)

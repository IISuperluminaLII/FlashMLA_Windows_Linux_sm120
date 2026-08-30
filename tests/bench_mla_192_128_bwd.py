"""
Benchmark: 192/128 MLA backward -- fused WMMA kernel vs ATen fallback.

The path is selected by env FLASH_MLA_SM120_FUSED_MLA_BWD (1=fused, else fallback).
Run twice to compare:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/bench_mla_192_128_bwd.py            # fallback
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 FLASH_MLA_SM120_FUSED_MLA_BWD=1 python tests/bench_mla_192_128_bwd.py  # fused

Times ONLY the backward (out.backward), which is the path that differs. Moderate
tensors so it is safe alongside device-0 training.
"""
import os
import torch

torch.manual_seed(0)
DEV = "cuda"
# Default-ON / opt-out: only "0" disables the fused kernel (matches the C++ dispatcher).
_e = os.environ.get("FLASH_MLA_SM120_FUSED_MLA_BWD")
MODE = "FALLBACK" if _e == "0" else "FUSED"

import flash_mla
from flash_mla.flash_mla_interface import flash_attn_varlen_func, FLASH_MLA_LOADED_VARIANT

print(f"[INFO] variant={FLASH_MLA_LOADED_VARIANT} device={torch.cuda.get_device_name(0)} MODE={MODE}")


def bench(batch, S, H, Dqk=192, Dvo=128, causal=True, iters=10, warmup=3):
    scale = Dqk ** -0.5
    total = batch * S
    cu = torch.arange(0, total + 1, S, device=DEV, dtype=torch.int32)
    q = torch.randn(total, H, Dqk, device=DEV, dtype=torch.bfloat16).mul_(0.5).requires_grad_(True)
    k = torch.randn(total, H, Dqk, device=DEV, dtype=torch.bfloat16).mul_(0.5).requires_grad_(True)
    v = torch.randn(total, H, Dvo, device=DEV, dtype=torch.bfloat16).mul_(0.5).requires_grad_(True)
    go = torch.randn(total, H, Dvo, device=DEV, dtype=torch.bfloat16).mul_(0.5)

    out, _ = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=causal, is_varlen=True)

    def one():
        q.grad = None; k.grad = None; v.grad = None
        out.backward(go, retain_graph=True)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        one()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"  [{MODE:8s}] batch={batch} S={S} H={H} causal={causal}: {ms:8.3f} ms/bwd")
    return ms


if __name__ == "__main__":
    for (batch, S, H) in [(2, 512, 8), (2, 1024, 8), (1, 2048, 8)]:
        bench(batch, S, H, causal=True)
    print("[DONE]", MODE)

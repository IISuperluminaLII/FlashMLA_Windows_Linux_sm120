"""Forward-only microbenchmark (isolates the fused forward from the backward).
Times flash_attn_varlen_func forward under no_grad. Small tensors (device-0 safe).
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/bench_fwd_only.py
Set FLASH_MLA_SM120_FUSED_FWD=0 to time the ATen fallback forward for A/B.
"""
import os
import torch
torch.manual_seed(0)
DEV = "cuda"
MODE = "ATEN" if os.environ.get("FLASH_MLA_SM120_FUSED_FWD") == "0" else "FUSED"
import flash_mla
from flash_mla.flash_mla_interface import flash_attn_varlen_func, FLASH_MLA_LOADED_VARIANT
print(f"[INFO] variant={FLASH_MLA_LOADED_VARIANT} dev={torch.cuda.get_device_name(0)} MODE={MODE}")


@torch.no_grad()
def bench(batch, S, H, Dqk, Dvo, causal=True, iters=30, warmup=8):
    scale = Dqk ** -0.5
    total = batch * S
    cu = torch.arange(0, total + 1, S, device=DEV, dtype=torch.int32)
    q = torch.randn(total, H, Dqk, device=DEV, dtype=torch.bfloat16).mul_(0.5)
    k = torch.randn(total, H, Dqk, device=DEV, dtype=torch.bfloat16).mul_(0.5)
    v = torch.randn(total, H, Dvo, device=DEV, dtype=torch.bfloat16).mul_(0.5)

    def one():
        flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=causal, is_varlen=True)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    # Per-iter timing; report MIN (least-contended) since device 0 runs training at 99%.
    best = float("inf")
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); one(); en.record(); torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en))
    ms = best
    # attention fwd flops ~ 2 * b * H * S^2 * (Dqk + Dvo) ; causal ~ half
    flops = 2.0 * batch * H * S * S * (Dqk + Dvo) * (0.5 if causal else 1.0)
    tflops = flops / (ms / 1000.0) / 1e12
    print(f"  [{MODE}] b={batch} S={S} H={H} {Dqk}/{Dvo} causal={causal}: min {ms:7.3f} ms  {tflops:6.2f} TFLOPS")
    return ms


if __name__ == "__main__":
    print("--- MLA 192/128 (model) ---")
    for S in [512, 1024, 2048]:
        bench(2, S, 8, 192, 128, causal=True)
    print("--- 128/128 ---")
    for S in [512, 1024, 2048]:
        bench(2, S, 8, 128, 128, causal=True)

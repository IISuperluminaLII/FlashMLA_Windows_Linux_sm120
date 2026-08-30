"""
Forward FLOPS vs the ceiling, contention-robust.

Times OUR fused forward against torch SDPA (cuDNN flash backend) on the SAME GPU at the
SAME instant, so the 99%-training contention cancels in the RATIO. cuDNN SDPA on sm_120
is ~96-97% of speed-of-light (gau-nernst FA-5090), so:
    our_%_of_SOL  ~=  (our_TFLOPS / sdpa_TFLOPS) * 96.5%
This estimates how far we are from the author/mma.sync SM80-class ceiling WITHOUT needing
an uncontended GPU. 128/128 only (SDPA needs symmetric head dims).
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/bench_fwd_vs_sdpa.py
Set FLASH_MLA_SM120_FUSED_FWD=0 to compare the ATen forward instead.
"""
import os
import torch
import torch.nn.functional as F
torch.manual_seed(0)
DEV = "cuda"
MODE = "ATEN" if os.environ.get("FLASH_MLA_SM120_FUSED_FWD") == "0" else "FUSED"
import flash_mla
from flash_mla.flash_mla_interface import flash_attn_varlen_func, FLASH_MLA_LOADED_VARIANT
PEAK = 251.9          # RTX PRO 6000 Blackwell dense bf16 TFLOPS (NVIDIA datasheet)
SDPA_SOL = 0.965      # cuDNN SDPA ~= 96.5% of SOL on sm_120 (gau-nernst)
print(f"[INFO] variant={FLASH_MLA_LOADED_VARIANT} dev={torch.cuda.get_device_name(0)} MODE={MODE}")
print(f"[INFO] ceiling={PEAK} TFLOPS dense bf16; SDPA assumed ~{SDPA_SOL*100:.0f}% SOL")


def min_ms(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


@torch.no_grad()
def run(batch, S, H, D=128, causal=True):
    scale = D ** -0.5
    flops = 2.0 * batch * H * S * S * (2 * D) * (0.5 if causal else 1.0)  # QK + PV
    total = batch * S
    cu = torch.arange(0, total + 1, S, device=DEV, dtype=torch.int32)
    q = torch.randn(total, H, D, device=DEV, dtype=torch.bfloat16).mul_(0.5)
    k = torch.randn(total, H, D, device=DEV, dtype=torch.bfloat16).mul_(0.5)
    v = torch.randn(total, H, D, device=DEV, dtype=torch.bfloat16).mul_(0.5)
    ours = min_ms(lambda: flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=causal, is_varlen=True))

    qb = q.view(batch, S, H, D).transpose(1, 2).contiguous()
    kb = k.view(batch, S, H, D).transpose(1, 2).contiguous()
    vb = v.view(batch, S, H, D).transpose(1, 2).contiguous()
    sdpa = min_ms(lambda: F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal))

    our_tf = flops / (ours / 1e3) / 1e12
    sdpa_tf = flops / (sdpa / 1e3) / 1e12
    ratio = our_tf / sdpa_tf
    # PRIMARY honest metric = our_tf / hardware PEAK (always <=100%). The SDPA-derived
    # "%SOL" is only valid when SDPA itself saturates (~96.5% SOL); at small/unsaturated
    # sizes SDPA runs well below peak, so ratio*SDPA_SOL can exceed 100% and is NOT real
    # SOL -- report it only as an upper bound when sdpa is near peak.
    our_peak = our_tf / PEAK * 100.0
    sdpa_peak = sdpa_tf / PEAK * 100.0
    print(f"  b={batch} S={S} H={H}: OURS {ours:6.3f}ms {our_tf:7.2f}TF ({our_peak:4.1f}%peak) | "
          f"SDPA {sdpa:6.3f}ms {sdpa_tf:7.2f}TF ({sdpa_peak:4.1f}%peak) | ours/SDPA={ratio:5.2f}x"
          + (f"  (beats SDPA by {(ratio-1)*100:.0f}%)" if ratio > 1 else ""))


if __name__ == "__main__":
    for S in [1024, 2048, 4096]:
        run(2, S, 8)

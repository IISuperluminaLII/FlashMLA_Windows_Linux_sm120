"""Reference points on THIS silicon: torch SDPA with the FLASH (FlashAttention-2 sm80 mma.sync)
and CUDNN backends, fwd-only and fwd+bwd, 128/128 causal, b=2 H=8 and b=8 H=32 (the CuTile
paper shape). TFLOPS use the FA2 convention (causal halved; bwd = 2.5x fwd).
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_bench_sdpa_backends.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
torch.manual_seed(0)
def min_ms(fn, iters=30, warmup=8):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); best = float("inf")
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e))
    return best
print(f"[ENV] {torch.cuda.get_device_name(0)} torch {torch.__version__}")
x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(30): x @ x
torch.cuda.synchronize()
for (b, h) in ((2, 8), (8, 32)):
    for S in (2048, 4096, 8192):
        D = 128
        q = torch.randn(b, h, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(b, h, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(b, h, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        fwd_flops = 4.0 * b * h * S * S * D * 0.5
        for name, be in (("FLASH(FA2 sm80)", SDPBackend.FLASH_ATTENTION), ("CUDNN", SDPBackend.CUDNN_ATTENTION)):
            try:
                with sdpa_kernel(be):
                    with torch.no_grad():
                        ms_f = min_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=True); g = torch.randn_like(o)
                    def fb():
                        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                        torch.autograd.grad(out, (q, k, v), g)
                    ms_fb = min_ms(fb, iters=15, warmup=5)
                print(f"  b={b} H={h} S={S} {name:16s}: fwd {ms_f:7.3f} ms {fwd_flops/ms_f/1e9:6.1f} TF | fwd+bwd {ms_fb:7.3f} ms {3.5*fwd_flops/ms_fb/1e9:6.1f} TF (bwd-only est {2.5*fwd_flops/max(ms_fb-ms_f,1e-6)/1e9:6.1f} TF)")
            except Exception as ex:
                print(f"  b={b} H={h} S={S} {name}: unavailable ({type(ex).__name__})")
        del q, k, v
print("SDPA_BACKENDS_DONE")

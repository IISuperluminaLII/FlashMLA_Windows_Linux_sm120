"""
Reproduce the CUDA-graph-capture failure of the FlashMLA fwd/bwd WITHOUT Inductor/nvcc, using raw
torch.cuda.graph capture. The backward allocates a ~70MB dynamic workspace via torch.empty; if that
allocation happens during capture it triggers cudaMalloc -> cudaErrorStreamCaptureUnsupported.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_cudagraph.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
from flash_mla.flash_mla_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward

DEV = "cuda"
S, H, Dqk, Dvo, B = 1024, 22, 192, 128, 4          # ~7b-ish: 22 heads, 192/128, 4 packed seqs
total = B * S
scale = Dqk ** -0.5
q = torch.randn(total, H, Dqk, device=DEV, dtype=torch.bfloat16)
k = torch.randn(total, H, Dqk, device=DEV, dtype=torch.bfloat16)
v = torch.randn(total, H, Dvo, device=DEV, dtype=torch.bfloat16)
cu = torch.arange(0, (B + 1) * S, S, device=DEV, dtype=torch.int32)


def fwd():
    return _flash_attn_varlen_forward(q, k, v, cu, cu, S, S, causal=True, softmax_scale=scale)


out, lse = fwd()
do = torch.randn_like(out)


def bwd():
    return _flash_attn_varlen_backward(do, q, k, v, out, lse, cu, cu, S, S, causal=True, softmax_scale=scale)


def _relerr(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-8)


def capture(fn, name):
    ref = tuple(t.clone() for t in fn())   # eager reference (same inputs)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()                      # warmup (allocations happen here, populate the pool)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            out = fn()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[{name}] CUDA graph capture FAILED: {type(e).__name__}: {str(e)[:150]}")
        return False
    # REPLAY and compare to eager -> catches stale-pointer / workspace-lifetime corruption
    for t in out:
        t.zero_()
    g.replay()
    torch.cuda.synchronize()
    errs = [_relerr(o, r) for o, r in zip(out, ref)]
    ok = all(e < 1e-2 for e in errs)
    print(f"[{name}] capture OK, replay-vs-eager rel={['%.2e' % e for e in errs]} -> {'OK' if ok else 'CORRUPT'}")
    return ok


if __name__ == "__main__":
    print(f"[INFO] dev={torch.cuda.get_device_name(0)} bwd workspace ~70MB (B={B} S={S} H={H} Dqk={Dqk})")
    f_ok = capture(fwd, "FWD")
    b_ok = capture(bwd, "BWD")
    print("RESULT:", "both capture OK" if (f_ok and b_ok) else "BWD (or FWD) NOT cudagraph-safe")
    raise SystemExit(0 if (f_ok and b_ok) else 1)

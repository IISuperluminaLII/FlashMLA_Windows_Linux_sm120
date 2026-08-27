"""
Verify flash_attn_varlen_func is torch.compile(fullgraph=True) compatible after wrapping the
pybind kernel as a torch.library custom op (fake + autograd). fullgraph=True ERRORS on any graph
break, so a clean run + grads matching eager proves the fix.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_compile.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
from flash_mla.flash_mla_interface import flash_attn_varlen_func

DEV = "cuda"


def relerr(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().mean().item() / (b.abs().mean().item() + 1e-8)


S, H, Dqk, Dvo = 128, 2, 192, 128
cu = torch.tensor([0, S], device=DEV, dtype=torch.int32)


def mk(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(S, H, Dqk, device=DEV, dtype=torch.bfloat16, generator=g) * 0.5).requires_grad_(True)
    k = (torch.randn(S, H, Dqk, device=DEV, dtype=torch.bfloat16, generator=g) * 0.5).requires_grad_(True)
    v = (torch.randn(S, H, Dvo, device=DEV, dtype=torch.bfloat16, generator=g) * 0.5).requires_grad_(True)
    return q, k, v


def fn(q, k, v):
    o, _ = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=Dqk ** -0.5, causal=True)
    return o


if __name__ == "__main__":
    print(f"[INFO] dev={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    qe, ke, ve = mk(1)
    oe = fn(qe, ke, ve)
    oe.float().pow(2).mean().backward()

    # Match the user's config: dynamic=False GLOBAL, but mark_dynamic on the token dim (what the
    # power-of-2 bucketed packing loader does) so that dim becomes a symint. aot_eager isolates the
    # graph-break fix from Inductor/nvcc. fullgraph=True -> errors on ANY graph break.
    cfn = torch.compile(fn, fullgraph=True, backend="aot_eager", dynamic=False)
    qc, kc, vc = mk(1)
    torch._dynamo.mark_dynamic(qc, 0)   # token dim dynamic via mark_dynamic (the bucketed-packing path)
    torch._dynamo.mark_dynamic(kc, 0)
    torch._dynamo.mark_dynamic(vc, 0)
    oc = cfn(qc, kc, vc)
    oc.float().pow(2).mean().backward()

    e_o = relerr(oc, oe)
    e_dq = relerr(qc.grad, qe.grad)
    e_dk = relerr(kc.grad, ke.grad)
    e_dv = relerr(vc.grad, ve.grad)
    print(f"[COMPILE] fullgraph=True traced OK (no graph break)")
    print(f"[COMPILE] fwd rel={e_o:.2e}  dq={e_dq:.2e} dk={e_dk:.2e} dv={e_dv:.2e}  (compiled vs eager)")
    ok = max(e_o, e_dq, e_dk, e_dv) < 1e-2
    print("RESULT:", "OK compile==eager (fullgraph compatible)" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)

"""Correctness of the raw mma.sync BACKWARD (FLASH_MLA_SM120_BWD_MMA=1), split dims.
Compares dQ/dK/dV from _flash_attn_varlen_backward against a torch autograd reference
(fp32 attention). Red->green: a layout bug in any of the 5 matmuls corrupts the matching
gradient. Covers 128/128 AND 192/128 (model config). Run with FLASH_MLA_SM120_BWD_MMA=1."""
import os, sys, torch
torch.manual_seed(0)
DEV = "cuda"
import flash_mla
from flash_mla.flash_mla_interface import flash_attn_varlen_func, _flash_attn_varlen_backward, FLASH_MLA_LOADED_VARIANT
print("[INFO] variant:", FLASH_MLA_LOADED_VARIANT, "BWD_MMA:", os.environ.get("FLASH_MLA_SM120_BWD_MMA"))


def ref_bwd(q, k, v, do, scale, causal):  # q,k:[S,H,Dqk] ; v,do:[S,H,Dvo]
    S = q.shape[0]
    qr = q.float().detach().requires_grad_(True)
    kr = k.float().detach().requires_grad_(True)
    vr = v.float().detach().requires_grad_(True)
    scores = torch.einsum("ihd,jhd->hij", qr, kr) * scale
    if causal:
        i = torch.arange(S, device=q.device).view(1, S, 1)
        j = torch.arange(S, device=q.device).view(1, 1, S)
        scores = scores.masked_fill(j > i, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("hij,jhd->ihd", p, vr)
    out.backward(do.float())
    return qr.grad, kr.grad, vr.grad


def relerr(a, b):
    d = (a - b).abs()
    return d.max().item(), (d.mean() / (b.abs().mean() + 1e-8)).item()


def run(S, H, Dqk, Dvo, causal=False, tol=0.03):
    scale = Dqk ** -0.5
    q = torch.randn(S, H, Dqk, device=DEV, dtype=torch.bfloat16) * 0.5
    k = torch.randn(S, H, Dqk, device=DEV, dtype=torch.bfloat16) * 0.5
    v = torch.randn(S, H, Dvo, device=DEV, dtype=torch.bfloat16) * 0.5
    do = torch.randn(S, H, Dvo, device=DEV, dtype=torch.bfloat16) * 0.5
    cu = torch.tensor([0, S], device=DEV, dtype=torch.int32)
    with torch.no_grad():
        out, lse = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=causal, is_varlen=True)
    dq, dk, dv = _flash_attn_varlen_backward(do, q, k, v, out, lse, cu, cu, S, S,
                                             causal=causal, softmax_scale=scale, is_varlen=True)
    rq, rk, rv = ref_bwd(q, k, v, do, scale, causal)
    print(f"\n[CASE] BWD S={S} H={H} Dqk={Dqk} Dvo={Dvo} causal={causal}")
    ok = True
    for name, a, b in [("dQ", dq, rq), ("dK", dk, rk), ("dV", dv, rv)]:
        mx, rl = relerr(a.float(), b)
        nan = bool(torch.isnan(a.float()).any())
        good = (rl < tol) and not nan
        ok = ok and good
        print(f"   {name} max_abs={mx:.3e} rel={rl:.3e} nan={nan} [{'OK' if good else 'FAIL'}]")
    print(f"   => {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    ok = True
    for Dqk, Dvo in [(192, 128), (128, 128)]:
        for cfg in [dict(S=64, H=2), dict(S=128, H=2), dict(S=100, H=2), dict(S=200, H=1),
                    dict(S=256, H=2), dict(S=512, H=1)]:
            for causal in (False, True):
                ok = run(Dqk=Dqk, Dvo=Dvo, causal=causal, **cfg) and ok
    print("\n[RESULT]", "ALL PASSED (mma backward correct)" if ok else "FAILURES DETECTED")
    sys.exit(0 if ok else 1)

"""Correctness of the swizzled mma forward at the BENCH shapes (S up to 4096, H=8,
causal, 128/128) vs an fp32 reference -- guards against a 'fast because it skipped
work / swizzle-corrupted' false positive. Run with FLASH_MLA_SM120_FWD_MMA=1."""
import os, torch
torch.manual_seed(0)
DEV = "cuda"
import flash_mla
from flash_mla.flash_mla_interface import flash_attn_varlen_func, FLASH_MLA_LOADED_VARIANT
print("[INFO] variant:", FLASH_MLA_LOADED_VARIANT, "FWD_MMA:", os.environ.get("FLASH_MLA_SM120_FWD_MMA"))


def ref_fwd(q, k, v, scale, causal):
    S = q.shape[0]
    scores = torch.einsum("ihd,jhd->hij", q.float(), k.float()) * scale
    if causal:
        i = torch.arange(S, device=q.device).view(1, S, 1)
        j = torch.arange(S, device=q.device).view(1, 1, S)
        scores = scores.masked_fill(j > i, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("hij,jhd->ihd", p, v.float())
    return out, lse.transpose(0, 1).contiguous()


def relerr(a, b):
    d = (a - b).abs()
    return d.max().item(), (d.mean() / (b.abs().mean() + 1e-8)).item()


def run(S, H, D=128, causal=True, tol=0.03):
    scale = D ** -0.5
    q = torch.randn(S, H, D, device=DEV, dtype=torch.bfloat16) * 0.5
    k = torch.randn(S, H, D, device=DEV, dtype=torch.bfloat16) * 0.5
    v = torch.randn(S, H, D, device=DEV, dtype=torch.bfloat16) * 0.5
    cu = torch.tensor([0, S], device=DEV, dtype=torch.int32)
    with torch.no_grad():
        out, lse = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=causal, is_varlen=True)
    r_out, r_lse = ref_fwd(q, k, v, scale, causal)
    omx, orel = relerr(out.float(), r_out)
    finite = torch.isfinite(r_lse)
    lmx, lrel = relerr(lse.float()[finite], r_lse[finite])
    nan = bool(torch.isnan(out).any() or torch.isnan(lse.float()[finite]).any())
    ok = (orel < tol) and (lrel < tol) and not nan
    print(f"[CASE] S={S} H={H} causal={causal}: O rel={orel:.3e} LSE rel={lrel:.3e} nan={nan} -> {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    ok = True
    for S in [1024, 2048, 4096]:
        for causal in (True, False):
            ok = run(S, 8, causal=causal) and ok
    print("[RESULT]", "ALL PASSED (bench-shape correctness)" if ok else "FAILURES DETECTED")

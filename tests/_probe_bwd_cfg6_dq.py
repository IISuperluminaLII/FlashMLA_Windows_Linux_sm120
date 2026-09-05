"""CFG=6 dQ debug: compare dq at CFG=2 vs the tier in FLASH_MLA_SM120_BWD_MMA_CFG (run twice, once per tier)."""
import os, sys
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("FLASH_MLA_SM120_FWD_MMA", "1"); os.environ.setdefault("FLASH_MLA_SM120_BWD_MMA", "1")
import torch
from flash_mla import flash_attn_varlen_func
torch.manual_seed(0)
cfg = os.environ.get("FLASH_MLA_SM120_BWD_MMA_CFG", "unset")
out_path = sys.argv[1]
for (dqk, dvo) in ((128, 128), (192, 128)):
    sq = skv = 96; H = 2
    q = torch.randn(sq, H, dqk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(skv, H, dqk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(skv, H, dvo, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, sq], device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_func(q, k, v, cu, cu, sq, skv, causal=False)
    o = out[0] if isinstance(out, (tuple, list)) else out
    o.float().sum().backward()
    dq = q.grad.float().cpu()
    print(f"[CFG={cfg}] {dqk}/{dvo}: |dq| mean={dq.abs().mean():.4e} max={dq.abs().max():.4e} zero_frac={(dq == 0).float().mean():.3f}")
    torch.save(dq, f"{out_path}_{dqk}.pt")
print("PROBE_DONE")

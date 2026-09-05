"""
Probe: causal backward with skv < sq (the forward masks with BOTTOM-RIGHT alignment, so queries
0..sq-skv-1 are fully masked and carry LSE = -inf). Reports NaN presence per BWD CFG so a
convention mismatch is attributed to the kernel FAMILY, not to one rung. Run once per CFG:
  FLASH_MLA_SM120_BWD_MMA_CFG=<n> python tests/_probe_bwd_causal_convention.py
"""
import os, sys
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("FLASH_MLA_SM120_FWD_MMA", "1"); os.environ.setdefault("FLASH_MLA_SM120_BWD_MMA", "1")
import torch
from flash_mla import flash_attn_varlen_func

torch.manual_seed(0)
cfg = os.environ.get("FLASH_MLA_SM120_BWD_MMA_CFG", "unset")
for (dqk, dvo) in ((128, 128), (192, 128)):
    for (sq, skv) in ((96, 40), (40, 96), (96, 96)):
        H = 2
        q = torch.randn(sq, H, dqk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(skv, H, dqk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(skv, H, dvo, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        cu_q = torch.tensor([0, sq], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, skv], device="cuda", dtype=torch.int32)
        out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, sq, skv, causal=True)
        o = out[0] if isinstance(out, (tuple, list)) else out
        o.float().sum().backward()
        nan = tuple(bool(torch.isnan(t.grad.float()).any()) for t in (q, k, v))
        print(f"[CFG={cfg}] {dqk}/{dvo} sq={sq} skv={skv} causal: out_nan={bool(torch.isnan(o.float()).any())} grad_nan(dq,dk,dv)={nan}")
print("PROBE_DONE")

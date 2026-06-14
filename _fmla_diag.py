import torch, math
torch.manual_seed(0)
dev = "cuda"
S, H, Dq, Dv = 6, 1, 192, 128
scale = Dq ** -0.5
import flash_mla
from flash_mla import flash_attn_varlen_func

q0 = torch.randn(S, H, Dq, device=dev, dtype=torch.bfloat16) * 0.5
k0 = torch.randn(S, H, Dq, device=dev, dtype=torch.bfloat16) * 0.5
v0 = torch.randn(S, H, Dv, device=dev, dtype=torch.bfloat16) * 0.5
go = torch.randn(S, H, Dv, device=dev, dtype=torch.bfloat16) * 0.5
cu = torch.tensor([0, S], device=dev, dtype=torch.int32)

q = q0.clone().requires_grad_(True); k = k0.clone().requires_grad_(True); v = v0.clone().requires_grad_(True)
out, _ = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=True, is_varlen=True)
out.backward(go)
fdq, fdk, fdv = q.grad.float(), k.grad.float(), v.grad.float()

qr = q0.float().clone().requires_grad_(True); kr = k0.float().clone().requires_grad_(True); vr = v0.float().clone().requires_grad_(True)
sc = torch.einsum("ihd,jhd->hij", qr, kr) * scale
i = torch.arange(S, device=dev).view(1, S, 1); j = torch.arange(S, device=dev).view(1, 1, S)
sc = sc.masked_fill(j > i, float("-inf"))
p = sc.softmax(-1)
ro = torch.einsum("hij,jhd->ihd", p, vr)
ro.backward(go.float())
rdq, rdk, rdv = qr.grad, kr.grad, vr.grad

torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)
print("=== dV  (rows=kv token, first 4 of 128 dims), head 0 ===")
print("fmla:\n", fdv[:, 0, :4])
print("ref :\n", rdv[:, 0, :4])
print("ratio fmla/ref (dim 0):", (fdv[:, 0, 0] / (rdv[:, 0, 0] + 1e-9)).tolist())
print("\n=== dQ (rows=q token, first 4 of 192 dims), head 0 ===")
print("fmla:\n", fdq[:, 0, :4])
print("ref :\n", rdq[:, 0, :4])
print("\n=== per-row dq L2 ratio (fmla/ref) ===")
print((fdq[:, 0].norm(dim=-1) / (rdq[:, 0].norm(dim=-1) + 1e-9)).tolist())

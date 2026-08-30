import torch, flash_mla
from flash_mla.flash_mla_interface import flash_attn_varlen_func
torch.manual_seed(0)
S, H, D = 64, 1, 128
scale = D ** -0.5
q = torch.randn(S, H, D, device='cuda', dtype=torch.bfloat16) * 0.5
k = torch.randn(S, H, D, device='cuda', dtype=torch.bfloat16) * 0.5
v = torch.randn(S, H, D, device='cuda', dtype=torch.bfloat16) * 0.5
cu = torch.tensor([0, S], device='cuda', dtype=torch.int32)
with torch.no_grad():
    o, _ = flash_attn_varlen_func(q, k, v, cu, cu, S, S, softmax_scale=scale, causal=False, is_varlen=True)
ref = (q[:, 0, :].float() @ k[:, 0, :].float().T) * scale      # [S,S] true scores
got = o[:, 0, :32].float()                                      # kernel dumped S[:, :32]
ref32 = ref[:, :32]
print("max|got-ref|      :", (got - ref32).abs().max().item())
print("got[0,:6]:", [round(x, 3) for x in got[0, :6].tolist()])
print("ref[0,:6]:", [round(x, 3) for x in ref32[0, :6].tolist()])
print("got[1,:6]:", [round(x, 3) for x in got[1, :6].tolist()])
print("ref[1,:6]:", [round(x, 3) for x in ref32[1, :6].tolist()])
print("got[8,:6]:", [round(x, 3) for x in got[8, :6].tolist()])
print("ref[8,:6]:", [round(x, 3) for x in ref32[8, :6].tolist()])
# probe common layout errors
print("max vs transposed-within-tile (got[r,c] vs ref[c,r] for r,c<32):",
      (got[:32, :32] - ref[:32, :32].T).abs().max().item())

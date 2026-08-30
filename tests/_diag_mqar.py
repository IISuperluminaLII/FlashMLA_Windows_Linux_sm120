"""
Diagnostic (NOT a pytest; underscore-prefixed so it is not collected): isolate what breaks
the MQAR retrieval plateau at acc ~= 1/n_pairs. Reuses the EXACT task from
test_needle_haystack_10m (make_mqar_batch / _loss_acc) so the construction can't diverge.

Hypothesis: the previous-token head is too blurry under learned absolute position
embeddings to tag each value with its specific key, so the induction head attends across
all values uniformly -> copies a random one -> acc = 1/n_pairs. RoPE makes "attend to the
previous token" a fixed sharp relative rotation; weight decay blurs the QK match. Isolate
each.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_mqar.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_needle_haystack_10m import (  # noqa: E402
    make_mqar_batch, _loss_acc, n_params, DEV,
    VOCAB, D_MODEL, N_LAYERS, N_HEADS, DQK, DVO, FFN, MAX_SEQ,
)
from flash_mla.flash_mla_interface import flash_attn_varlen_func  # noqa: E402


# ---- RoPE -------------------------------------------------------------------
def build_rope(seq_len, dim, device, base=10000.0):
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # [S, half]
    return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]  # [1,S,1,half]


def apply_rope(x, cos, sin):
    # x: [B,S,H,D] fp32; GPT-NeoX rotate-half on the two halves.
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---- model with optional RoPE / abs-pos -------------------------------------
class Attn(nn.Module):
    def __init__(self, use_rope):
        super().__init__()
        self.q = nn.Linear(D_MODEL, N_HEADS * DQK, bias=False)
        self.k = nn.Linear(D_MODEL, N_HEADS * DQK, bias=False)
        self.v = nn.Linear(D_MODEL, N_HEADS * DVO, bias=False)
        self.o = nn.Linear(N_HEADS * DVO, D_MODEL, bias=False)
        self.scale = DQK ** -0.5
        self.use_rope = use_rope

    def forward(self, x, backend, rope):
        B, S, _ = x.shape
        q = self.q(x).view(B, S, N_HEADS, DQK).float()
        k = self.k(x).view(B, S, N_HEADS, DQK).float()
        if self.use_rope:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        q = q.to(torch.bfloat16); k = k.to(torch.bfloat16)
        v = self.v(x).view(B, S, N_HEADS, DVO).to(torch.bfloat16)
        if backend == "flash":
            cu = torch.arange(0, (B + 1) * S, S, device=x.device, dtype=torch.int32)
            o, _ = flash_attn_varlen_func(
                q.reshape(B * S, N_HEADS, DQK), k.reshape(B * S, N_HEADS, DQK),
                v.reshape(B * S, N_HEADS, DVO), cu, cu, S, S,
                softmax_scale=self.scale, causal=True, is_varlen=True)
            o = o.reshape(B, S, N_HEADS * DVO)
        else:
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, scale=self.scale).transpose(1, 2).reshape(B, S, N_HEADS * DVO)
        return self.o(o.to(torch.float32))


class Blk(nn.Module):
    def __init__(self, use_rope):
        super().__init__()
        self.n1 = nn.LayerNorm(D_MODEL); self.attn = Attn(use_rope)
        self.n2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, FFN), nn.GELU(), nn.Linear(FFN, D_MODEL))

    def forward(self, x, backend, rope):
        x = x + self.attn(self.n1(x), backend, rope)
        return x + self.mlp(self.n2(x))


class Net(nn.Module):
    def __init__(self, use_rope, use_abspos):
        super().__init__()
        self.use_rope = use_rope; self.use_abspos = use_abspos
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(MAX_SEQ, D_MODEL) if use_abspos else None
        self.blocks = nn.ModuleList([Blk(use_rope) for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self._rope_cache = {}

    def rope(self, S):
        if S not in self._rope_cache:
            self._rope_cache[S] = build_rope(S, DQK, DEV)
        return self._rope_cache[S]

    def forward(self, idx, backend="flash"):
        B, S = idx.shape
        x = self.tok(idx)
        if self.use_abspos:
            x = x + self.pos(torch.arange(S, device=idx.device))[None]
        rope = self.rope(S) if self.use_rope else None
        for blk in self.blocks:
            x = blk(x, backend, rope)
        return self.head(self.norm(x))


def lr_at(step, peak, warmup, total, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return floor * peak + (1.0 - floor) * peak * 0.5 * (1.0 + math.cos(math.pi * prog))


def train(tag, use_rope, use_abspos, steps, B, n_pairs, n_query, lr, warmup, wd,
          backend="flash", seed=1, target=0.95):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(123)
    model = Net(use_rope, use_abspos).to(DEV)
    npar = n_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    acc = 0.0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, lr, warmup, steps)
        seq, qpos, qvals = make_mqar_batch(B, n_pairs, n_query, gen)
        loss, _ = _loss_acc(model(seq, backend), qpos, qvals)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                ev = make_mqar_batch(256, n_pairs, n_query, gen)
                _, acc_t = _loss_acc(model(ev[0], backend), ev[1], ev[2])
                acc = acc_t.item()
            print(f"[{tag}] step {step:4d} lr {opt.param_groups[0]['lr']:.2e} "
                  f"loss {loss.item():.3f} acc {acc:.3f}", flush=True)
            if acc >= target:
                print(f"[{tag}] HIT target {target} at step {step}", flush=True)
                break
    print(f"[RESULT] {tag} params={npar/1e6:.2f}M final_acc={acc:.3f}\n", flush=True)
    return acc


def run_both(tag, **kw):
    """Train SDPA and FlashMLA with IDENTICAL settings (same seed -> same init + same data
    sequence). The ONLY difference is the attention kernel, so equal final accuracy proves
    FlashMLA is a correct drop-in; a gap is a real FlashMLA bug. SDPA is the measure."""
    a_s = train(tag + "/sdpa",  backend="sdpa",  **kw)
    a_f = train(tag + "/flash", backend="flash", **kw)
    gap = abs(a_s - a_f)
    print(f"[PARITY] {tag}: sdpa={a_s:.3f} flash={a_f:.3f} gap={gap:.3f} "
          f"{'OK' if gap <= 0.05 else 'DIVERGENT'}\n", flush=True)
    return a_s, a_f


if __name__ == "__main__":
    print(f"[INFO] dev={torch.cuda.get_device_name(0)}", flush=True)
    STEPS = int(os.environ.get("DIAG_STEPS", "2000"))
    common = dict(steps=STEPS, B=64, n_pairs=8, n_query=8, lr=2e-3, warmup=150)
    results = {}
    # A: control (current test model) -- expect BOTH backends plateau ~0.13 (not a backend issue)
    results["A_abspos_wd0.1"] = run_both("A_abspos_wd0.1", use_rope=False, use_abspos=True, wd=0.1, **common)
    # C: RoPE (sharp relative previous-token head) -- expect the task becomes solvable
    results["C_rope_wd0.1"]   = run_both("C_rope_wd0.1",   use_rope=True,  use_abspos=True, wd=0.1, **common)
    # D: RoPE + low weight decay
    results["D_rope_wd0.01"]  = run_both("D_rope_wd0.01",  use_rope=True,  use_abspos=True, wd=0.01, **common)
    print("==== SUMMARY (sdpa is the measure; flash must match) ====", flush=True)
    for tag, (a_s, a_f) in results.items():
        print(f"  {tag:18s} sdpa={a_s:.3f} flash={a_f:.3f} gap={abs(a_s-a_f):.3f}", flush=True)
    print("[DONE] diag", flush=True)

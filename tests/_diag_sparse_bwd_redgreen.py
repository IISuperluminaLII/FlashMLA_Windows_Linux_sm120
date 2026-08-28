"""
Red/green test for sm120 sparse prefill BACKWARD (flash_mla_sparse_bwd).

Reference = torch autograd over the AUTHORS' forward semantics (reference_torch in
test_flash_mla_prefill.py): gather kv rows by per-position indices, scores masked -inf
at invalid indices, global softmax over the full topk, out = P @ kv[:, :d_v].
Gradient contract: d(kv_row) = dk_row + [dv_row, 0] since V aliases KV[:, :512];
kernel returns (dq, dk, dv) separately, so we compare dk + pad(dv) vs kv.grad.

RED on the current kernel is guaranteed by construction:
  case A: topk=128 (2 topk-blocks) -> kernel's per-block LOCAL softmax != global softmax
  case A: per-position DISTINCT indices with s_q=32 -> kernel uses only the first
          query-in-block's indices ("assumes same for block", bwd.cu:462)
  case B: s_q=1, topk=64 (single block+query) -> boundary where local==global softmax;
          documents the discrimination boundary (may pass on broken code by accident)
  case C: duplicates + invalid indices -> atomic accumulation + masking semantics
  case D: ragged topk tail (topk % 64 != 0) -> the k >= topk lanes must act invalid
  case E: all-invalid positions (idx = -1) -> fwd lse = -inf sink + bwd !isfinite(lse)
          guard; a missing guard produces non-finite grads and trips the isfinite assert

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_sparse_bwd_redgreen.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import math
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from flash_mla import flash_mla_sparse_fwd
from flash_mla.flash_mla_interface import flash_mla_sparse_bwd


def make_case(s_q, s_kv, topk, h_q=128, seed=0, invalid_frac=0.1, dup_frac=0.0,
              all_invalid_rows=()):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn((s_q, h_q, 576), dtype=torch.bfloat16, device="cuda", generator=g) / 10)
    kv = (torch.randn((s_kv, 1, 576), dtype=torch.bfloat16, device="cuda", generator=g) / 10)
    # Per-position DISTINCT indices (each s_q row gets its own permutation)
    idx = torch.empty((s_q, 1, topk), dtype=torch.int32, device="cuda")
    for s in range(s_q):
        perm = torch.randperm(s_kv, device="cuda", generator=g)
        if topk <= s_kv:
            row = perm[:topk]
        else:
            row = torch.cat([perm, torch.full((topk - s_kv,), s_kv, device="cuda", dtype=torch.long)])
        if dup_frac > 0 and topk >= 4 and s_kv >= 2:
            ndup = max(1, int(topk * dup_frac))
            row[:ndup] = row[ndup:2 * ndup]  # force duplicates
        if invalid_frac > 0:
            inv = torch.rand(topk, device="cuda", generator=g) < invalid_frac
            row = row.clone()
            row[inv] = s_kv + 7  # invalid sentinel (>= s_kv)
        idx[s, 0] = row.to(torch.int32)
    # Force whole positions invalid with the NEGATIVE form (-1): exercises the
    # forward's lse = -inf sink and the backward's !isfinite(lse) guard.
    for s in all_invalid_rows:
        idx[s, 0].fill_(-1)
    return q, kv, idx


def reference_grads(q, kv, idx, d_o, sm_scale):
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = idx.shape[-1]
    d_v = 512
    qf = q.float().detach().requires_grad_(True)
    kvf = kv.float().detach().requires_grad_(True)
    indices = idx[:, 0, :].long()                    # [s_q, topk]
    invalid = (indices < 0) | (indices >= s_kv)
    safe = indices.masked_fill(invalid, 0)
    gathered = kvf[:, 0, :][safe.flatten()].view(s_q, topk, d_qk)      # [s_q, topk, 576]
    scores = torch.einsum("shd,std->sht", qf, gathered) * sm_scale     # [s_q, h_q, topk]
    scores = scores.masked_fill(invalid.unsqueeze(1), float("-inf"))
    # All-invalid positions: softmax(all -inf) = NaN and would NaN-poison autograd.
    # The kernel contract (fwd lse = -inf sink; bwd !isfinite(lse) guard) is out = 0
    # and ZERO gradients for such a position. Encode that closed-form and NaN-free:
    # give the row finite dummy scores (detached constant zero), then multiply P by a
    # 0/1 row mask -- the mask constant zeroes both the output and every gradient
    # path through that row, exactly the kernel semantics.
    all_inv = invalid.all(dim=-1)                                      # [s_q]
    scores = torch.where(all_inv.view(s_q, 1, 1), torch.zeros_like(scores), scores)
    p = torch.softmax(scores, dim=-1)
    p = p * (~all_inv).view(s_q, 1, 1).float()
    out = torch.einsum("sht,std->shd", p, gathered[:, :, :d_v])
    out.backward(d_o.float())
    return qf.grad, kvf.grad                          # [s_q,h_q,576], [s_kv,1,576]


def cosdiff(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0 if (a.norm() == b.norm()) else 1.0
    return (1 - (a @ b) / denom).item()


def check(name, ans, ref, cos_tol=1e-5, rel_tol=0.04):
    cd = cosdiff(ans, ref)
    denom = ref.float().abs().mean().item() + 1e-8
    rel = (ans.float() - ref.float()).abs().mean().item() / denom
    ok = (cd < cos_tol) and (rel < rel_tol) and torch.isfinite(ans.float()).all().item()
    print(f"  [{'OK' if ok else 'FAILED'}] {name}: cos_diff={cd:.3e} mean_rel={rel:.3e}")
    return ok


def run_case(tag, s_q, s_kv, topk, seed, invalid_frac=0.1, dup_frac=0.0, must_discriminate=True,
             all_invalid_rows=()):
    print(f"[CASE {tag}] s_q={s_q} s_kv={s_kv} topk={topk} invalid={invalid_frac} dup={dup_frac}"
          + (f" all_invalid_rows={list(all_invalid_rows)}" if all_invalid_rows else ""))
    q, kv, idx = make_case(s_q, s_kv, topk, seed=seed, invalid_frac=invalid_frac, dup_frac=dup_frac,
                           all_invalid_rows=all_invalid_rows)
    sm_scale = 1.0 / math.sqrt(576)
    out, max_logits, lse = flash_mla_sparse_fwd(q, kv, idx, sm_scale)
    g = torch.Generator(device="cuda").manual_seed(seed + 1000)
    d_o = torch.randn(out.shape, dtype=torch.bfloat16, device="cuda", generator=g) / 10
    dq, dk, dv = flash_mla_sparse_bwd(d_o, q, kv, out, lse, idx, sm_scale)
    torch.cuda.synchronize()
    dq_ref, dkv_ref = reference_grads(q, kv, idx, d_o, sm_scale)
    dkv_ans = dk.float()
    dkv_ans[:, :, :512] += dv.float()
    ok = True
    ok &= check("dq ", dq, dq_ref)
    ok &= check("dkv", dkv_ans, dkv_ref)
    return ok


if __name__ == "__main__":
    torch.cuda.set_device(0)
    results = []
    # A: kills local-softmax (2 blocks) + first-query-indices bug (distinct per position)
    results.append(("A multi-block multi-query", run_case("A", 32, 1024, 128, seed=0)))
    # A2: deeper topk (4 blocks), more positions
    results.append(("A2 four blocks", run_case("A2", 62, 2048, 256, seed=1)))
    # B: discrimination boundary (single block, single query)
    results.append(("B single block+query", run_case("B", 1, 256, 64, seed=2)))
    # C: duplicates + heavy invalid
    results.append(("C dup+invalid", run_case("C", 8, 64, 64, seed=3, invalid_frac=0.3, dup_frac=0.1)))
    # D: RAGGED topk tail (topk % 64 != 0): the kernel's k >= topk lanes must act as
    # invalid. A mishandled tail reads garbage indices -> cos check fails (discriminates).
    results.append(("D ragged topk tail", run_case("D", 16, 512, 96, seed=4)))
    # E: ALL-INVALID positions (idx = -1): fwd lse = -inf, bwd !isfinite(lse) guard must
    # zero P/dS. If the guard were absent, P = exp2(S - (-inf)) = inf -> non-finite
    # grads -> the isfinite assert in check() fails (discriminates by construction).
    results.append(("E all-invalid rows", run_case("E", 8, 256, 64, seed=5,
                                                   all_invalid_rows=(0, 3))))

    failed = [n for n, ok in results if not ok]
    print(f"\n[RESULT] {len(results) - len(failed)} passed, {len(failed)} failed of {len(results)}")
    for n in failed:
        print(f"  [FAILED] {n}")
    raise SystemExit(1 if failed else 0)

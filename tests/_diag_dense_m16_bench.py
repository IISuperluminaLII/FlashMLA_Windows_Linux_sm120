"""
Dense decode SMALL-HEAD A/B (authors' harness verbatim; correctness asserted in
the same run). The CFG env is latched per process -- the runner executes this
once per tier.

Shapes (b=128, varlen, causal and not):
  - h16-sq1 : h_q=16, s_q=1  -> q_seq_per_hk=16: the M16 single-pass tier
              fires at CFG=2 (BM=64 kernel wastes 4x mma + re-reads pages)
  - h64-sq1 : h_q=64, s_q=1  -> q_seq_per_hk=64: M16 inert, BM=64 tier at all
              CFG>=1 (gate-boundary control)

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  FLASH_MLA_SM120_DENSE_DECODE_CFG=<n> python tests/_diag_dense_m16_bench.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from test_flash_mla_decoding import TestParam, test_flash_mla

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    cfg = os.environ.get("FLASH_MLA_SM120_DENSE_DECODE_CFG", "0")
    cases = [
        ("h16-sq1-s4k",  TestParam(128, 1, 4096, is_varlen=True, is_causal=False,
                                   is_fp8=False, topk=None, h_q=16, test_performance=True)),
        ("h16-sq1-s16k", TestParam(128, 1, 16384, is_varlen=True, is_causal=False,
                                   is_fp8=False, topk=None, h_q=16, test_performance=True)),
        ("h16-sq1-c4k",  TestParam(128, 1, 4096, is_varlen=True, is_causal=True,
                                   is_fp8=False, topk=None, h_q=16, test_performance=True)),
        # M32 tier targets (16 < q_seq_per_hk <= 32): h22 = the 7b model's head
        # count; h32 = the gate boundary; h16-sq2 = causal row folding at
        # q_seq_per_hk = 32 with q_head_per_hk = 16.
        ("h22-sq1-s4k",  TestParam(128, 1, 4096, is_varlen=True, is_causal=False,
                                   is_fp8=False, topk=None, h_q=22, test_performance=True)),
        ("h22-sq1-s16k", TestParam(128, 1, 16384, is_varlen=True, is_causal=False,
                                   is_fp8=False, topk=None, h_q=22, test_performance=True)),
        ("h32-sq1-s4k",  TestParam(128, 1, 4096, is_varlen=True, is_causal=False,
                                   is_fp8=False, topk=None, h_q=32, test_performance=True)),
        ("h32-sq1-c4k",  TestParam(128, 1, 4096, is_varlen=True, is_causal=True,
                                   is_fp8=False, topk=None, h_q=32, test_performance=True)),
        ("h16-sq2-c4k",  TestParam(128, 2, 4096, is_varlen=True, is_causal=True,
                                   is_fp8=False, topk=None, h_q=16, test_performance=True)),
        ("h64-sq1-s4k",  TestParam(128, 1, 4096, is_varlen=True, is_causal=False,
                                   is_fp8=False, topk=None, h_q=64, test_performance=True)),
    ]
    failed = 0
    for name, t in cases:
        print(f"[CASE] CFG={cfg} {name}")
        try:
            test_flash_mla(t)
        except Exception as e:
            failed += 1
            print(f"[FAILED] {name}: {type(e).__name__}: {str(e)[:140]}")
    print(f"[RESULT] {len(cases) - failed} passed, {failed} failed of {len(cases)}")
    raise SystemExit(1 if failed else 0)

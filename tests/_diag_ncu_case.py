"""
Minimal single-launch driver for ncu profiling of the dense decode kernels.
One correctness pass (test_performance=False) = exactly one dense-kernel launch
(+ metadata + combine), so `ncu --launch-count 1` profiles a clean instance.
Small batch keeps replay passes short on the shared device.

  NCU_CASE=h128|h16  FLASH_MLA_SM120_DENSE_DECODE_CFG=<n>  python tests/_diag_ncu_case.py
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

    case = os.environ.get("NCU_CASE", "h128")
    h_q = 16 if case == "h16" else 128
    t = TestParam(32, 1, 4096, is_varlen=True, is_causal=False,
                  is_fp8=False, topk=None, h_q=h_q, test_performance=False)
    print(f"[NCU-CASE] {case} h_q={h_q} b=32 s_k=4096")
    test_flash_mla(t)
    print("[RESULT] PASSED")

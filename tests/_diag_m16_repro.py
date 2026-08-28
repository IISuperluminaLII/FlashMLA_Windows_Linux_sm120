"""
M16 tier failure repro (single case, FULL output -- the A/B chain's grep filter
dropped the exception text). Correctness only; the assert fires before perf.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  FLASH_MLA_SM120_DENSE_DECODE_CFG=2 python tests/_diag_m16_repro.py
"""
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from test_flash_mla_decoding import TestParam, test_flash_mla

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    cfg = os.environ.get("FLASH_MLA_SM120_DENSE_DECODE_CFG", "0")
    print(f"[REPRO] CFG={cfg} h16-sq1-s4k varlen non-causal (correctness only)")
    t = TestParam(128, 1, 4096, is_varlen=True, is_causal=False,
                  is_fp8=False, topk=None, h_q=16, test_performance=False)
    try:
        test_flash_mla(t)
        print("[RESULT] PASSED")
    except Exception:
        traceback.print_exc()
        print("[RESULT] FAILED")
        raise SystemExit(1)

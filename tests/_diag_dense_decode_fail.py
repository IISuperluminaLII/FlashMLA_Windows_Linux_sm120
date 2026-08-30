"""
Isolate WHY the authors' decode test fails on sm120 dense at scale:
  (a) s_k >= 16384 fails in every config
  (b) causal + s_q=2 fails at every s_k
Runs the two minimal representative failing cases with FULL error output.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_dense_decode_fail.py
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

    cases = [
        ("A s_k=16384 noncausal", TestParam(128, 1, 16384, is_varlen=True, is_causal=False,
                                            is_fp8=False, topk=None, test_performance=False)),
        ("B causal s_q=2 s_k=4096", TestParam(128, 2, 4096, is_varlen=True, is_causal=True,
                                              is_fp8=False, topk=None, test_performance=False)),
    ]
    for tag, t in cases:
        print(f"===== CASE {tag} =====")
        try:
            test_flash_mla(t)
            print(f"[OK] {tag}")
        except Exception:
            print(f"[FAILED] {tag} -- full traceback:")
            traceback.print_exc()
        torch.cuda.synchronize()

"""
Extended sparse-decode regression cases the AUTHORS' ORACLE DOES NOT COVER
(audit/design-sparse-decode.md section 7.4), run with the authors' runner verbatim:

 (a) int64 page-address overflow: pool of 65,536 pages -> top page byte offset
     65,535 * 41,984 = 2.75e9 > INT32_MAX. ~13 GB VRAM footprint - check nvidia-smi first.
 (b) ragged head block: h_q=96 -> second m-block has num_valid_rows = 32.
 (c) authors' third all-invalid corner (topk=4096), absent from the 34-case diag.
 (d) topk >> seqlen: 20 valid of 128 -> heavy -1 padding at a tiling boundary.
 (e) s_q=4: hardest exercise of row0 = s_q_idx*q_head_per_hk + head_start.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tests/_diag_sparse_decode_ext.py
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

    cases = [
        ("a int64-page-overflow", TestParam(64, 2, 65536, is_varlen=False, is_causal=False,
                                            is_fp8=True, topk=2048, test_performance=False)),
        ("b ragged-head h_q=96", TestParam(64, 2, 4096, is_varlen=True, is_causal=False,
                                           is_fp8=True, topk=2048, test_performance=False, h_q=96)),
        ("c all-invalid topk=4096", TestParam(128, 2, 4096, is_varlen=True, is_causal=False,
                                              is_fp8=True, topk=4096, test_performance=False,
                                              is_all_indices_invalid=True)),
        ("d topk>>seqlen", TestParam(64, 4, 20, is_varlen=True, is_causal=False,
                                     is_fp8=True, topk=128, test_performance=False)),
        ("e s_q=4", TestParam(6, 4, 4096, is_varlen=True, is_causal=False,
                              is_fp8=True, topk=2048, test_performance=False)),
    ]

    passed, failed = 0, []
    for tag, t in cases:
        try:
            test_flash_mla(t)
            passed += 1
            print(f"[OK] {tag}")
        except Exception as e:
            failed.append(tag)
            print(f"[FAILED] {tag}: {type(e).__name__}: {str(e)[:140]}")

    print(f"\n[RESULT] {passed} passed, {len(failed)} failed of {len(cases)}")
    raise SystemExit(1 if failed else 0)

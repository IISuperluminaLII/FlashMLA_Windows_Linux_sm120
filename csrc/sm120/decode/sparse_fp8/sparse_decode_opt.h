#pragma once
// Single-instance latch for FLASH_MLA_SM120_SPARSE_DECODE_OPT -- the sparse fp8 decode
// OPTIMIZATION ladder (audit/design-sparse-decode-opt.md), ORTHOGONAL to the CFG variant
// ladder in sparse_decode_cfg.h: OPT rungs apply to whichever mma.sync CFG kernel runs
// (CFG=1 batch-parallel, CFG=2/4 split-KV; the CFG=3 crossover is measured-closed and
// ignores OPT). Same C++17 `inline` function-local-static pattern as sparse_decode_cfg.h:
// exactly ONE instance across every TU, read once, clamped.
//
// Ladder (cumulative; 0 = legacy, byte-identical to the certified kernels):
//   1  K-path register-fragment dequant: the QK mma B fragment is built straight from the
//      linear fp8 staging row (2 x LDS.U16 -> fp8x2 -> f32 -> *scale -> bf16x2, the same
//      fp32-domain arithmetic as dequant.h), no sKV round trip and ONE barrier per K tile
//      (16 -> 8 QK barriers per block); the fp8 stage rows are XOR-swizzled at 16 B granule
//      level so the 8-row fragment reads are bank-conflict-free (0 B of padding).
//      The rope tile moves into the (QK-dead) sKV; V keeps ldmatrix.x2.trans + smem dequant.
//   2  never-drain copy stream: V tile 0 is gathered at dt=7 (before the rope mma/softmax),
//      the NEXT block's K tile 0 / scales at vt=6 / vt=7 with the next block's token
//      pointers resolved one block early (parity double-buffered sTokPtr/sValid, +640 B).
//   3  reserved (V-path half-tile dequant/mma software pipelining); NOT implemented yet --
//      the launchers route OPT=3 to the highest implemented rung (2) and say so once.
#include <cstdlib>

namespace sm120 {

inline int sparse_decode_opt() {
    static const int opt = [] {
        const char* e = std::getenv("FLASH_MLA_SM120_SPARSE_DECODE_OPT");
        const int v = e ? std::atoi(e) : 0;
        return v < 0 ? 0 : (v > 3 ? 3 : v);
    }();
    return opt;
}

}  // namespace sm120

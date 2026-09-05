#pragma once
// Single-instance latch for FLASH_MLA_SM120_SPARSE_DECODE_CFG, shared by the
// metadata arm (pybind.cpp get_attn_impl_meta) and the kernel launcher
// (splitkv_mla.cu). C++17 `inline` guarantees ONE function-local static across
// every TU, so the two consumers can NEVER observe different values (the
// metadata=split/launcher=batch-parallel direction would let the combine kernel
// merge uninitialized accum -- same closed loop as dense_decode_cfg.h).
//
// Ladder: 0 = legacy WMMA batch-parallel (default, byte-identical)
//         1 = raw mma.sync batch-parallel direct-write (certified tier)
//         2 = mma.sync + authors' split-KV (metadata walk + partials + combine)
//         3 = 2 + 2-CTA cluster DSM crossover (half-topk dequant sharing;
//             sm90 H800 design transformed to plain sm_120; documented
//             NEGATIVE on this L2-rich part -- kept opt-in for the record)
//         4 = the CFG=2 split-KV kernel with the split cap raised 64 -> 192
//             (combine gained 128/192 tiers; authors' 64 encoded H800's 132
//             SMs). NOT the crossover: rung 4 routes to the splitkv kernel.
//             Serving m_blocks=1 s_q=1 shapes go 64 -> 188 CTAs.
//         5 = FP8 TENSOR-MATH batch-parallel (e4m3 x e4m3 mma.sync, whole block resident,
//             no dequant, no V re-gather; audit/design-sparse-decode-fp8mma.md). NOT the
//             authors' bit-faithful numerics (Q per-row e4m3, P' per-(row,group) e4m3).
//             Compiled only with the build opt-in FLASH_MLA_SM120_FP8_MMA=1; otherwise the
//             launcher warns once and runs rung 1 (same batch-parallel schedule).
//         6 = FP8 tensor-math split-KV with the 192 cap (schedule of rung 4; falls back to 4).
// The metadata arm derives split-ness from the rung (2/3/4/6 split; 0/1/5 batch-parallel)
// so a build without the fp8-math kernels still sees a consistent schedule after fallback.
#include <cstdlib>

namespace sm120 {

inline int sparse_decode_cfg() {
    static const int cfg = [] {
        const char* e = std::getenv("FLASH_MLA_SM120_SPARSE_DECODE_CFG");
        const int v = e ? std::atoi(e) : 0;
        return v < 0 ? 0 : (v > 6 ? 6 : v);
    }();
    return cfg;
}

inline bool sparse_decode_cfg_is_split(int cfg) { return cfg == 2 || cfg == 3 || cfg == 4 || cfg == 6; }
inline int  sparse_decode_cfg_split_cap(int cfg) { return (cfg == 4 || cfg == 6) ? 192 : 64; }

}  // namespace sm120

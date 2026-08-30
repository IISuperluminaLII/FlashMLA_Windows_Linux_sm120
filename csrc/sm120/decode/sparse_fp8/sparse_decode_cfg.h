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
#include <cstdlib>

namespace sm120 {

inline int sparse_decode_cfg() {
    static const int cfg = [] {
        const char* e = std::getenv("FLASH_MLA_SM120_SPARSE_DECODE_CFG");
        const int v = e ? std::atoi(e) : 0;
        return v < 0 ? 0 : (v > 4 ? 4 : v);
    }();
    return cfg;
}

}  // namespace sm120

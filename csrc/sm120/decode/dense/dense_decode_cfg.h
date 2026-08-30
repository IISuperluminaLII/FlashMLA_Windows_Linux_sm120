#pragma once
// Single-instance latch for FLASH_MLA_SM120_DENSE_DECODE_CFG, shared by the
// metadata arm (pybind.cpp get_attn_impl_meta) and the kernel launcher
// (splitkv_mla.cu). C++17 `inline` guarantees ONE function-local static across
// every TU in the extension, so the two consumers can NEVER observe different
// values. (Two per-TU statics could desync if the env were mutated between
// their first calls; the metadata=1/launcher=0 direction would emit a real
// multi-split schedule that only the mma tier writes partials for, letting the
// combine kernel merge uninitialized accum.)
//
// Ladder: 0 = legacy batch-parallel WMMA kernel (default, byte-identical to
//             the pre-ladder build; num_sm_parts stays 1 so the combine kernel
//             early-returns for every batch),
//         1 = raw mma.sync + the authors' split-KV tile scheduler
//             (splitkv_mla_mma.cuh; h_k == 1 -- MLA -- only, else legacy),
//         2 = 1 + the SMALL-M single-pass tier for q_seq_per_hk <= 16
//             (retained V tiles, 9-deep pipeline, no page re-read; larger
//             shapes keep the CFG=1 kernel),
//         3 = 2 + the split cap raised 64 -> 192 (combine kernel gained
//             128/192 MLA_NUM_SPLITS_SWITCH tiers; the authors' 64 encoded
//             H800's 132 SMs). s_q=1 h_q=128 rows go 128 -> 188 CTAs
//             (num_m_blocks=2 x 94 parts); q_seq_per_hk <= 64 shapes go
//             64 -> 188. Kernels are parts-agnostic (metadata walk), so the
//             tier set is exactly CFG=2's,
//         4 = 3 + the M32 single-pass tier for 16 < q_seq_per_hk <= 32
//             (32-token half-page retained-V tiles, 78,848 B smem; covers the
//             7b model's h=22 and h=32 serving; q_seq_per_hk <= 16 keeps M16),
//         5 = 4 with the M32 gate lifted: EVERY dense shape > 16 rows runs as
//             single-pass 32-row bands (grid.x = ceil(q_seq_per_hk/32); the
//             metadata arm divides occupancy by 32-row blocks). Concurrent
//             bands L2-dedup their page streams like BM=64's m-blocks but skip
//             that tier's V re-read; BM=64 remains reachable at CFG=3/4.
#include <cstdlib>

namespace sm120 {

inline int dense_decode_cfg() {
    static const int cfg = [] {
        const char* e = std::getenv("FLASH_MLA_SM120_DENSE_DECODE_CFG");
        const int v = e ? std::atoi(e) : 0;
        return v < 0 ? 0 : (v > 5 ? 5 : v);
    }();
    return cfg;
}

}  // namespace sm120

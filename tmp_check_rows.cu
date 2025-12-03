#include <cstdio>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "csrc/sm120/prefill/dense/collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp"

using KernelTraits = flash::Sm120WorkstationConfig;
using StrideQ = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>;
using StrideV = StrideK;
using Mask = cutlass::fmha::collective::CausalMask<false>;

using Mainloop = cutlass::fmha::collective::Sm100MlaFwdMainloopTmaWarpspecialized<
    cutlass::bfloat16_t,
    float,
    float,
    KernelTraits::TileShapeMlaFwd,
    StrideQ,
    StrideK,
    StrideV,
    Mask,
    KernelTraits::ThreadShape>;

int main() {
  constexpr int rows = decltype(cute::size<0>(typename Mainloop::TileShapePV{}))::value;
  constexpr int cols = decltype(cute::size<1>(typename Mainloop::TileShapePV{}))::value;
  std::printf("TileShapePV rows=%d cols=%d\n", rows, cols);
  return 0;
}

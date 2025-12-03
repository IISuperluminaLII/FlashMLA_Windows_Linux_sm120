#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <type_traits>

#include "../../../../csrc/sm120/prefill/dense/collective/fmha_fusion.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"

using KernelTraits = flash::Sm120WorkstationConfig;
using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using TileShape = KernelTraits::TileShapeFmhaFwd;
using Mask = cutlass::fmha::collective::CausalMask<false>;

using StrideQ = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>;
using StrideV = StrideK;

using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
    Element, ElementAcc, ElementAcc, TileShape, StrideQ, StrideK, StrideV,
    Mask, KernelTraits::ThreadShape, cute::false_type>;

__global__ void fwd_correction_tmem_load_rank_compile() {}

int main() {
  static_assert(std::is_same_v<typename Mainloop::TMEM_LOAD_V,
                               cute::SM100_TMEM_LOAD_16dp32b16x>);
  static_assert(std::is_same_v<typename Mainloop::TMEM_STORE_V,
                               cute::SM100_TMEM_STORE_16dp32b16x>);
  return 0;
}

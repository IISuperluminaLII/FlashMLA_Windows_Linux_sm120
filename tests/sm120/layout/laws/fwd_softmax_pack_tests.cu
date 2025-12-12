#include <cute/tensor.hpp>

#include "../../../csrc/sm120/prefill/dense/collective/fmha_common.hpp"
#include "../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"

using Element = cutlass::bfloat16_t;
using ElementQK = float;
using ElementPV = float;
using TileShape = flash::Sm120WorkstationConfig::TileShapeFmhaFwd;
using StrideQ = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>;
using StrideV = StrideK;
using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
    Element, ElementQK, ElementPV, TileShape, StrideQ, StrideK, StrideV,
    cutlass::fmha::collective::CausalMask<false>, flash::Sm120WorkstationConfig::ThreadShape>;

__global__ void softmax_pack_compile() {
  using TMEM_STORE_V = SM100_TMEM_STORE_32dp32b2x;
  using DstLayout = typename cute::Copy_Traits<TMEM_STORE_V>::DstLayout;
  auto tDstV = cute::make_tensor(cute::make_tmem_ptr<ElementQK>(0), DstLayout{});
  auto tReg = cute::make_tensor_like<ElementQK>(tDstV);
  (void)tDstV;
  (void)tReg;
}

int main() {
  return 0;
}

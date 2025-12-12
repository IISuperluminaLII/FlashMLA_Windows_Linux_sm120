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

using TileShapeQK = typename Mainloop::TileShapeQK;

int main() {
  using namespace cute;

  static_assert(size<1>(TileShapeQK{}) == _16{},
                "SM120 TileShapeQK N must stay at 16 columns");

  auto tileN = size<1>(TileShapeQK{});

  Tensor tStS_v = make_tensor<uint32_t>(make_shape(tileN, _2{}));
  Tensor tScS_v = make_identity_tensor(make_shape(tileN, _2{}));

  using LayoutStV = decltype(tStS_v.layout());
  using LayoutScV = decltype(tScS_v.layout());

  static_assert(size<0>(LayoutStV{}) == size<1>(TileShapeQK{}),
                "Correction TMEM V rows must match TileShapeQK N dimension");
  static_assert(size<0>(LayoutScV{}) == size<1>(TileShapeQK{}),
                "Correction TMEM coord rows must match TileShapeQK N dimension");

  return 0;
}

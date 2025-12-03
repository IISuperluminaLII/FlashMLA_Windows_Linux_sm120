#include <cute/tensor.hpp>
#include <type_traits>

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

__global__ void softmax_stats_stride_compile() {
  using StoreVTraits = typename Mainloop::StoreVTraits;
  using StatsLayoutVStore = typename Mainloop::StatsLayoutVStore;

  auto tScS = cute::make_identity_tensor(cute::make_shape(cute::_128{}, cute::_128{}));
  auto tStS = cute::make_tensor(cute::make_tmem_ptr<uint32_t>(0),
                                cute::make_layout(cute::make_shape(cute::_128{}, cute::_128{})));
  tStS.data() = uint32_t(0);

  auto [tStS_v, tScS_v, tStS_P, tScS_P, tStS_load, tScS_load] =
      Mainloop::make_softmax_stats_views(cute::_0{}, tStS, tScS);
  (void)tScS_v;
  (void)tStS_P;
  (void)tScS_P;
  (void)tStS_load;
  (void)tScS_load;

  // Stats views have explicit layouts matching Copy_Traits ValID
  static_assert(std::is_same_v<decltype(tStS_v.layout()), StatsLayoutVStore>);
  (void)tStS_v;
}

int main() {
  return 0;
}

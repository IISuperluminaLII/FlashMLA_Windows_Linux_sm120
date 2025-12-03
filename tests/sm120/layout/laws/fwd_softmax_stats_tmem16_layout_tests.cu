#include <cute/tensor.hpp>
#include <type_traits>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

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

__global__ void softmax_stats_tmem16_layout_kernel() {
  // Guard that we are exercising the 16dp SM120 small-tile path.
  static_assert(Mainloop::kIsSm120SmallTile, "Test only targets the small-tile SM120 path");

  using StoreVTraits = typename Mainloop::StoreVTraits;
  using LoadTraits = typename Mainloop::StatsLoadTraits;
  using StatsLayoutVStore = typename Mainloop::StatsLayoutVStore;
  using StatsLayoutVLoad = typename Mainloop::StatsLayoutVLoad;

  auto tScS = cute::make_identity_tensor(cute::make_shape(cute::_128{}, cute::_128{}));
  auto tStS = cute::make_tensor(
      cute::make_tmem_ptr<uint32_t>(0),
      cute::make_layout(cute::make_shape(cute::_128{}, cute::_128{})));

  auto [tStS_v, tScS_v, tStS_P, tScS_P, tStS_load, tScS_load] =
      Mainloop::make_softmax_stats_views(cute::_0{}, tStS, tScS);
  (void)tStS_P;
  (void)tScS_P;

  // Stats views have explicit layouts matching Copy_Traits ValID (upcast to element type)
  static_assert(std::is_same_v<decltype(tStS_v.layout()), StatsLayoutVStore>);
  static_assert(std::is_same_v<decltype(tStS_load.layout()), StatsLayoutVLoad>);
  // Also verify register layouts are upcast
  using StatsRegLayoutStore = typename Mainloop::StatsRegLayoutStore;
  using StatsRegLayoutLoad = typename Mainloop::StatsRegLayoutLoad;
  static_assert(std::is_same_v<decltype(tScS_v.layout()), StatsRegLayoutStore>);
  static_assert(std::is_same_v<decltype(tScS_load.layout()), StatsRegLayoutLoad>);
  (void)tStS_v;
  (void)tStS_load;
  (void)tScS_v;
  (void)tScS_load;
}

int main() {
  softmax_stats_tmem16_layout_kernel<<<1, 128>>>();
  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return 1;
  }
  return 0;
}

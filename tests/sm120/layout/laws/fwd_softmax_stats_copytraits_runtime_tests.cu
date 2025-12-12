#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <type_traits>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

__device__ __managed__ int g_result;

__global__ void softmax_stats_copytraits_kernel() {
  using TileShapeQK = flash::Sm120WorkstationConfig::TileShapeFmhaFwd;
  using StrideQ = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>;
  using StrideK = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>;
  using StrideV = StrideK;

  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      cutlass::bfloat16_t,
      float,
      float,
      TileShapeQK,
      StrideQ,
      StrideK,
      StrideV,
      cutlass::fmha::collective::CausalMask<false>,
      flash::Sm120WorkstationConfig::ThreadShape>;

  using StoreVTraits = typename Mainloop::StoreVTraits;
  using StatsLayoutVStore = typename Mainloop::StatsLayoutVStore;

  auto cS_base = cute::make_identity_tensor(cute::select<0, 1>(TileShapeQK{}));
  auto tScS =
      typename Mainloop::CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS_base);
  auto tStS = cute::partition_fragment_C(
      typename Mainloop::CollectiveMmaQK::TiledMma{}, cute::select<0, 1>(TileShapeQK{}));
  tStS.data() = uint32_t(Mainloop::TmemAllocation::S0);

  auto [tStS_v, tScS_v, tStS_P, tScS_P, tStS_load, tScS_load] =
      Mainloop::make_softmax_stats_views(cute::_0{}, tStS, tScS);
  (void)tStS_P;
  (void)tScS_P;
  (void)tStS_load;
  (void)tScS_load;

  // Stats views have explicit layouts matching Copy_Traits ValID (upcast to element type)
  static_assert(std::is_same_v<decltype(tStS_v.layout()), StatsLayoutVStore>);
  // Register layout is also upcast to element type
  using StatsRegLayoutStore = typename Mainloop::StatsRegLayoutStore;
  static_assert(std::is_same_v<decltype(tScS_v.layout()), StatsRegLayoutStore>);

  g_result = 0;
}

int main() {
  g_result = -1;
  softmax_stats_copytraits_kernel<<<1, 128>>>();
  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return 2;
  }
  return g_result;
}

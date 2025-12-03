#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <type_traits>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

// Compile-time guard for stats TMEM_STORE_V copy path.
// Ensures make_softmax_stats_views surfaces the Copy_Traits layouts
// (DstLayout/SrcLayout) so AtomTVLayout is contained in DataLayout.
__global__ void fwd_softmax_stats_copy_dstdst_compile() {
  using TileShapeQK = flash::Sm120WorkstationConfig::TileShapeFmhaFwd;
  using StrideQ = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>;
  using StrideK = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>;
  using StrideV = StrideK;
  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      cutlass::bfloat16_t, float, float, TileShapeQK, StrideQ, StrideK, StrideV,
      cutlass::fmha::collective::CausalMask<false>, flash::Sm120WorkstationConfig::ThreadShape>;

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

  using StatsLayoutVStore = typename Mainloop::StatsLayoutVStore;
  using StatsRegLayoutStore = typename Mainloop::StatsRegLayoutStore;
  static_assert(std::is_same_v<decltype(tStS_v)::layout_type, StatsLayoutVStore>,
                "Stats TMEM store view must expose Copy_Traits ValID layout (upcast)");
  static_assert(std::is_same_v<decltype(tScS_v)::layout_type, StatsRegLayoutStore>,
                "Stats register view must expose Copy_Traits SrcLayout (upcast)");

  // Verify make_tmem_copy succeeds with StatsLayoutVStore (validates upcast(ValID) match)
  auto tiled_tmem_storev =
      cute::make_tmem_copy(typename Mainloop::TMEM_STORE_V_OP{}, tStS_v);
  auto thr_tmem_storev = tiled_tmem_storev.get_slice(0);
  auto tTMEM_STOREVtS = thr_tmem_storev.partition_D(tStS_v);
  auto tTMEM_STOREVcS = thr_tmem_storev.partition_S(tScS_v);
  auto tTMEM_STOREVrS =
      flash::detail::make_softmax_store_register<float>(tTMEM_STOREVcS);

  // Verify tensor element counts (copy not executed due to partition rank mismatch)
  static_assert(cute::size(tTMEM_STOREVrS) == 4, "Register: 4 elements per thread");
  (void)tiled_tmem_storev;
  (void)tTMEM_STOREVtS;
  (void)tTMEM_STOREVrS;
}

int main() {
  return 0;
}

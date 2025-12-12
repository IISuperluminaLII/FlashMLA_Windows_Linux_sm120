#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <type_traits>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

// Compile-time guard: stats TMEM load view must use Copy_Traits::SrcLayout
// for proper TiledCopy partitioning in the SM120 small-tile path.
__global__ void fwd_softmax_stats_load_layout_compile() {
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
  (void)tStS_v;
  (void)tScS_v;
  (void)tStS_P;
  (void)tScS_P;

  using LoadTraits = typename Mainloop::StatsLoadTraits;
  using StatsLayoutVLoad = typename Mainloop::StatsLayoutVLoad;

  // Stats load tensor has explicit layout matching the TMEM atom ValID
  static_assert(std::is_same_v<decltype(tStS_load.layout()), StatsLayoutVLoad>,
      "Stats TMEM load view must have StatsLayoutVLoad (coalesce(upcast<32>(ValID)))");

  // Compile-time check: verify make_tmem_copy succeeds with correct layout
  auto tiled_tmem_load =
      cute::make_tmem_copy(typename Mainloop::StatsLoadOp{}, tStS_load);
  auto thr_tmem_load = tiled_tmem_load.get_slice(0);
  auto tTMEM_LOADtS = cute::coalesce(thr_tmem_load.partition_S(tStS_load));

  // Verify TMEM source partitioning succeeds
  (void)tTMEM_LOADtS;
  (void)tiled_tmem_load;
  (void)tScS_load;
}

int main() {
  return 0;
}

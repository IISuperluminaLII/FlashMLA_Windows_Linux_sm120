#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

// Compile-time reproduction for the SM120 small-tile softmax stats TMEM store
// path (uses SM100_TMEM_STORE_16dp32b16x). Mirrors the mainloop layouts and
// per-thread partitions to catch rank/pack mismatches.
__global__ void fwd_softmax_store_copy_small_compile() {
  using TileShapeQK = flash::Sm120WorkstationConfig::TileShapeFmhaFwd;
  static_assert(decltype(cute::size<1>(TileShapeQK{}))::value == 16, "N must be 16 for SM120 small tile");

  using StrideQ = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>;
  using StrideK = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>;
  using StrideV = StrideK;
  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      cutlass::bfloat16_t, float, float, TileShapeQK, StrideQ, StrideK, StrideV,
      cutlass::fmha::collective::CausalMask<false>, flash::Sm120WorkstationConfig::ThreadShape>;

  auto cS_base = cute::make_identity_tensor(cute::select<0, 1>(TileShapeQK{}));
  auto tScS = typename Mainloop::CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS_base);
  auto tStS = cute::partition_fragment_C(
      typename Mainloop::CollectiveMmaQK::TiledMma{}, cute::select<0, 1>(TileShapeQK{}));
  tStS.data() = uint32_t(Mainloop::TmemAllocation::S0);

  auto [tStS_v, tScS_v, tStS_P, tScS_P, tStS_load, tScS_load] =
      Mainloop::make_softmax_stats_views(cute::_0{}, tStS, tScS);
  (void)tStS_P;
  (void)tScS_P;
  (void)tStS_load;
  (void)tScS_load;

  using TMEM_STORE_V_OP = typename Mainloop::TMEM_STORE_V_OP;
  auto tStS_v_tmem = cute::coalesce(tStS_v);
  auto tiled_tmem_storev = cute::make_tmem_copy(TMEM_STORE_V_OP{}, tStS_v_tmem);
  auto thr_tmem_storev = tiled_tmem_storev.get_slice(0);

  auto tTMEM_STOREVtS = thr_tmem_storev.partition_D(tStS_v_tmem);
  auto tTMEM_STOREVcS = thr_tmem_storev.partition_S(tScS_v);
  auto tTMEM_STOREVrS =
      flash::detail::make_softmax_store_register<float>(tTMEM_STOREVcS);

  tTMEM_STOREVrS(0) = 0.0f;
  cute::copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);
}

int main() {
  return 0;
}

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

// Compile-time guard: P-buffer TMEM store views must use the Copy_Traits
// layouts required by SM100_TMEM_STORE_32dp32b16x for SM120 small tiles.
__global__ void fwd_softmax_p_store_layout_compile() {
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
  (void)tStS_load;
  (void)tScS_load;

  using StoreTraits = cute::Copy_Traits<cute::SM100_TMEM_STORE_32dp32b16x>;

  static_assert(std::is_same_v<decltype(tStS_P.layout()), typename StoreTraits::DstLayout>);
  static_assert(std::is_same_v<decltype(tScS_P.layout()), typename StoreTraits::SrcLayout>);
}

int main() {
  return 0;
}

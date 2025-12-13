#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

// Compile-time guard: P-buffer TMEM store views use explicit ValID-derived layouts
// (H1 approach) that match coalesce(upcast<32>(ValID)) for TMEM_STORE_P atom.
// P-buffer: 64x16 = 1024 elements for SM120 small tiles.
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

  // H1 approach: P-buffer tensors have explicit ValID-derived layouts
  // - tStS_P has PLayoutTmem = coalesce(upcast<32>(TMEM_STORE_P::ValID))
  // - tScS_P has PLayoutReg = make_layout(shape(PLayoutTmem))
  using PLayoutTmem = typename Mainloop::PLayoutTmem;
  using PLayoutReg = typename Mainloop::PLayoutReg;

  static_assert(std::is_same_v<decltype(tStS_P.layout()), PLayoutTmem>,
                "P-buffer TMEM tensor must use PLayoutTmem (ValID-derived)");
  static_assert(std::is_same_v<decltype(tScS_P.layout()), PLayoutReg>,
                "P-buffer register tensor must use PLayoutReg (contiguous)");

  // Verify P-buffer has correct element count: 64x16 = 1024 elements
  static_assert(cute::size(PLayoutTmem{}) == 1024, "P-buffer: 64 rows x 16 cols = 1024 elements");
}

int main() {
  return 0;
}

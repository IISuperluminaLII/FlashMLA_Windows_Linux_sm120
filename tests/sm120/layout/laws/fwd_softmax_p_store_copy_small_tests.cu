#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

// Compile-time reproduction of the TMEM_STORE (P buffer) copy path for SM120
// small tiles. Ensures the register tensor is flattened to avoid ambiguous
// scatter with the TMEM store atom.
__global__ void fwd_softmax_p_store_copy_small_compile() {
  using TileShapeQK = flash::Sm120WorkstationConfig::TileShapeFmhaFwd;
  constexpr bool kSmall =
      (decltype(cute::size<0>(TileShapeQK{}))::value == 64) &&
      (decltype(cute::size<1>(TileShapeQK{}))::value == 16);
  static_assert(kSmall, "SM120 small-tile path expected");

  using TMEM_STORE = cute::SM100_TMEM_STORE_32dp32b16x;
  using StoreTraits = cute::Copy_Traits<TMEM_STORE>;
  using LoadTraits = cute::Copy_Traits<cute::SM100_TMEM_LOAD_32dp32b16x>;

  // Ensure rank and shape are well-formed for the small-tile store atom.
  static_assert(decltype(cute::rank(typename StoreTraits::SrcLayout{}))::value ==
                decltype(cute::rank(typename StoreTraits::DstLayout{}))::value);
  static_assert(std::is_same_v<typename StoreTraits::SrcLayout, typename LoadTraits::DstLayout>);
  static_assert(std::is_same_v<typename StoreTraits::DstLayout, typename LoadTraits::SrcLayout>);
}

int main() {
  return 0;
}

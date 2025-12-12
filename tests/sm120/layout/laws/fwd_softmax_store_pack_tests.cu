#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>

#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

// Compile-time guard: ensure TMEM store register pack uses a register tensor
// built from the TMEM store copy layout (no shape-only construction).
__global__ void fwd_softmax_store_pack_compile() {
  using TileShapeQK = flash::Sm120WorkstationConfig::TileShapeFmhaFwd;
  constexpr int kTileN = decltype(cute::size<1>(TileShapeQK{}))::value;

  static_assert(kTileN == 16, "SM120 softmax store pack only validated for N=16");

  using TMEM_STORE = cute::SM100_TMEM_STORE_16dp32b16x;
  using TMEM_STORE_V = cute::SM100_TMEM_STORE_16dp32b16x;
  using PackLayout = typename cute::Copy_Traits<TMEM_STORE>::SrcLayout;
  using PackLayoutV = typename cute::Copy_Traits<TMEM_STORE_V>::SrcLayout;

  auto tTMEM_STOREcS = cute::make_tensor<uint32_t>(PackLayout{});
  auto tTMEM_STORErS_x4 =
      flash::detail::make_softmax_store_register<uint32_t>(tTMEM_STOREcS);

  static_assert(cute::size(tTMEM_STORErS_x4) == cute::size(tTMEM_STOREcS));
  static_assert(decltype(cute::rank(tTMEM_STORErS_x4))::value ==
                decltype(cute::rank(tTMEM_STOREcS))::value);
  (void)tTMEM_STORErS_x4;

  auto tTMEM_STOREVcS = cute::make_tensor<float>(PackLayoutV{});
  auto tTMEM_STOREVrS =
      flash::detail::make_softmax_store_register<float>(tTMEM_STOREVcS);
  static_assert(cute::size(tTMEM_STOREVrS) == cute::size(tTMEM_STOREVcS));
  static_assert(decltype(cute::rank(tTMEM_STOREVrS))::value ==
                decltype(cute::rank(tTMEM_STOREVcS))::value);
  (void)tTMEM_STOREVrS;
}

int main() {
  return 0;
}

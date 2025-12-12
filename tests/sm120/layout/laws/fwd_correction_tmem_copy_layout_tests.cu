#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"

// Compile-time guard: correction TMEM load/store copies use SM100 32dp atoms
// with matching Src/Dst layouts and ranks.
__global__ void fwd_correction_tmem_copy_layout_compile() {
  using Load = cute::SM100_TMEM_LOAD_32dp32b32x;
  using Store = cute::SM100_TMEM_STORE_32dp32b32x;
  using LoadTraits = cute::Copy_Traits<Load>;
  using StoreTraits = cute::Copy_Traits<Store>;
  using DPStride = decltype(cute::TMEM::DP<float>{});

  auto tmem_layout = cute::make_layout(
      cute::make_shape(cute::make_shape(cute::_32{}, cute::_4{}),
                       cute::make_shape(cute::C<32>{}, cute::C<32>{})),
      cute::make_stride(cute::make_stride(cute::_0{}, cute::TMEM::DP_b{}),
                        cute::make_stride(cute::_1{}, DPStride{})));

  // TMEM load (tmem -> registers)
  auto tLoadSrc = cute::make_tensor(cute::make_tmem_ptr<float>(0), tmem_layout);
  auto tLoadDst = cute::make_tensor<float>(typename LoadTraits::DstLayout{});
  auto tiledLoad = cute::make_tmem_copy(Load{}, tLoadSrc);
  auto thrLoad = tiledLoad.get_slice(0);
  auto tLoadS = thrLoad.partition_S(tLoadSrc);
  auto tLoadD = thrLoad.partition_D(tLoadDst);
  (void)tLoadS;
  (void)tLoadD;

  // TMEM store (registers -> tmem)
  auto tStoreDst = cute::make_tensor(cute::make_tmem_ptr<float>(0), tmem_layout);
  auto tStoreSrc = cute::make_tensor<float>(typename StoreTraits::SrcLayout{});
  auto tiledStore = cute::make_tmem_copy(Store{}, tStoreDst);
  auto thrStore = tiledStore.get_slice(0);
  auto tStoreD = thrStore.partition_D(tStoreDst);
  auto tStoreS = thrStore.partition_S(tStoreSrc);
  (void)tStoreS;
  (void)tStoreD;
}

int main() {
  return 0;
}

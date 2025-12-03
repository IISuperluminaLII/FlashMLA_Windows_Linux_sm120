#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>

#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

struct DummyStoreSlice {
  template <class CoordTensor>
  CUTE_HOST_DEVICE auto partition_S(CoordTensor const& coords) const {
    return cute::make_tensor(cute::make_tmem_ptr<uint32_t>(), cute::layout(coords));
  }
};

// Compile-only guard: CTAD for the TMEM STORE_V partition mirrors softmax_step.
__global__ void fwd_softmax_store_partition_compile() {
  using TileShapeQK = flash::Sm120WorkstationConfig::TileShapeFmhaFwd;
  constexpr int kTileN = decltype(cute::size<1>(TileShapeQK{}))::value;

  using TMEM_STORE_V = cute::SM100_TMEM_STORE_32dp32b2x;
  using DataLayoutV = typename cute::Copy_Traits<TMEM_STORE_V>::DstLayout;
  (void)kTileN;  // TMEM_STORE_V is fixed for stats.

  auto tScS_v = cute::make_identity_tensor(cute::shape(DataLayoutV{}));
  Tensor tTMEM_STOREVcS = DummyStoreSlice{}.partition_S(tScS_v);
  (void)tTMEM_STOREVcS;
}

int main() { return 0; }

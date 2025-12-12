#include <type_traits>

#define FLASH_MLA_ENABLE_SM120_BWD_KERNEL_IMPL 0

#include "../../../../csrc/sm120/prefill/dense/collective/fmha_fusion.hpp"
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/kernel/sm120_fmha_bwd_kernel_tma_warpspecialized.hpp"

using KernelTraits = flash::Sm120WorkstationConfig;
using ProblemShape = cute::tuple<cutlass::fmha::collective::VariableLength,
                                 cutlass::fmha::collective::VariableLength, int, int,
                                 cute::tuple<int32_t, int32_t>>;
using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using TileShape = KernelTraits::TileShapeFmhaBwd;
using Mask = cutlass::fmha::collective::CausalForBackwardMask<false>;

using Kernel = cutlass::fmha::kernel::Sm120FmhaBwdKernelTmaWarpSpecialized<
    KernelTraits, ProblemShape, Element, ElementAcc, TileShape, Mask>;

int main() {
  static_assert(std::is_same_v<typename Kernel::SoftmaxLoadOp,
                               cute::SM100_TMEM_LOAD_16dp32b16x>);
  static_assert(std::is_same_v<typename Kernel::SoftmaxStoreOp,
                               cute::SM100_TMEM_STORE_16dp32b8x>);
  return 0;
}

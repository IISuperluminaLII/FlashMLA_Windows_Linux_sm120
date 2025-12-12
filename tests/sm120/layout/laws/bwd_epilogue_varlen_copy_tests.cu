#include <cute/tensor.hpp>

#define FLASH_MLA_ENABLE_SM120_BWD_KERNEL_IMPL 0

#include "../../../../csrc/sm120/prefill/dense/collective/fmha_fusion.hpp"
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/kernel/sm120_fmha_bwd_kernel_tma_warpspecialized.hpp"

using KernelTraits = flash::Sm120WorkstationConfig;
using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using TileShape = KernelTraits::TileShapeFmhaBwd;
using Mask = cutlass::fmha::collective::CausalForBackwardMask<false>;

using ProblemShape = cute::tuple<cutlass::fmha::collective::VariableLength,
                                 cutlass::fmha::collective::VariableLength, int, int,
                                 cute::tuple<int32_t, int32_t>>;

using Kernel = cutlass::fmha::kernel::Sm120FmhaBwdKernelTmaWarpSpecialized<
    KernelTraits, ProblemShape, Element, ElementAcc, TileShape, Mask>;

__global__ void bwd_epilogue_varlen_copy_compile() {
  // We only need type instantiation; the body is empty.
}

int main() { return 0; }

#include "../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../csrc/sm120/prefill/dense/collective/fmha_fusion.hpp"
#include "../../../csrc/sm120/prefill/dense/kernel/sm120_fmha_bwd_mla_kernel_tma_warpspecialized.hpp"

using KernelTraits = flash::Sm120WorkstationConfig;
using ProblemShape = cute::tuple<
    cutlass::fmha::collective::VariableLength,
    cutlass::fmha::collective::VariableLength,
    int,
    int,
    cute::tuple<int32_t, int32_t>>;

using Kernel = cutlass::fmha::kernel::Sm120FmhaBwdMlaKernelTmaWarpSpecialized<
    KernelTraits,
    ProblemShape,
    cutlass::bfloat16_t,
    float,
    KernelTraits::TileShapeMlaBwd,
    cutlass::fmha::collective::CausalForBackwardMask<false>>;

static_assert(Kernel::MaxThreadsPerBlock > 0, "BWD kernel must instantiate");

int main() { return 0; }

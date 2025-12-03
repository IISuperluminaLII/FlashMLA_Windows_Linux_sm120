#include "../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../csrc/sm120/prefill/dense/kernel/sm120_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "../../../csrc/sm120/prefill/dense/collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp"
#include "../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "../../../csrc/sm120/prefill/dense/kernel/fmha_tile_scheduler.hpp"

using KernelTraits = flash::Sm120WorkstationConfig;
using ProblemShape = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

// Minimal Collective aliases matching sm120_fmha_fwd_kernel_tma_warpspecialized.hpp wiring
using CollectiveMainloop = cutlass::fmha::collective::Sm100MlaFwdMainloopTmaWarpspecialized<
    cutlass::bfloat16_t,
    float,
    float,
    KernelTraits::TileShapeMlaFwd,
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>,
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>,
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>,
    cutlass::fmha::collective::CausalMask<false>,
    KernelTraits::ThreadShape>;

using CollectiveEpilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
    cutlass::bfloat16_t,
    float,
    KernelTraits::TileShapeMlaFwd,
    cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>,
    cute::tuple<cute::_1, cute::tuple<cute::tuple<int, int>, int>>,
    cute::false_type>;

using TileScheduler = cutlass::fmha::kernel::IndividualTileScheduler;

using Kernel = cutlass::fmha::kernel::Sm120FmhaFwdKernelTmaWarpspecialized<
    KernelTraits,
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler>;

// Compile-time guard: shared storage must not exceed kSharedMemLimit
static_assert(Kernel::SharedStorageSize <= KernelTraits::kSharedMemLimit,
              "SM120 forward shared memory exceeds limit");

int main() { return 0; }

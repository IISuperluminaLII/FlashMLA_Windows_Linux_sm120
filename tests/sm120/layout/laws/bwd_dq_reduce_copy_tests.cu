#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>

#define FLASH_MLA_ENABLE_SM120_BWD_KERNEL_IMPL 0

#include "../../../../csrc/sm120/prefill/dense/collective/fmha_fusion.hpp"
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/kernel/sm120_fmha_bwd_mla_kernel_tma_warpspecialized.hpp"

using KernelTraits = flash::Sm120WorkstationConfig;
using ProblemShape =
    cute::tuple<int, int, int, int, cute::tuple<int32_t, int32_t>>;
using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using TileShape = KernelTraits::TileShapeMlaBwd;
using Mask = cutlass::fmha::collective::ResidualMaskForBackward;

using Kernel = cutlass::fmha::kernel::Sm120FmhaBwdMlaKernelTmaWarpSpecialized<
    KernelTraits, ProblemShape, Element, ElementAcc, TileShape, Mask>;

__global__ void bwd_reduce_copy_compile() {
  using LoadOp = cute::SM100_TMEM_LOAD_16dp32b16x;
  using X = cute::Underscore;

  auto tDQtDQ =
      cute::partition_fragment_C(typename Kernel::TiledMmaDSK{},
                                 cute::select<0, 1>(typename Kernel::TileShapeDSK{}))(
          cute::make_coord(X{}, X{}), cute::_0{}, cute::_0{});
  tDQtDQ.data() = Kernel::TmemAllocation::kDQ;

  auto tiled_t2r = cute::make_tmem_copy(LoadOp{}, tDQtDQ);
  auto thread_t2r = tiled_t2r.get_slice(0);

  auto cDQ =
      cute::make_identity_tensor(cute::take<0, 2>(typename Kernel::TileShapeDSK{}));
  auto sDQ = cute::make_tensor(
      cute::make_smem_ptr<ElementAcc>(static_cast<ElementAcc *>(nullptr)),
      typename Kernel::SmemLayoutDQ{});

  auto tTR_cDQ = thread_t2r.partition_D(cDQ);
  auto tTR_sDQ = thread_t2r.partition_D(sDQ);
  auto tTR_tDQ = thread_t2r.partition_S(tDQtDQ);

  auto tTR_rDQ = cute::make_tensor<ElementAcc>(cute::shape(tTR_cDQ));

  cute::copy(tiled_t2r, tTR_tDQ, tTR_rDQ);
  cute::copy(tTR_rDQ(_, _, cute::_0{}),
             tTR_sDQ(_, _, cute::_0{}, cute::_0{}));
}

int main() {
  return 0;
}

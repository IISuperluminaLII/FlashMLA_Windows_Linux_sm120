#include <type_traits>
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

__global__ void bwd_compute_varlen_tmem_copy_compile() {
  using LoadOp = typename Kernel::EpilogueLoadOp;
  using X = cute::Underscore;

  auto tDKtDK = cute::partition_fragment_C(typename Kernel::TiledMmaDSQ{},
                                           cute::select<0, 1>(typename Kernel::TileShapeDSQ{}))(
      cute::make_coord(X{}, X{}), cute::_0{}, cute::_0{});
  tDKtDK.data() = Kernel::TmemAllocation::kDK;

  auto tDVtDV = cute::partition_fragment_C(typename Kernel::TiledMmaPDO{},
                                           cute::select<0, 1>(typename Kernel::TileShapePDO{}))(
      cute::make_coord(X{}, X{}), cute::_0{}, cute::_0{});
  tDVtDV.data() = Kernel::TmemAllocation::kDV;

  constexpr int kNumWarpgroups = Kernel::kNumComputeWarps / 4;
  int dp_idx = threadIdx.x % 128;
  int wg_idx = (threadIdx.x % (Kernel::kNumComputeWarps * cutlass::NumThreadsPerWarp)) / 128;

  auto split_wg = [&](auto const& t) {
    if constexpr (decltype(cute::rank(t))::value == 3) {
      auto p = t.compose(cute::make_layout(
          cute::make_shape(cute::size<0>(t), cute::size<1>(t),
                           cute::make_shape(cute::Int<kNumWarpgroups>{},
                                            cute::size<2>(t) / cute::Int<kNumWarpgroups>{}))));
      return p(_, _, cute::make_coord(wg_idx, _));
    } else {
      auto p = t.compose(cute::make_layout(
          cute::make_shape(cute::size<0>(t), cute::size<1>(t), cute::size<2>(t),
                           cute::make_shape(cute::Int<kNumWarpgroups>{},
                                            cute::size<3>(t) / cute::Int<kNumWarpgroups>{}))));
      return p(_, _, _, cute::make_coord(wg_idx, _));
    }
  };

  auto tiled_t2r_dk = cute::make_tmem_copy(LoadOp{}, tDKtDK);
  auto thread_t2r_dk = tiled_t2r_dk.get_slice(dp_idx);

  auto cDK = cute::make_identity_tensor(cute::take<0, 2>(typename Kernel::TileShapeDSQ{}));
  auto tTR_cDK = split_wg(thread_t2r_dk.partition_D(cDK));
  auto tTR_tDK = split_wg(thread_t2r_dk.partition_S(tDKtDK));
  auto tTR_rDK = cute::make_tensor<ElementAcc>(cute::shape(tTR_cDK));

  cute::copy(tiled_t2r_dk, tTR_tDK, tTR_rDK);

  auto tiled_t2r_dv = cute::make_tmem_copy(LoadOp{}, tDVtDV);
  auto thread_t2r_dv = tiled_t2r_dv.get_slice(dp_idx);

  auto cDV = cute::make_identity_tensor(cute::take<0, 2>(typename Kernel::TileShapePDO{}));
  auto tTR_cDV = split_wg(thread_t2r_dv.partition_D(cDV));
  auto tTR_tDV = split_wg(thread_t2r_dv.partition_S(tDVtDV));
  auto tTR_rDV = cute::make_tensor<ElementAcc>(cute::shape(tTR_cDV));

  cute::copy(tiled_t2r_dv, tTR_tDV, tTR_rDV);
}

int main() {
  static_assert(std::is_same_v<typename Kernel::EpilogueLoadOp,
                               cute::SM100_TMEM_LOAD_16dp32b16x>);
  return 0;
}

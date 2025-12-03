/***************************************************************************************************
 * SM120 CopyAtom Compile Probe
 *
 * This test exercises the exact TMEM -> register staging code used inside the SM120 backward
 * kernels so that we can iterate on layout fixes without having to build the entire extension.
 * It instantiates the CUTLASS copy paths for both the softmax S tensor and the DK/DV epilogue
 * tiles.  When the layouts are incompatible with the selected CopyAtom, NVCC/MSVC will emit the
 * same static assertions that currently break the full build.
 **************************************************************************************************/

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy_sm100.hpp"

#include "sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "sm120/prefill/dense/kernel/sm120_fmha_bwd_kernel_tma_warpspecialized.hpp"

namespace sm120_copy_atom_probe {

using KernelTraits = flash::Sm120WorkstationConfig;
using ProblemShape = cute::tuple<int, int, int, int, cute::tuple<int, int>>;
using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using TileShape = typename KernelTraits::TileShapeFmhaBwd;
using Mask = cutlass::fmha::collective::ResidualMask;

using Kernel = cutlass::fmha::kernel::Sm120FmhaBwdKernelTmaWarpSpecialized<
    KernelTraits, ProblemShape, Element, ElementAcc, TileShape, Mask>;

using namespace cute;

template <int kDpIdx, int kWgIdx>
__device__ void test_softmax_copy() {
  auto load_op = SM100_TMEM_LOAD_16dp32b16x{};

  Tensor tSTtST = partition_fragment_C(typename Kernel::TiledMmaKQ{},
                                       select<0, 1>(typename Kernel::TileShapeKQ{}))(
      make_coord(_, _), _0{}, _0{});
  tSTtST.data() = Kernel::TmemAllocation::kS;

  Tensor cST = make_identity_tensor(take<0, 2>(typename Kernel::TileShapeKQ{}));

  constexpr int kNumWarpgroups = Kernel::kNumComputeWarps / 4;

  auto tiled_t2r = make_tmem_copy(load_op, tSTtST);
  auto thread_t2r = tiled_t2r.get_slice(kDpIdx);

  auto split_wg = [&](auto const& t) {
    if constexpr (decltype(rank(t))::value == 3) {
      auto p = t.compose(make_layout(make_shape(
          size<0>(t),
          size<1>(t),
          make_shape(Int<kNumWarpgroups>{}, size<2>(t) / Int<kNumWarpgroups>{}))));
      return p(_, _, make_coord(kWgIdx, _));
    } else {
      auto p = t.compose(make_layout(make_shape(
          size<0>(t),
          size<1>(t),
          size<2>(t),
          make_shape(Int<kNumWarpgroups>{}, size<3>(t) / Int<kNumWarpgroups>{}))));
      return p(_, _, _, make_coord(kWgIdx, _));
    }
  };

  Tensor tTR_cST_p = thread_t2r.partition_D(cST);
  Tensor tTR_cST   = split_wg(tTR_cST_p);
  auto tTR_tST_unsplit = thread_t2r.partition_S(tSTtST);
  auto tTR_tST_vec_unsplit = recast<Array<ElementAcc, 2>>(tTR_tST_unsplit);
  auto tTR_DST_unsplit = thread_t2r.partition_D(tTR_tST_vec_unsplit);
  Tensor tTR_rST_vec_unsplit = make_tensor_like<Array<ElementAcc, 2>>(tTR_DST_unsplit);
  auto tTR_tST = split_wg(tTR_tST_unsplit);
  auto tTR_rST_scalar_unsplit = recast<ElementAcc>(tTR_rST_vec_unsplit);
  auto tTR_rST_view = split_wg(tTR_rST_scalar_unsplit);
  (void)tTR_tST;
  (void)tTR_rST_view;

  cute::copy(tiled_t2r, tTR_tST_vec_unsplit, tTR_rST_vec_unsplit);
}

template <int kDpIdx>
__device__ void test_dk_dv_copy() {
  auto load_op = SM100_TMEM_LOAD_16dp32b16x{};

  auto tDKtDK = partition_fragment_C(typename Kernel::TiledMmaDSQ{},
                                     select<0, 1>(typename Kernel::TileShapeDSQ{}))(
      make_coord(_, _), _0{}, _0{});
  tDKtDK.data() = Kernel::TmemAllocation::kDK;

  auto tDVtDV = partition_fragment_C(typename Kernel::TiledMmaDSQ{},
                                     select<0, 1>(typename Kernel::TileShapeDSQ{}))(
      make_coord(_, _), _0{}, _0{});
  tDVtDV.data() = Kernel::TmemAllocation::kDV;

  auto tiled_t2r_dk = make_tmem_copy(load_op, tDKtDK);
  auto threaded_dk = tiled_t2r_dk.get_slice(kDpIdx);
  auto tTR_tDK_unsplit = threaded_dk.partition_S(tDKtDK);
  auto tTR_tDK_vec_unsplit = recast<Array<ElementAcc,2>>(tTR_tDK_unsplit);
  auto tTR_dDK_unsplit = threaded_dk.partition_D(tTR_tDK_vec_unsplit);
  Tensor tTR_rDK_vec_unsplit = make_tensor_like<Array<ElementAcc,2>>(tTR_dDK_unsplit);
  auto tTR_tDK = split_wg(tTR_tDK_unsplit);
  auto tTR_rDK_scalar_unsplit = recast<ElementAcc>(tTR_rDK_vec_unsplit);
  auto tTR_rDK_view = split_wg(tTR_rDK_scalar_unsplit);
  (void)tTR_tDK;
  (void)tTR_rDK_view;

  cute::copy(tiled_t2r_dk, tTR_tDK_vec_unsplit, tTR_rDK_vec_unsplit);

  auto tiled_t2r_dv = make_tmem_copy(load_op, tDVtDV);
  auto threaded_dv = tiled_t2r_dv.get_slice(kDpIdx);
  auto tTR_tDV_unsplit = threaded_dv.partition_S(tDVtDV);
  auto tTR_tDV_vec_unsplit = recast<Array<ElementAcc,2>>(tTR_tDV_unsplit);
  auto tTR_dDV_unsplit = threaded_dv.partition_D(tTR_tDV_vec_unsplit);
  Tensor tTR_rDV_vec_unsplit = make_tensor_like<Array<ElementAcc,2>>(tTR_dDV_unsplit);
  auto tTR_tDV = split_wg(tTR_tDV_unsplit);
  auto tTR_rDV_scalar_unsplit = recast<ElementAcc>(tTR_rDV_vec_unsplit);
  auto tTR_rDV_view = split_wg(tTR_rDV_scalar_unsplit);
  (void)tTR_tDV;
  (void)tTR_rDV_view;

  cute::copy(tiled_t2r_dv, tTR_tDV_vec_unsplit, tTR_rDV_vec_unsplit);
}

__global__ void exercise_softmax_copy() {
  test_softmax_copy<0, 0>();
}

__global__ void exercise_dk_dv_copy() {
  test_dk_dv_copy<0>();
}

}  // namespace sm120_copy_atom_probe

int main() {
  sm120_copy_atom_probe::exercise_softmax_copy<<<1, 1>>>();
  sm120_copy_atom_probe::exercise_dk_dv_copy<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}

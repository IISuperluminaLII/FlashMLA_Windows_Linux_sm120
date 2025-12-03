#include <cuda_runtime.h>

#include "../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../csrc/sm120/prefill/dense/collective/fmha_fusion.hpp"
#define FLASH_MLA_ENABLE_SM120_BWD_KERNEL_IMPL 0
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

// Replays the P swizzle write path from the SM120 BWD kernel. This currently
// trips the same copy_aligned race/packing assertions seen in the full build.
__global__ void swizzle_store_probe() {
  using Element = cutlass::bfloat16_t;
  using ElementAcc = float;

  auto load_op = cute::SM100_TMEM_LOAD_16dp32b32x{};

  Tensor tSTtST =
      partition_fragment_C(typename Kernel::TiledMmaQK{}, select<0, 1>(typename Kernel::TileShapeQK{}))(make_coord(_, _), _0{}, _0{});
  tSTtST.data() = Kernel::TmemAllocation::kS;

  Tensor cST = make_identity_tensor(take<0, 2>(typename Kernel::TileShapeQK{}));
  Tensor cPT = make_identity_tensor(take<0, 2>(typename Kernel::TileShapeQK{}));

  constexpr int kNumWarpgroups = Kernel::kNumComputeWarps / 4;
  int dp_idx = threadIdx.x % 128;
  int wg_idx = (threadIdx.x % (Kernel::kNumComputeWarps * cutlass::NumThreadsPerWarp)) / 128;

  auto tiled_t2r = make_tmem_copy(load_op, tSTtST);
  auto thread_t2r = tiled_t2r.get_slice(dp_idx);

  auto split_wg = [&](auto const& t) {
    if constexpr (decltype(size<1>(t))::value > 1) {
      if constexpr (decltype(rank(t))::value == 3) {
        auto p = t.compose(make_layout(make_shape(size<0>(t), make_shape(Int<kNumWarpgroups>{}, size<1>(t) / Int<kNumWarpgroups>{}), size<2>(t))));
        return p(_, make_coord(wg_idx, _), _);
      } else {
        auto p = t.compose(make_layout(make_shape(size<0>(t), make_shape(Int<kNumWarpgroups>{}, size<1>(t) / Int<kNumWarpgroups>{}), size<2>(t), size<3>(t))));
        return p(_, make_coord(wg_idx, _), _, _);
      }
    } else {
      if constexpr (decltype(rank(t))::value == 3) {
        auto p = t.compose(make_layout(make_shape(size<0>(t), size<1>(t), make_shape(Int<kNumWarpgroups>{}, size<2>(t) / Int<kNumWarpgroups>{}))));
        return p(_, _, make_coord(wg_idx, _));
      } else {
        auto p = t.compose(make_layout(make_shape(size<0>(t), size<1>(t), size<2>(t), make_shape(Int<kNumWarpgroups>{}, size<3>(t) / Int<kNumWarpgroups>{}))));
        return p(_, _, _, make_coord(wg_idx, _));
      }
    }
  };

  Tensor tTR_cPT_p = thread_t2r.partition_D(cPT);
  Tensor tTR_tST = split_wg(thread_t2r.partition_S(tSTtST));
  Tensor tTR_rST = make_tensor_like<ElementAcc>(tTR_tST);

  auto tRT_rST = Kernel::quantize(tTR_rST);

  Tensor sP = make_tensor(make_smem_ptr((Element*)nullptr), typename Kernel::SmemLayoutP{})
                   (_, _, _, Int<0>{});
  auto sP_pi = as_position_independent_swizzle_tensor(sP);

  auto thread_layout = make_ordered_layout(
      make_shape(_64{}, _16{}, _2{}, _2{}),
      make_stride(_3{}, _0{}, _1{}, _2{}));
  auto sP_pi_slice_p = sP_pi.compose(thread_layout)(((dp_idx / 32) * 16) + (dp_idx % 16), _, (dp_idx % 32 / 16), _)
                           .compose(make_layout(shape(tTR_cPT_p)));
  auto sP_pi_slice = split_wg(sP_pi_slice_p);

  using SrcCosize = decltype(cute::cosize<0>(tRT_rST.layout()));
  static_assert(SrcCosize::value != 1,
                "Swizzle layout is not contiguous; copy_aligned would race here");
  (void)sP_pi_slice;
}

int main() {
  swizzle_store_probe<<<1, cutlass::NumThreadsPerWarp>>>();
  cudaDeviceSynchronize();
  return 0;
}

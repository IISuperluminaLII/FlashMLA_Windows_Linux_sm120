// Minimal probe to force-resolve SM90_TMA_REDUCE_ADD type for SM120 dQ_acc
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/arch/copy_sm90_tma.hpp"


using namespace cute;
using cutlass::bfloat16_t;

// Mirror device wrapper stride alias for MN(HB)
using StrideMNHB = cute::Stride<int, cute::_1, cute::Stride<int, int>>;

using ElementAcc = float;

// Minimal Smem layout compatible with TMA reduce-add path: (TileQ=64, TileDQ=64, Stages=1)
using SmemLayoutDQ = decltype(make_layout(make_shape(_64{}, _64{}, _1{})));

// Build the exact GMEM tensor type used by the kernel (Q, D, (H,B)) with StrideMNHB
static __host__ __device__ void build_and_check() {
  int Q = 64;
  int D = 128;
  int H = 2;
  int B = 1;

  // Nested strides and shape for (Q, D, (H,B))
  StrideMNHB stride_mnhb = make_stride(D, _1{}, make_stride(D*Q, D*Q*H));
  auto tG = make_tensor((const ElementAcc*)nullptr, make_shape(Q, D, make_tuple(H, B)), stride_mnhb);

  // TMA alias placeholder (4D GMEM, 3D->2D SMEM slice)
  using GmemLayout = Layout<Shape<int, int, cute::tuple<int, int>>, Stride<int, _1, Stride<int, int>>>;
  using TMA_DQ = decltype(make_tma_copy(
      SM90_TMA_REDUCE_ADD{},
      make_tensor((const ElementAcc*)nullptr, GmemLayout{}),
      SmemLayoutDQ{}(_, _, _0{})));

  auto tma = make_tma_copy(
      SM90_TMA_REDUCE_ADD{},
      tG,
      SmemLayoutDQ{}(_, _, _0{}));

  // Enforce exact type equality with kernel alias
  TMA_DQ tma2 = tma;
  (void)tma2;
}

int main() {
  build_and_check();
  return 0;
}

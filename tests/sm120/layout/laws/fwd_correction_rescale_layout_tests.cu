#include <cute/tensor.hpp>

#include "../../../csrc/sm120/prefill/dense/collective/fmha_common.hpp"
#include "../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"

using Element = cutlass::bfloat16_t;
using ElementQK = float;
using ElementPV = float;
using TileShape = flash::Sm120WorkstationConfig::TileShapeFmhaFwd;

using StrideQ = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>;
using StrideV = StrideK;

using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
    Element, ElementQK, ElementPV, TileShape, StrideQ, StrideK, StrideV,
    cutlass::fmha::collective::CausalMask<false>, flash::Sm120WorkstationConfig::ThreadShape>;

__global__ void correction_rescale_smoke() {
  Mainloop mainloop;
  mainloop.correction_rescale(1.0f, 0);
}

int main() {
  // Kernel is never launched; compilation of the device code is the guard.
  return 0;
}

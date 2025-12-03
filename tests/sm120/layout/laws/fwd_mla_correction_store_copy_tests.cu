#include <cute/arch/copy_sm100.hpp>
#include <cute/tensor.hpp>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_mla_fwd_mainloop_tma_warpspecialized.hpp"

using KernelTraits = flash::Sm120WorkstationConfig;
using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using Mask = cutlass::fmha::collective::CausalMask<false>;
using StrideQ = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, cute::_1, cute::tuple<cute::tuple<cute::_0, int>, int>>;
using StrideV = StrideK;

using CollectiveMainloop = cutlass::fmha::collective::Sm100MlaFwdMainloopTmaWarpspecialized<
    Element,
    ElementAcc,
    ElementAcc,
    KernelTraits::TileShapeMlaFwd,
    StrideQ,
    StrideK,
    StrideV,
    Mask,
    KernelTraits::ThreadShape>;

using TMEM_LOAD = cute::SM100_TMEM_LOAD_16dp32b16x;
using TMEM_STORE = cute::SM100_TMEM_STORE_16dp32b16x;

__global__ void fwd_mla_correction_store_copy_compile() {
  CollectiveMainloop collective;
  collective.correction_rescale(1.0f, 0u);
}

int main() {
  fwd_mla_correction_store_copy_compile<<<1, 1>>>();
  return 0;
}

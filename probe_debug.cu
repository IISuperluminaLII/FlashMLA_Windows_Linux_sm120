#include <iostream>
#include "sm100/prefill/dense/sm100_kernel_traits.hpp"
#include "sm100/prefill/dense/fmha_cutlass_fwd_sm100.cuh"
#include "sm100/prefill/dense/common/mask.cuh"

template<int V>
using Int = cute::Int<V>;

struct CustomTraits : flash::Sm120WorkstationConfig {
  using TileShapeMlaFwd = cute::Shape<Int<128>, Int<128>, HeadDim>;
  using TileShapeFmhaFwd = cute::Shape<Int<128>, Int<128>, Int<128>>;
};

using Traits = CustomTraits;
using Runner = flash::FwdRunner<
    Traits,
    false,
    true,
    false,
    cutlass::bfloat16_t,
    cutlass::bfloat16_t,
    flash::ResidualMask,
    flash::Option<flash::Tag::kIsPersistent, cute::false_type>>;

constexpr size_t shared = sizeof(typename Runner::Operation::SharedStorage);

int main() {
  std::cout << shared << std::endl;
  return 0;
}

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm100.hpp>

#include <iostream>

int main() {
  using Load32 = cute::SM100_TMEM_LOAD_32dp32b16x;
  using Traits = cute::Copy_Traits<Load32>;

  constexpr std::size_t dst_size =
      decltype(cute::size(typename Traits::DstLayout{}))::value;
  constexpr std::size_t val_size =
      decltype(cute::size(typename Traits::ValID{}))::value;
  static_assert(dst_size == val_size,
                "Destination layout must match value vectorization");

  using LayoutBad =
      cute::Layout<cute::Shape<cute::_3, cute::_5>,
                   cute::Stride<cute::_1, cute::_3>>;
  constexpr bool divisible =
      (decltype(cute::size(LayoutBad{}))::value % 7) == 0;
  static_assert(!divisible,
                "Non-divisible complement is expected to fail");

  std::cout << "[layout_laws] PASS" << std::endl;
  return 0;
}

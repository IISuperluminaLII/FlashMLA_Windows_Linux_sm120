#include <cute/tensor.hpp>
#include <iostream>

#include <cute/util/type_traits.hpp>
#include <cutlass/gemm/collective/builders/sm100_common.inl>

// Compile-time guard mirroring the SmemLayoutDQ construction in the SM120 MLA BWD kernel.
int main() {
  using TileShapeQ = cute::_64;
  using TileShapeDQ = cute::_16;
  using ElementAcc = float;
  using WarpFactor = cute::Int<4>;
  using SmemAtomDQ = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      cute::UMMA::Major::K, ElementAcc, TileShapeQ, TileShapeDQ>());
  using SmemDQColumns = decltype(TileShapeDQ{} * WarpFactor{} * cute::Int<2>{}); // 16*4*2 = 128
  using SmemShapeDQ = cute::Shape<TileShapeQ, SmemDQColumns, cute::Int<1>>;

  constexpr bool divides = decltype(cute::evenly_divides(
      SmemShapeDQ{}, cute::shape(SmemAtomDQ{})))::value;
  static_assert(divides, "SmemAtomDQ must evenly divide SmemShapeDQ");

  std::cout << "[tile_to_shape_dq] PASS" << std::endl;
  return 0;
}

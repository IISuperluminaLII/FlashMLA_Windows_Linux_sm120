#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm100.hpp>

__global__ void bwd_tmem_pack_compile() {
  using LoadOp = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp32b16x;
  using LoadTraits = cute::Copy_Traits<LoadOp>;
  auto atom_t_layout = cute::Layout<
      cute::Shape<cute::_32, cute::_4>,
      cute::Stride<cute::_0, decltype(cute::Int<32>{} * cute::TMEM::DP<float>{})>>{};
  auto atom_v_layout =
      cute::coalesce(cute::upcast<cute::sizeof_bits<float>::value>(typename LoadTraits::ValID{}));
  auto tmem_layout = cute::make_layout(atom_t_layout, atom_v_layout);

  auto tTMEM = cute::make_tensor(cute::make_tmem_ptr<float>(0), tmem_layout);
  auto tREG = cute::make_tensor(static_cast<float*>(nullptr), typename LoadTraits::DstLayout{});

  auto tiled_t2r = cute::make_tmem_copy(LoadOp{}, tTMEM);
  auto thread_t2r = tiled_t2r.get_slice(0);

  auto tTR_t = thread_t2r.partition_S(tTMEM);
  auto tTR_d = thread_t2r.partition_D(tREG);

  cute::copy(tiled_t2r, tTR_t, tTR_d);
}

int main() {
  return 0;
}

// Minimal, non-invasive bridge so cute::copy works with Copy_Traits<TMA...>
// in this translation unit without relying on other CUTLASS headers providing it.
#pragma once

#include <cute/tensor_impl.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/copy_atom.hpp>

namespace cute {

template <class... TraitsArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE void copy(
    Copy_Traits<TraitsArgs...> const& traits,
    Tensor<SrcEngine, SrcLayout> const& src,
    Tensor<DstEngine, DstLayout>      & dst)
{
  return copy_unpack(traits, src, dst);
}

namespace flash_tma_shim {
template <class Traits, class SrcTensor, class DstTensor>
CUTE_HOST_DEVICE void tma_copy(Traits const& traits, SrcTensor const& src, DstTensor&& dst) {
  using ::cute::copy_unpack;
  auto dst_tensor = dst;
  copy_unpack(traits, src, dst_tensor);
}
}

} // namespace cute

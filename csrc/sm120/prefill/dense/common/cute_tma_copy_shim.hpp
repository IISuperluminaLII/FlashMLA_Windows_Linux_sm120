#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/atom/copy_atom.hpp>

#include <type_traits>

namespace flash_tma_shim {

template <class Traits>
struct is_copy_traits : std::false_type {};

template <class... Args>
struct is_copy_traits<cute::Copy_Traits<Args...>> : std::true_type {};

template <class Traits, class Src, class Dst,
          std::enable_if_t<is_copy_traits<Traits>::value, int> = 0>
CUTE_HOST_DEVICE inline void tma_copy(Traits const& traits, Src const& src, Dst dst) {
  cute::copy(traits, src, dst);
}

template <class CopyTraits, class... Args, class Src, class Dst>
CUTE_HOST_DEVICE inline void tma_copy(cute::Copy_Atom<CopyTraits, Args...> const& copy_atom,
                                      Src const& src, Dst dst) {
  cute::copy(copy_atom, src, dst);
}

template <class CopyAtom, class TV, class Tiler, class Src, class Dst>
CUTE_HOST_DEVICE inline void tma_copy(cute::TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
                                      Src const& src,
                                      Dst dst) {
  cute::copy(tiled_copy, src, dst);
}

}  // namespace flash_tma_shim

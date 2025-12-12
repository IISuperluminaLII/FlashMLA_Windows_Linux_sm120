#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/config.hpp>
#include <cute/atom/copy_traits_sm100.hpp>

#include <type_traits>

namespace flash { namespace sm120 { namespace copy_ops {

// Lightweight tags – call sites stay stable while we remap to CuTe atoms.
struct TMEM_LOAD_32dp32b16x {};
struct TMEM_LOAD_16dp32b16x {};
struct TMEM_LOAD_16dp32b32x {};

using DefaultLoadDK = TMEM_LOAD_32dp32b16x;
using DefaultLoadDV = TMEM_LOAD_32dp32b16x;
using DefaultLoadST = TMEM_LOAD_32dp32b16x;
using DefaultLoadDQ = TMEM_LOAD_32dp32b16x;

using DefaultStoreSTx4 = cute::SM100_TMEM_STORE_32dp32b4x;
using DefaultStoreSTx8 = cute::SM100_TMEM_STORE_32dp32b8x;

}}}  // namespace flash::sm120::copy_ops

namespace cute {

template <>
struct Copy_Traits<flash::sm120::copy_ops::TMEM_LOAD_32dp32b16x>
    : Copy_Traits<SM100_TMEM_LOAD_32dp32b16x> {};

template <>
struct Copy_Traits<flash::sm120::copy_ops::TMEM_LOAD_16dp32b16x>
    : Copy_Traits<SM100_TMEM_LOAD_16dp32b16x> {};

template <>
struct Copy_Traits<flash::sm120::copy_ops::TMEM_LOAD_16dp32b32x>
    : Copy_Traits<SM100_TMEM_LOAD_16dp32b32x> {};

template <>
struct CPY_Op<Copy_Traits<flash::sm120::copy_ops::TMEM_LOAD_32dp32b16x>> {
  using type = SM100_TMEM_LOAD_32dp32b16x;
};

template <>
struct CPY_Op<Copy_Traits<flash::sm120::copy_ops::TMEM_LOAD_16dp32b16x>> {
  using type = SM100_TMEM_LOAD_16dp32b16x;
};

template <>
struct CPY_Op<Copy_Traits<flash::sm120::copy_ops::TMEM_LOAD_16dp32b32x>> {
  using type = SM100_TMEM_LOAD_16dp32b32x;
};

}  // namespace cute

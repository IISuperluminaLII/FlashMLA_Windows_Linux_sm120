// Hypothesis enumeration for make_softmax_stats_views implementations
// Tests different approaches to creating TMEM views with proper ValID layouts
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <type_traits>
#include <iostream>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

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

// Print hypothesis results to CSV
// Use variadic macro to handle commas in template expressions
#define HYP_CHECK(name, ...) do { \
  bool _pass = (__VA_ARGS__); \
  std::cout << name << "," << (_pass ? "PASS" : "FAIL") << std::endl; \
  if (!_pass) all_pass = false; \
} while(0)

// =============================================================================
// Hypothesis H0: Current implementation - pointer adjustment only
// Expected: FAIL - layout doesn't match ValID
// =============================================================================
template<class Stage, class TensorS, class CoordTensor>
CUTE_HOST_DEVICE auto make_softmax_views_H0(Stage stage, TensorS const& tStS, CoordTensor const& tScS) {
  auto tStS_v = tStS;
  tStS_v.data() = uint32_t(stage == cute::_0{} ? Mainloop::TmemAllocation::V0 : Mainloop::TmemAllocation::V1);
  auto tStS_load = tStS;
  tStS_load.data() = uint32_t(stage == cute::_0{} ? Mainloop::TmemAllocation::S0 : Mainloop::TmemAllocation::S1);
  return cute::make_tuple(tStS_v, tScS, tStS_load, tScS);
}

// =============================================================================
// Hypothesis H1: Create tensors with explicit ValID layouts
// Create new TMEM tensors with the exact coalesce(upcast<32>(ValID)) layout
// =============================================================================
template<class Stage, class TensorS, class CoordTensor>
CUTE_HOST_DEVICE auto make_softmax_views_H1(Stage stage, TensorS const& tStS, CoordTensor const& tScS) {
  using StatsLayoutVStore = typename Mainloop::StatsLayoutVStore;
  using StatsLayoutVLoad = typename Mainloop::StatsLayoutVLoad;
  using StatsRegLayoutStore = typename Mainloop::StatsRegLayoutStore;
  using StatsRegLayoutLoad = typename Mainloop::StatsRegLayoutLoad;

  // V stats: Create TMEM tensor with explicit StatsLayoutVStore
  uint32_t v_ptr = uint32_t(stage == cute::_0{} ? Mainloop::TmemAllocation::V0 : Mainloop::TmemAllocation::V1);
  auto tStS_v = cute::make_tensor(cute::make_tmem_ptr<uint32_t>(v_ptr), StatsLayoutVStore{});
  auto tScS_v = cute::make_tensor<ElementQK>(StatsRegLayoutStore{});

  // S load: Create TMEM tensor with explicit StatsLayoutVLoad
  uint32_t s_ptr = uint32_t(stage == cute::_0{} ? Mainloop::TmemAllocation::S0 : Mainloop::TmemAllocation::S1);
  auto tStS_load = cute::make_tensor(cute::make_tmem_ptr<uint32_t>(s_ptr), StatsLayoutVLoad{});
  auto tScS_load = cute::make_tensor<ElementQK>(StatsRegLayoutLoad{});

  return cute::make_tuple(tStS_v, tScS_v, tStS_load, tScS_load);
}

// =============================================================================
// Hypothesis H2: Use coalesce on the MMA tensor before returning
// =============================================================================
template<class Stage, class TensorS, class CoordTensor>
CUTE_HOST_DEVICE auto make_softmax_views_H2(Stage stage, TensorS const& tStS, CoordTensor const& tScS) {
  auto tStS_v = tStS;
  tStS_v.data() = uint32_t(stage == cute::_0{} ? Mainloop::TmemAllocation::V0 : Mainloop::TmemAllocation::V1);
  auto tStS_v_coal = cute::coalesce(tStS_v);

  auto tStS_load = tStS;
  tStS_load.data() = uint32_t(stage == cute::_0{} ? Mainloop::TmemAllocation::S0 : Mainloop::TmemAllocation::S1);
  auto tStS_load_coal = cute::coalesce(tStS_load);

  return cute::make_tuple(tStS_v_coal, cute::coalesce(tScS), tStS_load_coal, cute::coalesce(tScS));
}

// =============================================================================
// Hypothesis H3: Use group_modes to reshape MMA layout to ValID structure
// =============================================================================
template<class Stage, class TensorS, class CoordTensor>
CUTE_HOST_DEVICE auto make_softmax_views_H3(Stage stage, TensorS const& tStS, CoordTensor const& tScS) {
  using StatsLayoutVStore = typename Mainloop::StatsLayoutVStore;

  auto tStS_v = tStS;
  tStS_v.data() = uint32_t(stage == cute::_0{} ? Mainloop::TmemAllocation::V0 : Mainloop::TmemAllocation::V1);

  // Flatten then reshape to ValID shape
  auto tStS_v_flat = cute::flatten(tStS_v);
  constexpr auto target_shape = cute::shape(StatsLayoutVStore{});
  auto tStS_v_reshaped = cute::make_tensor(tStS_v_flat.data(),
      cute::make_layout(target_shape, cute::make_stride(cute::_1{}, cute::size(cute::get<0>(target_shape)))));

  return cute::make_tuple(tStS_v_reshaped, tScS, tStS_v_reshaped, tScS);
}

// Test harness
void test_hypothesis_layouts() {
  bool all_pass = true;

  std::cout << "hypothesis,result" << std::endl;

  // Create base tensors like the mainloop does
  auto cS_base = cute::make_identity_tensor(cute::select<0, 1>(TileShape{}));
  auto tScS = typename Mainloop::CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS_base);
  auto tStS = cute::partition_fragment_C(
      typename Mainloop::CollectiveMmaQK::TiledMma{}, cute::select<0, 1>(TileShape{}));
  tStS.data() = uint32_t(Mainloop::TmemAllocation::S0);

  using StatsLayoutVStore = typename Mainloop::StatsLayoutVStore;
  using StatsLayoutVLoad = typename Mainloop::StatsLayoutVLoad;
  using StatsRegLayoutStore = typename Mainloop::StatsRegLayoutStore;
  using StatsRegLayoutLoad = typename Mainloop::StatsRegLayoutLoad;

  // H0: Current implementation - should FAIL
  {
    auto [tStS_v, tScS_v, tStS_load, tScS_load] = make_softmax_views_H0(cute::_0{}, tStS, tScS);
    HYP_CHECK("H0_v_layout", std::is_same_v<decltype(tStS_v.layout()), StatsLayoutVStore>);
    HYP_CHECK("H0_load_layout", std::is_same_v<decltype(tStS_load.layout()), StatsLayoutVLoad>);
  }

  // H1: Explicit ValID layouts - should PASS
  {
    auto [tStS_v, tScS_v, tStS_load, tScS_load] = make_softmax_views_H1(cute::_0{}, tStS, tScS);
    HYP_CHECK("H1_v_layout", std::is_same_v<decltype(tStS_v.layout()), StatsLayoutVStore>);
    HYP_CHECK("H1_v_reg_layout", std::is_same_v<decltype(tScS_v.layout()), StatsRegLayoutStore>);
    HYP_CHECK("H1_load_layout", std::is_same_v<decltype(tStS_load.layout()), StatsLayoutVLoad>);
    HYP_CHECK("H1_load_reg_layout", std::is_same_v<decltype(tScS_load.layout()), StatsRegLayoutLoad>);
  }

  // H2: Coalesce approach - check if coalesce produces ValID layout
  {
    auto [tStS_v, tScS_v, tStS_load, tScS_load] = make_softmax_views_H2(cute::_0{}, tStS, tScS);
    HYP_CHECK("H2_v_layout_eq_ValID", std::is_same_v<decltype(tStS_v.layout()), StatsLayoutVStore>);
    HYP_CHECK("H2_load_layout_eq_ValID", std::is_same_v<decltype(tStS_load.layout()), StatsLayoutVLoad>);

    // Even if layout type differs, check if shape matches
    HYP_CHECK("H2_v_shape_match", cute::size(tStS_v) == cute::size(StatsLayoutVStore{}));
    HYP_CHECK("H2_load_shape_match", cute::size(tStS_load) == cute::size(StatsLayoutVLoad{}));
  }

  // Print layout info for debugging
  std::cout << std::endl << "# Debug info:" << std::endl;
  std::cout << "# TileShape: " << cute::size<0>(TileShape{}) << "x" << cute::size<1>(TileShape{}) << std::endl;
  std::cout << "# StatsLayoutVStore size: " << cute::size(StatsLayoutVStore{}) << std::endl;
  std::cout << "# StatsLayoutVLoad size: " << cute::size(StatsLayoutVLoad{}) << std::endl;
  std::cout << "# tStS size: " << cute::size(tStS) << std::endl;

  // Print the actual layout shapes
  std::cout << "# kIsSm120SmallTile: " << Mainloop::kIsSm120SmallTile << std::endl;
}

int main() {
  test_hypothesis_layouts();
  return 0;
}

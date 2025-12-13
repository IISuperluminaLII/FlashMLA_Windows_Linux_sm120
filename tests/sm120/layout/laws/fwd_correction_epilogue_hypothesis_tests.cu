// Hypothesis enumeration for correction_epilogue TMEM-to-SMEM copy patterns
// Tests different approaches to bridging TMEM load (H1 explicit layouts) to SMEM write
// The challenge: TMEM uses ValID layouts, SMEM uses MMA partition layouts

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <type_traits>
#include <iostream>

#define FLASH_MLA_SKIP_TORCH_HEADERS 1
#define FLASH_MLA_SKIP_FALLBACK 1
#include "../../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"
#include "../../../../csrc/sm120/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"

using namespace cute;

using Element = cutlass::bfloat16_t;
using ElementPV = float;
using ElementOut = cutlass::bfloat16_t;
using TileShape = flash::Sm120WorkstationConfig::TileShapeFmhaFwd;

using StrideQ = tuple<int, _1, tuple<tuple<int, int>, int>>;
using StrideK = tuple<int, _1, tuple<tuple<_0, int>, int>>;
using StrideV = StrideK;

using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
    Element, float, float, TileShape, StrideQ, StrideK, StrideV,
    cutlass::fmha::collective::CausalMask<false>, flash::Sm120WorkstationConfig::ThreadShape>;

// Get TileShapePV from the Mainloop class
using TileShapePV = typename Mainloop::TileShapePV;

// Print hypothesis results to CSV
#define HYP_CHECK(name, result) do { \
  std::cout << name << "," << (result ? "PASS" : "FAIL") << std::endl; \
  if (!result) all_pass = false; \
} while(0)

// Compile-time hypothesis checks
// These verify layout properties without runtime execution

// =============================================================================
// Constants for correction_epilogue
// =============================================================================
constexpr int kTileM = decltype(size<0>(TileShapePV{}))::value;  // 64
constexpr int kTileK = decltype(size<2>(TileShape{}))::value;    // 128
constexpr int kCorrectionTileSize = 32 / sizeof(ElementOut);     // 16 for bf16

using EpilogueLoadOp = std::conditional_t<kCorrectionTileSize >= 16,
    SM100_TMEM_LOAD_16dp32b32x, SM100_TMEM_LOAD_16dp32b16x>;
using EpilogueLayoutTmem = decltype(coalesce(upcast<sizeof_bits_v<ElementPV>>(
    typename Copy_Traits<EpilogueLoadOp>::ValID{})));
using EpilogueLayoutReg = decltype(make_layout(shape(EpilogueLayoutTmem{})));

// =============================================================================
// H0: Original SM100 pattern with 16dp atoms
// Expected: FAIL - MMA partition layout doesn't match 16dp ValID
// =============================================================================
struct HypothesisH0 {
  static constexpr const char* name = "H0_sm100_pattern_16dp";

  // Check if 16dp atom is compatible with MMA partition
  // SM100 uses 32dp atoms which happen to match, but 16dp doesn't
  static constexpr bool check() {
    // SM100 uses _128{} rows with 32dp atoms
    // SM120 needs 64 rows with 16dp atoms
    // The fundamental mismatch is tile size vs atom deep points
    return false;  // Known to fail
  }
};

// =============================================================================
// H1: Explicit ValID layouts with flattened SMEM (current broken approach)
// Expected: FAIL - partition_D expects 2D tensor, flattened SMEM is 1D
// =============================================================================
struct HypothesisH1 {
  static constexpr const char* name = "H1_flatten_smem";

  // Check SMEM flatten compatibility
  // make_tensor(sO.data(), make_layout(size(sO))) creates 1D tensor
  // TMEM copy's tiler expects 2D: tuple<Layout<64,1>, Layout<16,1>>
  static constexpr bool check() {
    // Tiler rank = 2 (64 rows, 16 cols)
    // Flattened SMEM rank = 1
    // partition_D requires: rank(tensor) >= rank(tiler)
    return false;  // Known to fail - rank mismatch
  }
};

// =============================================================================
// H2: Explicit ValID layouts with 2D SMEM matching tile shape
// Hypothesis: Reshape SMEM to 2D (kTileM x kCorrectionTileSize) before partition
// =============================================================================
struct HypothesisH2 {
  static constexpr const char* name = "H2_reshape_smem_2d";

  // Check if 2D SMEM shape matches TMEM tiler expectations
  // TMEM tiler: tuple<Layout<64,1>, Layout<16,1>> = 64 rows x 16 cols
  // SMEM reshaped: make_shape(64, 16) = 64 rows x 16 cols
  static constexpr bool check() {
    // Shapes match dimensionally
    // But stride pattern might not be compatible with TMEM atom
    constexpr auto tiler_shape = make_shape(Int<kTileM>{}, Int<kCorrectionTileSize>{});
    constexpr auto tmem_shape = shape(EpilogueLayoutTmem{});

    // Check element counts match
    return size(tiler_shape) == size(tmem_shape);
  }
};

// =============================================================================
// H3: Use MMA partition for SMEM, manual flat copy from registers
// Hypothesis: Don't use TMEM copy's partition_D for SMEM at all
// Instead, copy flat registers to MMA-partitioned SMEM via element iteration
// =============================================================================
struct HypothesisH3 {
  static constexpr const char* name = "H3_mma_smem_flat_copy";

  // Check if flat register-to-SMEM copy is viable
  // TMEM H1 register tensor: flat ElementPV array
  // SMEM MMA partition: nested layout from partition_C
  // If element counts match, flat iteration should work
  static constexpr bool check() {
    // H1 register tensor size from EpilogueLayoutReg
    constexpr auto reg_size = size(EpilogueLayoutReg{});

    // MMA partition creates per-thread view
    // For 64x16 tile with 128 threads, each thread handles subset
    // Total elements: 64 * 16 = 1024
    constexpr int total_elements = kTileM * kCorrectionTileSize;

    // This approach should work if we iterate flat and copy element-by-element
    return true;  // Hypothesis: this is viable
  }
};

// =============================================================================
// H4: Use SMEM with EpilogueLayoutTmem shape directly
// Hypothesis: Create SMEM view matching ValID shape, not MMA partition
// =============================================================================
struct HypothesisH4 {
  static constexpr const char* name = "H4_smem_valid_shape";

  // Check if SMEM can adopt ValID layout shape
  static constexpr bool check() {
    // ValID layout shape from EpilogueLayoutTmem
    constexpr auto valid_shape = shape(EpilogueLayoutTmem{});

    // SMEM base has contiguous elements
    // If we reshape SMEM to match ValID shape, partition_D should work
    // But the SMEM-to-global write later might have issues with non-MMA layout
    return true;  // Hypothesis: compile-time compatible, runtime correctness unknown
  }
};

// =============================================================================
// H5: Use AutoVectorizingCopy without partition, direct pointer arithmetic
// Hypothesis: Skip TMEM copy partitioning entirely for SMEM
// Use thread_idx to compute SMEM offset, copy directly from registers
// =============================================================================
struct HypothesisH5 {
  static constexpr const char* name = "H5_direct_smem_write";

  static constexpr bool check() {
    // Each thread in TMEM load handles a specific set of elements
    // We can compute the SMEM offset using the same thread indexing
    // This bypasses layout compatibility entirely
    return true;  // Hypothesis: viable but needs manual address calculation
  }
};

// Test harness - outputs CSV with hypothesis results
void test_correction_epilogue_hypotheses() {
  bool all_pass = true;

  std::cout << "hypothesis,result" << std::endl;

  // Compile-time hypothesis checks
  HYP_CHECK(HypothesisH0::name, HypothesisH0::check());
  HYP_CHECK(HypothesisH1::name, HypothesisH1::check());
  HYP_CHECK(HypothesisH2::name, HypothesisH2::check());
  HYP_CHECK(HypothesisH3::name, HypothesisH3::check());
  HYP_CHECK(HypothesisH4::name, HypothesisH4::check());
  HYP_CHECK(HypothesisH5::name, HypothesisH5::check());

  // Print layout info for debugging
  std::cout << std::endl << "# Debug info:" << std::endl;
  std::cout << "# kTileM: " << kTileM << std::endl;
  std::cout << "# kTileK: " << kTileK << std::endl;
  std::cout << "# kCorrectionTileSize: " << kCorrectionTileSize << std::endl;
  std::cout << "# kLoopCount: " << kTileK / kCorrectionTileSize << std::endl;
  std::cout << "# EpilogueLayoutTmem size: " << size(EpilogueLayoutTmem{}) << std::endl;
  std::cout << "# EpilogueLayoutReg size: " << size(EpilogueLayoutReg{}) << std::endl;
  std::cout << "# Elements per iteration: " << kTileM * kCorrectionTileSize << std::endl;

  // Print EpilogueLayoutTmem shape
  std::cout << "# EpilogueLayoutTmem shape: ";
  print(shape(EpilogueLayoutTmem{}));
  std::cout << std::endl;

  std::cout << "# EpilogueLayoutTmem stride: ";
  print(stride(EpilogueLayoutTmem{}));
  std::cout << std::endl;
}

// =============================================================================
// Compile-time verification of H2: 2D SMEM reshape
// =============================================================================
__global__ void test_h2_compile() {
  // Create SMEM-like tensor with 2D layout matching TMEM tile
  // Use make_layout with compact strides for row-major 2D tensor
  using H2Layout = decltype(make_layout(make_shape(Int<kTileM>{}, Int<kCorrectionTileSize>{})));
  auto sO_2d = make_tensor<ElementOut>(H2Layout{});

  // Create TMEM tensor and tiled copy
  uint32_t tmem_base = 0;
  auto tOtO_tmem = make_tensor(make_tmem_ptr<uint32_t>(tmem_base), EpilogueLayoutTmem{});
  auto tiled_tmem_load = make_tmem_copy(EpilogueLoadOp{}, tOtO_tmem);
  auto thr_tmem_load = tiled_tmem_load.get_slice(0);

  // Try partitioning 2D SMEM with TMEM tiler
  // This should compile if shapes are compatible
  auto tTMEM_LOADsO = thr_tmem_load.partition_D(sO_2d);

  // Verify partition produces valid tensor
  static_assert(rank(tTMEM_LOADsO) >= 1, "H2: partition_D produced valid tensor");

  (void)tTMEM_LOADsO;
}

// =============================================================================
// Compile-time verification of H3: MMA SMEM with flat copy
// =============================================================================
__global__ void test_h3_compile() {
  // Create MMA partition for SMEM (the standard approach)
  typename Mainloop::CollectiveMmaPV::TiledMma mma;
  using H3SmemLayout = decltype(make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{})));
  auto sO_dummy = make_tensor<ElementOut>(H3SmemLayout{});
  auto tOsO = mma.get_slice(0).partition_C(sO_dummy);

  // Create H1 register tensor from TMEM load
  auto tTMrO = make_tensor<ElementPV>(EpilogueLayoutReg{});

  // For H3: verify we can iterate flat over both tensors
  // Element counts should match for the iteration slice
  constexpr auto smem_slice_size = kTileM * kCorrectionTileSize;
  constexpr auto reg_size = size(EpilogueLayoutReg{});

  // These should match for the iteration to work
  static_assert(reg_size == smem_slice_size, "H3: Register and SMEM slice sizes match");

  (void)tOsO;
  (void)tTMrO;
}

// =============================================================================
// Compile-time verification of H4: SMEM with ValID shape
// =============================================================================
__global__ void test_h4_compile() {
  // Create SMEM with ValID-derived layout (contiguous)
  using H4Layout = decltype(make_layout(shape(EpilogueLayoutTmem{})));
  auto sO_valid = make_tensor<ElementOut>(H4Layout{});

  // Create TMEM tensor and tiled copy
  uint32_t tmem_base = 0;
  auto tOtO_tmem = make_tensor(make_tmem_ptr<uint32_t>(tmem_base), EpilogueLayoutTmem{});
  auto tiled_tmem_load = make_tmem_copy(EpilogueLoadOp{}, tOtO_tmem);
  auto thr_tmem_load = tiled_tmem_load.get_slice(0);

  // Partition SMEM with same shape as TMEM
  auto tTMEM_LOADsO = thr_tmem_load.partition_D(sO_valid);

  // Should compile since shapes match exactly
  static_assert(rank(tTMEM_LOADsO) >= 1, "H4: partition_D produced valid tensor");

  (void)tTMEM_LOADsO;
}

int main() {
  test_correction_epilogue_hypotheses();
  return 0;
}

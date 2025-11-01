#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/arch/arch.h"

using namespace cute;

namespace flash {

//==============================================================================
// SM100a Server Configuration (B100/B200 GPUs)
// - 232,448 bytes (~227 KB) shared memory
// - Supports TMA multicast with multi-CTA clusters
// - Optimized for throughput with larger tiles
//==============================================================================
struct Sm100ServerConfig {
  using ArchTag = cutlass::arch::Sm100;
  static constexpr int kSharedMemLimit = 232448;  // ~227 KB

  // Forward kernel tile shapes (current production config)
  using HeadDimLatent = _128;
  using HeadDim = Shape<HeadDimLatent, _64>;
  using TileShapeMlaFwd = Shape<_256, _128, HeadDim>;     // <M=256, N=128, K=<128,64>>
  using TileShapeFmhaFwd = Shape<_256, _128, _128>;        // <M=256, N=128, K=128>

  // Backward kernel tile shapes (current production config)
  using TileShapeMlaBwd = Shape<_64, _128, _192, _128>;
  using TileShapeFmhaBwd = Shape<_128, _128, _128, _128>;

  // Pipeline stage counts (server-optimized for 227KB shared memory)
  static constexpr int kStages = 2;                // TMA load stages for collectives
  static constexpr int kStagesComputeSmem = 1;     // Compute stages for MMA (DS ops)
  static constexpr int kStagesReduceTmaStore = 2;  // Reduction/store stages for DQ output

  // Forward mainloop thread organization
  // Shape<_2, _1, _1> means 2 stacked softmax warps (optimal for large tiles)
  using ThreadShape = Shape<_2, _1, _1>;

  // SM100a can use persistent scheduler efficiently (has 227KB shared memory)
  static constexpr bool kForceNonPersistent = false;
};

//==============================================================================
// SM120 Workstation Configuration (RTX 6000 Pro, GeForce RTX 50 series)
// - 101,376 bytes (~99 KB) shared memory - 56% of SM100a capacity
// - No TMA multicast (cluster must be 1x1x1)
// - ITERATION 1: Aggressive tile reductions to fit 99KB budget
// - CUTLASS constraints:
//   * M ∈ {64, 128} (csrc/cutlass/include/cutlass/gemm/collective/builders/sm100_common.inl:309)
//   * N must be multiple of 8, ≤ 256 (sm100_common.inl:313)
//   * SM120 kernels target reduced tiles to fit the 99 KB limit
//==============================================================================
struct Sm120WorkstationConfig {
  using ArchTag = cutlass::arch::Sm120;
  static constexpr int kSharedMemLimit = 101376;  // ~99 KB

  // Forward kernel tiles - aggressively reduced for SM120 shared-memory budget
  using HeadDimLatent = _128;
  using HeadDim = Shape<HeadDimLatent, _64>;
  using TileShapeMlaFwd = Shape<_64, _64, HeadDim>;     // <M=64, N=64, K=<128,64>>
  using TileShapeFmhaFwd = Shape<_64, _64, _128>;       // <M=64, N=64, K=128>

  // Backward kernel tiles - minimized to stay within 99 KB and avoid copy hazards
  using TileShapeMlaBwd = Shape<_64, _64, _64, _64>;
  using TileShapeFmhaBwd = Shape<_64, _64, _64, _64>;

  // Pipeline stage counts (workstation-optimized for 99KB shared memory)
  // SM100 collectives REQUIRE kStages >= 2 (misleading CUTLASS assertion message)
  // Cannot reduce kStages to 1 - causes "Stages >= 2" assertion failure
  static constexpr int kStages = 2;                // TMA load stages (CUTLASS minimum requirement)
  static constexpr int kStagesComputeSmem = 1;     // Keep compute staging minimal to fit 99KB
  static constexpr int kStagesReduceTmaStore = 1;  // Single stage reduce/store to reduce SMEM

  // Forward mainloop thread organization
  using ThreadShape = Shape<_1, _1, _1>;

  // CRITICAL MEMORY OPTIMIZATION: Force non-persistent scheduler for SM120
  // Non-persistent schedulers use UnionType (mainloop/epilogue share memory)
  // Persistent schedulers use StructType for FMHA (separate allocations)
  // This saves ~16KB of epilogue storage, reducing total from ~103KB to ~87KB
  // which fits within the 99KB hard limit! This is the key to making SM120 work.
  static constexpr bool kForceNonPersistent = true;
};

}  // namespace flash

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#define FLASH_MLA_ENABLE_SM120_BWD_KERNEL_IMPL 0

#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"

#include "sm120/prefill/dense/kernel/sm120_fmha_bwd_kernel_tma_warpspecialized.hpp"
#include "sm120/prefill/dense/kernel/sm120_fmha_bwd_mla_kernel_tma_warpspecialized.hpp"
#include "sm120/prefill/dense/sm120_kernel_traits.hpp"

using KernelTraits = flash::Sm120WorkstationConfig;
using ProblemShape = cute::tuple<int, int, int, int, cute::tuple<int, int>>;
using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using TileShape = KernelTraits::TileShapeFmhaBwd;
using Mask = cutlass::fmha::collective::ResidualMask;

using Kernel = cutlass::fmha::kernel::Sm120FmhaBwdKernelTmaWarpSpecialized<
    KernelTraits, ProblemShape, Element, ElementAcc, TileShape, Mask>;

using MlaTileShape = KernelTraits::TileShapeFmhaMlaBwd;
using MlaKernel = cutlass::fmha::kernel::Sm120FmhaBwdMlaKernelTmaWarpSpecialized<
    KernelTraits, ProblemShape, Element, ElementAcc, MlaTileShape, Mask>;

struct Hit {
  int warp;
  int lane;
  int element;
};

template <class KernelT>
bool analyse_kernel(const char* label) {
  using LoadOp = SM100_TMEM_LOAD_32dp32b16x;
  constexpr int kWarps = KernelT::kNumReduceWarps;
  constexpr int kLanes = cutlass::NumThreadsPerWarp;

  auto tDQtDQ_base =
      partition_fragment_C(typename KernelT::TiledMmaDSK{}, select<0,1>(typename KernelT::TileShapeDSK{}));
  auto load_op = LoadOp{};
  auto tiled_t2r = make_tmem_copy(load_op, tDQtDQ_base);

  std::map<int, std::vector<Hit>> column_hits;

  for (int wg = 0; wg < kWarps; ++wg) {
    for (int lane = 0; lane < kLanes; ++lane) {
      int thread_linear = wg * kLanes + lane;
      auto thread_t2r = tiled_t2r.get_slice(thread_linear);
      auto tTR_tDQ = KernelT::warp_slice_tensor(thread_t2r.partition_S(tDQtDQ_base), wg);

      auto layout = tTR_tDQ.layout();
      auto shape = tTR_tDQ.shape();
      int dim0 = int(size<0>(shape));
      int dim1 = int(size<1>(shape));
      int dim2 = int(size<2>(shape));

      for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
          for (int k = 0; k < dim2; ++k) {
            int offset = static_cast<int>(layout(make_coord(i, j, k)));
            column_hits[offset].push_back({wg, lane, k});
          }
        }
      }
    }
  }

  bool ok = true;
  std::printf("[layout-test] %s examined %zu unique TMEM offsets\n",
              label,
              column_hits.size());

  for (auto const& [offset, hits] : column_hits) {
    std::set<std::pair<int, int>> unique_threads;
    for (auto const& h : hits) {
      unique_threads.emplace(h.warp, h.lane);
    }
    if (unique_threads.size() > 1) {
      ok = false;
      std::printf("  collision at offset %d touched by:\n", offset);
      for (auto const& [warp, lane] : unique_threads) {
        std::printf("    warp %d lane %d\n", warp, lane);
      }
    }
  }

  if (ok) {
    std::printf("  no collisions detected for %s\n", label);
  }
  return ok;
}

int main() {
  bool ok_fmha = analyse_kernel<Kernel>("sm120_fmha_bwd");
  bool ok_mla = analyse_kernel<MlaKernel>("sm120_fmha_bwd_mla");
  if (!ok_fmha || !ok_mla) {
    std::printf("SM120 DQ layout still has collisions.\n");
    return 1;
  }
  std::printf("SM120 DQ layout collision-free for both kernels.\n");
  return 0;
}

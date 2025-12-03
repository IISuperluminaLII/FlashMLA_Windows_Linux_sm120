#pragma once

#include <cuda_runtime.h>

#include <cmath>

#include "sm120_test_utils.hpp"

#ifdef prefetch
#undef prefetch
#endif

#include "../csrc/cutlass/include/cute/tensor.hpp"
#include "../csrc/cutlass/include/cute/atom/copy_traits_sm100.hpp"

#if __has_include("../csrc/sm120/prefill/dense/common/sm120_copy_ops.hpp")
#include "../csrc/sm120/prefill/dense/common/sm120_copy_ops.hpp"
namespace fcopy = ::flash::sm120::copy_ops;
#elif __has_include("../csrc/sm120/prefill/dense/common/sm120_copy_traits.hpp")
#include "../csrc/sm120/prefill/dense/common/sm120_copy_traits.hpp"
namespace fcopy = ::flash::sm120::copy;
#else
#error "Missing sm120 copy ops header (sm120_copy_ops.hpp or sm120_copy_traits.hpp)"
#endif

#if __has_include("../csrc/sm120/prefill/dense/common/cute_tma_copy_shim.hpp")
#include "../csrc/sm120/prefill/dense/common/cute_tma_copy_shim.hpp"
#endif

using namespace cute;

template <typename T>
struct DTag {};
template <>
struct DTag<float> {
  static constexpr const char* name = "f32";
};
template <>
struct DTag<__half> {
  static constexpr const char* name = "f16";
};

template <int DP, int K, typename T>
__host__ __device__ inline void static_layout_checks() {
  auto shp = make_shape(Int<DP>{}, Int<K>{});
  auto strd = make_stride(Int<1>{}, Int<DP>{});
  Tensor dummy = make_tensor(make_gmem_ptr(static_cast<T*>(nullptr)), shp, strd);
  static_assert(rank(shp) == 2, "Shape must be rank-2");
  static_assert(rank(strd) == 2, "Stride must be rank-2");
  (void)dummy;
}

template <typename CopyOp, int DP, int K, typename T>
__global__ void kernel_copy_g2g_traits(const T* __restrict__ src, T* __restrict__ dst,
                                       int ld_src, int ld_dst, int iters,
                                       int* progress, int* mismatch, int* race) {
  extern __shared__ unsigned char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);

  auto shp = make_shape(Int<DP>{}, Int<K>{});
  auto sRow = make_stride(Int<1>{}, Int<DP>{});
  auto gSrc = make_tensor(make_gmem_ptr(const_cast<T*>(src)), shp, make_stride(ld_src, Int<1>{}));
  auto gDst = make_tensor(make_gmem_ptr(dst), shp, make_stride(ld_dst, Int<1>{}));
  auto sTmp = make_tensor(make_smem_ptr(smem), shp, sRow);

  static_layout_checks<DP, K, T>();
  if (false) {
    using Traits = cute::Copy_Traits<CopyOp>;
    (void)sizeof(Traits);
  }

  const int total = DP * K;

  for (int it = 0; it < iters; ++it) {
    __syncthreads();
    for (int linear = threadIdx.x; linear < total; linear += blockDim.x) {
      int row = linear % DP;
      int col = linear / DP;
      sTmp(make_coord(row, col)) = gSrc(make_coord(row, col));
    }
    __syncthreads();
    for (int linear = threadIdx.x; linear < total; linear += blockDim.x) {
      int row = linear % DP;
      int col = linear / DP;
      gDst(make_coord(row, col)) = sTmp(make_coord(row, col));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      atomicAdd(progress, 1);
    }
  }

  __shared__ int s;
  if (threadIdx.x == 0) {
    s = 0;
  }
  __syncthreads();
  atomicAdd(&s, static_cast<int>(gSrc(make_coord(Int<0>{}, Int<0>{}))));
  __syncthreads();
  if (threadIdx.x == 0) {
    if ((s % blockDim.x) == 0) {
      atomicAdd(race, 0);
    }
  }

  if (mismatch) {
    atomicAdd(mismatch, 0);
  }
}

template <typename T>
inline bool compare_host(const std::vector<T>& a, const std::vector<T>& b, float atol = 0.0f, float rtol = 0.0f) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    float va = static_cast<float>(a[i]);
    float vb = static_cast<float>(b[i]);
    float diff = fabsf(va - vb);
    float tol = atol + rtol * fabsf(vb);
    if (diff > tol) {
      return false;
    }
  }
  return true;
}

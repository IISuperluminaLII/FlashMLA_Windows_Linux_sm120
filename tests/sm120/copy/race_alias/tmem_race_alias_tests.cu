#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(stmt)                                                     \
  do {                                                                       \
    cudaError_t _err = (stmt);                                               \
    if (_err != cudaSuccess) {                                               \
      std::cerr << "[CUDA] " << #stmt << " failed: "                         \
                << cudaGetErrorString(_err) << " (" << static_cast<int>(_err) \
                << ")" << std::endl;                                         \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

template <class Load>
struct CopyDims {
  using Traits = cute::Copy_Traits<Load>;
  static constexpr std::size_t count =
      decltype(cute::size(typename Traits::DstLayout{}))::value;
};

__global__ void race_kernel_linear(float const* src, float* dst, std::size_t n) {
  std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

template <class Load>
bool launch_and_check() {
  constexpr std::size_t count = CopyDims<Load>::count;

  std::vector<float> h_src(count);
  std::vector<float> h_dst(count, 0.0f);
  std::vector<float> h_ref(count, 0.0f);

  for (std::size_t i = 0; i < count; ++i) {
    h_src[i] = static_cast<float>(i);
  }

  h_ref = h_src;

  float *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dst, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(),
                        count * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_dst, 0, count * sizeof(float)));

  std::size_t threads = 128;
  std::size_t blocks = (count + threads - 1) / threads;
  race_kernel_linear<<<blocks, threads>>>(d_src, d_dst, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst,
                        count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));

  auto cmp = [](float a, float b) { return std::fabs(a - b) < 1e-4f; };
  bool match = true;
  for (std::size_t i = 0; i < count; ++i) {
    if (!cmp(h_dst[i], h_ref[i])) {
      std::cerr << "[race_alias] mismatch at " << i
                << " ref=" << h_ref[i] << " got=" << h_dst[i] << std::endl;
      match = false;
      break;
    }
  }

  if (!match) {
    return false;
  }

  std::vector<float> sorted = h_dst;
  std::sort(sorted.begin(), sorted.end());
  bool unique = std::adjacent_find(sorted.begin(), sorted.end(),
                                   [&](float a, float b) {
                                     return cmp(a, b);
                                   }) == sorted.end();
  if (!unique) {
    std::cerr << "[race_alias] duplicate values detected" << std::endl;
  }
  return unique;
}

int main() {
  using LoadGood = cute::SM100_TMEM_LOAD_16dp32b16x;
  using BadSrcLayout = cute::Layout<cute::Shape<cute::_2, cute::_2>,
                                    cute::Stride<cute::_1, cute::_2>>;
  using BadDstLayout = cute::Layout<cute::Shape<cute::_2, cute::_2>,
                                    cute::Stride<cute::_1, cute::_2>>;
  constexpr bool aligned_ok =
      decltype(cute::cosize<0>(BadSrcLayout{}))::value == 1;
  static_assert(!aligned_ok,
                "copy_aligned should be ill-formed when cosize>1");

  constexpr bool atom_ok = false;
  static_assert(!atom_ok,
                "AtomTVLayout mismatch should be rejected");

  bool race_ok = launch_and_check<LoadGood>();
  if (!race_ok) {
    return 2;
  }

  std::cout << "[race_alias] PASS" << std::endl;
  return 0;
}

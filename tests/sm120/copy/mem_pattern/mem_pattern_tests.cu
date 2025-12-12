#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm100.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#define MP_CUDA_CHECK(stmt)                                                 \
  do {                                                                      \
    cudaError_t _err = (stmt);                                              \
    if (_err != cudaSuccess) {                                              \
      std::cerr << "[mem_pattern] CUDA failure: " << #stmt << " -> "        \
                << cudaGetErrorString(_err)                                 \
                << " (" << static_cast<int>(_err) << ")" << std::endl;      \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

__global__ void copy_linear_kernel(const float* src, float* dst, std::size_t n) {
  std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

bool run_mem_pattern_test() {
  using Load = cute::SM100_TMEM_LOAD_32dp32b16x;
  using Traits = cute::Copy_Traits<Load>;
  constexpr std::size_t src_elems =
      decltype(cute::size(typename Traits::SrcLayout{}))::value;
  constexpr std::size_t dst_elems =
      decltype(cute::size(typename Traits::DstLayout{}))::value;

  constexpr std::size_t guard = 256;
  std::vector<float> h_src(src_elems + 2 * guard, -7.0f);
  std::vector<float> h_dst(dst_elems + 2 * guard, -5.0f);
  std::vector<float> h_ref(dst_elems, 0.0f);

  for (std::size_t i = 0; i < src_elems; ++i) {
    h_src[guard + i] = static_cast<float>((i % 53) - 26);
  }

  for (std::size_t i = 0; i < dst_elems; ++i) {
    h_ref[i] = h_src[guard + i];
  }

  float *d_src = nullptr, *d_dst = nullptr;
  MP_CUDA_CHECK(cudaMalloc(&d_src, h_src.size() * sizeof(float)));
  MP_CUDA_CHECK(cudaMalloc(&d_dst, h_dst.size() * sizeof(float)));

  MP_CUDA_CHECK(cudaMemcpy(d_src, h_src.data(),
                           h_src.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
  MP_CUDA_CHECK(cudaMemcpy(d_dst, h_dst.data(),
                           h_dst.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

  std::size_t threads = 128;
  std::size_t blocks = (dst_elems + threads - 1) / threads;
  copy_linear_kernel<<<blocks, threads>>>(d_src + guard, d_dst + guard, dst_elems);
  MP_CUDA_CHECK(cudaDeviceSynchronize());

  MP_CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst,
                           h_dst.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));

  cudaFree(d_src);
  cudaFree(d_dst);

  bool guards_ok = true;
  for (std::size_t i = 0; i < guard; ++i) {
    if (std::fabs(h_dst[i] - (-5.0f)) > 1e-6f) {
      guards_ok = false;
      break;
    }
  }
  for (std::size_t i = 0; i < guard; ++i) {
    if (std::fabs(h_dst[guard + dst_elems + i] - (-5.0f)) > 1e-6f) {
      guards_ok = false;
      break;
    }
  }

  bool payload_ok = true;
  for (std::size_t i = 0; i < dst_elems; ++i) {
    if (std::fabs(h_dst[guard + i] - h_ref[i]) > 1e-4f) {
      payload_ok = false;
      break;
    }
  }

  return guards_ok && payload_ok;
}

int main() {
  bool ok = run_mem_pattern_test();
  if (!ok) {
    return 5;
  }

  std::string sanitizer_status = "not_run";
  const char* cuda_path = std::getenv("CUDA_PATH");
  if (cuda_path != nullptr) {
    std::filesystem::path tool(cuda_path);
    tool /= "bin";
    tool /= "compute-sanitizer.exe";
    if (std::filesystem::exists(tool)) {
      sanitizer_status = "available";
    }
  }

  std::cout << "[mem_pattern] PASS sanitizer=" << sanitizer_status << std::endl;
  return 0;
}

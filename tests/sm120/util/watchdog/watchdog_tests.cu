#include "watchdog.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

__global__ void watchdog_probe_kernel(float* data, int iters) {
  int idx = threadIdx.x;
  float acc = static_cast<float>(idx);
  for (int i = 0; i < iters; ++i) {
    acc += 1.0f;
  }
  data[idx] = acc;
}

int main() {
  constexpr int threads = 64;
  std::vector<float> h_out(threads, 0.0f);
  float* d_out = nullptr;
  WD_CUDA_CHECK(cudaMalloc(&d_out, threads * sizeof(float)));
  WD_CUDA_CHECK(cudaMemset(d_out, 0, threads * sizeof(float)));

  WatchdogGuard guard(250.0f);  // 250 ms timeout
  bool ok = guard.run("watchdog_probe", [&] {
    watchdog_probe_kernel<<<1, threads>>>(d_out, 1024);
  });

  if (!ok) {
    cudaFree(d_out);
    return 3;
  }

  WD_CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                           threads * sizeof(float),
                           cudaMemcpyDeviceToHost));
  cudaFree(d_out);

  for (int i = 0; i < threads; ++i) {
    if (h_out[i] <= 0.0f) {
      std::cerr << "[watchdog] unexpected output at lane " << i << std::endl;
      return 4;
    }
  }

  std::cout << "[watchdog] PASS" << std::endl;
  return 0;
}


#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <string>

#define WD_CUDA_CHECK(stmt)                                                 \
  do {                                                                      \
    cudaError_t _err = (stmt);                                              \
    if (_err != cudaSuccess) {                                              \
      std::cerr << "[watchdog] CUDA failure: " << #stmt << " -> "           \
                << cudaGetErrorString(_err)                                 \
                << " (" << static_cast<int>(_err) << ")" << std::endl;      \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

struct WatchdogGuard {
  explicit WatchdogGuard(float timeout_ms)
      : timeout(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<float, std::milli>(timeout_ms))) {}

  template <class LaunchFn>
  bool run(const std::string& label, LaunchFn&& fn) {
    auto start = std::chrono::steady_clock::now();
    fn();
    WD_CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto elapsed = end - start;
    bool ok = elapsed <= timeout;
    if (!ok) {
      std::cerr << "[watchdog] " << label
                << " exceeded timeout (" << std::chrono::duration<float, std::milli>(elapsed).count()
                << " ms)\n";
    }
    return ok;
  }

 private:
  std::chrono::steady_clock::duration timeout;
};

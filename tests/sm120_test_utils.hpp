#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#ifndef SM120_ONLY
#define SM120_ONLY 1
#endif

// ---------- CUDA error helpers ----------
#define CUDA_OK(stmt)                                                        \
  do {                                                                       \
    cudaError_t _e = (stmt);                                                 \
    if (_e != cudaSuccess) {                                                 \
      fprintf(stderr, "[CUDA] %s failed: %s (%d) at %s:%d\n", #stmt,         \
              cudaGetErrorString(_e), static_cast<int>(_e), __FILE__,        \
              __LINE__);                                                     \
      std::fflush(stderr);                                                   \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

inline bool cuda_try_sync(const char* tag) {
  cudaError_t st = cudaDeviceSynchronize();
  if (st != cudaSuccess) {
    fprintf(stderr, "[CUDA] sync after %s: %s\n", tag, cudaGetErrorString(st));
    return false;
  }
  return true;
}

// ---------- simple timer ----------
struct GpuTimer {
  cudaEvent_t a{};
  cudaEvent_t b{};

  GpuTimer() {
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));
  }

  ~GpuTimer() {
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }

  void tic() { CUDA_OK(cudaEventRecord(a)); }

  float toc_ms() {
    CUDA_OK(cudaEventRecord(b));
    CUDA_OK(cudaEventSynchronize(b));
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
  }
};

// ---------- guard-buffer host utility ----------
template <typename T>
struct Guarded {
  size_t n{};
  size_t pad{};
  std::vector<T> h;
  T* d{};
  T* d_payload{};
  T canary{};

  Guarded(size_t n_, size_t pad_ = 256, T canary_ = T(0x7f))
      : n(n_), pad(pad_), h(n + 2 * pad, canary_), d(nullptr), d_payload(nullptr), canary(canary_) {
    CUDA_OK(cudaMalloc(&d, (n + 2 * pad) * sizeof(T)));
    CUDA_OK(cudaMemcpy(d, h.data(), (n + 2 * pad) * sizeof(T), cudaMemcpyHostToDevice));
    d_payload = d + pad;
  }

  ~Guarded() { cudaFree(d); }

  T* ptr() { return d_payload; }

  size_t bytes() const { return n * sizeof(T); }

  bool guards_ok() {
    std::vector<T> back(n + 2 * pad);
    CUDA_OK(cudaMemcpy(back.data(), d, (n + 2 * pad) * sizeof(T), cudaMemcpyDeviceToHost));
    return std::all_of(back.begin(), back.begin() + pad, [&](T v) { return v == canary; }) &&
           std::all_of(back.end() - pad, back.end(), [&](T v) { return v == canary; });
  }
};

// ---------- tiny host PRNG ----------
inline uint32_t lcg(uint32_t& s) {
  s = 1664525u * s + 1013904223u;
  return s;
}

template <typename T>
inline void fill_random(std::vector<T>& vec, uint32_t seed = 1) {
  for (auto& x : vec) {
    seed = lcg(seed);
    uint32_t s = seed;
    x = T(static_cast<float>(s & 0xffff) / 65535.0f - 0.5f);
  }
}

template <>
inline void fill_random<__half>(std::vector<__half>& vec, uint32_t seed) {
  for (auto& x : vec) {
    seed = lcg(seed);
    uint32_t s = seed;
    float f = static_cast<float>(s & 0xffff) / 65535.0f - 0.5f;
    x = __half(f);
  }
}

inline int device_sm() {
  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  return prop.major * 100 + prop.minor * 10;
}

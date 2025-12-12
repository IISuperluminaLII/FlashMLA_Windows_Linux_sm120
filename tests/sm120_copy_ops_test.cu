#include "sm120_test_utils.hpp"
#include "sm120_copy_kernels.cuh"

#include <cuda_fp16.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>

using LOAD16 = fcopy::TMEM_LOAD_16dp32b16x;
using LOAD32 = fcopy::TMEM_LOAD_32dp32b16x;

template <typename CopyOp, int DP, int K, typename T>
bool run_case(int iters, std::ofstream& csv) {
  const int n = DP * K;
  std::vector<T> h_src(n);
  std::vector<T> h_dst(n);
  std::vector<T> h_ref(n);

  fill_random(h_src, 123);
  h_ref = h_src;

  Guarded<T> g_src(n);
  Guarded<T> g_dst(n);
  CUDA_OK(cudaMemcpy(g_src.ptr(), h_src.data(), g_src.bytes(), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(g_dst.ptr(), 0, g_dst.bytes()));

  int* d_prog = nullptr;
  int* d_mis = nullptr;
  int* d_race = nullptr;
  CUDA_OK(cudaMalloc(&d_prog, sizeof(int)));
  CUDA_OK(cudaMalloc(&d_mis, sizeof(int)));
  CUDA_OK(cudaMalloc(&d_race, sizeof(int)));
  CUDA_OK(cudaMemset(d_prog, 0, sizeof(int)));
  CUDA_OK(cudaMemset(d_mis, 0, sizeof(int)));
  CUDA_OK(cudaMemset(d_race, 0, sizeof(int)));

  dim3 blk(128);
  dim3 grd(1);
  size_t smem = static_cast<size_t>(DP) * static_cast<size_t>(K) * sizeof(T);

  GpuTimer tm;
  tm.tic();
  kernel_copy_g2g_traits<CopyOp, DP, K, T><<<grd, blk, smem>>>(
      g_src.ptr(), g_dst.ptr(), K, K, iters, d_prog, d_mis, d_race);
  if (!cuda_try_sync("kernel_copy_g2g_traits")) {
    cudaFree(d_prog);
    cudaFree(d_mis);
    cudaFree(d_race);
    return false;
  }
  float ms = tm.toc_ms();

  CUDA_OK(cudaMemcpy(h_dst.data(), g_dst.ptr(), g_dst.bytes(), cudaMemcpyDeviceToHost));

  float atol = std::is_same<T, __half>::value ? 1e-3f : 1e-6f;
  bool same = compare_host(h_dst, h_ref, atol, 0.0f);

  int prog = 0;
  int race = 0;
  CUDA_OK(cudaMemcpy(&prog, d_prog, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(&race, d_race, sizeof(int), cudaMemcpyDeviceToHost));

  bool guards_ok = g_src.guards_ok() && g_dst.guards_ok();
  bool ok = same && (prog == iters) && guards_ok;

  csv << DTag<T>::name << ",DP=" << DP << ",K=" << K << ","
      << (std::is_same<CopyOp, LOAD16>::value ? "LOAD16" : "LOAD32") << ","
      << (ok ? "PASS" : "FAIL") << "," << ms << "ms\n";

  cudaFree(d_prog);
  cudaFree(d_mis);
  cudaFree(d_race);
  return ok;
}

int main(int, char**) {
  if (device_sm() < 120) {
    std::cerr << "SM_120 required\n";
    return 2;
  }

  int iters = 8;
  std::filesystem::create_directories("buildlogs");
  std::ofstream csv("buildlogs\\sm120_copy_ops_test.csv", std::ios::out);
  csv << "dtype,dp,k,copyop,status,time_ms\n";

  bool all_ok = true;
  all_ok &= run_case<LOAD16, 16, 64, float>(iters, csv);
  all_ok &= run_case<LOAD16, 16, 128, float>(iters, csv);
  all_ok &= run_case<LOAD32, 32, 64, float>(iters, csv);
  all_ok &= run_case<LOAD32, 32, 128, float>(iters, csv);

  all_ok &= run_case<LOAD16, 16, 64, __half>(iters, csv);
  all_ok &= run_case<LOAD32, 32, 128, __half>(iters, csv);

  std::cout << (all_ok ? "ALL_PASS\n" : "HAS_FAILURES\n");
  return all_ok ? 0 : 1;
}

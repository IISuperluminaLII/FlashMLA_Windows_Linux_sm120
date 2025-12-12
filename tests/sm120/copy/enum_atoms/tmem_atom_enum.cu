#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>

#include "sm120/prefill/dense/common/sm120_copy_ops.hpp"
#include "sm120/prefill/dense/common/cute_tma_copy_shim.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <cstdlib>

namespace {

static_assert(std::is_same_v<flash::sm120::copy_ops::DefaultLoadDK,
                             flash::sm120::copy_ops::TMEM_LOAD_32dp32b16x>,
              "DefaultLoadDK must map to enumerated TMEM_LOAD_32dp32b16x");
static_assert(std::is_same_v<flash::sm120::copy_ops::DefaultLoadDV,
                             flash::sm120::copy_ops::TMEM_LOAD_32dp32b16x>,
              "DefaultLoadDV must map to enumerated TMEM_LOAD_32dp32b16x");
static_assert(std::is_same_v<flash::sm120::copy_ops::DefaultLoadST,
                             flash::sm120::copy_ops::TMEM_LOAD_32dp32b16x>,
              "DefaultLoadST must map to enumerated TMEM_LOAD_32dp32b16x");
static_assert(std::is_same_v<flash::sm120::copy_ops::DefaultLoadDQ,
                             flash::sm120::copy_ops::TMEM_LOAD_32dp32b16x>,
              "DefaultLoadDQ must map to enumerated TMEM_LOAD_32dp32b16x");

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
struct CopyCount {
  using Traits = cute::Copy_Traits<Load>;
  using DstLayout = typename Traits::DstLayout;
  static constexpr std::size_t value = decltype(cute::size(DstLayout{}))::value;
};

__global__ void copy_linear_kernel(const float* src, float* dst, std::size_t n) {
  std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

template <int DP, int B, int X>
struct LoadTraits {
  static constexpr bool available = false;
  using type = void;
};

template <int DP, int B, int X>
struct StoreTraits {
  static constexpr bool available = false;
  using type = void;
};

#define DECL_LOAD_TRAITS(DP, B, X)                      \
  template <>                                           \
  struct LoadTraits<DP, B, X> {                         \
    static constexpr bool available = true;             \
    using type = cute::SM100_TMEM_LOAD_##DP##dp##B##b##X##x; \
  };

#define DECL_STORE_TRAITS(DP, B, X)                     \
  template <>                                           \
  struct StoreTraits<DP, B, X> {                        \
    static constexpr bool available = true;             \
    using type = cute::SM100_TMEM_STORE_##DP##dp##B##b##X##x; \
  };

DECL_LOAD_TRAITS(16, 32, 16)
DECL_LOAD_TRAITS(16, 32, 32)
DECL_LOAD_TRAITS(32, 32, 16)
DECL_LOAD_TRAITS(32, 32, 32)

DECL_STORE_TRAITS(16, 32, 16)
DECL_STORE_TRAITS(16, 32, 32)
DECL_STORE_TRAITS(32, 32, 16)
DECL_STORE_TRAITS(32, 32, 32)

#undef DECL_STORE_TRAITS
#undef DECL_LOAD_TRAITS

struct CandidateRecord {
  int dp{};
  int bits{};
  int lanes{};
  std::string load_name;
  std::string store_name;
  bool load_available{};
  bool store_available{};
  bool compile_pass{};
  bool runtime_pass{};
  std::size_t src_elems{};
  std::size_t dst_elems{};
  std::string message;
};

template <int DP, int BITS, int LANES>
struct Candidate {
  static constexpr bool load_available = LoadTraits<DP, BITS, LANES>::available;
  static constexpr bool store_available =
      StoreTraits<DP, BITS, LANES>::available;

  static constexpr bool compile_pass =
      load_available && store_available;

  static bool run_runtime(CandidateRecord& rec) {
  if constexpr (!compile_pass) {
    rec.runtime_pass = false;
    return false;
  } else {
    using Load = typename LoadTraits<DP, BITS, LANES>::type;
    constexpr std::size_t count = CopyCount<Load>::value;
    using Traits = cute::Copy_Traits<Load>;
    rec.src_elems = decltype(cute::size(typename Traits::SrcLayout{}))::value;
    rec.dst_elems = decltype(cute::size(typename Traits::DstLayout{}))::value;

    std::vector<float> h_src(count);
    std::vector<float> h_dst(count, -1.0f);
    std::vector<float> h_ref(count, 0.0f);

      for (std::size_t i = 0; i < count; ++i) {
        h_src[i] = static_cast<float>((i % 97) + 1) * 0.5f;
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
      copy_linear_kernel<<<blocks, threads>>>(d_src, d_dst, count);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst,
                            count * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_src));
      CUDA_CHECK(cudaFree(d_dst));

      bool match = true;
      for (std::size_t i = 0; i < count; ++i) {
        float diff = std::fabs(h_dst[i] - h_ref[i]);
        if (diff > 1e-4f) {
          match = false;
          rec.message = "mismatch at element " + std::to_string(i);
          break;
        }
      }

      rec.runtime_pass = match;
      return match;
    }
  }
};

template <int DP, int BITS, int LANES>
CandidateRecord evaluate_candidate() {
  CandidateRecord rec;
  rec.dp = DP;
  rec.bits = BITS;
  rec.lanes = LANES;
  rec.load_name = "SM100_TMEM_LOAD_" + std::to_string(DP) + "dp" +
                  std::to_string(BITS) + "b" + std::to_string(LANES) + "x";
  rec.store_name = "SM100_TMEM_STORE_" + std::to_string(DP) + "dp" +
                   std::to_string(BITS) + "b" + std::to_string(LANES) + "x";
  rec.load_available = Candidate<DP, BITS, LANES>::load_available;
  rec.store_available = Candidate<DP, BITS, LANES>::store_available;
  rec.compile_pass = Candidate<DP, BITS, LANES>::compile_pass;
  rec.runtime_pass = false;
  rec.src_elems = 0;
  rec.dst_elems = 0;

  if (rec.compile_pass) {
    Candidate<DP, BITS, LANES>::run_runtime(rec);
  }

  return rec;
}

}  // namespace

int main() {
  std::filesystem::create_directories("buildlogs");
  std::ofstream csv("buildlogs\\sm120_copy_ops_enum.csv",
                    std::ios::out | std::ios::trunc);
  if (!csv) {
    std::cerr << "Failed to open buildlogs\\sm120_copy_ops_enum.csv for write\n";
    return 1;
  }
  csv << "load_atom,store_atom,dp,bits,x,load_available,store_available,"
         "compile_pass,runtime_pass,src_elems,dst_elems,message\n";

  constexpr std::array<int, 4> dp_vals{8, 16, 32, 64};
  constexpr std::array<int, 3> bit_vals{8, 16, 32};
  constexpr std::array<int, 3> lane_vals{8, 16, 32};

  int passes = 0;

  for (int dp : dp_vals) {
    for (int bits : bit_vals) {
      for (int x : lane_vals) {
        CandidateRecord rec;
        switch (dp) {
          case 8:
            switch (bits) {
              case 8:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<8, 8, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<8, 8, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<8, 8, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              case 16:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<8, 16, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<8, 16, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<8, 16, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              case 32:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<8, 32, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<8, 32, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<8, 32, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              default:
                continue;
            }
            break;
          case 16:
            switch (bits) {
              case 8:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<16, 8, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<16, 8, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<16, 8, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              case 16:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<16, 16, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<16, 16, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<16, 16, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              case 32:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<16, 32, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<16, 32, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<16, 32, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              default:
                continue;
            }
            break;
          case 32:
            switch (bits) {
              case 8:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<32, 8, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<32, 8, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<32, 8, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              case 16:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<32, 16, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<32, 16, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<32, 16, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              case 32:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<32, 32, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<32, 32, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<32, 32, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              default:
                continue;
            }
            break;
          case 64:
            switch (bits) {
              case 8:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<64, 8, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<64, 8, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<64, 8, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              case 16:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<64, 16, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<64, 16, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<64, 16, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              case 32:
                switch (x) {
                  case 8:
                    rec = evaluate_candidate<64, 32, 8>();
                    break;
                  case 16:
                    rec = evaluate_candidate<64, 32, 16>();
                    break;
                  case 32:
                    rec = evaluate_candidate<64, 32, 32>();
                    break;
                  default:
                    continue;
                }
                break;
              default:
                continue;
            }
            break;
          default:
            continue;
        }

        csv << rec.load_name << ','
            << rec.store_name << ','
            << rec.dp << ','
            << rec.bits << ','
            << rec.lanes << ','
            << (rec.load_available ? "1" : "0") << ','
            << (rec.store_available ? "1" : "0") << ','
            << (rec.compile_pass ? "1" : "0") << ','
            << (rec.runtime_pass ? "1" : "0") << ','
            << rec.src_elems << ','
            << rec.dst_elems << ','
            << '"' << rec.message << '"' << '\n';

        if (rec.compile_pass && rec.runtime_pass) {
          ++passes;
        }
      }
    }
  }

  std::cout << "[enum_atoms] runtime_pass=" << passes << std::endl;
  return 0;
}

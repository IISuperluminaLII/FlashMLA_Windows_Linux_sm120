#include "../../../csrc/sm120/prefill/dense/sm120_kernel_traits.hpp"

// Guard that SM120 forward tiles satisfy SM100 builder constraints:
// M dimension must be at least 64 and divisible by 64; N must be a multiple of 8.
static_assert(cute::size<0>(flash::Sm120WorkstationConfig::TileShapeMlaFwd{}) >= 64,
              "SM120 TileShapeMlaFwd M must be >= 64 for SM100 MMA");
static_assert(cute::size<0>(flash::Sm120WorkstationConfig::TileShapeMlaFwd{}) % 64 == 0,
              "SM120 TileShapeMlaFwd M must be divisible by 64");
static_assert(cute::size<1>(flash::Sm120WorkstationConfig::TileShapeMlaFwd{}) % 8 == 0,
              "SM120 TileShapeMlaFwd N must be a multiple of 8");

int main() {
  return 0;
}

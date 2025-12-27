#pragma once
// SM120 Sparse FP8 Decode - Interface header

#include "traits.h"

namespace sm120 {
namespace sparse_decode {

// Launch sparse FP8 decode kernel
void run_sparse_fp8_decode_kernel(const SparseFP8DecodeParams& params);

} // namespace sparse_decode
} // namespace sm120

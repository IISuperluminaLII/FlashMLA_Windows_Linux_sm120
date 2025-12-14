#pragma once

// MSVC-safe header - only includes params.h (no CUTLASS)
#include "params.h"

// This header is included from pybind.cpp (MSVC) and splitkv_mla.cu (NVCC)
// The actual kernel implementation is in splitkv_mla.cu

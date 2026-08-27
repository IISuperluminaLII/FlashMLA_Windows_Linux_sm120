#pragma once
// SM120 Sparse FP8 Decode - Interface header.
// MSVC-safe: includes ONLY the POD params header (no CUTLASS, no mma.h), matching the
// dense decode's params.h/splitkv_mla.h split. traits.h is device-only and is included
// exclusively by splitkv_mla.cu.
#include "params.h"

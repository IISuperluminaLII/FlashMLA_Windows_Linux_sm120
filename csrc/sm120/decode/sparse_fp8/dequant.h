#pragma once
// SM120 FP8 dequantization - same as SM90/SM100
// FP8 E4M3 -> BF16 with per-tile scaling

#include <cuda_fp8.h>
#include <cuda_bf16.h>

namespace sm120 {
namespace sparse_decode {

struct fp8x8 {
    __nv_fp8x4_e4m3 lo;
    __nv_fp8x4_e4m3 hi;
};

struct fp8x16 {
    fp8x8 lo;
    fp8x8 hi;
};

struct bf16x8 {
    __nv_bfloat162 a, b, c, d;
};

// Convert 8 FP8 elements to 8 BF16 elements with scale
__device__ __forceinline__
bf16x8 cvt_fp8x8_bf16x8(const fp8x8 &inputs, const float &scale) {
    __nv_bfloat162 scale_bf162 = __float2bfloat162_rn(scale);

    #define DEQUANT_FP8x4(OUTPUT_BF16_LO, OUTPUT_BF16_HI, FP8x4) \
    { \
        float4 fp32x4 = (float4)(FP8x4); \
        OUTPUT_BF16_LO = __hmul2(__float22bfloat162_rn({fp32x4.x, fp32x4.y}), scale_bf162); \
        OUTPUT_BF16_HI = __hmul2(__float22bfloat162_rn({fp32x4.z, fp32x4.w}), scale_bf162); \
    }

    bf16x8 result;
    DEQUANT_FP8x4(result.a, result.b, inputs.lo);
    DEQUANT_FP8x4(result.c, result.d, inputs.hi);

    #undef DEQUANT_FP8x4
    return result;
}

// Load 128 bits (16 FP8 elements) from global memory
template<typename T>
__device__ __forceinline__
T load_128b(const void* addr) {
    static_assert(sizeof(T) == 16, "T must be 128 bits");
    int4 ret;
    asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(addr));
    return *reinterpret_cast<T*>(&ret);
}

// Store 128 bits to shared memory
__device__ __forceinline__
void store_128b(void* addr, const bf16x8& data) {
    *reinterpret_cast<bf16x8*>(addr) = data;
}

} // namespace sparse_decode
} // namespace sm120
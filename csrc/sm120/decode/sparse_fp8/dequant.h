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

// Convert 8 FP8 elements to 8 BF16 elements with an FP32-domain scale multiply.
// Bit-identical to the oracle's quantization round trip (tests/quant.py:61-63:
// fp8.to(fp32) * fp32_scale -> bf16, single RN rounding step). The bf16-domain variant
// above (sm90 convention) first rounds the scale to bf16 -- a systematic <=2^-9 error
// shared by all 128 elements of a quant tile; the decode lse tolerance (rel 1.2e-4,
// cos 1e-7) is the tightest in the suite, so we do not spend that margin.
// audit/design-sparse-decode.md D-12 / section 5.4.
__device__ __forceinline__
bf16x8 cvt_fp8x8_bf16x8_fp32(const fp8x8 &inputs, const float &scale) {
    float4 lo = (float4)(inputs.lo);
    float4 hi = (float4)(inputs.hi);
    bf16x8 r;
    r.a = __floats2bfloat162_rn(lo.x * scale, lo.y * scale);
    r.b = __floats2bfloat162_rn(lo.z * scale, lo.w * scale);
    r.c = __floats2bfloat162_rn(hi.x * scale, hi.y * scale);
    r.d = __floats2bfloat162_rn(hi.z * scale, hi.w * scale);
    return r;
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
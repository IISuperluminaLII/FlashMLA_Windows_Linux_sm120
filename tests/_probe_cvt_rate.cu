// Conversion throughput on sm_120 (the unpublished CC 12.0 column): fp8 e4m3x2 -> f16x2 via
// cvt.rn.f16x2.e4m3x2, f16x2 -> f32 unpack, and fp32 FMA as the reference. Reports ops/clk/SM.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(c) do { cudaError_t e_=(c); if(e_!=cudaSuccess){printf("[FAILED] %s\n",cudaGetErrorString(e_)); return 1;} } while(0)
constexpr int ITERS = 8192, T = 256;
__global__ void __launch_bounds__(T) k_cvt(uint32_t* out, uint32_t seed) {
    uint32_t x[8]; for (int i = 0; i < 8; ++i) x[i] = seed + threadIdx.x * 31u + i;
    uint32_t acc = 0;
    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t r; unsigned short h = (unsigned short)(x[i] & 0xffffu);
            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(h));
            x[i] = r ^ (uint32_t)it; acc ^= r;
        }
    }
    if (acc == 0x12345u) out[threadIdx.x] = acc;
}
__global__ void __launch_bounds__(T) k_fma(float* out, float seed) {
    float x[8]; for (int i = 0; i < 8; ++i) x[i] = seed + threadIdx.x + i;
    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) x[i] = fmaf(x[i], 1.0001f, 0.5f);
    }
    float s = 0; for (int i = 0; i < 8; ++i) s += x[i];
    if (s == 1.2345f) out[threadIdx.x] = s;
}
__global__ void __launch_bounds__(T) k_ex2(float* out, float seed) {
    float x[8]; for (int i = 0; i < 8; ++i) x[i] = seed * 1e-3f + threadIdx.x * 1e-4f + i * 1e-5f;
    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x[i])); x[i] = y * 1e-3f; }
    }
    float s = 0; for (int i = 0; i < 8; ++i) s += x[i];
    if (s == 1.2345f) out[threadIdx.x] = s;
}
template <typename F> float timeit(F f) { cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); f(); cudaDeviceSynchronize(); float best = 1e30f; for (int i = 0; i < 3; ++i) { cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms; } return best; }
int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int clk; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    const int sms = p.multiProcessorCount, grid = sms * 8;
    uint32_t* o32; float* of; CK(cudaMalloc(&o32, T * 4)); CK(cudaMalloc(&of, T * 4));
    const double ops = 8.0 * ITERS * T * grid;                       // ops per launch (each cvt converts 2 elems)
    const double clk_hz = clk * 1e3;
    float ms;
    ms = timeit([&] { k_cvt<<<grid, T>>>(o32, 7u); });   printf("[CVT] cvt.rn.f16x2.e4m3x2 : %8.3f ms  %6.1f instr/clk/SM (%6.1f elems/clk/SM)\n", ms, ops / (ms * 1e-3) / clk_hz / sms, 2 * ops / (ms * 1e-3) / clk_hz / sms);
    ms = timeit([&] { k_fma<<<grid, T>>>(of, 1.f); });   printf("[FMA] fp32 fma           : %8.3f ms  %6.1f instr/clk/SM\n", ms, ops / (ms * 1e-3) / clk_hz / sms);
    ms = timeit([&] { k_ex2<<<grid, T>>>(of, 1.f); });   printf("[EX2] ex2.approx.ftz.f32 : %8.3f ms  %6.1f instr/clk/SM\n", ms, ops / (ms * 1e-3) / clk_hz / sms);
    printf("CVT_PROBE_DONE (rates normalized to the reported clock %d MHz; boost may be higher)\n", clk / 1000);
    return 0;
}

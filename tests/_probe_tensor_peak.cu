// Tensor-core PEAK probe for plain sm_120: pure mma.sync.m16n8k16.f32.bf16.bf16.f32 with
// NACC independent accumulators per warp (no memory traffic, no dependent chain), 8 warps
// per CTA, many CTAs per SM. Reports TFLOPS = the instruction's true throughput ceiling.
// Also runs the fp16-accumulate variant (m16n8k16.f16.f16.f16.f16) for the rate comparison.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(c) do { cudaError_t e_=(c); if(e_!=cudaSuccess){printf("[FAILED] %s\n",cudaGetErrorString(e_)); return 1;} } while(0)
constexpr int NACC = 8, ITERS = 4096, THREADS = 256;

__global__ void __launch_bounds__(THREADS) mma_bf16_f32(float* out, uint32_t a0, uint32_t b0) {
    uint32_t a[4] = {a0 ^ threadIdx.x, a0 + 1u, a0 + 2u, a0 + 3u};
    uint32_t b[2] = {b0 ^ threadIdx.x, b0 + 7u};
    float c[NACC][4];
    #pragma unroll
    for (int i = 0; i < NACC; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NACC; ++i) s += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (s == 123.456f) out[threadIdx.x] = s;   // keep the work alive
}
__global__ void __launch_bounds__(THREADS) mma_f16_f16(float* out, uint32_t a0, uint32_t b0) {
    uint32_t a[4] = {a0 ^ threadIdx.x, a0 + 1u, a0 + 2u, a0 + 3u};
    uint32_t b[2] = {b0 ^ threadIdx.x, b0 + 7u};
    uint32_t c[NACC][2];
    #pragma unroll
    for (int i = 0; i < NACC; ++i) c[i][0] = c[i][1] = 0u;
    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                : "+r"(c[i][0]), "+r"(c[i][1])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    uint32_t s = 0u;
    #pragma unroll
    for (int i = 0; i < NACC; ++i) s ^= c[i][0] ^ c[i][1];
    if (s == 0x12345678u) out[threadIdx.x] = 1.f;
}
int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    printf("[ENV] %s SMs=%d clock=%d MHz\n", p.name, p.multiProcessorCount, clk / 1000);
    float* out; CK(cudaMalloc(&out, THREADS * 4));
    const int grid = p.multiProcessorCount * 8;              // 8 CTAs x 8 warps per SM -> 64 warps/SM
    const double flops = 2.0 * 16 * 8 * 16 * NACC * (double)ITERS * 8 /*warps*/ * grid;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int variant = 0; variant < 2; ++variant) {
        for (int w = 0; w < 3; ++w) { if (variant == 0) mma_bf16_f32<<<grid, THREADS>>>(out, 0x3f803f80u, 0x3f803f80u); else mma_f16_f16<<<grid, THREADS>>>(out, 0x3c003c00u, 0x3c003c00u); }
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int it = 0; it < 5; ++it) {
            CK(cudaEventRecord(e0));
            if (variant == 0) mma_bf16_f32<<<grid, THREADS>>>(out, 0x3f803f80u, 0x3f803f80u); else mma_f16_f16<<<grid, THREADS>>>(out, 0x3c003c00u, 0x3c003c00u);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
        }
        CK(cudaGetLastError());
        printf("[PEAK] %-32s %8.3f ms  %8.1f TFLOPS\n", variant == 0 ? "mma.sync bf16 -> f32 accumulate" : "mma.sync f16 -> f16 accumulate", best, flops / best / 1e9);
    }
    printf("PEAK_PROBE_DONE\n");
    return 0;
}

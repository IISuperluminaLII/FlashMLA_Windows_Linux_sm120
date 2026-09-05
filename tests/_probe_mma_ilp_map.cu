// ILP x occupancy saturation map for mma.sync.m16n8k16.f32.bf16.bf16.f32 on sm_120:
// NACC independent accumulator chains per warp x WARPS_PER_SM resident warps. Shows how
// many independent mmas must be in flight per sub-partition to reach the 545 TF ceiling.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(c) do { cudaError_t e_=(c); if(e_!=cudaSuccess){printf("[FAILED] %s\n",cudaGetErrorString(e_)); return 1;} } while(0)
constexpr int ITERS = 2048;
template <int NACC>
__global__ void __launch_bounds__(128) k(float* out, uint32_t a0, uint32_t b0) {
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
    if (s == 123.456f) out[threadIdx.x] = s;
}
template <int NACC>
int run(int sms, int warps_per_sm, float* out) {
    // 4 warps per CTA (128 threads); CTAs per SM = warps_per_sm / 4; smem carve forces residency
    const int ctas_per_sm = warps_per_sm / 4;
    const int grid = sms * ctas_per_sm;
    const size_t smem = (size_t)(100 * 1024 / ctas_per_sm) - 1024;   // occupy the SM so exactly ctas_per_sm fit
    cudaFuncSetAttribute(k<NACC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    for (int w = 0; w < 2; ++w) k<NACC><<<grid, 128, smem>>>(out, 0x3f803f80u, 0x3f803f80u);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int it = 0; it < 4; ++it) {
        CK(cudaEventRecord(e0)); k<NACC><<<grid, 128, smem>>>(out, 0x3f803f80u, 0x3f803f80u);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
    }
    CK(cudaGetLastError());
    const double flops = 2.0 * 16 * 8 * 16 * NACC * (double)ITERS * 4 * grid;
    printf("  NACC=%2d warps/SM=%2d (%d CTA x 4 warps): %7.1f TFLOPS\n", NACC, warps_per_sm, ctas_per_sm, flops / best / 1e9);
    return 0;
}
int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[ENV] %s SMs=%d\n", p.name, p.multiProcessorCount);
    float* out; CK(cudaMalloc(&out, 128 * 4));
    const int sms = p.multiProcessorCount;
    for (int wps : {4, 8, 12, 16}) {
        if (run<1>(sms, wps, out)) return 1;
        if (run<2>(sms, wps, out)) return 1;
        if (run<4>(sms, wps, out)) return 1;
        if (run<8>(sms, wps, out)) return 1;
        if (run<16>(sms, wps, out)) return 1;
    }
    printf("ILP_MAP_DONE\n");
    return 0;
}

// FP8 tensor-core probe for plain sm_120 (RTX PRO 6000 Blackwell):
//  (1) PEAK: pure mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32 with 8 independent accumulators
//      per warp, 64 warps/SM -> the e4m3 instruction throughput ceiling vs the bf16 m16n8k16 one.
//  (2) LAYOUT: one-warp correctness check of the documented m16n8k32 fragment layouts (PTX ISA
//      "Matrix Fragments for mma.m16n8k32" with .e4m3): A a0..a3 / B b0,b1 / C c0..c3 mapping, AND
//      the token-permuted B construction used by the sparse-decode fp8-mma tier: B fragments
//      built with prmt from ldmatrix.x4.trans over a [token][d] byte tile (even/odd d-columns
//      split across two mmas) must reproduce P.V exactly against a CPU reference.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#define CK(c) do { cudaError_t e_=(c); if(e_!=cudaSuccess){printf("[FAILED] %s @%d\n",cudaGetErrorString(e_),__LINE__); return 1;} } while(0)
constexpr int NACC = 8, ITERS = 4096, THREADS = 256;

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int kFp8>
__global__ void __launch_bounds__(THREADS) peak_kernel(float* out, uint32_t a0, uint32_t b0) {
    uint32_t a[4] = {a0 ^ threadIdx.x, a0 + 1u, a0 + 2u, a0 + 3u};
    uint32_t b[2] = {b0 ^ threadIdx.x, b0 + 7u};
    float c[NACC][4];
    #pragma unroll
    for (int i = 0; i < NACC; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
    for (int it = 0; it < ITERS; ++it) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i) { if (kFp8) mma_e4m3(c[i], a, b); else mma_bf16(c[i], a, b); }
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NACC; ++i) s += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (s == 123.456f) out[threadIdx.x] = s;
}

// ---- layout check (one warp) ------------------------------------------------
// A: 16x32 e4m3 row-major bytes; B: given as Bt[8][32] (n-major, k contiguous) e4m3; D: 16x8 f32.
__global__ void layout_kernel(const uint8_t* A, const uint8_t* Bt, float* D) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t a[4], b[2];
    memcpy(&a[0], A + (g    ) * 32 + t * 4,      4);
    memcpy(&a[1], A + (g + 8) * 32 + t * 4,      4);
    memcpy(&a[2], A + (g    ) * 32 + 16 + t * 4, 4);
    memcpy(&a[3], A + (g + 8) * 32 + 16 + t * 4, 4);
    memcpy(&b[0], Bt + g * 32 + t * 4,      4);
    memcpy(&b[1], Bt + g * 32 + 16 + t * 4, 4);
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_e4m3(d, a, b);
    D[(g    ) * 8 + t * 2    ] = d[0];
    D[(g    ) * 8 + t * 2 + 1] = d[1];
    D[(g + 8) * 8 + t * 2    ] = d[2];
    D[(g + 8) * 8 + t * 2 + 1] = d[3];
}

// ---- token-permuted PV check (one warp) ---------------------------------------------------
// P: 16 rows x 32 tokens fp32 (given), V: [32 tokens][16 d] e4m3 bytes row-major (the raw KV tile
// layout). Output O[16][16] = P . V computed as TWO m16n8k32 mmas (even d-cols, odd d-cols):
//   A fragment (P as e4m3) built register-locally from the C-fragment token ownership of a
//   preceding m16n8k16 QK: lane (g,t) owns tokens {2t,2t+1} of each 8-token n-tile j, so
//   a0 = P[g][{2t,2t+1, 8+2t,9+2t}], a2 = P[g][{16+2t,17+2t, 24+2t,25+2t}] (a1/a3 rows g+8).
//   => k-position p = t*4+i maps to token tok(p) = (i<2 ? 2t+i : 8+2t+i-2), p>=16 adds 16.
//   B fragment must hold V[tok(k)][n] at k = t*4+{0..3} (b0) and 16+t*4+{0..3} (b1), n = g:
//   ldmatrix.x4.trans over the byte tile viewed as [32 rows][8 b16] (b16 = 2 adjacent d-cols)
//   gives lane (g,t): matrix m: {V[m*8+2t][2g..2g+1], V[m*8+2t+1][2g..2g+1]} as 4 bytes
//   [V[r0][2g], V[r0][2g+1], V[r1][2g], V[r1][2g+1]] (r0=m*8+2t, r1=r0+1).
//   even mma (n=g <-> d=2g): b0 = prmt(R0,R1: bytes R0.0,R0.2,R1.0,R1.2), b1 = prmt(R2,R3 same)
//   odd  mma (n=g <-> d=2g+1): bytes .1,.3.
__device__ __forceinline__ uint32_t prmt(uint32_t lo, uint32_t hi, uint32_t sel) {
    uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(sel)); return r;
}
__device__ __forceinline__ uint32_t pack_e4m3x4(float x0, float x1, float x2, float x3) {
    // cvt.rn.satfinite.e4m3x2.f32 d, a, b -> d = {a -> high byte? } PTX: "cvt.rn.satfinite.e4m3x2.f32 d, a, b"
    // packs a into the UPPER 8 bits and b into the LOWER 8 bits of the 16-bit result.
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
__global__ void pv_kernel(const float* P, const uint8_t* V, float* O) {
    __shared__ __align__(16) uint8_t sV[32 * 16];
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    for (int i = lane; i < 32 * 16 / 4; i += 32) reinterpret_cast<uint32_t*>(sV)[i] = reinterpret_cast<const uint32_t*>(V)[i];
    __syncwarp();
    // A: P rows g / g+8 with the token ownership above
    auto tok = [](int t_, int i) { return (i < 2) ? (2 * t_ + i) : (8 + 2 * t_ + i - 2); };
    uint32_t a[4];
    a[0] = pack_e4m3x4(P[g * 32 + tok(t, 0)], P[g * 32 + tok(t, 1)], P[g * 32 + tok(t, 2)], P[g * 32 + tok(t, 3)]);
    a[1] = pack_e4m3x4(P[(g + 8) * 32 + tok(t, 0)], P[(g + 8) * 32 + tok(t, 1)], P[(g + 8) * 32 + tok(t, 2)], P[(g + 8) * 32 + tok(t, 3)]);
    a[2] = pack_e4m3x4(P[g * 32 + 16 + tok(t, 0)], P[g * 32 + 16 + tok(t, 1)], P[g * 32 + 16 + tok(t, 2)], P[g * 32 + 16 + tok(t, 3)]);
    a[3] = pack_e4m3x4(P[(g + 8) * 32 + 16 + tok(t, 0)], P[(g + 8) * 32 + 16 + tok(t, 1)], P[(g + 8) * 32 + 16 + tok(t, 2)], P[(g + 8) * 32 + 16 + tok(t, 3)]);
    // B via ldmatrix.x4.trans: 4 matrices = token rows {0-7, 8-15, 16-23, 24-31}, 8 b16 cols = 16 d
    uint32_t R[4];
    {
        const int m = lane >> 3, r = lane & 7;           // lane provides row address of matrix m, row r
        const uint32_t addr = (uint32_t)__cvta_generic_to_shared(sV + (m * 8 + r) * 16);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3]) : "r"(addr));
    }
    uint32_t be[2], bo[2];
    be[0] = prmt(R[0], R[1], 0x6420u); be[1] = prmt(R[2], R[3], 0x6420u);
    bo[0] = prmt(R[0], R[1], 0x7531u); bo[1] = prmt(R[2], R[3], 0x7531u);
    float de[4] = {0, 0, 0, 0}, dout[4] = {0, 0, 0, 0};
    mma_e4m3(de, a, be);
    mma_e4m3(dout, a, bo);
    // C col n=g' where g' = t*2+{0,1}: even mma -> d = 2*(t*2+j), odd -> 2*(t*2+j)+1
    O[g * 16 + 2 * (t * 2 + 0)]         = de[0];
    O[g * 16 + 2 * (t * 2 + 1)]         = de[1];
    O[(g + 8) * 16 + 2 * (t * 2 + 0)]   = de[2];
    O[(g + 8) * 16 + 2 * (t * 2 + 1)]   = de[3];
    O[g * 16 + 2 * (t * 2 + 0) + 1]       = dout[0];
    O[g * 16 + 2 * (t * 2 + 1) + 1]       = dout[1];
    O[(g + 8) * 16 + 2 * (t * 2 + 0) + 1] = dout[2];
    O[(g + 8) * 16 + 2 * (t * 2 + 1) + 1] = dout[3];
}

static float e4m3_to_f(uint8_t v) { __nv_fp8_e4m3 x; memcpy(&x, &v, 1); return (float)x; }
static uint8_t f_to_e4m3(float f) { __nv_fp8_e4m3 x(f); uint8_t v; memcpy(&v, &x, 1); return v; }

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[ENV] %s SMs=%d CC=%d.%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
    // ---- (2) layout check first (cheap, decisive)
    {
        uint8_t hA[16 * 32], hBt[8 * 32]; float ref[16 * 8] = {};
        uint32_t seed = 12345u;
        auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float)((seed >> 9) & 0xFF) / 64.f - 2.f; };
        for (int i = 0; i < 16 * 32; ++i) hA[i] = f_to_e4m3(rnd());
        for (int i = 0; i < 8 * 32; ++i) hBt[i] = f_to_e4m3(rnd());
        for (int m = 0; m < 16; ++m) for (int n = 0; n < 8; ++n) { float s = 0; for (int k = 0; k < 32; ++k) s += e4m3_to_f(hA[m * 32 + k]) * e4m3_to_f(hBt[n * 32 + k]); ref[m * 8 + n] = s; }
        uint8_t *dA, *dB; float* dD; CK(cudaMalloc(&dA, 512)); CK(cudaMalloc(&dB, 256)); CK(cudaMalloc(&dD, 128 * 4));
        CK(cudaMemcpy(dA, hA, 512, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, hBt, 256, cudaMemcpyHostToDevice));
        layout_kernel<<<1, 32>>>(dA, dB, dD); CK(cudaDeviceSynchronize());
        float hD[128]; CK(cudaMemcpy(hD, dD, 512, cudaMemcpyDeviceToHost));
        int bad = 0; float maxe = 0;
        for (int i = 0; i < 128; ++i) { float e = fabsf(hD[i] - ref[i]); maxe = fmaxf(maxe, e); if (e > 1e-3f * (1.f + fabsf(ref[i]))) ++bad; }
        printf("[LAYOUT] m16n8k32 e4m3 A/B/C fragment map: %s (max abs err %.3g, %d/128 bad)\n", bad ? "FAILED" : "OK", maxe, bad);
        if (bad) return 1;
    }
    // ---- (2b) token-permuted PV via ldmatrix.trans + prmt
    {
        float hP[16 * 32]; uint8_t hV[32 * 16]; float ref[16 * 16] = {};
        uint32_t seed = 777u;
        auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float)((seed >> 9) & 0xFF) / 256.f; };
        for (int i = 0; i < 16 * 32; ++i) hP[i] = e4m3_to_f(f_to_e4m3(rnd()));            // exactly representable P
        for (int i = 0; i < 32 * 16; ++i) hV[i] = f_to_e4m3(rnd() * 4.f - 2.f);
        for (int m = 0; m < 16; ++m) for (int n = 0; n < 16; ++n) { float s = 0; for (int k = 0; k < 32; ++k) s += hP[m * 32 + k] * e4m3_to_f(hV[k * 16 + n]); ref[m * 16 + n] = s; }
        float* dP; uint8_t* dV; float* dO; CK(cudaMalloc(&dP, sizeof(hP))); CK(cudaMalloc(&dV, 512)); CK(cudaMalloc(&dO, 256 * 4));
        CK(cudaMemcpy(dP, hP, sizeof(hP), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dV, hV, 512, cudaMemcpyHostToDevice));
        pv_kernel<<<1, 32>>>(dP, dV, dO); CK(cudaDeviceSynchronize());
        float hO[256]; CK(cudaMemcpy(hO, dO, 1024, cudaMemcpyDeviceToHost));
        int bad = 0; float maxe = 0;
        for (int i = 0; i < 256; ++i) { float e = fabsf(hO[i] - ref[i]); maxe = fmaxf(maxe, e); if (e > 1e-3f * (1.f + fabsf(ref[i]))) ++bad; }
        printf("[PV-PERMUTE] ldmatrix.x4.trans + prmt even/odd B construction: %s (max abs err %.3g, %d/256 bad)\n", bad ? "FAILED" : "OK", maxe, bad);
        if (bad) return 1;
    }
    // ---- (1) peak
    float* out; CK(cudaMalloc(&out, THREADS * 4));
    const int grid = p.multiProcessorCount * 8;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int v = 0; v < 2; ++v) {
        const double flops = 2.0 * 16 * 8 * (v ? 32 : 16) * NACC * (double)ITERS * 8 * grid;
        for (int w = 0; w < 3; ++w) { if (v) peak_kernel<1><<<grid, THREADS>>>(out, 0x38383838u, 0x38383838u); else peak_kernel<0><<<grid, THREADS>>>(out, 0x3f803f80u, 0x3f803f80u); }
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int it = 0; it < 5; ++it) {
            CK(cudaEventRecord(e0));
            if (v) peak_kernel<1><<<grid, THREADS>>>(out, 0x38383838u, 0x38383838u); else peak_kernel<0><<<grid, THREADS>>>(out, 0x3f803f80u, 0x3f803f80u);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
        }
        CK(cudaGetLastError());
        printf("[PEAK] %-40s %8.3f ms  %8.1f TFLOPS\n", v ? "mma.sync m16n8k32 e4m3 -> f32 accumulate" : "mma.sync m16n8k16 bf16 -> f32 accumulate", best, flops / best / 1e9);
    }
    printf("FP8_PROBE_DONE\n");
    return 0;
}

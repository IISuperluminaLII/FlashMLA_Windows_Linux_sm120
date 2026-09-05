// Throughput probe: fp32 dQ-style tile reductions into an L2-resident accumulator
// on plain sm_120, modeling the single-kernel backward's traffic pattern:
// 64 "KV-block" CTAs per (b,h) plane each reduce a 32x128 fp32 tile (16KB) into
// all 128 q-blocks of the plane (S=4096). Modes:
//   0  cp.reduce.async.bulk (per-row 512B bulk reductions, 32 lanes issue)
//   1  red.global.add.v4.f32 from all 128 threads (register-direct, 16B)
//   2  scalar atomicAdd (RED) from all threads (the 1.9x-cost baseline)
//   3  plain st.global.v4 (NON-reducing store = bandwidth ceiling reference)
// Also VERIFIES mode 0 sums correctly (every element == 64 after one launch).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("[FAILED] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

constexpr int ROWS = 32, COLS = 128;            // fp32 tile 32x128 = 16KB (128/128, BM=32)
constexpr int S = 4096, QB = S / ROWS;          // 128 q-blocks per plane
constexpr int THREADS = 128;
constexpr int KV_CTAS = 64;                     // S / BN(64)

__global__ void __launch_bounds__(THREADS) reduce_probe(float* __restrict__ dq, int mode) {
    extern __shared__ float tile[];
    float* base = dq + (size_t)blockIdx.y * S * COLS;
    for (int i = threadIdx.x; i < ROWS * COLS; i += THREADS) tile[i] = 1.0f;
    __syncthreads();
    const int r = threadIdx.x / 4, c0 = (threadIdx.x % 4) * 32;   // modes 1-3 ownership
    for (int qb = 0; qb < QB; ++qb) {
        float* dst = base + (size_t)qb * ROWS * COLS;
        if (mode == 0) {
            if (threadIdx.x < ROWS) {
                uint32_t src = (uint32_t)__cvta_generic_to_shared(tile + threadIdx.x * COLS);
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                             :: "l"(dst + threadIdx.x * COLS), "r"(src), "n"(COLS * 4) : "memory");
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                asm volatile("cp.async.bulk.wait_group.read 4;" ::: "memory");
            }
        } else if (mode == 1) {
            float* d = dst + r * COLS + c0; const float* s = tile + r * COLS + c0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float4 v = *reinterpret_cast<const float4*>(s + j * 4);
                asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};"
                             :: "l"(d + j * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
            }
        } else if (mode == 2) {
            float* d = dst + r * COLS + c0; const float* s = tile + r * COLS + c0;
            #pragma unroll
            for (int j = 0; j < 32; ++j) atomicAdd(d + j, s[j]);
        } else {
            float* d = dst + r * COLS + c0; const float* s = tile + r * COLS + c0;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                *reinterpret_cast<float4*>(d + j * 4) = *reinterpret_cast<const float4*>(s + j * 4);
        }
    }
    if (mode == 0 && threadIdx.x < ROWS) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

int main() {
    int dev = 0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    printf("[ENV] %s cc=%d.%d L2=%d MB\n", p.name, p.major, p.minor, p.l2CacheSize >> 20);
    const int HEADS = 32;                                        // planes: 32 x 2MB = 64MB (L2-resident)
    const size_t n = (size_t)HEADS * S * COLS;
    float* dq = nullptr; CK(cudaMalloc(&dq, n * sizeof(float)));
    const double bytes_per_launch = (double)KV_CTAS * HEADS * QB * ROWS * COLS * 4;  // 4 GB
    const char* names[4] = {"cp.reduce.async.bulk", "red.global.add.v4.f32", "atomicAdd scalar", "st.global.v4 (no reduce)"};
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    dim3 grid(KV_CTAS, HEADS), block(THREADS);
    const size_t smem = ROWS * COLS * sizeof(float);
    for (int mode = 0; mode < 4; ++mode) {
        for (int w = 0; w < 2; ++w) reduce_probe<<<grid, block, smem>>>(dq, mode);
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int it = 0; it < 5; ++it) {
            CK(cudaEventRecord(e0));
            reduce_probe<<<grid, block, smem>>>(dq, mode);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
        }
        CK(cudaGetLastError());
        printf("[MODE %d] %-26s %8.3f ms  %8.1f GB/s reduce traffic\n", mode, names[mode], best, bytes_per_launch / best / 1e6);
    }
    // correctness of the bulk reduce: fresh zero buffer, ONE launch -> every element == KV_CTAS
    CK(cudaMemset(dq, 0, n * sizeof(float)));
    reduce_probe<<<grid, block, smem>>>(dq, 0);
    CK(cudaDeviceSynchronize());
    float* h = new float[n];
    CK(cudaMemcpy(h, dq, n * sizeof(float), cudaMemcpyDeviceToHost));
    size_t bad = 0; for (size_t i = 0; i < n; ++i) if (h[i] != (float)KV_CTAS) ++bad;
    printf("[VERIFY] bulk reduce: %zu / %zu elements wrong (expect 0; each == %d)\n", bad, n, KV_CTAS);
    printf(bad == 0 ? "[OK] cp.reduce.async.bulk adds correctly on this silicon\n" : "[FAILED] bulk reduce numerics\n");
    delete[] h; CK(cudaFree(dq));
    printf("PROBE_DONE\n");
    return bad == 0 ? 0 : 1;
}

// Architecture-limit probes the structural designs hinge on (query, do not assume):
//  (1) aggregate L2 -> smem streaming bandwidth (all SMs, cp.async.cg 16 B, L2-resident 48 MB working set)
//  (2) same with 1-D cp.async.bulk 16 KB chunks (TMA engine path)
//  (3) per-SM ldmatrix.x4 shared-memory read bandwidth (B/clk/SM) -- the "128 B/clk" assumption
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("[FAILED] %s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)
__device__ __forceinline__ uint32_t cvta(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// (1) every thread streams 16-B granules of the CTA slice through a 2-stage smem ring with cp.async
__global__ void __launch_bounds__(256) l2_stream_cpasync(const int4* __restrict__ src, size_t bytes_per_cta, int iters, uint32_t* sink) {
    __shared__ __align__(16) int4 ring[2][256 * 4];    // 2 x 16 KB
    const int4* base = src + (static_cast<size_t>(blockIdx.x) * bytes_per_cta) / 16;
    const size_t n16 = bytes_per_cta / 16;
    uint32_t acc = 0;
    for (int it = 0; it < iters; ++it) {
        for (size_t off = 0; off < n16; off += 256 * 4) {
            const int st = (off / (256 * 4)) & 1;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const size_t i = off + j * 256 + threadIdx.x;
                uint32_t dst = cvta(&ring[st][j * 256 + threadIdx.x]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(base + (i < n16 ? i : 0)) : "memory");
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            acc ^= ring[st ^ 1][threadIdx.x].x;
        }
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    if (acc == 0xdeadbeef) sink[blockIdx.x] = acc;
}
// (2) one lane issues 16 KB bulk copies into a 2-stage ring, mbarrier-completed
__global__ void __launch_bounds__(256) l2_stream_bulk(const char* __restrict__ src, size_t bytes_per_cta, int iters, uint32_t* sink) {
    __shared__ __align__(128) char ring[2][16384];
    __shared__ __align__(8) uint64_t mbar[2];
    if (threadIdx.x == 0) {
        for (int s = 0; s < 2; ++s) asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "r"(cvta(&mbar[s])));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    const char* base = src + static_cast<size_t>(blockIdx.x) * bytes_per_cta;
    const size_t nchunks = bytes_per_cta / 16384;
    uint32_t acc = 0; uint32_t phase[2] = {0, 0};
    for (int it = 0; it < iters; ++it) {
        for (size_t c = 0; c < nchunks; ++c) {
            const int s = c & 1;
            if (threadIdx.x == 0) {
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(cvta(&mbar[s])), "r"(16384) : "memory");
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                             :: "r"(cvta(ring[s])), "l"(base + c * 16384), "r"(16384), "r"(cvta(&mbar[s])) : "memory");
            }
            uint32_t done = 0;
            do { asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }\n" : "=r"(done) : "r"(cvta(&mbar[s])), "r"(phase[s]) : "memory"); } while (!done);
            phase[s] ^= 1;
            acc ^= reinterpret_cast<uint32_t*>(ring[s])[threadIdx.x];
            __syncthreads();   // everyone done reading stage s before it is refilled two chunks later
        }
    }
    if (acc == 0xdeadbeef) sink[blockIdx.x] = acc;
}
// (3) ldmatrix.x4 read bandwidth: 8 warps hammer a 32 KB tile (conflict-free: 8 rows at 512-B stride, XOR granule rotation)
__global__ void __launch_bounds__(256) ldsm_bw(int iters, uint32_t* sink) {
    __shared__ __align__(128) uint32_t tile[8192];   // 32 KB = 64 rows x 512 B
    for (int i = threadIdx.x; i < 8192; i += 256) tile[i] = i * 2654435761u;
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll 8
        for (int k = 0; k < 64; ++k) {
            const int row = ((k * 8 + warp) + (lane & 7)) & 63;          // 8 consecutive rows per matrix
            const int gran = ((lane >> 3) + (row & 7)) & 31;               // 16-B granule, rotated by row -> distinct banks
            uint32_t addr = cvta(&tile[row * 128 + gran * 4]);
            uint32_t r0, r1, r2, r3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
            acc0 ^= r0; acc1 ^= r1; acc2 ^= r2; acc3 ^= r3;
        }
    }
    if ((acc0 ^ acc1 ^ acc2 ^ acc3) == 0xdeadbeef) sink[blockIdx.x] = acc0;
}
int main() {
    int dev = 0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    printf("device %s SMs=%d\n", p.name, p.multiProcessorCount);
    const int nsm = p.multiProcessorCount;
    const size_t total = 48ull << 20;                       // 48 MB working set (L2 is 128 MB)
    const size_t per_cta = total / nsm / 16384 * 16384;     // multiple of 16 KB
    char* d = nullptr; uint32_t* sink = nullptr;
    CK(cudaMalloc(&d, per_cta * nsm + 65536)); CK(cudaMemset(d, 1, per_cta * nsm + 65536)); CK(cudaMalloc(&sink, nsm * 8 * 4));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    l2_stream_cpasync<<<nsm, 256>>>(reinterpret_cast<const int4*>(d), per_cta, 4, sink); CK(cudaDeviceSynchronize());   // warm-up
    for (int ctas_per_sm = 1; ctas_per_sm <= 2; ++ctas_per_sm) {
        const int grid = nsm * ctas_per_sm; const size_t pc = per_cta / ctas_per_sm / 16384 * 16384; const int iters = 20;
        float ms = 0;
        CK(cudaEventRecord(e0)); l2_stream_cpasync<<<grid, 256>>>(reinterpret_cast<const int4*>(d), pc, iters, sink); CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
        CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("[L2->smem cp.async 16B] %d CTA/SM: %.0f GB/s aggregate (%.1f B/clk/SM at 2.61 GHz)\n", ctas_per_sm, static_cast<double>(pc) * grid * iters / (ms * 1e6), static_cast<double>(pc) * grid * iters / (ms * 1e-3) / nsm / 2.61e9);
        CK(cudaEventRecord(e0)); l2_stream_bulk<<<grid, 256>>>(d, pc, iters, sink); CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
        CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("[L2->smem bulk 16KB   ] %d CTA/SM: %.0f GB/s aggregate (%.1f B/clk/SM)\n", ctas_per_sm, static_cast<double>(pc) * grid * iters / (ms * 1e6), static_cast<double>(pc) * grid * iters / (ms * 1e-3) / nsm / 2.61e9);
    }
    for (int ctas_per_sm = 1; ctas_per_sm <= 2; ++ctas_per_sm) {
        const int iters = 2000; const int grid = nsm * ctas_per_sm;
        float ms = 0;
        CK(cudaEventRecord(e0)); ldsm_bw<<<grid, 256>>>(iters, sink); CK(cudaEventRecord(e1)); CK(cudaDeviceSynchronize());
        CK(cudaEventElapsedTime(&ms, e0, e1));
        const double bytes = static_cast<double>(grid) * 8 * iters * 64 * 512;   // 8 warps x 64 x4 (512 B) per iter
        printf("[ldmatrix.x4 smem read ] %d CTA/SM (%d warps/SM): %.1f B/clk/SM at 2.61 GHz (%.0f GB/s aggregate)\n", ctas_per_sm, 8 * ctas_per_sm, bytes / (ms * 1e-3) / nsm / 2.61e9, bytes / (ms * 1e6));
    }
    printf("L2_SMEM_BW_PROBE_DONE\n");
    return 0;
}

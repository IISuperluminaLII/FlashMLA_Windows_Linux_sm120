// ptxas acceptance probe for the fwd rung-8/9 warp-specialized pipeline primitives on PLAIN sm_120.
#include <cuda_runtime.h>
#include <cstdint>
__global__ void __launch_bounds__(288, 1) probe(float* out, const float* in, const uint64_t* tmap) {
    __shared__ __align__(8) uint64_t mbar[4];
    __shared__ __align__(128) float buf[1024];
    uint32_t mb = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[0]));
    uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(&buf[0]));
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 32;\n" :: "r"(mb));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x >= 256) {   // producer warp
        // (A) cp.async 16B tracked by the mbarrier (noinc)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sb + (threadIdx.x - 256) * 16), "l"(in), "r"(16) : "memory");
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(mb) : "memory");
        // (B) 1-D bulk row copy with complete_tx
        if (threadIdx.x == 256) asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(mb), "r"(384) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(sb + 1024), "l"(in), "r"(384), "r"(mb) : "memory");
        // (C) TMA tensor-map load (rank 3) -- proven by FROST on this GPU, never probed in-tree
        int c0 = 0, c1 = 0, c2 = 0;
        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
                     :: "r"(sb + 2048), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(mb) : "memory");
        // producer waits for consumers on a named barrier (id/count form)
        asm volatile("bar.sync 1, 288;\n" ::: "memory");
    } else {
        // consumer: parity wait + fire-and-forget arrive
        uint32_t ph = 0, done = 0;
        do {
            asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }\n"
                         : "=r"(done) : "r"(mb), "r"(ph) : "memory");
        } while (!done);
        asm volatile("bar.arrive 1, 288;\n" ::: "memory");
        // (D) packed f32x2 FMA / MUL for the softmax micro-structure
        float a = in[threadIdx.x], b = in[threadIdx.x + 1], c = in[threadIdx.x + 2], d = in[threadIdx.x + 3];
        uint64_t pa, pb, pc, pr;
        asm("mov.b64 %0, {%1, %2};\n" : "=l"(pa) : "f"(a), "f"(b));
        asm("mov.b64 %0, {%1, %2};\n" : "=l"(pb) : "f"(c), "f"(d));
        asm("mov.b64 %0, {%1, %2};\n" : "=l"(pc) : "f"(b), "f"(a));
        asm("fma.rn.f32x2 %0, %1, %2, %3;\n" : "=l"(pr) : "l"(pa), "l"(pb), "l"(pc));
        asm("mul.rn.f32x2 %0, %1, %2;\n" : "=l"(pr) : "l"(pr), "l"(pb));
        float r0, r1;
        asm("mov.b64 {%0, %1}, %2;\n" : "=f"(r0), "=f"(r1) : "l"(pr));
        out[threadIdx.x] = r0 + r1 + buf[threadIdx.x];
    }
}

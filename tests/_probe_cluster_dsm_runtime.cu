// Runtime smoke test: 2-CTA cluster + DSM crossover primitives on plain sm_120.
// Verifies ON LIVE SILICON (not just ptxas):
//   1. cluster launch (dim3(2,1,1)) with 99KB opt-in dynamic smem per CTA
//   2. mapa.shared::cluster peer address of a dynamic-smem mbarrier + data slot
//   3. st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 peer store
//   4. mbarrier expect_tx handshake: consumer waits for 16 bytes from its peer
//   5. cluster_sync ordering
// Each CTA writes (rank ^ pattern) into its PEER's smem slot; both CTAs then read
// their OWN slot and report what the peer delivered. PASS iff both directions land.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(call)                                                                  \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            printf("[FAILED] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            return 1;                                                             \
        }                                                                         \
    } while (0)

constexpr size_t SMEM_BYTES = 99 * 1024;   // the kernels' real opt-in budget

__global__ void __cluster_dims__(2, 1, 1) dsm_probe_kernel(int64_t* out, int* ok) {
    extern __shared__ char smem_raw[];
    // Layout: [0..15] data slot (16B), [16..23] mbarrier (8B). Rest untouched
    // (presence of the full 99KB allocation is part of what is being probed).
    int64_t* slot = reinterpret_cast<int64_t*>(smem_raw);
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem_raw + 16);

    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));

    const uint32_t slot_sa = static_cast<uint32_t>(__cvta_generic_to_shared(slot));
    const uint32_t bar_sa  = static_cast<uint32_t>(__cvta_generic_to_shared(bar));

    if (threadIdx.x == 0) {
        slot[0] = -1; slot[1] = -1;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_sa));
    }
    // Barrier-init visible cluster-wide before any peer touches it.
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");

    if (threadIdx.x == 0) {
        // Arm my own barrier: expect 16 bytes (from the peer's st.async).
        uint64_t st;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], 16;"
                     : "=l"(st) : "r"(bar_sa));
        // Peer addresses via mapa (rank ^ 1).
        uint32_t peer_slot, peer_bar;
        const uint32_t peer = rank ^ 1u;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer_slot) : "r"(slot_sa), "r"(peer));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peer_bar) : "r"(bar_sa), "r"(peer));
        // Ship 16 bytes into the peer's slot, crediting the peer's tx barrier.
        const int64_t payload0 = 0x1000 + (int64_t)rank;
        const int64_t payload1 = 0x2000 + (int64_t)rank;
        asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 "
                     "[%0], {%1, %2}, [%3];"
                     :: "r"(peer_slot), "l"(payload0), "l"(payload1), "r"(peer_bar));
        // Wait for MY barrier: peer's 16 bytes arrived.
        uint32_t done = 0;
        while (!done) {
            asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; "
                         "selp.u32 %0, 1, 0, p;}" : "=r"(done) : "r"(bar_sa));
        }
        // What did the peer deliver? Expect payloads stamped with THEIR rank.
        const int64_t got0 = slot[0], got1 = slot[1];
        const int64_t want0 = 0x1000 + (int64_t)(rank ^ 1u);
        const int64_t want1 = 0x2000 + (int64_t)(rank ^ 1u);
        out[rank * 2 + 0] = got0;
        out[rank * 2 + 1] = got1;
        if (got0 == want0 && got1 == want1) atomicAdd(ok, 1);
    }
    // Both CTAs alive until every transfer completes (smem lifetime).
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

int main() {
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    CK(cudaFuncSetAttribute(dsm_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)SMEM_BYTES));

    int64_t* out; int* ok;
    CK(cudaMalloc(&out, 4 * sizeof(int64_t)));
    CK(cudaMalloc(&ok, sizeof(int)));
    CK(cudaMemset(out, 0, 4 * sizeof(int64_t)));
    CK(cudaMemset(ok, 0, sizeof(int)));

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2, 1, 1);
    cfg.blockDim = dim3(32, 1, 1);
    cfg.dynamicSmemBytes = SMEM_BYTES;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, dsm_probe_kernel, out, ok));
    CK(cudaDeviceSynchronize());

    int64_t h_out[4]; int h_ok = 0;
    CK(cudaMemcpy(h_out, out, sizeof(h_out), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&h_ok, ok, sizeof(int), cudaMemcpyDeviceToHost));
    printf("cta0 got: %llx %llx | cta1 got: %llx %llx | ok=%d\n",
           (long long)h_out[0], (long long)h_out[1],
           (long long)h_out[2], (long long)h_out[3], h_ok);
    printf(h_ok == 2 ? "[OK] DSM crossover primitives WORK on this silicon\n"
                     : "[FAILED] peer payload mismatch\n");
    CK(cudaFree(out)); CK(cudaFree(ok));
    return h_ok == 2 ? 0 : 1;
}

// Runtime probe for fwd rung 10: a TMA tensor-map (cp.async.bulk.tensor.3d, SWIZZLE_128B) load of a
// [64 rows x 64 cols] bf16 box from a [total_k, H, D] K tensor on PLAIN sm_120, verifying that the
// smem image equals our XOR swizzle swz(r,c) = r*64 + (c ^ ((r&7)<<3)) and that OOB rows zero-fill.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("[FAILED] %s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

__device__ __forceinline__ uint32_t cvta(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__global__ void tma_probe(const __grid_constant__ CUtensorMap tmap, uint16_t* out, int c0, int c1, int c2) {
    __shared__ __align__(1024) uint16_t buf[64 * 64];      // 8 KB box; SWIZZLE_128B needs 1024-B alignment
    __shared__ __align__(8) uint64_t mbar;
    const uint32_t mb = cvta(&mbar), sb = cvta(buf);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "r"(mb));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(mb), "r"(64 * 64 * 2) : "memory");
        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
                     :: "r"(sb), "l"(&tmap), "r"(c0), "r"(c1), "r"(c2), "r"(mb) : "memory");
    }
    uint32_t done = 0;
    do {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }\n"
                     : "=r"(done) : "r"(mb), "r"(0) : "memory");
    } while (!done);
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) out[i] = buf[i];
}

typedef CUresult (*PFN_encodeTiled_t)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                       const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                       CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

int main() {
    const int TK = 256, H = 2, D = 192;
    std::vector<uint16_t> hK(static_cast<size_t>(TK) * H * D);
    for (int r = 0; r < TK; ++r) for (int h = 0; h < H; ++h) for (int d = 0; d < D; ++d)
        hK[(static_cast<size_t>(r) * H + h) * D + d] = static_cast<uint16_t>((r * 7 + h * 3001 + d * 13 + 1) & 0x7fff);
    uint16_t* dK = nullptr; uint16_t* dOut = nullptr;
    CK(cudaMalloc(&dK, hK.size() * 2)); CK(cudaMalloc(&dOut, 64 * 64 * 2));
    CK(cudaMemcpy(dK, hK.data(), hK.size() * 2, cudaMemcpyHostToDevice));

    void* fn = nullptr; cudaDriverEntryPointQueryResult q;
    CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q));
    if (fn == nullptr || q != cudaDriverEntryPointSuccess) { printf("[FAILED] no cuTensorMapEncodeTiled entry point\n"); return 1; }
    PFN_encodeTiled_t encode = reinterpret_cast<PFN_encodeTiled_t>(fn);

    alignas(64) CUtensorMap tmap;
    cuuint64_t dims[3]    = {static_cast<cuuint64_t>(D), static_cast<cuuint64_t>(H), static_cast<cuuint64_t>(TK)};
    cuuint64_t strides[2] = {static_cast<cuuint64_t>(D) * 2, static_cast<cuuint64_t>(H) * D * 2};   // bytes, dims 1..2
    cuuint32_t box[3]     = {64, 1, 64};
    cuuint32_t estr[3]    = {1, 1, 1};
    CUresult cr = encode(&tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dK, dims, strides, box, estr,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (cr != CUDA_SUCCESS) { printf("[FAILED] cuTensorMapEncodeTiled rc=%d\n", static_cast<int>(cr)); return 1; }

    int fails = 0;
    // case A: interior box (chunk 1 = cols 64..127, head 1, rows 96..159); case B: OOB tail (rows 224..287)
    const int cases[2][3] = {{64, 1, 96}, {128, 0, 224}};
    for (int t = 0; t < 2; ++t) {
        const int c0 = cases[t][0], c1 = cases[t][1], c2 = cases[t][2];
        tma_probe<<<1, 128>>>(tmap, dOut, c0, c1, c2);
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> img(64 * 64);
        CK(cudaMemcpy(img.data(), dOut, 64 * 64 * 2, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) {
            const int row = c2 + r;
            const uint16_t expect = (row < TK) ? hK[(static_cast<size_t>(row) * H + c1) * D + c0 + c] : 0;
            const uint16_t got = img[r * 64 + (c ^ ((r & 7) << 3))];     // our swz(r, c) on a 64-wide row
            if (got != expect) { if (bad < 3) printf("  mismatch case %d r=%d c=%d got=%u expect=%u\n", t, r, c, got, expect); ++bad; }
        }
        printf("[%s] case %d (c0=%d h=%d row0=%d): %d/4096 mismatches vs swz XOR image%s\n",
               bad ? "FAILED" : "OK", t, c0, c1, c2, bad, t == 1 ? " (rows 256..287 OOB -> zero-fill checked)" : "");
        fails += bad;
    }
    printf("TMA_TENSORMAP_PROBE %s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}

#pragma once
/***************************************************************************************************
 * SM120 Sparse-FP8 Decode Traits - device-only constants + shared-memory plan.
 * Included ONLY by splitkv_mla.cu (never by host TUs; pybind sees params.h via splitkv_mla.h).
 *
 * Design: audit/design-sparse-decode.md sections 1-3. Byte-exact smem accounting:
 *   sQ   bf16 [64][512]  65,536 B   resident Q nope half (reused 9 x num_topk_blocks times)
 *   sKV  bf16 [64][64]    8,192 B   gathered+dequantized K d-tile / V column tile
 *   sAcc fp32 [64][68]   17,408 B   QK staging | softmax input | PV staging (ACC_LD=68 pad
 *                                   makes the phase-C read pattern (4r+sub+4j) mod 32 cover
 *                                   all banks exactly once -> conflict-free)
 *   sPQ  bf16 [64][64]    8,192 B   UNION: Q-rope tile (QK d-tile 8) | P probabilities
 *   sM/sL/sScale f32[64]    768 B   online-softmax state (M2 base-2 domain, -1e30 init)
 *   sTokPtr ptr[64]         512 B   per-token gather base (nullptr == invalid index)
 *   sValid  i8[64]           64 B
 *   TOTAL = 100,736 B = 98.375 KiB <= 101,376 B (99 KiB opt-in) -> exactly 1 CTA/SM
 *   (per-SM budget 102,400 B; live-verified via cudaDeviceGetAttribute).
 **************************************************************************************************/

#include <cuda_bf16.h>
#include <cstdint>

namespace sm120 {
namespace sparse_decode {

// Block geometry
static constexpr int BLOCK_M         = 64;    // query HEADS per CTA (rows of the attention tile)
static constexpr int TOPK_BLOCK_SIZE = 64;    // selected KV tokens per iteration
static constexpr int PAGE_BLOCK_SIZE = 64;    // compile-time; host-enforced page size

// MLA dims
static constexpr int HEAD_DIM_K    = 576;
static constexpr int HEAD_DIM_NOPE = 512;     // == HEAD_DIM_V; also the fp8-quantized span
static constexpr int HEAD_DIM_V    = 512;
static constexpr int HEAD_DIM_ROPE = 64;
static constexpr int QUANT_TILE_SIZE = 128;   // elements per fp32 scale
static constexpr int NUM_SCALES      = 4;
static constexpr int NUM_BYTES_PER_TOKEN = 656;  // 512 fp8 + 16 scale bytes + 128 rope bytes

// Threads
static constexpr int NUM_THREADS = 256;
static constexpr int NUM_WARPS   = NUM_THREADS / 32;          // 8
static constexpr int TILES_PER_WARP = 16 / NUM_WARPS;         // 2 (16 WMMA tiles per 64x64 GEMM)

// Staging leading dim (fp32): 64 + 4 pad for bank-conflict-free softmax reads
static constexpr int ACC_LD = 68;

// Output accumulator: O[64][512] fp32 across 256 threads
static constexpr int O_PER_THREAD = (BLOCK_M * HEAD_DIM_V) / NUM_THREADS;   // 128

// Softmax constants (base-2 domain). -1e30f, NOT -INFINITY: keeps (m_old - m_new) finite so
// exp2f never sees (-inf)-(-inf) = NaN in all-invalid blocks (sm90 convention).
static constexpr float MAX_INIT_VAL = -1e30f;
static constexpr float LOG2E = 1.4426950408889634f;   // bit-identical to (float)M_LOG2E

struct SharedMemoryPlan {
    __align__(128) __nv_bfloat16 sQ  [BLOCK_M * HEAD_DIM_NOPE];   // [64][512] ld 512
    __align__(128) __nv_bfloat16 sKV [TOPK_BLOCK_SIZE * 64];      // [64][64]  K d-tile | V col-tile
    __align__(128) float         sAcc[BLOCK_M * ACC_LD];          // [64][68]  staging
    __align__(128) __nv_bfloat16 sPQ [BLOCK_M * 64];              // [64][64]  P | Q-rope (UNION)
    __align__(128) float         sM     [BLOCK_M];
    __align__(128) float         sL     [BLOCK_M];
    __align__(128) float         sScale [BLOCK_M];
    __align__(128) const uint8_t* sTokPtr[TOPK_BLOCK_SIZE];
    __align__(128) int8_t        sValid [TOPK_BLOCK_SIZE];
};
static_assert(sizeof(SharedMemoryPlan) <= 99 * 1024, "exceeds SM120 99KB opt-in smem");

}  // namespace sparse_decode
}  // namespace sm120

//! Metal Shading Language (MSL) kernel sources for zgml GPU backend.
//!
//! Each kernel is a comptime string literal that gets compiled at runtime via
//! `newLibraryWithSource:`. Three GEMM variants are provided:
//!
//!   - `f32_matmul`  — Threadgroup-tiled FP32 GEMM (32x32 tiles, shared memory, FMA)
//!   - `q8_matmul`   — Dequantize-on-the-fly GEMM for Q8_0 blocks (f16 scale + 32 i8)
//!   - `q4_matmul`   — Dequantize-on-the-fly GEMM for Q4_0 blocks (f16 scale + 16 nibble bytes)

// ---------------------------------------------------------------------------
// f32_matmul — tiled GEMM with 32x32 threadgroup tiles
// ---------------------------------------------------------------------------

pub const f32_matmul_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\// Tile dimensions
    \\constant uint TILE = 32;
    \\constant uint THREAD_TILE = 8;
    \\// Threads per threadgroup: 32 x (TILE/THREAD_TILE) = 32 x 4 = 128
    \\// Each thread covers a THREAD_TILE-tall strip within the 32x32 output tile.
    \\
    \\kernel void f32_matmul(
    \\    device const float* A       [[buffer(0)]],
    \\    device const float* B       [[buffer(1)]],
    \\    device       float* C       [[buffer(2)]],
    \\    constant     uint&  M       [[buffer(3)]],
    \\    constant     uint&  N       [[buffer(4)]],
    \\    constant     uint&  K       [[buffer(5)]],
    \\    uint2 gid  [[threadgroup_position_in_grid]],
    \\    uint2 tid  [[thread_position_in_threadgroup]])
    \\{
    \\    // Shared memory tiles
    \\    threadgroup float A_tile[TILE][TILE];
    \\    threadgroup float B_tile[TILE][TILE];
    \\
    \\    // Global row/col for this thread's output element(s)
    \\    const uint row_base = gid.y * TILE + tid.y * THREAD_TILE;
    \\    const uint col      = gid.x * TILE + tid.x;
    \\
    \\    // Accumulators — each thread computes THREAD_TILE rows for one column
    \\    float acc[THREAD_TILE];
    \\    for (uint i = 0; i < THREAD_TILE; i++) acc[i] = 0.0f;
    \\
    \\    // Walk over K in TILE-sized steps
    \\    const uint num_tiles = (K + TILE - 1) / TILE;
    \\    for (uint t = 0; t < num_tiles; t++) {
    \\        // Cooperative load of A_tile and B_tile.
    \\        // Each thread loads THREAD_TILE elements to cover the 32x32 tile.
    \\        for (uint i = 0; i < THREAD_TILE; i++) {
    \\            const uint a_row = row_base + i;
    \\            const uint a_col = t * TILE + tid.x;
    \\            A_tile[tid.y * THREAD_TILE + i][tid.x] =
    \\                (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
    \\
    \\            const uint b_row = t * TILE + tid.y * THREAD_TILE + i;
    \\            const uint b_col = col;
    \\            B_tile[tid.y * THREAD_TILE + i][tid.x] =
    \\                (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
    \\        }
    \\
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        // Multiply-accumulate within the tile
    \\        for (uint k = 0; k < TILE; k++) {
    \\            float b_val = B_tile[k][tid.x];
    \\            for (uint i = 0; i < THREAD_TILE; i++) {
    \\                acc[i] = fma(A_tile[tid.y * THREAD_TILE + i][k], b_val, acc[i]);
    \\            }
    \\        }
    \\
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    // Write results
    \\    for (uint i = 0; i < THREAD_TILE; i++) {
    \\        const uint r = row_base + i;
    \\        if (r < M && col < N) {
    \\            C[r * N + col] = acc[i];
    \\        }
    \\    }
    \\}
;

// ---------------------------------------------------------------------------
// q8_matmul — dequantize Q8_0 blocks on the fly during GEMM
// ---------------------------------------------------------------------------

pub const q8_matmul_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\// Q8_0 block layout: 2 bytes f16 scale + 32 i8 quants = 34 bytes, 32 elements
    \\constant uint Q8_BLOCK_SIZE  = 32;
    \\constant uint Q8_BLOCK_BYTES = 34;
    \\
    \\// Dequantize one element from a Q8_0 weight buffer.
    \\inline float q8_dequant(device const uchar* W, uint flat_idx) {
    \\    const uint blk      = flat_idx / Q8_BLOCK_SIZE;
    \\    const uint within   = flat_idx % Q8_BLOCK_SIZE;
    \\    const uint blk_off  = blk * Q8_BLOCK_BYTES;
    \\    // Read f16 scale (first 2 bytes of block).
    \\    const half scale = *reinterpret_cast<device const half*>(W + blk_off);
    \\    const char q     = *reinterpret_cast<device const char*>(W + blk_off + 2 + within);
    \\    return float(q) * float(scale);
    \\}
    \\
    \\kernel void q8_matmul(
    \\    device const float* A       [[buffer(0)]],   // input  [M, K] f32
    \\    device const uchar* W       [[buffer(1)]],   // weight [K, N] Q8_0 blocks
    \\    device       float* C       [[buffer(2)]],   // output [M, N] f32
    \\    constant     uint&  M       [[buffer(3)]],
    \\    constant     uint&  N       [[buffer(4)]],
    \\    constant     uint&  K       [[buffer(5)]],
    \\    uint2 gid [[thread_position_in_grid]])
    \\{
    \\    const uint row = gid.y;
    \\    const uint col = gid.x;
    \\    if (row >= M || col >= N) return;
    \\
    \\    float acc = 0.0f;
    \\
    \\    // Process K in block-aligned chunks for better data locality
    \\    for (uint k = 0; k < K; k++) {
    \\        const float a_val = A[row * K + k];
    \\        const uint  w_idx = k * N + col;
    \\        acc = fma(a_val, q8_dequant(W, w_idx), acc);
    \\    }
    \\
    \\    C[row * N + col] = acc;
    \\}
;

// ---------------------------------------------------------------------------
// q4_matmul — nibble unpacking + dequant Q4_0 blocks during GEMM
// ---------------------------------------------------------------------------

pub const q4_matmul_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\// Q4_0 block layout: 2 bytes f16 scale + 16 packed nibble bytes = 18 bytes, 32 elements
    \\// Each nibble byte holds two 4-bit unsigned values; signed = nibble - 8.
    \\constant uint Q4_BLOCK_SIZE  = 32;
    \\constant uint Q4_BLOCK_BYTES = 18;
    \\
    \\// Dequantize one element from a Q4_0 weight buffer.
    \\inline float q4_dequant(device const uchar* W, uint flat_idx) {
    \\    const uint blk      = flat_idx / Q4_BLOCK_SIZE;
    \\    const uint within   = flat_idx % Q4_BLOCK_SIZE;
    \\    const uint blk_off  = blk * Q4_BLOCK_BYTES;
    \\    // Read f16 scale.
    \\    const half scale = *reinterpret_cast<device const half*>(W + blk_off);
    \\    // Each byte holds 2 nibbles: low nibble = even element, high nibble = odd.
    \\    const uint byte_idx = within / 2;
    \\    const uchar packed  = W[blk_off + 2 + byte_idx];
    \\    int nibble;
    \\    if (within % 2 == 0) {
    \\        nibble = int(packed & 0x0F) - 8;
    \\    } else {
    \\        nibble = int(packed >> 4) - 8;
    \\    }
    \\    return float(nibble) * float(scale);
    \\}
    \\
    \\kernel void q4_matmul(
    \\    device const float* A       [[buffer(0)]],   // input  [M, K] f32
    \\    device const uchar* W       [[buffer(1)]],   // weight [K, N] Q4_0 blocks
    \\    device       float* C       [[buffer(2)]],   // output [M, N] f32
    \\    constant     uint&  M       [[buffer(3)]],
    \\    constant     uint&  N       [[buffer(4)]],
    \\    constant     uint&  K       [[buffer(5)]],
    \\    uint2 gid [[thread_position_in_grid]])
    \\{
    \\    const uint row = gid.y;
    \\    const uint col = gid.x;
    \\    if (row >= M || col >= N) return;
    \\
    \\    float acc = 0.0f;
    \\
    \\    for (uint k = 0; k < K; k++) {
    \\        const float a_val = A[row * K + k];
    \\        const uint  w_idx = k * N + col;
    \\        acc = fma(a_val, q4_dequant(W, w_idx), acc);
    \\    }
    \\
    \\    C[row * N + col] = acc;
    \\}
;

// ---------------------------------------------------------------------------
// Combined source — all three kernels in one compilation unit
// ---------------------------------------------------------------------------

pub const all_shaders = f32_matmul_source ++ "\n" ++ q8_matmul_source ++ "\n" ++ q4_matmul_source;

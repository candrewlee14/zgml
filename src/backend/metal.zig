//! Metal GPU backend for macOS / Apple Silicon.
//!
//! Uses shared memory (MTLResourceStorageModeShared) so upload/download
//! are plain memcpy — CPU and GPU see the same physical pages.
//! Each dispatch is synchronous (commit + waitUntilCompleted).

const std = @import("std");
const backend_mod = @import("../backend.zig");
const profile_mod = @import("../profile.zig");
const reference = @import("reference.zig");
const program_mod = @import("program.zig");
const c = @cImport(@cInclude("metal_shim.h"));

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.Io.Threaded.global_single_threaded.io()).nanoseconds;
}

// ── Tile size for simdgroup kernel ────────────────────────────────

const TILE: u32 = 32; // output tile per threadgroup (TILE x TILE)
// 4 simdgroups per threadgroup (128 threads), each handles 8x8 sub-tiles
// Shared memory per K step: TILE*8 + 8*TILE = 512 floats = 2 KB

// ── Metal shader source (compiled at init time) ───────────────────

const shader_source =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\constant uint TILE = 32;
    \\constant uint NSUB = 4; // 2x2 arrangement of 8x8 sub-tiles per simdgroup
    \\
    \\struct MatMulParams {
    \\    uint M; uint N; uint K;
    \\    uint a_row_stride; uint a_col_stride;
    \\    uint b_row_stride; uint b_col_stride;
    \\    uint a_offset; uint b_offset;
    \\    uint dst_offset; uint dst_row_stride;
    \\};
    \\
    \\// Simdgroup-accelerated matmul: 4 simdgroups per threadgroup,
    \\// each computing a 2x2 grid of 8x8 tiles = 16x16 per simdgroup,
    \\// 32x32 per threadgroup. Uses hardware matrix multiply-accumulate.
    \\kernel void matmul_f32(
    \\    device const float* A [[buffer(0)]],
    \\    device const float* B [[buffer(1)]],
    \\    device float* C       [[buffer(2)]],
    \\    constant MatMulParams& p [[buffer(3)]],
    \\    uint2 group_id  [[threadgroup_position_in_grid]],
    \\    uint  simd_idx  [[simdgroup_index_in_threadgroup]],
    \\    uint  lane      [[thread_index_in_simdgroup]],
    \\    uint  tid       [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint gRow = group_id.y * TILE;
    \\    const uint gCol = group_id.x * TILE;
    \\
    \\    // Each simdgroup owns a 16x16 quadrant (2x2 of 8x8 sub-tiles).
    \\    const uint sRow = (simd_idx / 2) * 16;
    \\    const uint sCol = (simd_idx % 2) * 16;
    \\
    \\    simdgroup_float8x8 acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\
    \\    threadgroup float tA[TILE * 8];
    \\    threadgroup float tB[8 * TILE];
    \\
    \\    for (uint kt = 0; kt < p.K; kt += 8) {
    \\        // Cooperatively load A[gRow:gRow+32, kt:kt+8] — 256 elems, 128 threads → 2 each
    \\        for (uint i = tid; i < TILE * 8; i += 128) {
    \\            uint r = i / 8, c = i % 8;
    \\            uint ar = gRow + r, ac = kt + c;
    \\            tA[i] = (ar < p.M && ac < p.K)
    \\                ? A[p.a_offset + ar * p.a_row_stride + ac * p.a_col_stride] : 0.0f;
    \\        }
    \\        // Cooperatively load B[kt:kt+8, gCol:gCol+32] — 256 elems
    \\        for (uint i = tid; i < 8 * TILE; i += 128) {
    \\            uint r = i / TILE, c = i % TILE;
    \\            uint br = kt + r, bc = gCol + c;
    \\            tB[i] = (br < p.K && bc < p.N)
    \\                ? B[p.b_offset + br * p.b_row_stride + bc * p.b_col_stride] : 0.0f;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        // Each simdgroup: 2x2 sub-tiles within its 16x16 quadrant
    \\        simdgroup_float8x8 a0, a1, b0, b1;
    \\        simdgroup_load(a0, tA + (sRow + 0) * 8, 8);
    \\        simdgroup_load(a1, tA + (sRow + 8) * 8, 8);
    \\        simdgroup_load(b0, tB + (sCol + 0), TILE);
    \\        simdgroup_load(b1, tB + (sCol + 8), TILE);
    \\
    \\        simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
    \\        simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
    \\        simdgroup_multiply_accumulate(acc[2], a1, b0, acc[2]);
    \\        simdgroup_multiply_accumulate(acc[3], a1, b1, acc[3]);
    \\
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    // Store 2x2 sub-tiles via threadgroup memory
    \\    threadgroup float tC[TILE * TILE];
    \\    simdgroup_store(acc[0], tC + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[1], tC + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(acc[2], tC + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[3], tC + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Write to global — 1024 elems, 128 threads → 8 each
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, cc = gCol + c;
    \\        if (cr < p.M && cc < p.N)
    \\            C[p.dst_offset + cr * p.dst_row_stride + cc] = tC[i];
    \\    }
    \\}
    \\
    \\struct QMatMulParams {
    \\    uint M; uint N; uint K;
    \\    uint block_size;
    \\    uint input_offset;
    \\    uint input_row_stride;
    \\    uint dst_offset;
    \\    uint dst_row_stride;
    \\};
    \\
    \\// Simdgroup-accelerated quantized matmul: dequantize during tile load.
    \\kernel void qmatmul_f32(
    \\    device const char*  weight_data   [[buffer(0)]],
    \\    device const float* weight_scales [[buffer(1)]],
    \\    device const float* input         [[buffer(2)]],
    \\    device float*       output        [[buffer(3)]],
    \\    constant QMatMulParams& p [[buffer(4)]],
    \\    uint2 group_id  [[threadgroup_position_in_grid]],
    \\    uint  simd_idx  [[simdgroup_index_in_threadgroup]],
    \\    uint  lane      [[thread_index_in_simdgroup]],
    \\    uint  tid       [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint gRow = group_id.y * TILE;
    \\    const uint gCol = group_id.x * TILE;
    \\    const uint sRow = (simd_idx / 2) * 16;
    \\    const uint sCol = (simd_idx % 2) * 16;
    \\
    \\    simdgroup_float8x8 acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\
    \\    threadgroup float tI[TILE * 8];
    \\    threadgroup float tW[8 * TILE];
    \\
    \\    for (uint kt = 0; kt < p.K; kt += 8) {
    \\        for (uint i = tid; i < TILE * 8; i += 128) {
    \\            uint r = i / 8, c = i % 8;
    \\            uint ir = gRow + r, ic = kt + c;
    \\            tI[i] = (ir < p.M && ic < p.K) ? input[p.input_offset + ir * p.input_row_stride + ic] : 0.0f;
    \\        }
    \\        for (uint i = tid; i < 8 * TILE; i += 128) {
    \\            uint r = i / TILE, c = i % TILE;
    \\            uint kr = kt + r, nc = gCol + c;
    \\            if (kr < p.K && nc < p.N) {
    \\                uint w_idx = kr * p.N + nc;
    \\                tW[i] = float(weight_data[w_idx]) * weight_scales[w_idx / p.block_size];
    \\            } else {
    \\                tW[i] = 0.0f;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        simdgroup_float8x8 a0, a1, b0, b1;
    \\        simdgroup_load(a0, tI + (sRow + 0) * 8, 8);
    \\        simdgroup_load(a1, tI + (sRow + 8) * 8, 8);
    \\        simdgroup_load(b0, tW + (sCol + 0), TILE);
    \\        simdgroup_load(b1, tW + (sCol + 8), TILE);
    \\
    \\        simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
    \\        simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
    \\        simdgroup_multiply_accumulate(acc[2], a1, b0, acc[2]);
    \\        simdgroup_multiply_accumulate(acc[3], a1, b1, acc[3]);
    \\
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    threadgroup float tC[TILE * TILE];
    \\    simdgroup_store(acc[0], tC + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[1], tC + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(acc[2], tC + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[3], tC + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, cc = gCol + c;
    \\        if (cr < p.M && cc < p.N)
    \\            output[p.dst_offset + cr * p.dst_row_stride + cc] = tC[i];
    \\    }
    \\}
    \\
    \\constant uint MAX_QMATMUL_BATCH = 4;
    \\
    \\struct QMatmulBatch4Params {
    \\    uint n_ops; uint max_tiles_y;
    \\    uint M[MAX_QMATMUL_BATCH];
    \\    uint N[MAX_QMATMUL_BATCH];
    \\    uint K[MAX_QMATMUL_BATCH];
    \\    uint block_size[MAX_QMATMUL_BATCH];
    \\    uint input_offset[MAX_QMATMUL_BATCH];
    \\    uint input_row_stride[MAX_QMATMUL_BATCH];
    \\    uint dst_offset[MAX_QMATMUL_BATCH];
    \\    uint dst_row_stride[MAX_QMATMUL_BATCH];
    \\    uint has_slice[MAX_QMATMUL_BATCH];
    \\    uint slice_rows[MAX_QMATMUL_BATCH];
    \\    uint slice_cols[MAX_QMATMUL_BATCH];
    \\    uint slice_src_col_start[MAX_QMATMUL_BATCH];
    \\    uint slice_dst_offset[MAX_QMATMUL_BATCH];
    \\    uint slice_dst_row_stride[MAX_QMATMUL_BATCH];
    \\    uint slice_dst_col_stride[MAX_QMATMUL_BATCH];
    \\};
    \\
    \\kernel void qmatmul_batch4_f32(
    \\    device const char*  w0 [[buffer(0)]],
    \\    device const float* s0 [[buffer(1)]],
    \\    device const float* i0 [[buffer(2)]],
    \\    device float*       o0 [[buffer(3)]],
    \\    device float*       sd0 [[buffer(4)]],
    \\    device const char*  w1 [[buffer(5)]],
    \\    device const float* s1 [[buffer(6)]],
    \\    device const float* i1 [[buffer(7)]],
    \\    device float*       o1 [[buffer(8)]],
    \\    device float*       sd1 [[buffer(9)]],
    \\    device const char*  w2 [[buffer(10)]],
    \\    device const float* s2 [[buffer(11)]],
    \\    device const float* i2 [[buffer(12)]],
    \\    device float*       o2 [[buffer(13)]],
    \\    device float*       sd2 [[buffer(14)]],
    \\    device const char*  w3 [[buffer(15)]],
    \\    device const float* s3 [[buffer(16)]],
    \\    device const float* i3 [[buffer(17)]],
    \\    device float*       o3 [[buffer(18)]],
    \\    device float*       sd3 [[buffer(19)]],
    \\    constant QMatmulBatch4Params& p [[buffer(20)]],
    \\    uint2 group_id [[threadgroup_position_in_grid]],
    \\    uint  simd_idx [[simdgroup_index_in_threadgroup]],
    \\    uint  lane     [[thread_index_in_simdgroup]],
    \\    uint  tid      [[thread_index_in_threadgroup]]
    \\) {
    \\    uint slot = group_id.y / p.max_tiles_y;
    \\    uint row_tile = group_id.y - slot * p.max_tiles_y;
    \\    if (slot >= p.n_ops) return;
    \\
    \\    device const char* w = w0;
    \\    device const float* s = s0;
    \\    device const float* input = i0;
    \\    device float* output = o0;
    \\    device float* slice_dst = sd0;
    \\    switch (slot) {
    \\        case 1: w = w1; s = s1; input = i1; output = o1; slice_dst = sd1; break;
    \\        case 2: w = w2; s = s2; input = i2; output = o2; slice_dst = sd2; break;
    \\        case 3: w = w3; s = s3; input = i3; output = o3; slice_dst = sd3; break;
    \\        default: break;
    \\    }
    \\
    \\    uint M = p.M[slot], N = p.N[slot], K = p.K[slot];
    \\    const uint gRow = row_tile * TILE;
    \\    const uint gCol = group_id.x * TILE;
    \\    const uint sRow = (simd_idx / 2) * 16;
    \\    const uint sCol = (simd_idx % 2) * 16;
    \\
    \\    simdgroup_float8x8 acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\
    \\    threadgroup float tI[TILE * 8];
    \\    threadgroup float tW[8 * TILE];
    \\
    \\    for (uint kt = 0; kt < K; kt += 8) {
    \\        for (uint i = tid; i < TILE * 8; i += 128) {
    \\            uint r = i / 8, c = i % 8;
    \\            uint ir = gRow + r, ic = kt + c;
    \\            tI[i] = (ir < M && ic < K) ? input[p.input_offset[slot] + ir * p.input_row_stride[slot] + ic] : 0.0f;
    \\        }
    \\        for (uint i = tid; i < 8 * TILE; i += 128) {
    \\            uint r = i / TILE, c = i % TILE;
    \\            uint kr = kt + r, nc = gCol + c;
    \\            if (kr < K && nc < N) {
    \\                uint w_idx = kr * N + nc;
    \\                tW[i] = float(w[w_idx]) * s[w_idx / p.block_size[slot]];
    \\            } else {
    \\                tW[i] = 0.0f;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        simdgroup_float8x8 a0, a1, b0, b1;
    \\        simdgroup_load(a0, tI + (sRow + 0) * 8, 8);
    \\        simdgroup_load(a1, tI + (sRow + 8) * 8, 8);
    \\        simdgroup_load(b0, tW + (sCol + 0), TILE);
    \\        simdgroup_load(b1, tW + (sCol + 8), TILE);
    \\
    \\        simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
    \\        simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
    \\        simdgroup_multiply_accumulate(acc[2], a1, b0, acc[2]);
    \\        simdgroup_multiply_accumulate(acc[3], a1, b1, acc[3]);
    \\
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    threadgroup float tC[TILE * TILE];
    \\    simdgroup_store(acc[0], tC + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[1], tC + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(acc[2], tC + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[3], tC + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, cc = gCol + c;
    \\        if (cr < M && cc < N) {
    \\            float val = tC[i];
    \\            output[p.dst_offset[slot] + cr * p.dst_row_stride[slot] + cc] = val;
    \\            uint slice_col_start = p.slice_src_col_start[slot];
    \\            if (p.has_slice[slot] != 0 && cc >= slice_col_start && cc < slice_col_start + p.slice_rows[slot] && cr < p.slice_cols[slot]) {
    \\                uint slice_row = cc - slice_col_start;
    \\                slice_dst[p.slice_dst_offset[slot] + slice_row * p.slice_dst_row_stride[slot] + cr * p.slice_dst_col_stride[slot]] = val;
    \\            }
    \\        }
    \\    }
    \\}
    \\
    \\// Decode-oriented qmatmul specialization: M==1, one thread per output element.
    \\kernel void qmatvec_f32(
    \\    device const char*  weight_data   [[buffer(0)]],
    \\    device const float* weight_scales [[buffer(1)]],
    \\    device const float* input         [[buffer(2)]],
    \\    device float*       output        [[buffer(3)]],
    \\    constant QMatMulParams& p [[buffer(4)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.N) return;
    \\    float sum = 0.0f;
    \\    for (uint k = 0; k < p.K; k++) {
    \\        uint w_idx = k * p.N + gid;
    \\        sum += input[p.input_offset + k] * float(weight_data[w_idx]) * weight_scales[w_idx / p.block_size];
    \\    }
    \\    output[p.dst_offset + gid] = sum;
    \\}
    \\
    \\constant uint MAX_QMATVEC_BATCH = 4;
    \\
    \\struct QMatvecBatch4Params {
    \\    uint n_ops; uint max_n;
    \\    uint N[MAX_QMATVEC_BATCH];
    \\    uint K[MAX_QMATVEC_BATCH];
    \\    uint block_size[MAX_QMATVEC_BATCH];
    \\    uint input_offset[MAX_QMATVEC_BATCH];
    \\    uint dst_offset[MAX_QMATVEC_BATCH];
    \\};
    \\
    \\kernel void qmatvec_batch4_f32(
    \\    device const char*  w0 [[buffer(0)]],
    \\    device const float* s0 [[buffer(1)]],
    \\    device const float* i0 [[buffer(2)]],
    \\    device float*       o0 [[buffer(3)]],
    \\    device const char*  w1 [[buffer(4)]],
    \\    device const float* s1 [[buffer(5)]],
    \\    device const float* i1 [[buffer(6)]],
    \\    device float*       o1 [[buffer(7)]],
    \\    device const char*  w2 [[buffer(8)]],
    \\    device const float* s2 [[buffer(9)]],
    \\    device const float* i2 [[buffer(10)]],
    \\    device float*       o2 [[buffer(11)]],
    \\    device const char*  w3 [[buffer(12)]],
    \\    device const float* s3 [[buffer(13)]],
    \\    device const float* i3 [[buffer(14)]],
    \\    device float*       o3 [[buffer(15)]],
    \\    constant QMatvecBatch4Params& p [[buffer(16)]],
    \\    uint2 gid [[thread_position_in_grid]]
    \\) {
    \\    uint col = gid.x;
    \\    uint slot = gid.y;
    \\    if (slot >= p.n_ops || col >= p.N[slot]) return;
    \\
    \\    device const char* w = w0;
    \\    device const float* s = s0;
    \\    device const float* input = i0;
    \\    device float* output = o0;
    \\    switch (slot) {
    \\        case 1: w = w1; s = s1; input = i1; output = o1; break;
    \\        case 2: w = w2; s = s2; input = i2; output = o2; break;
    \\        case 3: w = w3; s = s3; input = i3; output = o3; break;
    \\        default: break;
    \\    }
    \\
    \\    float sum = 0.0f;
    \\    for (uint k = 0; k < p.K[slot]; k++) {
    \\        uint w_idx = k * p.N[slot] + col;
    \\        sum += input[p.input_offset[slot] + k] * float(w[w_idx]) * s[w_idx / p.block_size[slot]];
    \\    }
    \\    output[p.dst_offset[slot] + col] = sum;
    \\}
    \\
    \\// ── F16 matvec: M==1, one thread per output element ────
    \\
    \\struct MatVecParams {
    \\    uint N; uint K;
    \\    uint a_offset; uint dst_offset;
    \\};
    \\
    \\kernel void matvec_f16(
    \\    device const float* A  [[buffer(0)]],
    \\    device const half*  B  [[buffer(1)]],
    \\    device float*       C  [[buffer(2)]],
    \\    constant MatVecParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.N) return;
    \\    float sum = 0.0f;
    \\    for (uint k = 0; k < p.K; k++) {
    \\        sum += A[p.a_offset + k] * float(B[k * p.N + gid]);
    \\    }
    \\    C[p.dst_offset + gid] = sum;
    \\}
    \\
    \\// ── F16 tiled matmul: M>1, simdgroup with half precision ─
    \\
    \\struct MatMulF16Params {
    \\    uint M; uint N; uint K;
    \\    uint a_row_stride; uint a_col_stride;
    \\    uint a_offset; uint dst_offset; uint dst_row_stride;
    \\};
    \\
    \\kernel void matmul_f16(
    \\    device const float* A  [[buffer(0)]],
    \\    device const half*  B  [[buffer(1)]],
    \\    device float*       C  [[buffer(2)]],
    \\    constant MatMulF16Params& p [[buffer(3)]],
    \\    uint2 group_id  [[threadgroup_position_in_grid]],
    \\    uint  simd_idx  [[simdgroup_index_in_threadgroup]],
    \\    uint  lane      [[thread_index_in_simdgroup]],
    \\    uint  tid       [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint gRow = group_id.y * TILE;
    \\    const uint gCol = group_id.x * TILE;
    \\    const uint sRow = (simd_idx / 2) * 16;
    \\    const uint sCol = (simd_idx % 2) * 16;
    \\
    \\    simdgroup_float8x8 acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\
    \\    threadgroup half tA[TILE * 8];
    \\    threadgroup half tB[8 * TILE];
    \\
    \\    for (uint kt = 0; kt < p.K; kt += 8) {
    \\        // Load A (f32 global) → half threadgroup
    \\        for (uint i = tid; i < TILE * 8; i += 128) {
    \\            uint r = i / 8, c = i % 8;
    \\            uint ar = gRow + r, ac = kt + c;
    \\            tA[i] = (ar < p.M && ac < p.K)
    \\                ? half(A[p.a_offset + ar * p.a_row_stride + ac * p.a_col_stride]) : half(0);
    \\        }
    \\        // Load B (f16 global, packed K×N) → half threadgroup
    \\        for (uint i = tid; i < 8 * TILE; i += 128) {
    \\            uint r = i / TILE, c = i % TILE;
    \\            uint br = kt + r, bc = gCol + c;
    \\            tB[i] = (br < p.K && bc < p.N)
    \\                ? B[br * p.N + bc] : half(0);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        simdgroup_half8x8 a0, a1, b0, b1;
    \\        simdgroup_load(a0, tA + (sRow + 0) * 8, 8);
    \\        simdgroup_load(a1, tA + (sRow + 8) * 8, 8);
    \\        simdgroup_load(b0, tB + (sCol + 0), TILE);
    \\        simdgroup_load(b1, tB + (sCol + 8), TILE);
    \\
    \\        simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
    \\        simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
    \\        simdgroup_multiply_accumulate(acc[2], a1, b0, acc[2]);
    \\        simdgroup_multiply_accumulate(acc[3], a1, b1, acc[3]);
    \\
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    threadgroup float tC[TILE * TILE];
    \\    simdgroup_store(acc[0], tC + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[1], tC + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(acc[2], tC + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[3], tC + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, cc = gCol + c;
    \\        if (cr < p.M && cc < p.N)
    \\            C[p.dst_offset + cr * p.dst_row_stride + cc] = tC[i];
    \\    }
    \\}
    \\
    \\// ── RoPE: rotary position encoding ─────────────────────
    \\
    \\struct RopeParams {
    \\    uint half_d; uint seq_len;
    \\    uint src_off; uint cs_off; uint dst_off;
    \\    uint src_rs; uint src_cs; uint cs_cs;
    \\};
    \\
    \\kernel void rope_f32(
    \\    device const float* src     [[buffer(0)]],
    \\    device const float* cos_sin [[buffer(1)]],
    \\    device float*       dst     [[buffer(2)]],
    \\    constant RopeParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.half_d * p.seq_len) return;
    \\    uint col = gid / p.half_d;
    \\    uint i   = gid % p.half_d;
    \\    uint d   = p.half_d * 2;
    \\
    \\    float cos_val = cos_sin[p.cs_off + col * p.cs_cs + i];
    \\    float sin_val = cos_sin[p.cs_off + col * p.cs_cs + p.half_d + i];
    \\    float x_lo = src[p.src_off + col * p.src_cs + i * p.src_rs];
    \\    float x_hi = src[p.src_off + col * p.src_cs + (i + p.half_d) * p.src_rs];
    \\
    \\    uint dst_base = p.dst_off + col * d;
    \\    dst[dst_base + i]            = x_lo * cos_val - x_hi * sin_val;
    \\    dst[dst_base + i + p.half_d] = x_hi * cos_val + x_lo * sin_val;
    \\}
    \\
    \\constant uint MAX_ROPE_BATCH = 16;
    \\
    \\struct RopeBatchParams {
    \\    uint n_ops; uint max_n;
    \\    uint half_d[MAX_ROPE_BATCH];
    \\    uint seq_len[MAX_ROPE_BATCH];
    \\    uint src_off[MAX_ROPE_BATCH];
    \\    uint cs_off[MAX_ROPE_BATCH];
    \\    uint dst_off[MAX_ROPE_BATCH];
    \\    uint src_rs[MAX_ROPE_BATCH];
    \\    uint src_cs[MAX_ROPE_BATCH];
    \\    uint cs_cs[MAX_ROPE_BATCH];
    \\};
    \\
    \\kernel void rope_batch_f32(
    \\    device const float* src     [[buffer(0)]],
    \\    device const float* cos_sin [[buffer(1)]],
    \\    device float*       dst     [[buffer(2)]],
    \\    constant RopeBatchParams& p [[buffer(3)]],
    \\    uint2 gid [[thread_position_in_grid]]
    \\) {
    \\    uint local = gid.x;
    \\    uint slot = gid.y;
    \\    if (slot >= p.n_ops || local >= p.half_d[slot] * p.seq_len[slot]) return;
    \\
    \\    uint col = local / p.half_d[slot];
    \\    uint i   = local % p.half_d[slot];
    \\    uint d   = p.half_d[slot] * 2;
    \\
    \\    float cos_val = cos_sin[p.cs_off[slot] + col * p.cs_cs[slot] + i];
    \\    float sin_val = cos_sin[p.cs_off[slot] + col * p.cs_cs[slot] + p.half_d[slot] + i];
    \\    float x_lo = src[p.src_off[slot] + col * p.src_cs[slot] + i * p.src_rs[slot]];
    \\    float x_hi = src[p.src_off[slot] + col * p.src_cs[slot] + (i + p.half_d[slot]) * p.src_rs[slot]];
    \\
    \\    uint dst_base = p.dst_off[slot] + col * d;
    \\    dst[dst_base + i]                  = x_lo * cos_val - x_hi * sin_val;
    \\    dst[dst_base + i + p.half_d[slot]] = x_hi * cos_val + x_lo * sin_val;
    \\}
    \\
    \\// ── RoPE directly into a strided slice destination ─────
    \\
    \\struct RopeSliceAssignParams {
    \\    uint half_d; uint seq_len;
    \\    uint src_off; uint cs_off;
    \\    uint src_rs; uint src_cs; uint cs_cs;
    \\    uint dst_offset; uint dst_row_stride; uint dst_col_stride;
    \\};
    \\
    \\kernel void rope_slice_assign_f32(
    \\    device const float* src     [[buffer(0)]],
    \\    device const float* cos_sin [[buffer(1)]],
    \\    device float*       dst     [[buffer(2)]],
    \\    constant RopeSliceAssignParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.half_d * p.seq_len) return;
    \\    uint col = gid / p.half_d;
    \\    uint i   = gid % p.half_d;
    \\
    \\    float cos_val = cos_sin[p.cs_off + col * p.cs_cs + i];
    \\    float sin_val = cos_sin[p.cs_off + col * p.cs_cs + p.half_d + i];
    \\    float x_lo = src[p.src_off + col * p.src_cs + i * p.src_rs];
    \\    float x_hi = src[p.src_off + col * p.src_cs + (i + p.half_d) * p.src_rs];
    \\    float y_lo = x_lo * cos_val - x_hi * sin_val;
    \\    float y_hi = x_hi * cos_val + x_lo * sin_val;
    \\
    \\    uint dst_col = p.dst_offset + col * p.dst_col_stride;
    \\    dst[dst_col + i * p.dst_row_stride] = y_lo;
    \\    dst[dst_col + (i + p.half_d) * p.dst_row_stride] = y_hi;
    \\}
    \\
    \\// ── QMatmul with an attached strided slice store ────────
    \\
    \\struct QMatmulSliceAssignParams {
    \\    uint M; uint N; uint K;
    \\    uint block_size;
    \\    uint input_offset;
    \\    uint input_row_stride;
    \\    uint dst_offset;
    \\    uint dst_row_stride;
    \\    uint slice_rows; uint slice_cols;
    \\    uint slice_src_col_start;
    \\    uint slice_dst_offset;
    \\    uint slice_dst_row_stride;
    \\    uint slice_dst_col_stride;
    \\};
    \\
    \\kernel void qmatmul_slice_assign_f32(
    \\    device const char*  weight_data   [[buffer(0)]],
    \\    device const float* weight_scales [[buffer(1)]],
    \\    device const float* input         [[buffer(2)]],
    \\    device float*       output        [[buffer(3)]],
    \\    device float*       slice_dst     [[buffer(4)]],
    \\    constant QMatmulSliceAssignParams& p [[buffer(5)]],
    \\    uint2 group_id  [[threadgroup_position_in_grid]],
    \\    uint  simd_idx  [[simdgroup_index_in_threadgroup]],
    \\    uint  lane      [[thread_index_in_simdgroup]],
    \\    uint  tid       [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint gRow = group_id.y * TILE;
    \\    const uint gCol = group_id.x * TILE;
    \\    const uint sRow = (simd_idx / 2) * 16;
    \\    const uint sCol = (simd_idx % 2) * 16;
    \\
    \\    simdgroup_float8x8 acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\
    \\    threadgroup float tI[TILE * 8];
    \\    threadgroup float tW[8 * TILE];
    \\
    \\    for (uint kt = 0; kt < p.K; kt += 8) {
    \\        for (uint i = tid; i < TILE * 8; i += 128) {
    \\            uint r = i / 8, c = i % 8;
    \\            uint ir = gRow + r, ic = kt + c;
    \\            tI[i] = (ir < p.M && ic < p.K) ? input[p.input_offset + ir * p.input_row_stride + ic] : 0.0f;
    \\        }
    \\        for (uint i = tid; i < 8 * TILE; i += 128) {
    \\            uint r = i / TILE, c = i % TILE;
    \\            uint kr = kt + r, nc = gCol + c;
    \\            if (kr < p.K && nc < p.N) {
    \\                uint w_idx = kr * p.N + nc;
    \\                tW[i] = float(weight_data[w_idx]) * weight_scales[w_idx / p.block_size];
    \\            } else {
    \\                tW[i] = 0.0f;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        simdgroup_float8x8 a0, a1, b0, b1;
    \\        simdgroup_load(a0, tI + (sRow + 0) * 8, 8);
    \\        simdgroup_load(a1, tI + (sRow + 8) * 8, 8);
    \\        simdgroup_load(b0, tW + (sCol + 0), TILE);
    \\        simdgroup_load(b1, tW + (sCol + 8), TILE);
    \\
    \\        simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
    \\        simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
    \\        simdgroup_multiply_accumulate(acc[2], a1, b0, acc[2]);
    \\        simdgroup_multiply_accumulate(acc[3], a1, b1, acc[3]);
    \\
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    threadgroup float tC[TILE * TILE];
    \\    simdgroup_store(acc[0], tC + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[1], tC + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(acc[2], tC + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[3], tC + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, cc = gCol + c;
    \\        if (cr < p.M && cc < p.N) {
    \\            float val = tC[i];
    \\            output[p.dst_offset + cr * p.dst_row_stride + cc] = val;
    \\            if (cc >= p.slice_src_col_start && cc < p.slice_src_col_start + p.slice_rows && cr < p.slice_cols) {
    \\                uint slice_row = cc - p.slice_src_col_start;
    \\                slice_dst[p.slice_dst_offset + slice_row * p.slice_dst_row_stride + cr * p.slice_dst_col_stride] = val;
    \\            }
    \\        }
    \\    }
    \\}
    \\
    \\struct QMatmulElementwiseParams {
    \\    uint M; uint N; uint K;
    \\    uint block_size;
    \\    uint input_offset;
    \\    uint input_row_stride;
    \\    uint dst_offset;
    \\    uint dst_row_stride;
    \\    uint ew_op;
    \\    uint ew_is_swapped;
    \\    uint ew_dst_offset;
    \\    uint ew_secondary_offset;
    \\};
    \\
    \\kernel void qmatmul_elementwise_f32(
    \\    device const char*  weight_data   [[buffer(0)]],
    \\    device const float* weight_scales [[buffer(1)]],
    \\    device const float* input         [[buffer(2)]],
    \\    device float*       output        [[buffer(3)]],
    \\    device const float* secondary     [[buffer(4)]],
    \\    device float*       ew_output     [[buffer(5)]],
    \\    constant QMatmulElementwiseParams& p [[buffer(6)]],
    \\    uint2 group_id  [[threadgroup_position_in_grid]],
    \\    uint  simd_idx  [[simdgroup_index_in_threadgroup]],
    \\    uint  lane      [[thread_index_in_simdgroup]],
    \\    uint  tid       [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint gRow = group_id.y * TILE;
    \\    const uint gCol = group_id.x * TILE;
    \\    const uint sRow = (simd_idx / 2) * 16;
    \\    const uint sCol = (simd_idx % 2) * 16;
    \\
    \\    simdgroup_float8x8 acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\
    \\    threadgroup float tI[TILE * 8];
    \\    threadgroup float tW[8 * TILE];
    \\
    \\    for (uint kt = 0; kt < p.K; kt += 8) {
    \\        for (uint i = tid; i < TILE * 8; i += 128) {
    \\            uint r = i / 8, c = i % 8;
    \\            uint ir = gRow + r, ic = kt + c;
    \\            tI[i] = (ir < p.M && ic < p.K) ? input[p.input_offset + ir * p.input_row_stride + ic] : 0.0f;
    \\        }
    \\        for (uint i = tid; i < 8 * TILE; i += 128) {
    \\            uint r = i / TILE, c = i % TILE;
    \\            uint kr = kt + r, nc = gCol + c;
    \\            if (kr < p.K && nc < p.N) {
    \\                uint w_idx = kr * p.N + nc;
    \\                tW[i] = float(weight_data[w_idx]) * weight_scales[w_idx / p.block_size];
    \\            } else {
    \\                tW[i] = 0.0f;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        simdgroup_float8x8 a0, a1, b0, b1;
    \\        simdgroup_load(a0, tI + (sRow + 0) * 8, 8);
    \\        simdgroup_load(a1, tI + (sRow + 8) * 8, 8);
    \\        simdgroup_load(b0, tW + (sCol + 0), TILE);
    \\        simdgroup_load(b1, tW + (sCol + 8), TILE);
    \\
    \\        simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
    \\        simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
    \\        simdgroup_multiply_accumulate(acc[2], a1, b0, acc[2]);
    \\        simdgroup_multiply_accumulate(acc[3], a1, b1, acc[3]);
    \\
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    threadgroup float tC[TILE * TILE];
    \\    simdgroup_store(acc[0], tC + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[1], tC + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(acc[2], tC + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[3], tC + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, cc = gCol + c;
    \\        if (cr < p.M && cc < p.N) {
    \\            float val = tC[i];
    \\            uint linear = cr * p.N + cc;
    \\            float other = secondary[p.ew_secondary_offset + linear];
    \\            float ew = val;
    \\            if (p.ew_op == 7) ew = (p.ew_is_swapped != 0) ? other + val : val + other;
    \\            else if (p.ew_op == 8) ew = (p.ew_is_swapped != 0) ? other * val : val * other;
    \\            output[p.dst_offset + cr * p.dst_row_stride + cc] = val;
    \\            ew_output[p.ew_dst_offset + linear] = ew;
    \\        }
    \\    }
    \\}
    \\
    \\kernel void qmatvec_slice_assign_f32(
    \\    device const char*  weight_data   [[buffer(0)]],
    \\    device const float* weight_scales [[buffer(1)]],
    \\    device const float* input         [[buffer(2)]],
    \\    device float*       output        [[buffer(3)]],
    \\    device float*       slice_dst     [[buffer(4)]],
    \\    constant QMatmulSliceAssignParams& p [[buffer(5)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.N) return;
    \\    float sum = 0.0f;
    \\    for (uint k = 0; k < p.K; k++) {
    \\        uint w_idx = k * p.N + gid;
    \\        sum += input[p.input_offset + k] * float(weight_data[w_idx]) * weight_scales[w_idx / p.block_size];
    \\    }
    \\
    \\    output[p.dst_offset + gid] = sum;
    \\    if (gid >= p.slice_src_col_start && gid < p.slice_src_col_start + p.slice_rows) {
    \\        uint row = gid - p.slice_src_col_start;
    \\        uint col = 0;
    \\        slice_dst[p.slice_dst_offset + row * p.slice_dst_row_stride + col * p.slice_dst_col_stride] = sum;
    \\    }
    \\}
    \\
    \\constant uint MAX_SLICE_ASSIGN_BATCH = 16;
    \\
    \\struct SliceAssignBatchParams {
    \\    uint n_ops; uint max_n;
    \\    uint rows[MAX_SLICE_ASSIGN_BATCH];
    \\    uint cols[MAX_SLICE_ASSIGN_BATCH];
    \\    uint dst_offset[MAX_SLICE_ASSIGN_BATCH];
    \\    uint dst_row_stride[MAX_SLICE_ASSIGN_BATCH];
    \\    uint dst_col_stride[MAX_SLICE_ASSIGN_BATCH];
    \\    uint src_offset[MAX_SLICE_ASSIGN_BATCH];
    \\    uint src_row_stride[MAX_SLICE_ASSIGN_BATCH];
    \\    uint src_col_stride[MAX_SLICE_ASSIGN_BATCH];
    \\};
    \\
    \\kernel void slice_assign_batch_f32(
    \\    device const float* src [[buffer(0)]],
    \\    device float* dst [[buffer(1)]],
    \\    constant SliceAssignBatchParams& p [[buffer(2)]],
    \\    uint2 gid [[thread_position_in_grid]]
    \\) {
    \\    uint local = gid.x;
    \\    uint slot = gid.y;
    \\    if (slot >= p.n_ops || local >= p.rows[slot] * p.cols[slot]) return;
    \\    uint row = local % p.rows[slot];
    \\    uint col = local / p.rows[slot];
    \\    dst[p.dst_offset[slot] + row * p.dst_row_stride[slot] + col * p.dst_col_stride[slot]] =
    \\        src[p.src_offset[slot] + row * p.src_row_stride[slot] + col * p.src_col_stride[slot]];
    \\}
    \\
    \\// ── RMSNorm with repeated scale and final multiply ──────
    \\
    \\struct RmsNormScaleParams {
    \\    uint rows; uint cols;
    \\    float eps;
    \\    uint src_offset;
    \\    uint norm_dst_offset;
    \\    uint scale_src_offset;
    \\    uint scale_repeat_dst_offset;
    \\    uint scaled_dst_offset;
    \\};
    \\
    \\kernel void rmsnorm_scale_f32(
    \\    device const float* src [[buffer(0)]],
    \\    device const float* scale_src [[buffer(1)]],
    \\    device float* norm_dst [[buffer(2)]],
    \\    device float* scale_repeat_dst [[buffer(3)]],
    \\    device float* scaled_dst [[buffer(4)]],
    \\    constant RmsNormScaleParams& p [[buffer(5)]],
    \\    uint row [[thread_position_in_grid]]
    \\) {
    \\    if (row >= p.rows) return;
    \\    uint src_base = p.src_offset + row * p.cols;
    \\    uint norm_base = p.norm_dst_offset + row * p.cols;
    \\    uint repeat_base = p.scale_repeat_dst_offset + row * p.cols;
    \\    uint scaled_base = p.scaled_dst_offset + row * p.cols;
    \\
    \\    float ss = 0.0f;
    \\    for (uint col = 0; col < p.cols; col++) {
    \\        float v = src[src_base + col];
    \\        ss += v * v;
    \\    }
    \\    float inv_rms = 1.0f / sqrt(ss / float(p.cols) + p.eps);
    \\    for (uint col = 0; col < p.cols; col++) {
    \\        float norm = src[src_base + col] * inv_rms;
    \\        float scale = scale_src[p.scale_src_offset + col];
    \\        norm_dst[norm_base + col] = norm;
    \\        scale_repeat_dst[repeat_base + col] = scale;
    \\        scaled_dst[scaled_base + col] = norm * scale;
    \\    }
    \\}
    \\
    \\// ── Attention: fused softmax(Q@K^T * scale + mask) @ V ─
    \\
    \\struct AttentionParams {
    \\    uint d_head; uint seq_q; uint seq_kv;
    \\    float scale;
    \\    uint q_off; uint k_off; uint v_off; uint mask_off; uint dst_off;
    \\    uint q_rs; uint q_cs; uint k_rs; uint k_cs; uint v_rs; uint v_cs;
    \\    uint mask_rs; uint mask_cs; uint dst_rs; uint dst_cs;
    \\};
    \\
    \\constant uint MAX_SEQ = 4096;
    \\constant uint MAX_ATTENTION_BATCH_HEADS = 16;
    \\
    \\// One threadgroup per query position. Decode is the seq_q=1 case.
    \\kernel void attention_f32(
    \\    device const float* Q    [[buffer(0)]],
    \\    device const float* K    [[buffer(1)]],
    \\    device const float* V    [[buffer(2)]],
    \\    device const float* mask [[buffer(3)]],
    \\    device float*       dst  [[buffer(4)]],
    \\    constant AttentionParams& p [[buffer(5)]],
    \\    uint q_col   [[threadgroup_position_in_grid]],
    \\    uint tid     [[thread_index_in_threadgroup]]
    \\) {
    \\    if (q_col >= p.seq_q) return;
    \\    const uint tg_size = 256;
    \\    threadgroup float scores[MAX_SEQ];
    \\    threadgroup float scratch[256];
    \\
    \\    // Phase 1: Compute Q·K scores (threads share the work across KV positions).
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size) {
    \\        float mv = mask[p.mask_off + s * p.mask_rs + q_col * p.mask_cs];
    \\        if (!isfinite(mv)) { scores[s] = -INFINITY; continue; }
    \\        float dot = 0.0f;
    \\        for (uint r = 0; r < p.d_head; r++)
    \\            dot += Q[p.q_off + r * p.q_rs + q_col * p.q_cs] * K[p.k_off + r * p.k_rs + s * p.k_cs];
    \\        scores[s] = dot * p.scale + mv;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Phase 2: Max reduction.
    \\    float local_max = -INFINITY;
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size)
    \\        local_max = max(local_max, scores[s]);
    \\    scratch[tid] = local_max;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
    \\        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float global_max = scratch[0];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Phase 3: Softmax (exp + sum reduction + normalize).
    \\    float local_sum = 0.0f;
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size) {
    \\        float w = (scores[s] == -INFINITY) ? 0.0f : exp(scores[s] - global_max);
    \\        scores[s] = w;
    \\        local_sum += w;
    \\    }
    \\    scratch[tid] = local_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
    \\        if (tid < stride) scratch[tid] += scratch[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float inv_sum = (scratch[0] > 0.0f) ? 1.0f / scratch[0] : 0.0f;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size)
    \\        scores[s] *= inv_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Phase 4: Weighted sum of V columns → output[d_head].
    \\    for (uint r = tid; r < p.d_head; r += tg_size) {
    \\        float val = 0.0f;
    \\        for (uint s = 0; s < p.seq_kv; s++)
    \\            val += scores[s] * V[p.v_off + r * p.v_rs + s * p.v_cs];
    \\        dst[p.dst_off + r * p.dst_rs + q_col * p.dst_cs] = val;
    \\    }
    \\}
    \\
    \\struct AttentionBatchParams {
    \\    uint n_heads; uint d_head; uint seq_q; uint seq_kv;
    \\    float scale;
    \\    uint q_off[MAX_ATTENTION_BATCH_HEADS];
    \\    uint k_off[MAX_ATTENTION_BATCH_HEADS];
    \\    uint v_off[MAX_ATTENTION_BATCH_HEADS];
    \\    uint mask_off[MAX_ATTENTION_BATCH_HEADS];
    \\    uint dst_off[MAX_ATTENTION_BATCH_HEADS];
    \\    uint q_rs; uint q_cs; uint k_rs; uint k_cs; uint v_rs; uint v_cs;
    \\    uint mask_rs; uint mask_cs; uint dst_rs; uint dst_cs;
    \\};
    \\
    \\kernel void attention_batch_f32(
    \\    device const float* Q    [[buffer(0)]],
    \\    device const float* K    [[buffer(1)]],
    \\    device const float* V    [[buffer(2)]],
    \\    device const float* mask [[buffer(3)]],
    \\    device float*       dst  [[buffer(4)]],
    \\    constant AttentionBatchParams& p [[buffer(5)]],
    \\    uint2 group [[threadgroup_position_in_grid]],
    \\    uint tid     [[thread_index_in_threadgroup]]
    \\) {
    \\    uint q_col = group.x;
    \\    uint head = group.y;
    \\    if (q_col >= p.seq_q) return;
    \\    if (head >= p.n_heads) return;
    \\    const uint tg_size = 256;
    \\
    \\    threadgroup float scores[MAX_SEQ];
    \\    threadgroup float scratch[256];
    \\
    \\    uint q_off = p.q_off[head];
    \\    uint k_off = p.k_off[head];
    \\    uint v_off = p.v_off[head];
    \\    uint mask_off = p.mask_off[head];
    \\    uint dst_off = p.dst_off[head];
    \\
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size) {
    \\        float mv = mask[mask_off + s * p.mask_rs + q_col * p.mask_cs];
    \\        if (!isfinite(mv)) { scores[s] = -INFINITY; continue; }
    \\        float dot = 0.0f;
    \\        for (uint r = 0; r < p.d_head; r++)
    \\            dot += Q[q_off + r * p.q_rs + q_col * p.q_cs] * K[k_off + r * p.k_rs + s * p.k_cs];
    \\        scores[s] = dot * p.scale + mv;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    float local_max = -INFINITY;
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size)
    \\        local_max = max(local_max, scores[s]);
    \\    scratch[tid] = local_max;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
    \\        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float global_max = scratch[0];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    float local_sum = 0.0f;
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size) {
    \\        float w = (scores[s] == -INFINITY) ? 0.0f : exp(scores[s] - global_max);
    \\        scores[s] = w;
    \\        local_sum += w;
    \\    }
    \\    scratch[tid] = local_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
    \\        if (tid < stride) scratch[tid] += scratch[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    float inv_sum = (scratch[0] > 0.0f) ? 1.0f / scratch[0] : 0.0f;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size)
    \\        scores[s] *= inv_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint r = tid; r < p.d_head; r += tg_size) {
    \\        float val = 0.0f;
    \\        for (uint s = 0; s < p.seq_kv; s++)
    \\            val += scores[s] * V[v_off + r * p.v_rs + s * p.v_cs];
    \\        dst[dst_off + r * p.dst_rs + q_col * p.dst_cs] = val;
    \\    }
    \\}
    \\
    \\// ── Unified compute kernel ──────────────────────────────
    \\
    \\struct ComputeParams {
    \\    uint op;
    \\    uint n_elements;
    \\    uint dst_ne[4];    uint dst_strides[4];   uint dst_offset;
    \\    uint src0_ne[4];   uint src0_strides[4];   uint src0_offset;
    \\    uint src1_ne[4];   uint src1_strides[4];   uint src1_offset;
    \\};
    \\
    \\// Op codes — elementwise match op.zig values directly:
    \\//   add=7 mul=8 neg=9 abs=10 sgn=11 step=12 relu=13
    \\//   sqrt=14 recip=15 exp=16 log=17 gelu=18
    \\//   sum=19 max=20 repeat=21 slice_assign=27
    \\// Fused ops use codes 100+:
    \\//   fused_softmax=100 fused_layernorm=101 fused_rmsnorm=102
    \\
    \\kernel void compute_f32(
    \\    device const float* src0 [[buffer(0)]],
    \\    device const float* src1 [[buffer(1)]],
    \\    device float* dst        [[buffer(2)]],
    \\    constant ComputeParams& p [[buffer(3)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.n_elements) return;
    \\    switch (p.op) {
    \\        // ── Elementwise (one thread per element) ──
    \\        case 7:  dst[p.dst_offset + gid] = src0[p.src0_offset + gid] + src1[p.src1_offset + gid]; break;
    \\        case 8:  dst[p.dst_offset + gid] = src0[p.src0_offset + gid] * src1[p.src1_offset + gid]; break;
    \\        case 9:  dst[p.dst_offset + gid] = -src0[p.src0_offset + gid]; break;
    \\        case 10: dst[p.dst_offset + gid] = abs(src0[p.src0_offset + gid]); break;
    \\        case 11: dst[p.dst_offset + gid] = sign(src0[p.src0_offset + gid]); break;
    \\        case 12: dst[p.dst_offset + gid] = (src0[p.src0_offset + gid] > 0.0f) ? 1.0f : 0.0f; break;
    \\        case 13: dst[p.dst_offset + gid] = max(src0[p.src0_offset + gid], 0.0f); break;
    \\        case 14: dst[p.dst_offset + gid] = sqrt(src0[p.src0_offset + gid]); break;
    \\        case 15: dst[p.dst_offset + gid] = 1.0f / src0[p.src0_offset + gid]; break;
    \\        case 16: dst[p.dst_offset + gid] = exp(src0[p.src0_offset + gid]); break;
    \\        case 17: dst[p.dst_offset + gid] = log(src0[p.src0_offset + gid]); break;
    \\        case 18: {
    \\            float a = src0[p.src0_offset + gid];
    \\            float c = 0.7978845608f * (a + 0.044715f * a * a * a);
    \\            dst[p.dst_offset + gid] = 0.5f * a * (1.0f + precise::tanh(c));
    \\            break;
    \\        }
    \\        // ── Reduce: sum(19) or max(20), one thread per output ──
    \\        case 19: case 20: {
    \\            uint reduce_size = p.src0_ne[0];
    \\            uint src_base = p.src0_offset + gid * reduce_size;
    \\            float val = (p.op == 20) ? -INFINITY : 0.0f;
    \\            for (uint k = 0; k < reduce_size; k++) {
    \\                float v = src0[src_base + k];
    \\                if (p.op == 19) val += v; else val = max(val, v);
    \\            }
    \\            dst[p.dst_offset + gid] = val;
    \\            break;
    \\        }
    \\        // ── Repeat: broadcast via modular indexing ──
    \\        case 21: {
    \\            uint idx = gid;
    \\            uint src_idx = p.src0_offset;
    \\            for (int d = 3; d >= 0; d--) {
    \\                uint ud = uint(d);
    \\                uint coord = idx / p.dst_strides[ud];
    \\                idx %= p.dst_strides[ud];
    \\                src_idx += (coord % p.src0_ne[ud]) * p.src0_strides[ud];
    \\            }
    \\            dst[p.dst_offset + gid] = src0[src_idx];
    \\            break;
    \\        }
    \\        // ── Slice assign: strided copy ──
    \\        case 27: {
    \\            uint row = gid % p.src0_ne[0];
    \\            uint col = gid / p.src0_ne[0];
    \\            dst[p.dst_offset + row * p.dst_strides[0] + col * p.dst_strides[1]] =
    \\                src0[p.src0_offset + row * p.src0_strides[0] + col * p.src0_strides[1]];
    \\            break;
    \\        }
    \\        // ── Fused softmax: one thread per row ──
    \\        // n_elements = rows, src0_ne[0] = cols
    \\        case 100: {
    \\            uint cols = p.src0_ne[0];
    \\            uint src_base = p.src0_offset + gid * cols;
    \\            uint dst_base = p.dst_offset + gid * cols;
    \\            float m = -INFINITY;
    \\            for (uint j = 0; j < cols; j++) m = max(m, src0[src_base + j]);
    \\            float s = 0.0f;
    \\            for (uint j = 0; j < cols; j++) {
    \\                float e = exp(src0[src_base + j] - m);
    \\                dst[dst_base + j] = e;
    \\                s += e;
    \\            }
    \\            float inv = 1.0f / s;
    \\            for (uint j = 0; j < cols; j++) dst[dst_base + j] *= inv;
    \\            break;
    \\        }
    \\        // ── Fused layer norm: one thread per row ──
    \\        // n_elements = rows, src0_ne[0] = cols
    \\        case 101: {
    \\            uint cols = p.src0_ne[0];
    \\            uint base = p.src0_offset + gid * cols;
    \\            uint dbase = p.dst_offset + gid * cols;
    \\            float eps = as_type<float>(p.src1_ne[0]);
    \\            float mu = 0.0f;
    \\            for (uint j = 0; j < cols; j++) mu += src0[base + j];
    \\            mu /= float(cols);
    \\            float v = 0.0f;
    \\            for (uint j = 0; j < cols; j++) {
    \\                float d = src0[base + j] - mu;
    \\                v += d * d;
    \\            }
    \\            float inv_std = 1.0f / sqrt(v / float(cols) + eps);
    \\            for (uint j = 0; j < cols; j++)
    \\                dst[dbase + j] = (src0[base + j] - mu) * inv_std;
    \\            break;
    \\        }
    \\        // ── Fused RMS norm: one thread per row ──
    \\        // n_elements = rows, src0_ne[0] = cols
    \\        case 102: {
    \\            uint cols = p.src0_ne[0];
    \\            uint base = p.src0_offset + gid * cols;
    \\            uint dbase = p.dst_offset + gid * cols;
    \\            float eps = as_type<float>(p.src1_ne[0]);
    \\            float ss = 0.0f;
    \\            for (uint j = 0; j < cols; j++) ss += src0[base + j] * src0[base + j];
    \\            float inv_rms = 1.0f / sqrt(ss / float(cols) + eps);
    \\            for (uint j = 0; j < cols; j++)
    \\                dst[dbase + j] = src0[base + j] * inv_rms;
    \\            break;
    \\        }
    \\        default: break;
    \\    }
    \\}
    \\
    \\constant uint MAX_FUSED_EW_STEPS = 8;
    \\constant uint MAX_FUSED_EW_SECONDARIES = 8;
    \\
    \\struct FusedEwParams {
    \\    uint n_elements;
    \\    uint n_steps;
    \\    uint dst_offset;
    \\    uint src_offset;
    \\    uint op[MAX_FUSED_EW_STEPS];
    \\    uint is_swapped[MAX_FUSED_EW_STEPS];
    \\    uint secondary_slot[MAX_FUSED_EW_STEPS];
    \\    uint secondary_offset[MAX_FUSED_EW_STEPS];
    \\};
    \\
    \\float fused_secondary(
    \\    uint slot,
    \\    uint idx,
    \\    device const float* s0,
    \\    device const float* s1,
    \\    device const float* s2,
    \\    device const float* s3,
    \\    device const float* s4,
    \\    device const float* s5,
    \\    device const float* s6,
    \\    device const float* s7
    \\) {
    \\    switch (slot) {
    \\        case 0: return s0[idx];
    \\        case 1: return s1[idx];
    \\        case 2: return s2[idx];
    \\        case 3: return s3[idx];
    \\        case 4: return s4[idx];
    \\        case 5: return s5[idx];
    \\        case 6: return s6[idx];
    \\        default: return s7[idx];
    \\    }
    \\}
    \\
    \\float fused_unary(uint op, float v) {
    \\    switch (op) {
    \\        case 9: return -v;
    \\        case 10: return abs(v);
    \\        case 11: return sign(v);
    \\        case 12: return (v > 0.0f) ? 1.0f : 0.0f;
    \\        case 13: return max(v, 0.0f);
    \\        case 14: return sqrt(v);
    \\        case 15: return 1.0f / v;
    \\        case 16: return exp(v);
    \\        case 17: return log(v);
    \\        case 18: {
    \\            float c = 0.7978845608f * (v + 0.044715f * v * v * v);
    \\            return 0.5f * v * (1.0f + precise::tanh(c));
    \\        }
    \\        default: return v;
    \\    }
    \\}
    \\
    \\float apply_elementwise(uint op, float a, float b) {
    \\    switch (op) {
    \\        case 7: return a + b;
    \\        case 8: return a * b;
    \\        default: return fused_unary(op, a);
    \\    }
    \\}
    \\
    \\constant uint MAX_ELEMENTWISE_BATCH = 8;
    \\
    \\struct ElementwiseBatchParams {
    \\    uint n_ops;
    \\    uint max_n;
    \\    uint op[MAX_ELEMENTWISE_BATCH];
    \\    uint n_elements[MAX_ELEMENTWISE_BATCH];
    \\    uint dst_offset[MAX_ELEMENTWISE_BATCH];
    \\    uint src0_offset[MAX_ELEMENTWISE_BATCH];
    \\    uint src1_offset[MAX_ELEMENTWISE_BATCH];
    \\};
    \\
    \\kernel void elementwise_batch8_f32(
    \\    device const float* src0_0 [[buffer(0)]],
    \\    device const float* src1_0 [[buffer(1)]],
    \\    device float* dst_0        [[buffer(2)]],
    \\    device const float* src0_1 [[buffer(3)]],
    \\    device const float* src1_1 [[buffer(4)]],
    \\    device float* dst_1        [[buffer(5)]],
    \\    device const float* src0_2 [[buffer(6)]],
    \\    device const float* src1_2 [[buffer(7)]],
    \\    device float* dst_2        [[buffer(8)]],
    \\    device const float* src0_3 [[buffer(9)]],
    \\    device const float* src1_3 [[buffer(10)]],
    \\    device float* dst_3        [[buffer(11)]],
    \\    device const float* src0_4 [[buffer(12)]],
    \\    device const float* src1_4 [[buffer(13)]],
    \\    device float* dst_4        [[buffer(14)]],
    \\    device const float* src0_5 [[buffer(15)]],
    \\    device const float* src1_5 [[buffer(16)]],
    \\    device float* dst_5        [[buffer(17)]],
    \\    device const float* src0_6 [[buffer(18)]],
    \\    device const float* src1_6 [[buffer(19)]],
    \\    device float* dst_6        [[buffer(20)]],
    \\    device const float* src0_7 [[buffer(21)]],
    \\    device const float* src1_7 [[buffer(22)]],
    \\    device float* dst_7        [[buffer(23)]],
    \\    constant ElementwiseBatchParams& p [[buffer(24)]],
    \\    uint2 gid [[thread_position_in_grid]]
    \\) {
    \\    uint slot = gid.y;
    \\    uint i = gid.x;
    \\    if (slot >= p.n_ops || i >= p.n_elements[slot]) return;
    \\
    \\    device const float* src0 = src0_0;
    \\    device const float* src1 = src1_0;
    \\    device float* dst = dst_0;
    \\    switch (slot) {
    \\        case 1: src0 = src0_1; src1 = src1_1; dst = dst_1; break;
    \\        case 2: src0 = src0_2; src1 = src1_2; dst = dst_2; break;
    \\        case 3: src0 = src0_3; src1 = src1_3; dst = dst_3; break;
    \\        case 4: src0 = src0_4; src1 = src1_4; dst = dst_4; break;
    \\        case 5: src0 = src0_5; src1 = src1_5; dst = dst_5; break;
    \\        case 6: src0 = src0_6; src1 = src1_6; dst = dst_6; break;
    \\        case 7: src0 = src0_7; src1 = src1_7; dst = dst_7; break;
    \\        default: break;
    \\    }
    \\
    \\    float a = src0[p.src0_offset[slot] + i];
    \\    float b = src1[p.src1_offset[slot] + i];
    \\    dst[p.dst_offset[slot] + i] = apply_elementwise(p.op[slot], a, b);
    \\}
    \\
    \\kernel void fused_elementwise_f32(
    \\    device const float* src [[buffer(0)]],
    \\    device float* dst [[buffer(1)]],
    \\    device const float* s0 [[buffer(2)]],
    \\    device const float* s1 [[buffer(3)]],
    \\    device const float* s2 [[buffer(4)]],
    \\    device const float* s3 [[buffer(5)]],
    \\    device const float* s4 [[buffer(6)]],
    \\    device const float* s5 [[buffer(7)]],
    \\    device const float* s6 [[buffer(8)]],
    \\    device const float* s7 [[buffer(9)]],
    \\    constant FusedEwParams& p [[buffer(10)]],
    \\    uint gid [[thread_position_in_grid]]
    \\) {
    \\    if (gid >= p.n_elements) return;
    \\    float v = src[p.src_offset + gid];
    \\    for (uint step = 0; step < p.n_steps; step++) {
    \\        uint op = p.op[step];
    \\        if (op == 7 || op == 8) {
    \\            uint sec_idx = p.secondary_offset[step] + gid;
    \\            float other = fused_secondary(p.secondary_slot[step], sec_idx, s0, s1, s2, s3, s4, s5, s6, s7);
    \\            if (op == 7) v = (p.is_swapped[step] != 0) ? other + v : v + other;
    \\            else v = (p.is_swapped[step] != 0) ? other * v : v * other;
    \\        } else {
    \\            v = fused_unary(op, v);
    \\        }
    \\    }
    \\    dst[p.dst_offset + gid] = v;
    \\}
    \\
    \\struct QMatmulFusedEwParams {
    \\    uint M; uint N; uint K;
    \\    uint block_size;
    \\    uint input_offset;
    \\    uint input_row_stride;
    \\    uint dst_offset;
    \\    uint dst_row_stride;
    \\    uint n_steps;
    \\    uint ew_dst_offset;
    \\    uint op[MAX_FUSED_EW_STEPS];
    \\    uint is_swapped[MAX_FUSED_EW_STEPS];
    \\    uint secondary_slot[MAX_FUSED_EW_STEPS];
    \\    uint secondary_offset[MAX_FUSED_EW_STEPS];
    \\};
    \\
    \\kernel void qmatmul_fused_elementwise_f32(
    \\    device const char*  weight_data   [[buffer(0)]],
    \\    device const float* weight_scales [[buffer(1)]],
    \\    device const float* input         [[buffer(2)]],
    \\    device float*       output        [[buffer(3)]],
    \\    device float*       ew_output     [[buffer(4)]],
    \\    device const float* s0 [[buffer(5)]],
    \\    device const float* s1 [[buffer(6)]],
    \\    device const float* s2 [[buffer(7)]],
    \\    device const float* s3 [[buffer(8)]],
    \\    device const float* s4 [[buffer(9)]],
    \\    device const float* s5 [[buffer(10)]],
    \\    device const float* s6 [[buffer(11)]],
    \\    device const float* s7 [[buffer(12)]],
    \\    constant QMatmulFusedEwParams& p [[buffer(13)]],
    \\    uint2 group_id  [[threadgroup_position_in_grid]],
    \\    uint  simd_idx  [[simdgroup_index_in_threadgroup]],
    \\    uint  lane      [[thread_index_in_simdgroup]],
    \\    uint  tid       [[thread_index_in_threadgroup]]
    \\) {
    \\    const uint gRow = group_id.y * TILE;
    \\    const uint gCol = group_id.x * TILE;
    \\    const uint sRow = (simd_idx / 2) * 16;
    \\    const uint sCol = (simd_idx % 2) * 16;
    \\
    \\    simdgroup_float8x8 acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\    threadgroup float tI[TILE * 8];
    \\    threadgroup float tW[8 * TILE];
    \\
    \\    for (uint kt = 0; kt < p.K; kt += 8) {
    \\        for (uint i = tid; i < TILE * 8; i += 128) {
    \\            uint r = i / 8, c = i % 8;
    \\            uint ir = gRow + r, ic = kt + c;
    \\            tI[i] = (ir < p.M && ic < p.K) ? input[p.input_offset + ir * p.input_row_stride + ic] : 0.0f;
    \\        }
    \\        for (uint i = tid; i < 8 * TILE; i += 128) {
    \\            uint r = i / TILE, c = i % TILE;
    \\            uint kr = kt + r, nc = gCol + c;
    \\            if (kr < p.K && nc < p.N) {
    \\                uint w_idx = kr * p.N + nc;
    \\                tW[i] = float(weight_data[w_idx]) * weight_scales[w_idx / p.block_size];
    \\            } else {
    \\                tW[i] = 0.0f;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        simdgroup_float8x8 a0, a1, b0, b1;
    \\        simdgroup_load(a0, tI + (sRow + 0) * 8, 8);
    \\        simdgroup_load(a1, tI + (sRow + 8) * 8, 8);
    \\        simdgroup_load(b0, tW + (sCol + 0), TILE);
    \\        simdgroup_load(b1, tW + (sCol + 8), TILE);
    \\        simdgroup_multiply_accumulate(acc[0], a0, b0, acc[0]);
    \\        simdgroup_multiply_accumulate(acc[1], a0, b1, acc[1]);
    \\        simdgroup_multiply_accumulate(acc[2], a1, b0, acc[2]);
    \\        simdgroup_multiply_accumulate(acc[3], a1, b1, acc[3]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    threadgroup float tC[TILE * TILE];
    \\    simdgroup_store(acc[0], tC + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[1], tC + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(acc[2], tC + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[3], tC + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, cc = gCol + c;
    \\        if (cr < p.M && cc < p.N) {
    \\            float base = tC[i];
    \\            uint linear = cr * p.N + cc;
    \\            float v = base;
    \\            for (uint step = 0; step < p.n_steps; step++) {
    \\                uint op = p.op[step];
    \\                if (op == 7 || op == 8) {
    \\                    uint sec_idx = p.secondary_offset[step] + linear;
    \\                    float other = fused_secondary(p.secondary_slot[step], sec_idx, s0, s1, s2, s3, s4, s5, s6, s7);
    \\                    if (op == 7) v = (p.is_swapped[step] != 0) ? other + v : v + other;
    \\                    else v = (p.is_swapped[step] != 0) ? other * v : v * other;
    \\                } else {
    \\                    v = fused_unary(op, v);
    \\                }
    \\            }
    \\            output[p.dst_offset + cr * p.dst_row_stride + cc] = base;
    \\            ew_output[p.ew_dst_offset + linear] = v;
    \\        }
    \\    }
    \\}
;

// ── Kernel param structs (must match MSL layout) ──────────────────

const MatMulParams = extern struct {
    M: u32,
    N: u32,
    K: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    dst_offset: u32,
    dst_row_stride: u32,
};

const QMatMulParams = extern struct {
    M: u32,
    N: u32,
    K: u32,
    block_size: u32,
    input_offset: u32,
    input_row_stride: u32,
    dst_offset: u32,
    dst_row_stride: u32,
};

const MAX_QMATMUL_BATCH: usize = 4;

const QMatmulBatch4Params = extern struct {
    n_ops: u32,
    max_tiles_y: u32,
    M: [MAX_QMATMUL_BATCH]u32,
    N: [MAX_QMATMUL_BATCH]u32,
    K: [MAX_QMATMUL_BATCH]u32,
    block_size: [MAX_QMATMUL_BATCH]u32,
    input_offset: [MAX_QMATMUL_BATCH]u32,
    input_row_stride: [MAX_QMATMUL_BATCH]u32,
    dst_offset: [MAX_QMATMUL_BATCH]u32,
    dst_row_stride: [MAX_QMATMUL_BATCH]u32,
    has_slice: [MAX_QMATMUL_BATCH]u32,
    slice_rows: [MAX_QMATMUL_BATCH]u32,
    slice_cols: [MAX_QMATMUL_BATCH]u32,
    slice_src_col_start: [MAX_QMATMUL_BATCH]u32,
    slice_dst_offset: [MAX_QMATMUL_BATCH]u32,
    slice_dst_row_stride: [MAX_QMATMUL_BATCH]u32,
    slice_dst_col_stride: [MAX_QMATMUL_BATCH]u32,
};

const MAX_QMATVEC_BATCH: usize = 4;

const QMatvecBatch4Params = extern struct {
    n_ops: u32,
    max_n: u32,
    N: [MAX_QMATVEC_BATCH]u32,
    K: [MAX_QMATVEC_BATCH]u32,
    block_size: [MAX_QMATVEC_BATCH]u32,
    input_offset: [MAX_QMATVEC_BATCH]u32,
    dst_offset: [MAX_QMATVEC_BATCH]u32,
};

const MatVecParams = extern struct {
    N: u32,
    K: u32,
    a_offset: u32,
    dst_offset: u32,
};

const MatMulF16Params = extern struct {
    M: u32,
    N: u32,
    K: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    a_offset: u32,
    dst_offset: u32,
    dst_row_stride: u32,
};

const RopeParams = extern struct {
    half_d: u32,
    seq_len: u32,
    src_off: u32,
    cs_off: u32,
    dst_off: u32,
    src_rs: u32,
    src_cs: u32,
    cs_cs: u32,
};

const MAX_ROPE_BATCH: usize = 16;

const RopeBatchParams = extern struct {
    n_ops: u32,
    max_n: u32,
    half_d: [MAX_ROPE_BATCH]u32,
    seq_len: [MAX_ROPE_BATCH]u32,
    src_off: [MAX_ROPE_BATCH]u32,
    cs_off: [MAX_ROPE_BATCH]u32,
    dst_off: [MAX_ROPE_BATCH]u32,
    src_rs: [MAX_ROPE_BATCH]u32,
    src_cs: [MAX_ROPE_BATCH]u32,
    cs_cs: [MAX_ROPE_BATCH]u32,
};

const RopeSliceAssignParams = extern struct {
    half_d: u32,
    seq_len: u32,
    src_off: u32,
    cs_off: u32,
    src_rs: u32,
    src_cs: u32,
    cs_cs: u32,
    dst_offset: u32,
    dst_row_stride: u32,
    dst_col_stride: u32,
};

const QMatmulSliceAssignParams = extern struct {
    M: u32,
    N: u32,
    K: u32,
    block_size: u32,
    input_offset: u32,
    input_row_stride: u32,
    dst_offset: u32,
    dst_row_stride: u32,
    slice_rows: u32,
    slice_cols: u32,
    slice_src_col_start: u32,
    slice_dst_offset: u32,
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
};

const QMatmulElementwiseParams = extern struct {
    M: u32,
    N: u32,
    K: u32,
    block_size: u32,
    input_offset: u32,
    input_row_stride: u32,
    dst_offset: u32,
    dst_row_stride: u32,
    ew_op: u32,
    ew_is_swapped: u32,
    ew_dst_offset: u32,
    ew_secondary_offset: u32,
};

const MAX_SLICE_ASSIGN_BATCH: usize = 16;

const SliceAssignBatchParams = extern struct {
    n_ops: u32,
    max_n: u32,
    rows: [MAX_SLICE_ASSIGN_BATCH]u32,
    cols: [MAX_SLICE_ASSIGN_BATCH]u32,
    dst_offset: [MAX_SLICE_ASSIGN_BATCH]u32,
    dst_row_stride: [MAX_SLICE_ASSIGN_BATCH]u32,
    dst_col_stride: [MAX_SLICE_ASSIGN_BATCH]u32,
    src_offset: [MAX_SLICE_ASSIGN_BATCH]u32,
    src_row_stride: [MAX_SLICE_ASSIGN_BATCH]u32,
    src_col_stride: [MAX_SLICE_ASSIGN_BATCH]u32,
};

const RmsNormScaleParams = extern struct {
    rows: u32,
    cols: u32,
    eps: f32,
    src_offset: u32,
    norm_dst_offset: u32,
    scale_src_offset: u32,
    scale_repeat_dst_offset: u32,
    scaled_dst_offset: u32,
};

const AttentionParams = extern struct {
    d_head: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    mask_off: u32,
    dst_off: u32,
    q_rs: u32,
    q_cs: u32,
    k_rs: u32,
    k_cs: u32,
    v_rs: u32,
    v_cs: u32,
    mask_rs: u32,
    mask_cs: u32,
    dst_rs: u32,
    dst_cs: u32,
};

const MAX_ATTENTION_BATCH_HEADS: usize = 16;

const AttentionBatchParams = extern struct {
    n_heads: u32,
    d_head: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    q_off: [MAX_ATTENTION_BATCH_HEADS]u32,
    k_off: [MAX_ATTENTION_BATCH_HEADS]u32,
    v_off: [MAX_ATTENTION_BATCH_HEADS]u32,
    mask_off: [MAX_ATTENTION_BATCH_HEADS]u32,
    dst_off: [MAX_ATTENTION_BATCH_HEADS]u32,
    q_rs: u32,
    q_cs: u32,
    k_rs: u32,
    k_cs: u32,
    v_rs: u32,
    v_cs: u32,
    mask_rs: u32,
    mask_cs: u32,
    dst_rs: u32,
    dst_cs: u32,
};

const ComputeParams = extern struct {
    op: u32,
    n_elements: u32,
    dst_ne: [4]u32,
    dst_strides: [4]u32,
    dst_offset: u32,
    src0_ne: [4]u32,
    src0_strides: [4]u32,
    src0_offset: u32,
    src1_ne: [4]u32,
    src1_strides: [4]u32,
    src1_offset: u32,
};

const MAX_FUSED_EW_STEPS: usize = 8;
const MAX_FUSED_EW_SECONDARIES: usize = 8;

const FusedEwParams = extern struct {
    n_elements: u32,
    n_steps: u32,
    dst_offset: u32,
    src_offset: u32,
    op: [MAX_FUSED_EW_STEPS]u32,
    is_swapped: [MAX_FUSED_EW_STEPS]u32,
    secondary_slot: [MAX_FUSED_EW_STEPS]u32,
    secondary_offset: [MAX_FUSED_EW_STEPS]u32,
};

const MAX_ELEMENTWISE_BATCH: usize = 8;

const ElementwiseBatchParams = extern struct {
    n_ops: u32,
    max_n: u32,
    op: [MAX_ELEMENTWISE_BATCH]u32,
    n_elements: [MAX_ELEMENTWISE_BATCH]u32,
    dst_offset: [MAX_ELEMENTWISE_BATCH]u32,
    src0_offset: [MAX_ELEMENTWISE_BATCH]u32,
    src1_offset: [MAX_ELEMENTWISE_BATCH]u32,
};

const QMatmulFusedEwParams = extern struct {
    M: u32,
    N: u32,
    K: u32,
    block_size: u32,
    input_offset: u32,
    input_row_stride: u32,
    dst_offset: u32,
    dst_row_stride: u32,
    n_steps: u32,
    ew_dst_offset: u32,
    op: [MAX_FUSED_EW_STEPS]u32,
    is_swapped: [MAX_FUSED_EW_STEPS]u32,
    secondary_slot: [MAX_FUSED_EW_STEPS]u32,
    secondary_offset: [MAX_FUSED_EW_STEPS]u32,
};

const WG_SIZE: u32 = 256;
const MATMUL_THREADS: u32 = 128;

const DispatchGrid = struct {
    gx: u32,
    gy: u32 = 1,
};

const ComputeDispatchSpec = struct {
    params: ComputeParams,
    src0: u16,
    src1: u16,
    dst: u16,
    grid: DispatchGrid,
};

fn matmulGrid(M: anytype, N: anytype) DispatchGrid {
    return .{
        .gx = @intCast((N + TILE - 1) / TILE),
        .gy = @intCast((M + TILE - 1) / TILE),
    };
}

fn linearGrid(n: anytype) u32 {
    return @intCast((n + WG_SIZE - 1) / WG_SIZE);
}

fn matmulParams(geom: backend_mod.MatMulGeometry) MatMulParams {
    return .{
        .M = @intCast(geom.M),
        .N = @intCast(geom.N),
        .K = @intCast(geom.K),
        .a_row_stride = @intCast(geom.a_row_stride),
        .a_col_stride = @intCast(geom.a_col_stride),
        .b_row_stride = @intCast(geom.b_row_stride),
        .b_col_stride = @intCast(geom.b_col_stride),
        .a_offset = @intCast(geom.a_offset),
        .b_offset = @intCast(geom.b_offset),
        .dst_offset = @intCast(geom.dst_offset),
        .dst_row_stride = @intCast(geom.dst_row_stride),
    };
}

fn qmatmulParams(q: anytype, block_size: usize) QMatMulParams {
    return .{
        .M = q.M,
        .N = q.N,
        .K = q.K,
        .block_size = @intCast(block_size),
        .input_offset = q.input_offset,
        .input_row_stride = if (q.input_row_stride != 0) q.input_row_stride else q.K,
        .dst_offset = q.dst_offset,
        .dst_row_stride = if (q.dst_row_stride != 0) q.dst_row_stride else q.N,
    };
}

fn baseComputeParams(op_code: u32, n_elements: u32, dst_offset: u32) ComputeParams {
    var p = std.mem.zeroes(ComputeParams);
    p.op = op_code;
    p.n_elements = n_elements;
    p.dst_offset = dst_offset;
    return p;
}

fn rowComputeParams(op_code: u32, rows: u32, cols: u32, src_offset: u32, dst_offset: u32) ComputeParams {
    var p = baseComputeParams(op_code, rows, dst_offset);
    p.src0_ne[0] = cols;
    p.src0_offset = src_offset;
    return p;
}

fn epsilonRowComputeParams(op_code: u32, rows: u32, cols: u32, eps: f32, src_offset: u32, dst_offset: u32) ComputeParams {
    var p = rowComputeParams(op_code, rows, cols, src_offset, dst_offset);
    p.src1_ne[0] = @bitCast(eps);
    return p;
}

fn elementwiseComputeParams(e: anytype) ComputeParams {
    var p = baseComputeParams(@intFromEnum(e.op), e.n, e.dst_offset);
    p.src0_offset = e.src0_offset;
    p.src1_offset = e.src1_offset;
    return p;
}

fn reduceComputeParams(r: anytype) ComputeParams {
    var p = baseComputeParams(@intFromEnum(r.op), r.n_out, r.dst_offset);
    p.src0_ne[0] = r.reduce_size;
    p.src0_offset = r.src_offset;
    return p;
}

fn repeatComputeParams(rp: anytype) ComputeParams {
    var p = baseComputeParams(@intFromEnum(backend_mod.Op.repeat), rp.n, rp.dst_offset);
    p.dst_ne = rp.dst_ne;
    p.dst_strides = rp.dst_strides;
    p.src0_ne = rp.src_ne;
    p.src0_strides = rp.src_strides;
    p.src0_offset = rp.src_offset;
    return p;
}

fn sliceAssignComputeParams(sa: anytype) ComputeParams {
    var p = std.mem.zeroes(ComputeParams);
    p.op = @intFromEnum(backend_mod.Op.slice_assign);
    p.n_elements = sa.rows * sa.cols;
    p.src0_ne[0] = sa.rows;
    p.dst_strides[0] = sa.dst_row_stride;
    p.dst_strides[1] = sa.dst_col_stride;
    p.dst_offset = sa.dst_offset;
    p.src0_strides[0] = sa.src_row_stride;
    p.src0_strides[1] = sa.src_col_stride;
    p.src0_offset = sa.src_offset;
    return p;
}

fn isSupportedElementwiseOp(op: backend_mod.Op) bool {
    return switch (op) {
        .add, .mul, .neg, .abs, .sgn, .step, .relu, .sqrt, .recip, .exp, .log, .gelu => true,
        else => false,
    };
}

fn computeDispatchSpec(op: backend_mod.DeviceOp) ?ComputeDispatchSpec {
    switch (op) {
        .elementwise => |e| {
            if (!isSupportedElementwiseOp(e.op)) return null;
            return .{
                .params = elementwiseComputeParams(e),
                .src0 = e.src0,
                .src1 = e.src1,
                .dst = e.dst,
                .grid = .{ .gx = linearGrid(e.n) },
            };
        },
        .softmax => |s| return .{
            .params = rowComputeParams(program_mod.compute_op_softmax, s.rows, s.cols, s.src_offset, s.dst_offset),
            .src0 = s.src,
            .src1 = s.src,
            .dst = s.dst,
            .grid = .{ .gx = linearGrid(s.rows) },
        },
        .layernorm => |l| return .{
            .params = epsilonRowComputeParams(program_mod.compute_op_layernorm, l.rows, l.cols, l.eps, l.src_offset, l.dst_offset),
            .src0 = l.src,
            .src1 = l.src,
            .dst = l.dst,
            .grid = .{ .gx = linearGrid(l.rows) },
        },
        .rmsnorm => |r| return .{
            .params = epsilonRowComputeParams(program_mod.compute_op_rmsnorm, r.rows, r.cols, r.eps, r.src_offset, r.dst_offset),
            .src0 = r.src,
            .src1 = r.src,
            .dst = r.dst,
            .grid = .{ .gx = linearGrid(r.rows) },
        },
        .reduce => |r| {
            if (r.op != .sum and r.op != .max) return null;
            return .{
                .params = reduceComputeParams(r),
                .src0 = r.src,
                .src1 = r.dst,
                .dst = r.dst,
                .grid = .{ .gx = linearGrid(r.n_out) },
            };
        },
        .repeat => |rp| return .{
            .params = repeatComputeParams(rp),
            .src0 = rp.src,
            .src1 = rp.src,
            .dst = rp.dst,
            .grid = .{ .gx = linearGrid(rp.n) },
        },
        .slice_assign => |sa| {
            const params = sliceAssignComputeParams(sa);
            return .{
                .params = params,
                .src0 = sa.src,
                .src1 = sa.src,
                .dst = sa.dst,
                .grid = .{ .gx = linearGrid(params.n_elements) },
            };
        },
        else => return null,
    }
}

fn attentionParams(att: anytype) AttentionParams {
    return .{
        .d_head = att.d_head,
        .seq_q = att.seq_q,
        .seq_kv = att.seq_kv,
        .scale = att.scale,
        .q_off = att.q_off,
        .k_off = att.k_off,
        .v_off = att.v_off,
        .mask_off = att.mask_off,
        .dst_off = att.dst_off,
        .q_rs = att.q_rs,
        .q_cs = att.q_cs,
        .k_rs = att.k_rs,
        .k_cs = att.k_cs,
        .v_rs = att.v_rs,
        .v_cs = att.v_cs,
        .mask_rs = att.mask_rs,
        .mask_cs = att.mask_cs,
        .dst_rs = att.dst_rs,
        .dst_cs = att.dst_cs,
    };
}

fn canEncodeAttention(att: anytype) bool {
    return att.seq_q >= 1 and
        att.seq_kv <= 4096 and
        att.d_head <= 512 and
        att.has_mask and
        att.q_rs == 1 and
        att.mask_rs == 1 and
        att.dst_rs == 1;
}

// ── MetalBackend ──────────────────────────────────────────────────

pub const MetalBackend = struct {
    device: *anyopaque,
    queue: *anyopaque,
    matmul_pipeline: *anyopaque,
    qmatmul_pipeline: *anyopaque,
    qmatmul_batch4_pipeline: *anyopaque,
    qmatvec_pipeline: *anyopaque,
    qmatvec_batch4_pipeline: *anyopaque,
    matvec_f16_pipeline: *anyopaque,
    matmul_f16_pipeline: *anyopaque,
    rope_pipeline: *anyopaque,
    rope_batch_pipeline: *anyopaque,
    rope_slice_assign_pipeline: *anyopaque,
    qmatmul_slice_assign_pipeline: *anyopaque,
    qmatmul_elementwise_pipeline: *anyopaque,
    qmatmul_fused_ew_pipeline: *anyopaque,
    qmatvec_slice_assign_pipeline: *anyopaque,
    slice_assign_batch_pipeline: *anyopaque,
    rmsnorm_scale_pipeline: *anyopaque,
    attention_pipeline: *anyopaque,
    attention_batch_pipeline: *anyopaque,
    compute_pipeline: *anyopaque,
    elementwise_batch8_pipeline: *anyopaque,
    fused_ew_pipeline: *anyopaque,
    library: *anyopaque,
    active_commands: ?*anyopaque = null,
    fine_grained_program_dispatch: bool = false,
    region_program_dispatch: bool = false,

    pub fn init() !MetalBackend {
        const device = c.mtl_create_device() orelse return error.MetalNotAvailable;
        errdefer c.mtl_release(device);

        const queue = c.mtl_create_queue(device) orelse return error.MetalInitFailed;
        errdefer c.mtl_release(queue);

        const library = c.mtl_compile_source(device, shader_source.ptr, shader_source.len) orelse return error.ShaderCompileFailed;
        errdefer c.mtl_release(library);

        const matmul_pipeline = c.mtl_create_pipeline(device, library, "matmul_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(matmul_pipeline);
        const qmatmul_pipeline = c.mtl_create_pipeline(device, library, "qmatmul_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(qmatmul_pipeline);
        const qmatmul_batch4_pipeline = c.mtl_create_pipeline(device, library, "qmatmul_batch4_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(qmatmul_batch4_pipeline);
        const qmatvec_pipeline = c.mtl_create_pipeline(device, library, "qmatvec_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(qmatvec_pipeline);
        const qmatvec_batch4_pipeline = c.mtl_create_pipeline(device, library, "qmatvec_batch4_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(qmatvec_batch4_pipeline);
        const matvec_f16_pipeline = c.mtl_create_pipeline(device, library, "matvec_f16") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(matvec_f16_pipeline);
        const matmul_f16_pipeline = c.mtl_create_pipeline(device, library, "matmul_f16") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(matmul_f16_pipeline);
        const rope_pipeline = c.mtl_create_pipeline(device, library, "rope_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(rope_pipeline);
        const rope_batch_pipeline = c.mtl_create_pipeline(device, library, "rope_batch_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(rope_batch_pipeline);
        const rope_slice_assign_pipeline = c.mtl_create_pipeline(device, library, "rope_slice_assign_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(rope_slice_assign_pipeline);
        const qmatmul_slice_assign_pipeline = c.mtl_create_pipeline(device, library, "qmatmul_slice_assign_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(qmatmul_slice_assign_pipeline);
        const qmatmul_elementwise_pipeline = c.mtl_create_pipeline(device, library, "qmatmul_elementwise_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(qmatmul_elementwise_pipeline);
        const qmatmul_fused_ew_pipeline = c.mtl_create_pipeline(device, library, "qmatmul_fused_elementwise_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(qmatmul_fused_ew_pipeline);
        const qmatvec_slice_assign_pipeline = c.mtl_create_pipeline(device, library, "qmatvec_slice_assign_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(qmatvec_slice_assign_pipeline);
        const slice_assign_batch_pipeline = c.mtl_create_pipeline(device, library, "slice_assign_batch_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(slice_assign_batch_pipeline);
        const rmsnorm_scale_pipeline = c.mtl_create_pipeline(device, library, "rmsnorm_scale_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(rmsnorm_scale_pipeline);
        const attention_pipeline = c.mtl_create_pipeline(device, library, "attention_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(attention_pipeline);
        const attention_batch_pipeline = c.mtl_create_pipeline(device, library, "attention_batch_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(attention_batch_pipeline);
        const compute_pipeline = c.mtl_create_pipeline(device, library, "compute_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(compute_pipeline);
        const elementwise_batch8_pipeline = c.mtl_create_pipeline(device, library, "elementwise_batch8_f32") orelse return error.PipelineCreateFailed;
        errdefer c.mtl_release(elementwise_batch8_pipeline);
        const fused_ew_pipeline = c.mtl_create_pipeline(device, library, "fused_elementwise_f32") orelse return error.PipelineCreateFailed;

        return .{
            .device = device,
            .queue = queue,
            .matmul_pipeline = matmul_pipeline,
            .qmatmul_pipeline = qmatmul_pipeline,
            .qmatmul_batch4_pipeline = qmatmul_batch4_pipeline,
            .qmatvec_pipeline = qmatvec_pipeline,
            .qmatvec_batch4_pipeline = qmatvec_batch4_pipeline,
            .matvec_f16_pipeline = matvec_f16_pipeline,
            .matmul_f16_pipeline = matmul_f16_pipeline,
            .rope_pipeline = rope_pipeline,
            .rope_batch_pipeline = rope_batch_pipeline,
            .rope_slice_assign_pipeline = rope_slice_assign_pipeline,
            .qmatmul_slice_assign_pipeline = qmatmul_slice_assign_pipeline,
            .qmatmul_elementwise_pipeline = qmatmul_elementwise_pipeline,
            .qmatmul_fused_ew_pipeline = qmatmul_fused_ew_pipeline,
            .qmatvec_slice_assign_pipeline = qmatvec_slice_assign_pipeline,
            .slice_assign_batch_pipeline = slice_assign_batch_pipeline,
            .rmsnorm_scale_pipeline = rmsnorm_scale_pipeline,
            .attention_pipeline = attention_pipeline,
            .attention_batch_pipeline = attention_batch_pipeline,
            .compute_pipeline = compute_pipeline,
            .elementwise_batch8_pipeline = elementwise_batch8_pipeline,
            .fused_ew_pipeline = fused_ew_pipeline,
            .library = library,
        };
    }

    pub fn deinit(self: *MetalBackend) void {
        self.flushCommands();
        c.mtl_release(self.fused_ew_pipeline);
        c.mtl_release(self.elementwise_batch8_pipeline);
        c.mtl_release(self.compute_pipeline);
        c.mtl_release(self.attention_batch_pipeline);
        c.mtl_release(self.attention_pipeline);
        c.mtl_release(self.rmsnorm_scale_pipeline);
        c.mtl_release(self.slice_assign_batch_pipeline);
        c.mtl_release(self.qmatvec_slice_assign_pipeline);
        c.mtl_release(self.qmatmul_fused_ew_pipeline);
        c.mtl_release(self.qmatmul_elementwise_pipeline);
        c.mtl_release(self.qmatmul_slice_assign_pipeline);
        c.mtl_release(self.rope_slice_assign_pipeline);
        c.mtl_release(self.rope_batch_pipeline);
        c.mtl_release(self.rope_pipeline);
        c.mtl_release(self.matmul_f16_pipeline);
        c.mtl_release(self.matvec_f16_pipeline);
        c.mtl_release(self.qmatvec_batch4_pipeline);
        c.mtl_release(self.qmatvec_pipeline);
        c.mtl_release(self.qmatmul_batch4_pipeline);
        c.mtl_release(self.qmatmul_pipeline);
        c.mtl_release(self.matmul_pipeline);
        c.mtl_release(self.library);
        c.mtl_release(self.queue);
        c.mtl_release(self.device);
    }

    /// Enable one-dispatch-per-DeviceOp execution for experiments and
    /// conformance tests. Decode keeps this off by default until layer-level
    /// fusion removes the tiny-dispatch overhead.
    pub fn setFineGrainedProgramDispatch(self: *MetalBackend, enabled: bool) void {
        self.fine_grained_program_dispatch = enabled;
    }

    /// Enable named schedule-region lowering experiments. This keeps the
    /// public DeviceProgram IR simple while allowing Metal to lower reusable
    /// transformer-shaped regions when it has a native implementation.
    pub fn setRegionProgramDispatch(self: *MetalBackend, enabled: bool) void {
        self.region_program_dispatch = enabled;
    }

    /// Ensure a command session is active, creating one if needed.
    fn ensureCommands(self: *MetalBackend) *anyopaque {
        if (self.active_commands == null) {
            self.active_commands = c.mtl_begin_commands(self.queue);
        }
        return self.active_commands.?;
    }

    /// Commit and wait on any active command session.
    fn flushCommands(self: *MetalBackend) void {
        if (self.active_commands) |cmds| {
            c.mtl_commit_and_wait(cmds);
            self.active_commands = null;
        }
    }

    pub fn backend(self: *MetalBackend) backend_mod.Backend {
        return .{
            .ctx = @ptrCast(self),
            .vtable = &vtable,
            .name_str = "metal",
            .device_type = .metal,
            .capabilities = backend_mod.Capabilities.metal,
        };
    }
};

// ── VTable implementation ─────────────────────────────────────────

fn getState(ctx: *anyopaque) *MetalBackend {
    return @ptrCast(@alignCast(ctx));
}

// Host kernel dispatch — delegate to BLAS.
fn denseMatMulF32(_: *anyopaque, spec: backend_mod.DenseMatMulSpecF32) bool {
    const forward = @import("../tensor/forward.zig");
    const g = spec.geom;
    forward.blasSgemm(spec.dst, spec.a, spec.b, g.M, g.N, g.K, g.a_row_stride, g.a_col_stride, g.b_row_stride, g.b_col_stride, g.a_offset, g.b_offset, g.dst_offset, g.dst_row_stride);
    return true;
}

// ── Compiled program execution ────────────────────────────────────

/// Opaque handle to a Metal buffer + size.
const DeviceBuffer = struct {
    ptr: *anyopaque,
    size: usize,
};

/// Device-resident quantized weight (data + scales as Metal buffers).
const DeviceQWeight = struct {
    data: DeviceBuffer,
    scales: DeviceBuffer,
    block_size: usize,
};

fn releaseDeviceBuffers(device_bufs: []const DeviceBuffer) void {
    for (device_bufs) |buf| c.mtl_release(buf.ptr);
}

fn releaseQWeightViews(qweight_views: []const DeviceQWeight) void {
    for (qweight_views) |qw| {
        c.mtl_release(qw.data.ptr);
        c.mtl_release(qw.scales.ptr);
    }
}

fn metalSchedulePolicy(fine_grained: bool) program_mod.SchedulePolicy {
    return .{
        .capabilities = backend_mod.Capabilities.metal,
        .native_kernels = .{
            .elementwise = fine_grained,
            .fused_elementwise = fine_grained,
            .row = fine_grained,
            .reduce = fine_grained,
            .movement = fine_grained,
            .matmul = true,
            .qmatvec = true,
            .qmatmul = true,
            .rope = fine_grained,
            .attention = fine_grained,
        },
        .fine_grained = fine_grained,
    };
}

const metal_pattern_decode_layer_stage: u32 = 0;
const metal_pattern_prefill_layer_stage: u32 = 1;
const metal_projection_anchors_per_stage: u32 = 7;

fn buildMetalRegionSchedule(alloc: std.mem.Allocator, schedule: []const program_mod.KernelItem) ![]program_mod.ScheduleUnit {
    const stages = [_]program_mod.StagePolicy{
        program_mod.StagePolicy.anchored(
            "decode-layer",
            metal_pattern_decode_layer_stage,
            program_mod.RegionPolicy.qmatvecCluster(),
            metal_projection_anchors_per_stage,
        ),
        program_mod.StagePolicy.anchored(
            "prefill-layer",
            metal_pattern_prefill_layer_stage,
            program_mod.RegionPolicy.qmatmulCluster(),
            metal_projection_anchors_per_stage,
        ),
    };
    return program_mod.buildStageRegionSchedule(alloc, schedule, &stages);
}

const CompiledProgram = struct {
    backend: *MetalBackend,
    device_bufs: []DeviceBuffer,
    ref_buffers: []reference.Buffer,
    qweight_views: []DeviceQWeight,
    ref_qweights: []reference.QWeight,
    ops: []const backend_mod.DeviceOp,
    schedule: []const program_mod.KernelItem,
    region_schedule: []const program_mod.ScheduleUnit,
    alloc: std.mem.Allocator,
    runtime_profile: profile_mod.RuntimeProfile = .{},

    fn deinit(self: *CompiledProgram) void {
        releaseDeviceBuffers(self.device_bufs);
        releaseQWeightViews(self.qweight_views);
        for (self.ref_qweights) |qw| reference.deinitTransposedQWeight(self.alloc, qw);
        self.alloc.free(self.ref_buffers);
        self.alloc.free(self.device_bufs);
        if (self.qweight_views.len > 0) self.alloc.free(self.qweight_views);
        if (self.ref_qweights.len > 0) self.alloc.free(self.ref_qweights);
        if (self.schedule.len > 0) self.alloc.free(self.schedule);
        if (self.region_schedule.len > 0) self.alloc.free(self.region_schedule);
        self.alloc.destroy(self);
    }

    fn execute(self: *CompiledProgram, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
        // Upload per-step inputs (token embed, pos, mask) via shared memory.
        reference.uploadToBuffers(self.ref_buffers, inputs);

        if (self.schedule.len == 0 and self.ops.len > 0) {
            self.executeUnscheduled();
        } else {
            self.executeScheduled();
        }
        self.flushCommandsProfiled();
        self.runtime_profile.call_count += 1;

        // Download outputs (logits) via shared memory.
        reference.downloadFromBuffers(self.ref_buffers, outputs);
    }

    fn executeUnscheduled(self: *CompiledProgram) void {
        for (self.ops) |op| {
            self.executeOp(op);
        }
    }

    fn executeScheduled(self: *CompiledProgram) void {
        if (self.backend.region_program_dispatch and self.region_schedule.len > 0) {
            self.executeRegionScheduled();
            return;
        }

        for (self.schedule) |item| {
            self.executeScheduleItem(item);
        }
    }

    fn executeRegionScheduled(self: *CompiledProgram) void {
        for (self.region_schedule) |unit| {
            switch (unit.kind) {
                .item => self.executeScheduleItem(self.schedule[@intCast(unit.start_item)]),
                .pattern_region => {
                    if (!self.tryEncodePatternRegion(unit)) {
                        self.executeScheduleItems(unit.start_item, unit.item_count);
                    }
                },
            }
        }
    }

    fn executeScheduleItems(self: *CompiledProgram, start_item: u32, item_count: u32) void {
        const start: usize = @intCast(start_item);
        const end = start + @as(usize, item_count);
        for (self.schedule[start..end]) |item| {
            self.executeScheduleItem(item);
        }
    }

    fn executeScheduleItem(self: *CompiledProgram, item: program_mod.KernelItem) void {
        const start: usize = @intCast(item.start);
        const end = start + @as(usize, item.len);
        switch (item.execution) {
            .backend => {
                for (self.ops[start..end]) |op| {
                    self.executeOp(op);
                }
            },
            .fallback => {
                self.flushCommandsProfiled();
                for (self.ops[start..end]) |op| {
                    self.executeFallbackOp(op);
                }
            },
        }
    }

    fn executeOp(self: *CompiledProgram, op: backend_mod.DeviceOp) void {
        const tag: usize = @intFromEnum(op);
        const t0 = nowNs();
        if (self.tryEncodeGpuOp(op)) {
            self.runtime_profile.backend_op_count +%= 1;
        } else {
            self.flushCommandsProfiled();
            reference.executeOp(self.ref_buffers, self.ref_qweights, op);
            self.runtime_profile.fallback_op_count +%= 1;
        }
        self.runtime_profile.time_ns[tag] +%= @intCast(nowNs() - t0);
    }

    fn executeFallbackOp(self: *CompiledProgram, op: backend_mod.DeviceOp) void {
        const tag: usize = @intFromEnum(op);
        const t0 = nowNs();
        reference.executeOp(self.ref_buffers, self.ref_qweights, op);
        self.runtime_profile.fallback_op_count +%= 1;
        self.runtime_profile.time_ns[tag] +%= @intCast(nowNs() - t0);
    }

    fn flushCommandsProfiled(self: *CompiledProgram) void {
        if (self.backend.active_commands == null) return;
        const t0 = nowNs();
        self.backend.flushCommands();
        self.runtime_profile.sync_time_ns +%= @intCast(nowNs() - t0);
        self.runtime_profile.sync_count +%= 1;
    }

    fn encode(
        self: *CompiledProgram,
        pipeline: *anyopaque,
        buffers: []const DeviceBuffer,
        params: *const anyopaque,
        params_size: usize,
        params_index: u32,
        grid: DispatchGrid,
        threads_x: u32,
    ) void {
        var raw_buffers: [32]?*anyopaque = undefined;
        std.debug.assert(buffers.len <= raw_buffers.len);
        for (buffers, 0..) |buf, i| raw_buffers[i] = buf.ptr;
        c.mtl_encode_dispatch(
            self.backend.ensureCommands(),
            pipeline,
            @ptrCast(&raw_buffers),
            @intCast(buffers.len),
            params,
            params_size,
            params_index,
            grid.gx,
            grid.gy,
            threads_x,
            1,
        );
        self.runtime_profile.backend_dispatch_count +%= 1;
    }

    fn encodeTyped(
        self: *CompiledProgram,
        comptime Params: type,
        pipeline: *anyopaque,
        buffers: []const DeviceBuffer,
        params: Params,
        params_index: u32,
        grid: DispatchGrid,
        threads_x: u32,
    ) void {
        self.encode(pipeline, buffers, &params, @sizeOf(Params), params_index, grid, threads_x);
    }

    fn canEncodeFusedElementwise(fe: anytype) bool {
        if (fe.steps.len > MAX_FUSED_EW_STEPS) return false;
        for (fe.steps) |step| {
            if (!isSupportedElementwiseOp(step.op)) return false;
        }
        return true;
    }

    fn encodeFusedElementwise(self: *CompiledProgram, fe: anytype) bool {
        if (!canEncodeFusedElementwise(fe)) return false;

        var params = std.mem.zeroes(FusedEwParams);
        params.n_elements = fe.n;
        params.n_steps = @intCast(fe.steps.len);
        params.dst_offset = fe.dst_offset;
        params.src_offset = fe.src_offset;

        var buffers: [2 + MAX_FUSED_EW_SECONDARIES]DeviceBuffer = undefined;
        buffers[0] = self.device_bufs[fe.src];
        buffers[1] = self.device_bufs[fe.dst];
        for (buffers[2..]) |*buf| buf.* = self.device_bufs[fe.src];

        var secondary_bufs: [MAX_FUSED_EW_SECONDARIES]u16 = undefined;
        var secondary_count: usize = 0;

        for (fe.steps, 0..) |step, i| {
            params.op[i] = @intFromEnum(step.op);
            params.is_swapped[i] = @intFromBool(step.is_swapped);
            params.secondary_offset[i] = step.secondary_offset;

            if (step.op.isBinary()) {
                const slot = for (secondary_bufs[0..secondary_count], 0..) |buf_idx, slot_idx| {
                    if (buf_idx == step.secondary_buf) break slot_idx;
                } else blk: {
                    if (secondary_count >= MAX_FUSED_EW_SECONDARIES) return false;
                    const next = secondary_count;
                    secondary_bufs[next] = step.secondary_buf;
                    buffers[2 + next] = self.device_bufs[step.secondary_buf];
                    secondary_count += 1;
                    break :blk next;
                };
                params.secondary_slot[i] = @intCast(slot);
            }
        }

        self.encodeTyped(
            FusedEwParams,
            self.backend.fused_ew_pipeline,
            buffers[0..],
            params,
            10,
            .{ .gx = linearGrid(fe.n) },
            WG_SIZE,
        );
        return true;
    }

    fn tryEncodeFusedElementwise(self: *CompiledProgram, fe: anytype) bool {
        if (!self.backend.fine_grained_program_dispatch) return false;
        return self.encodeFusedElementwise(fe);
    }

    fn encodeComputeDispatch(self: *CompiledProgram, spec: ComputeDispatchSpec) void {
        const buffers = [_]DeviceBuffer{
            self.device_bufs[spec.src0],
            self.device_bufs[spec.src1],
            self.device_bufs[spec.dst],
        };
        self.encodeTyped(ComputeParams, self.backend.compute_pipeline, &buffers, spec.params, 3, spec.grid, WG_SIZE);
    }

    fn canEncodeElementwiseBatchOp(e: anytype) bool {
        return program_mod.canBatchElementwiseOp(e);
    }

    fn encodeElementwiseBatch(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, indices: []const usize) void {
        const first_e = ops[indices[0]].elementwise;
        var buffers: [MAX_ELEMENTWISE_BATCH * 3]DeviceBuffer = undefined;
        for (0..MAX_ELEMENTWISE_BATCH) |slot| {
            buffers[slot * 3 + 0] = self.device_bufs[first_e.src0];
            buffers[slot * 3 + 1] = self.device_bufs[first_e.src1];
            buffers[slot * 3 + 2] = self.device_bufs[first_e.dst];
        }

        var params = std.mem.zeroes(ElementwiseBatchParams);
        params.n_ops = @intCast(indices.len);
        for (indices, 0..) |op_index, slot| {
            const e = ops[op_index].elementwise;
            buffers[slot * 3 + 0] = self.device_bufs[e.src0];
            buffers[slot * 3 + 1] = self.device_bufs[e.src1];
            buffers[slot * 3 + 2] = self.device_bufs[e.dst];
            params.op[slot] = @intFromEnum(e.op);
            params.n_elements[slot] = e.n;
            params.dst_offset[slot] = e.dst_offset;
            params.src0_offset[slot] = e.src0_offset;
            params.src1_offset[slot] = e.src1_offset;
            params.max_n = @max(params.max_n, e.n);
        }

        self.encodeTyped(
            ElementwiseBatchParams,
            self.backend.elementwise_batch8_pipeline,
            &buffers,
            params,
            24,
            .{ .gx = linearGrid(params.max_n), .gy = @intCast(indices.len) },
            WG_SIZE,
        );
    }

    fn encodeQMatvec(self: *CompiledProgram, q: anytype) bool {
        if (@as(usize, q.weight_idx) >= self.qweight_views.len) return false;
        const w = self.qweight_views[q.weight_idx];
        const buffers = [_]DeviceBuffer{
            w.data,
            w.scales,
            self.device_bufs[q.input],
            self.device_bufs[q.dst],
        };
        self.encodeTyped(QMatMulParams, self.backend.qmatvec_pipeline, &buffers, qmatmulParams(q, w.block_size), 4, .{ .gx = linearGrid(q.N) }, WG_SIZE);
        return true;
    }

    fn canEncodeQMatmulBatchOp(self: *CompiledProgram, q: anytype) bool {
        return q.M != 1 and @as(usize, q.weight_idx) < self.qweight_views.len;
    }

    fn encodeQMatmulBatch(
        self: *CompiledProgram,
        ops: []const backend_mod.DeviceOp,
        indices: []const usize,
        slice_indices: []const ?usize,
    ) void {
        const first_q = ops[indices[0]].qmatmul;
        const first_w = self.qweight_views[first_q.weight_idx];
        var buffers: [MAX_QMATMUL_BATCH * 5]DeviceBuffer = undefined;
        for (0..MAX_QMATMUL_BATCH) |slot| {
            buffers[slot * 5 + 0] = first_w.data;
            buffers[slot * 5 + 1] = first_w.scales;
            buffers[slot * 5 + 2] = self.device_bufs[first_q.input];
            buffers[slot * 5 + 3] = self.device_bufs[first_q.dst];
            buffers[slot * 5 + 4] = self.device_bufs[first_q.dst];
        }

        var params = std.mem.zeroes(QMatmulBatch4Params);
        params.n_ops = @intCast(indices.len);
        var max_tiles_x: u32 = 1;
        var max_tiles_y: u32 = 1;
        for (indices, 0..) |op_index, slot| {
            const q = ops[op_index].qmatmul;
            const w = self.qweight_views[q.weight_idx];
            const qparams = qmatmulParams(q, w.block_size);
            buffers[slot * 5 + 0] = w.data;
            buffers[slot * 5 + 1] = w.scales;
            buffers[slot * 5 + 2] = self.device_bufs[q.input];
            buffers[slot * 5 + 3] = self.device_bufs[q.dst];
            buffers[slot * 5 + 4] = self.device_bufs[q.dst];
            params.M[slot] = qparams.M;
            params.N[slot] = qparams.N;
            params.K[slot] = qparams.K;
            params.block_size[slot] = qparams.block_size;
            params.input_offset[slot] = qparams.input_offset;
            params.input_row_stride[slot] = qparams.input_row_stride;
            params.dst_offset[slot] = qparams.dst_offset;
            params.dst_row_stride[slot] = qparams.dst_row_stride;
            if (slice_indices[slot]) |slice_index| {
                const sa = ops[slice_index].slice_assign;
                params.has_slice[slot] = 1;
                params.slice_rows[slot] = sa.rows;
                params.slice_cols[slot] = sa.cols;
                params.slice_src_col_start[slot] = program_mod.qmatmulSliceSrcColStart(q, sa).?;
                params.slice_dst_offset[slot] = sa.dst_offset;
                params.slice_dst_row_stride[slot] = sa.dst_row_stride;
                params.slice_dst_col_stride[slot] = sa.dst_col_stride;
                buffers[slot * 5 + 4] = self.device_bufs[sa.dst];
            }
            max_tiles_x = @max(max_tiles_x, (qparams.N + TILE - 1) / TILE);
            max_tiles_y = @max(max_tiles_y, (qparams.M + TILE - 1) / TILE);
        }
        params.max_tiles_y = max_tiles_y;

        self.encodeTyped(
            QMatmulBatch4Params,
            self.backend.qmatmul_batch4_pipeline,
            &buffers,
            params,
            20,
            .{ .gx = max_tiles_x, .gy = max_tiles_y * @as(u32, @intCast(indices.len)) },
            MATMUL_THREADS,
        );
    }

    fn canEncodeQMatvecBatchOp(self: *CompiledProgram, q: anytype) bool {
        return q.M == 1 and
            @as(usize, q.weight_idx) < self.qweight_views.len and
            (q.input_row_stride == 0 or q.input_row_stride == q.K) and
            (q.dst_row_stride == 0 or q.dst_row_stride == q.N);
    }

    fn encodeQMatvecBatch(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, indices: []const usize) void {
        const first_q = ops[indices[0]].qmatmul;
        const first_w = self.qweight_views[first_q.weight_idx];
        var buffers: [MAX_QMATVEC_BATCH * 4]DeviceBuffer = undefined;
        for (0..MAX_QMATVEC_BATCH) |slot| {
            buffers[slot * 4 + 0] = first_w.data;
            buffers[slot * 4 + 1] = first_w.scales;
            buffers[slot * 4 + 2] = self.device_bufs[first_q.input];
            buffers[slot * 4 + 3] = self.device_bufs[first_q.dst];
        }

        var params = std.mem.zeroes(QMatvecBatch4Params);
        params.n_ops = @intCast(indices.len);
        for (indices, 0..) |op_index, slot| {
            const q = ops[op_index].qmatmul;
            const w = self.qweight_views[q.weight_idx];
            const qparams = qmatmulParams(q, w.block_size);
            buffers[slot * 4 + 0] = w.data;
            buffers[slot * 4 + 1] = w.scales;
            buffers[slot * 4 + 2] = self.device_bufs[q.input];
            buffers[slot * 4 + 3] = self.device_bufs[q.dst];
            params.N[slot] = qparams.N;
            params.K[slot] = qparams.K;
            params.block_size[slot] = qparams.block_size;
            params.input_offset[slot] = qparams.input_offset;
            params.dst_offset[slot] = qparams.dst_offset;
            params.max_n = @max(params.max_n, qparams.N);
        }

        self.encodeTyped(
            QMatvecBatch4Params,
            self.backend.qmatvec_batch4_pipeline,
            &buffers,
            params,
            16,
            .{ .gx = linearGrid(params.max_n), .gy = @intCast(indices.len) },
            WG_SIZE,
        );
    }

    fn encodeRope(self: *CompiledProgram, rr: anytype) void {
        const buffers = [_]DeviceBuffer{
            self.device_bufs[rr.src],
            self.device_bufs[rr.cos_sin],
            self.device_bufs[rr.dst],
        };
        const params = RopeParams{
            .half_d = rr.half_d,
            .seq_len = rr.seq_len,
            .src_off = rr.src_off,
            .cs_off = rr.cs_off,
            .dst_off = rr.dst_off,
            .src_rs = rr.src_rs,
            .src_cs = rr.src_cs,
            .cs_cs = rr.cs_cs,
        };
        self.encodeTyped(RopeParams, self.backend.rope_pipeline, &buffers, params, 3, .{ .gx = linearGrid(rr.half_d * rr.seq_len) }, WG_SIZE);
    }

    fn ropeBatchCompatible(first: anytype, next: anytype) bool {
        return program_mod.ropeBatchCompatible(first, next);
    }

    fn ropeBatchRunLen(ops: []const backend_mod.DeviceOp) usize {
        return program_mod.ropeBatchRunLen(ops, MAX_ROPE_BATCH);
    }

    fn encodeRopeBatch(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, n: usize) void {
        const first = ops[0].rope;
        const buffers = [_]DeviceBuffer{
            self.device_bufs[first.src],
            self.device_bufs[first.cos_sin],
            self.device_bufs[first.dst],
        };
        var params = std.mem.zeroes(RopeBatchParams);
        params.n_ops = @intCast(n);
        for (ops[0..n], 0..) |op, i| {
            const rr = op.rope;
            params.half_d[i] = rr.half_d;
            params.seq_len[i] = rr.seq_len;
            params.src_off[i] = rr.src_off;
            params.cs_off[i] = rr.cs_off;
            params.dst_off[i] = rr.dst_off;
            params.src_rs[i] = rr.src_rs;
            params.src_cs[i] = rr.src_cs;
            params.cs_cs[i] = rr.cs_cs;
            params.max_n = @max(params.max_n, rr.half_d * rr.seq_len);
        }
        self.encodeTyped(RopeBatchParams, self.backend.rope_batch_pipeline, &buffers, params, 3, .{ .gx = linearGrid(params.max_n), .gy = @intCast(n) }, WG_SIZE);
    }

    fn canFuseRopeSliceAssign(rr: anytype, sa: anytype) bool {
        return program_mod.ropeSliceAssignCompatible(rr, sa);
    }

    fn encodeRopeSliceAssign(self: *CompiledProgram, rr: anytype, sa: anytype) void {
        const buffers = [_]DeviceBuffer{
            self.device_bufs[rr.src],
            self.device_bufs[rr.cos_sin],
            self.device_bufs[sa.dst],
        };
        const params = RopeSliceAssignParams{
            .half_d = rr.half_d,
            .seq_len = rr.seq_len,
            .src_off = rr.src_off,
            .cs_off = rr.cs_off,
            .src_rs = rr.src_rs,
            .src_cs = rr.src_cs,
            .cs_cs = rr.cs_cs,
            .dst_offset = sa.dst_offset,
            .dst_row_stride = sa.dst_row_stride,
            .dst_col_stride = sa.dst_col_stride,
        };
        self.encodeTyped(RopeSliceAssignParams, self.backend.rope_slice_assign_pipeline, &buffers, params, 3, .{ .gx = linearGrid(rr.half_d * rr.seq_len) }, WG_SIZE);
    }

    fn canFuseQMatvecSliceAssign(self: *CompiledProgram, q: anytype, sa: anytype) bool {
        return @as(usize, q.weight_idx) < self.qweight_views.len and
            program_mod.qmatvecSliceSidecarCompatible(q, sa);
    }

    fn canFuseQMatmulSliceAssign(self: *CompiledProgram, q: anytype, sa: anytype) bool {
        return @as(usize, q.weight_idx) < self.qweight_views.len and
            program_mod.qmatmulSliceSidecarCompatible(q, sa);
    }

    fn encodeQMatvecSliceAssign(self: *CompiledProgram, q: anytype, sa: anytype) bool {
        if (!self.canFuseQMatvecSliceAssign(q, sa)) return false;
        const w = self.qweight_views[q.weight_idx];
        const qparams = qmatmulParams(q, w.block_size);
        const buffers = [_]DeviceBuffer{
            w.data,
            w.scales,
            self.device_bufs[q.input],
            self.device_bufs[q.dst],
            self.device_bufs[sa.dst],
        };
        const params = QMatmulSliceAssignParams{
            .M = qparams.M,
            .N = qparams.N,
            .K = qparams.K,
            .block_size = qparams.block_size,
            .input_offset = qparams.input_offset,
            .input_row_stride = qparams.input_row_stride,
            .dst_offset = qparams.dst_offset,
            .dst_row_stride = qparams.dst_row_stride,
            .slice_rows = sa.rows,
            .slice_cols = sa.cols,
            .slice_src_col_start = program_mod.qmatmulSliceSrcColStart(q, sa).?,
            .slice_dst_offset = sa.dst_offset,
            .slice_dst_row_stride = sa.dst_row_stride,
            .slice_dst_col_stride = sa.dst_col_stride,
        };
        self.encodeTyped(QMatmulSliceAssignParams, self.backend.qmatvec_slice_assign_pipeline, &buffers, params, 5, .{ .gx = linearGrid(q.N) }, WG_SIZE);
        return true;
    }

    fn encodeQMatmulSliceAssign(self: *CompiledProgram, q: anytype, sa: anytype) bool {
        if (!self.canFuseQMatmulSliceAssign(q, sa)) return false;
        const w = self.qweight_views[q.weight_idx];
        const qparams = qmatmulParams(q, w.block_size);
        const buffers = [_]DeviceBuffer{
            w.data,
            w.scales,
            self.device_bufs[q.input],
            self.device_bufs[q.dst],
            self.device_bufs[sa.dst],
        };
        const params = QMatmulSliceAssignParams{
            .M = qparams.M,
            .N = qparams.N,
            .K = qparams.K,
            .block_size = qparams.block_size,
            .input_offset = qparams.input_offset,
            .input_row_stride = qparams.input_row_stride,
            .dst_offset = qparams.dst_offset,
            .dst_row_stride = qparams.dst_row_stride,
            .slice_rows = sa.rows,
            .slice_cols = sa.cols,
            .slice_src_col_start = program_mod.qmatmulSliceSrcColStart(q, sa).?,
            .slice_dst_offset = sa.dst_offset,
            .slice_dst_row_stride = sa.dst_row_stride,
            .slice_dst_col_stride = sa.dst_col_stride,
        };
        self.encodeTyped(QMatmulSliceAssignParams, self.backend.qmatmul_slice_assign_pipeline, &buffers, params, 5, matmulGrid(q.M, q.N), MATMUL_THREADS);
        return true;
    }

    fn canFuseQMatmulElementwise(self: *CompiledProgram, q: anytype, e: anytype) bool {
        if (q.M == 1 or @as(usize, q.weight_idx) >= self.qweight_views.len) return false;
        if (e.op != .add and e.op != .mul) return false;
        if (e.n != q.M * q.N) return false;
        const dst_row_stride = if (q.dst_row_stride != 0) q.dst_row_stride else q.N;
        if (dst_row_stride != q.N) return false;
        return (e.src0 == q.dst and e.src0_offset == q.dst_offset) or
            (e.src1 == q.dst and e.src1_offset == q.dst_offset);
    }

    fn encodeQMatmulElementwise(self: *CompiledProgram, q: anytype, e: anytype) bool {
        if (!self.canFuseQMatmulElementwise(q, e)) return false;
        const q_is_src0 = e.src0 == q.dst and e.src0_offset == q.dst_offset;
        const secondary_buf = if (q_is_src0) e.src1 else e.src0;
        const secondary_offset = if (q_is_src0) e.src1_offset else e.src0_offset;
        const w = self.qweight_views[q.weight_idx];
        const qparams = qmatmulParams(q, w.block_size);
        const buffers = [_]DeviceBuffer{
            w.data,
            w.scales,
            self.device_bufs[q.input],
            self.device_bufs[q.dst],
            self.device_bufs[secondary_buf],
            self.device_bufs[e.dst],
        };
        const params = QMatmulElementwiseParams{
            .M = qparams.M,
            .N = qparams.N,
            .K = qparams.K,
            .block_size = qparams.block_size,
            .input_offset = qparams.input_offset,
            .input_row_stride = qparams.input_row_stride,
            .dst_offset = qparams.dst_offset,
            .dst_row_stride = qparams.dst_row_stride,
            .ew_op = @intFromEnum(e.op),
            .ew_is_swapped = if (q_is_src0) 0 else 1,
            .ew_dst_offset = e.dst_offset,
            .ew_secondary_offset = secondary_offset,
        };
        self.encodeTyped(QMatmulElementwiseParams, self.backend.qmatmul_elementwise_pipeline, &buffers, params, 6, matmulGrid(q.M, q.N), MATMUL_THREADS);
        return true;
    }

    fn canFuseQMatmulFusedElementwise(self: *CompiledProgram, q: anytype, fe: anytype) bool {
        if (q.M == 1 or @as(usize, q.weight_idx) >= self.qweight_views.len) return false;
        if (!canEncodeFusedElementwise(fe)) return false;
        if (fe.n != q.M * q.N) return false;
        const dst_row_stride = if (q.dst_row_stride != 0) q.dst_row_stride else q.N;
        if (dst_row_stride != q.N) return false;
        return fe.src == q.dst and fe.src_offset == q.dst_offset;
    }

    fn encodeQMatmulFusedElementwise(self: *CompiledProgram, q: anytype, fe: anytype) bool {
        if (!self.canFuseQMatmulFusedElementwise(q, fe)) return false;
        const w = self.qweight_views[q.weight_idx];
        const qparams = qmatmulParams(q, w.block_size);
        var params = std.mem.zeroes(QMatmulFusedEwParams);
        params.M = qparams.M;
        params.N = qparams.N;
        params.K = qparams.K;
        params.block_size = qparams.block_size;
        params.input_offset = qparams.input_offset;
        params.input_row_stride = qparams.input_row_stride;
        params.dst_offset = qparams.dst_offset;
        params.dst_row_stride = qparams.dst_row_stride;
        params.n_steps = @intCast(fe.steps.len);
        params.ew_dst_offset = fe.dst_offset;

        var buffers: [5 + MAX_FUSED_EW_SECONDARIES]DeviceBuffer = undefined;
        buffers[0] = w.data;
        buffers[1] = w.scales;
        buffers[2] = self.device_bufs[q.input];
        buffers[3] = self.device_bufs[q.dst];
        buffers[4] = self.device_bufs[fe.dst];
        for (buffers[5..]) |*buf| buf.* = self.device_bufs[fe.src];

        var secondary_bufs: [MAX_FUSED_EW_SECONDARIES]u16 = undefined;
        var secondary_count: usize = 0;

        for (fe.steps, 0..) |step, i| {
            params.op[i] = @intFromEnum(step.op);
            params.is_swapped[i] = @intFromBool(step.is_swapped);
            params.secondary_offset[i] = step.secondary_offset;

            if (step.op.isBinary()) {
                const slot = for (secondary_bufs[0..secondary_count], 0..) |buf_idx, slot_idx| {
                    if (buf_idx == step.secondary_buf) break slot_idx;
                } else blk: {
                    if (secondary_count >= MAX_FUSED_EW_SECONDARIES) return false;
                    const next = secondary_count;
                    secondary_bufs[next] = step.secondary_buf;
                    buffers[5 + next] = self.device_bufs[step.secondary_buf];
                    secondary_count += 1;
                    break :blk next;
                };
                params.secondary_slot[i] = @intCast(slot);
            }
        }

        self.encodeTyped(
            QMatmulFusedEwParams,
            self.backend.qmatmul_fused_ew_pipeline,
            &buffers,
            params,
            13,
            matmulGrid(q.M, q.N),
            MATMUL_THREADS,
        );
        return true;
    }

    fn sliceAssignBatchCompatible(first: anytype, next: anytype) bool {
        return program_mod.sliceAssignBatchCompatible(first, next);
    }

    fn sliceAssignBatchRunLen(ops: []const backend_mod.DeviceOp) usize {
        return program_mod.sliceAssignBatchRunLen(ops, MAX_SLICE_ASSIGN_BATCH);
    }

    fn encodeSliceAssignBatch(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, n: usize) void {
        const first = ops[0].slice_assign;
        const buffers = [_]DeviceBuffer{
            self.device_bufs[first.src],
            self.device_bufs[first.dst],
        };
        var params = std.mem.zeroes(SliceAssignBatchParams);
        params.n_ops = @intCast(n);
        for (ops[0..n], 0..) |op, i| {
            const sa = op.slice_assign;
            params.rows[i] = sa.rows;
            params.cols[i] = sa.cols;
            params.dst_offset[i] = sa.dst_offset;
            params.dst_row_stride[i] = sa.dst_row_stride;
            params.dst_col_stride[i] = sa.dst_col_stride;
            params.src_offset[i] = sa.src_offset;
            params.src_row_stride[i] = sa.src_row_stride;
            params.src_col_stride[i] = sa.src_col_stride;
            params.max_n = @max(params.max_n, sa.rows * sa.cols);
        }
        self.encodeTyped(SliceAssignBatchParams, self.backend.slice_assign_batch_pipeline, &buffers, params, 2, .{ .gx = linearGrid(params.max_n), .gy = @intCast(n) }, WG_SIZE);
    }

    fn canFuseRmsnormRepeatMul(rn: anytype, rp: anytype, e: anytype) bool {
        return program_mod.isRmsnormScaleChain(
            .{ .rmsnorm = rn },
            .{ .repeat = rp },
            .{ .elementwise = e },
        );
    }

    fn encodeRmsnormRepeatMul(self: *CompiledProgram, rn: anytype, rp: anytype, e: anytype) bool {
        if (!canFuseRmsnormRepeatMul(rn, rp, e)) return false;
        const buffers = [_]DeviceBuffer{
            self.device_bufs[rn.src],
            self.device_bufs[rp.src],
            self.device_bufs[rn.dst],
            self.device_bufs[rp.dst],
            self.device_bufs[e.dst],
        };
        const params = RmsNormScaleParams{
            .rows = rn.rows,
            .cols = rn.cols,
            .eps = rn.eps,
            .src_offset = rn.src_offset,
            .norm_dst_offset = rn.dst_offset,
            .scale_src_offset = rp.src_offset,
            .scale_repeat_dst_offset = rp.dst_offset,
            .scaled_dst_offset = e.dst_offset,
        };
        self.encodeTyped(RmsNormScaleParams, self.backend.rmsnorm_scale_pipeline, &buffers, params, 5, .{ .gx = linearGrid(rn.rows) }, WG_SIZE);
        return true;
    }

    fn encodeAttention(self: *CompiledProgram, att: anytype) void {
        const buffers = [_]DeviceBuffer{
            self.device_bufs[att.q],
            self.device_bufs[att.k],
            self.device_bufs[att.v],
            self.device_bufs[att.mask],
            self.device_bufs[att.dst],
        };
        self.encodeTyped(AttentionParams, self.backend.attention_pipeline, &buffers, attentionParams(att), 5, .{ .gx = att.seq_q }, WG_SIZE);
    }

    fn attentionBatchCompatible(first: anytype, next: anytype) bool {
        return program_mod.attentionBatchCompatible(first, next);
    }

    fn attentionBatchRunLen(self: *CompiledProgram, ops: []const backend_mod.DeviceOp) usize {
        _ = self;
        const n = program_mod.attentionBatchRunLen(ops, MAX_ATTENTION_BATCH_HEADS);
        if (n == 0) return 0;
        const first = switch (ops[0]) {
            .attention => |att| att,
            else => return 0,
        };
        if (!canEncodeAttention(first)) return 0;
        for (ops[1..n]) |op| {
            const next = switch (op) {
                .attention => |att| att,
                else => return 0,
            };
            if (!canEncodeAttention(next) or !attentionBatchCompatible(first, next)) return 0;
        }
        return n;
    }

    fn encodeAttentionBatch(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, n: usize) void {
        const first = ops[0].attention;
        const buffers = [_]DeviceBuffer{
            self.device_bufs[first.q],
            self.device_bufs[first.k],
            self.device_bufs[first.v],
            self.device_bufs[first.mask],
            self.device_bufs[first.dst],
        };
        var params = std.mem.zeroes(AttentionBatchParams);
        params.n_heads = @intCast(n);
        params.d_head = first.d_head;
        params.seq_q = first.seq_q;
        params.seq_kv = first.seq_kv;
        params.scale = first.scale;
        params.q_rs = first.q_rs;
        params.q_cs = first.q_cs;
        params.k_rs = first.k_rs;
        params.k_cs = first.k_cs;
        params.v_rs = first.v_rs;
        params.v_cs = first.v_cs;
        params.mask_rs = first.mask_rs;
        params.mask_cs = first.mask_cs;
        params.dst_rs = first.dst_rs;
        params.dst_cs = first.dst_cs;
        for (ops[0..n], 0..) |op, i| {
            const att = op.attention;
            params.q_off[i] = att.q_off;
            params.k_off[i] = att.k_off;
            params.v_off[i] = att.v_off;
            params.mask_off[i] = att.mask_off;
            params.dst_off[i] = att.dst_off;
        }
        self.encodeTyped(AttentionBatchParams, self.backend.attention_batch_pipeline, &buffers, params, 5, .{ .gx = first.seq_q, .gy = @intCast(n) }, WG_SIZE);
    }

    fn canEncodeRegionGpuOp(self: *CompiledProgram, op: backend_mod.DeviceOp) bool {
        if (computeDispatchSpec(op) != null) return true;
        return switch (op) {
            .matmul => true,
            .qmatmul => |q| @as(usize, q.weight_idx) < self.qweight_views.len and (q.M != 1 or self.canEncodeQMatvecBatchOp(q)),
            .rope => true,
            .attention => |att| canEncodeAttention(att),
            .fused_elementwise => |fe| canEncodeFusedElementwise(fe),
            else => false,
        };
    }

    fn recordRegionBackendOp(self: *CompiledProgram, op: backend_mod.DeviceOp, elapsed_ns: u64) void {
        const tag: usize = @intFromEnum(op);
        self.runtime_profile.backend_op_count +%= 1;
        self.runtime_profile.time_ns[tag] +%= elapsed_ns;
    }

    fn recordRegionFusedPair(self: *CompiledProgram, a: backend_mod.DeviceOp, b: backend_mod.DeviceOp, elapsed_ns: u64) void {
        const first = elapsed_ns / 2;
        self.recordRegionBackendOp(a, first);
        self.recordRegionBackendOp(b, elapsed_ns - first);
    }

    fn recordRegionFusedRun(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, elapsed_ns: u64) void {
        if (ops.len == 0) return;
        const per_op = elapsed_ns / ops.len;
        var used: u64 = 0;
        for (ops, 0..) |op, i| {
            const t = if (i + 1 == ops.len) elapsed_ns - used else per_op;
            self.recordRegionBackendOp(op, t);
            used += t;
        }
    }

    fn recordRegionFusedRunFromIndices(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, indices: []const usize, elapsed_ns: u64) void {
        if (indices.len == 0) return;
        const per_op = elapsed_ns / indices.len;
        var used: u64 = 0;
        for (indices, 0..) |idx, i| {
            const t = if (i + 1 == indices.len) elapsed_ns - used else per_op;
            self.recordRegionBackendOp(ops[idx], t);
            used += t;
        }
    }

    fn tryEncodeProjectionCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        switch (command.projection_kind) {
            .qmatvec => {
                for (command.anchorIndices()) |idx| {
                    if (!self.canEncodeQMatvecBatchOp(ops[idx].qmatmul)) return false;
                }

                const t0 = nowNs();
                self.encodeQMatvecBatch(ops, command.anchorIndices());
                self.recordRegionFusedRunFromIndices(ops, command.anchorIndices(), @intCast(nowNs() - t0));
                return true;
            },
            .qmatmul => {
                for (command.anchorIndices()) |idx| {
                    if (!self.canEncodeQMatmulBatchOp(ops[idx].qmatmul)) return false;
                }
                for (command.sidecarIndices(), 0..) |maybe_idx, slot| {
                    const idx = maybe_idx orelse continue;
                    if (!self.canFuseQMatmulSliceAssign(ops[command.indices[slot]].qmatmul, ops[idx].slice_assign)) return false;
                }

                const t0 = nowNs();
                self.encodeQMatmulBatch(ops, command.anchorIndices(), command.sidecarIndices());
                self.recordRegionFusedRunFromIndices(ops, command.anchorIndices(), @intCast(nowNs() - t0));
                for (command.sidecarIndices()) |maybe_idx| {
                    if (maybe_idx) |idx| self.recordRegionBackendOp(ops[idx], 0);
                }
                return true;
            },
        }
    }

    fn tryEncodeElementwiseBatchRun(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, start: usize, skipped: []bool) bool {
        if (start >= ops.len or skipped[start]) return false;
        const first = switch (ops[start]) {
            .elementwise => |e| e,
            else => return false,
        };
        if (!canEncodeElementwiseBatchOp(first)) return false;

        var indices: [MAX_ELEMENTWISE_BATCH]usize = undefined;
        indices[0] = start;
        var count: usize = 1;

        var scan = start + 1;
        while (scan < ops.len and count < MAX_ELEMENTWISE_BATCH) : (scan += 1) {
            if (skipped[scan]) continue;
            const e = switch (ops[scan]) {
                .elementwise => |e| e,
                else => continue,
            };
            if (!canEncodeElementwiseBatchOp(e)) continue;
            if (!program_mod.canHoistElementwiseTo(ops, start, scan, e)) continue;
            if (program_mod.elementwiseConflictsSelected(ops, indices[0..count], e)) continue;
            indices[count] = scan;
            count += 1;
        }

        if (count < 2) return false;

        const t0 = nowNs();
        self.encodeElementwiseBatch(ops, indices[0..count]);
        self.recordRegionFusedRunFromIndices(ops, indices[0..count], @intCast(nowNs() - t0));
        for (indices[0..count]) |idx| skipped[idx] = true;
        return true;
    }

    fn tryEncodeRegionGpuOp(self: *CompiledProgram, op: backend_mod.DeviceOp) bool {
        const t0 = nowNs();
        const encoded = if (computeDispatchSpec(op)) |spec| blk: {
            self.encodeComputeDispatch(spec);
            break :blk true;
        } else switch (op) {
            .matmul => |m| blk: {
                const buffers = [_]DeviceBuffer{
                    self.device_bufs[m.a],
                    self.device_bufs[m.b],
                    self.device_bufs[m.dst],
                };
                self.encodeTyped(MatMulParams, self.backend.matmul_pipeline, &buffers, matmulParams(m.geom), 3, matmulGrid(m.geom.M, m.geom.N), MATMUL_THREADS);
                break :blk true;
            },
            .qmatmul => |q| blk: {
                if (@as(usize, q.weight_idx) >= self.qweight_views.len) break :blk false;
                if (q.M == 1) break :blk self.encodeQMatvec(q);
                const w = self.qweight_views[q.weight_idx];
                const buffers = [_]DeviceBuffer{
                    w.data,
                    w.scales,
                    self.device_bufs[q.input],
                    self.device_bufs[q.dst],
                };
                self.encodeTyped(QMatMulParams, self.backend.qmatmul_pipeline, &buffers, qmatmulParams(q, w.block_size), 4, matmulGrid(q.M, q.N), MATMUL_THREADS);
                break :blk true;
            },
            .rope => |rr| blk: {
                self.encodeRope(rr);
                break :blk true;
            },
            .attention => |att| blk: {
                if (!self.canEncodeRegionGpuOp(op)) break :blk false;
                self.encodeAttention(att);
                break :blk true;
            },
            .fused_elementwise => |fe| self.encodeFusedElementwise(fe),
            else => false,
        };
        if (encoded) {
            self.recordRegionBackendOp(op, @intCast(nowNs() - t0));
        }
        return encoded;
    }

    fn tryEncodeRegionFusedPair(self: *CompiledProgram, a: backend_mod.DeviceOp, b: backend_mod.DeviceOp) bool {
        const t0 = nowNs();
        const encoded = switch (a) {
            .qmatmul => |q| switch (b) {
                .slice_assign => |sa| if (q.M == 1) self.encodeQMatvecSliceAssign(q, sa) else self.encodeQMatmulSliceAssign(q, sa),
                .elementwise => |e| self.encodeQMatmulElementwise(q, e),
                .fused_elementwise => |fe| self.encodeQMatmulFusedElementwise(q, fe),
                else => false,
            },
            .rope => |rr| switch (b) {
                .slice_assign => |sa| blk: {
                    if (!canFuseRopeSliceAssign(rr, sa)) break :blk false;
                    self.encodeRopeSliceAssign(rr, sa);
                    break :blk true;
                },
                else => false,
            },
            else => false,
        };
        if (encoded) {
            self.recordRegionFusedPair(a, b, @intCast(nowNs() - t0));
        }
        return encoded;
    }

    fn tryEncodeProgramCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, start: usize, skipped: []bool) usize {
        const command = program_mod.findProgramCommand(
            ops,
            start,
            program_mod.CommandStreamPolicy.metal(MAX_QMATVEC_BATCH, MAX_QMATMUL_BATCH),
            skipped,
        ) orelse return 0;
        const end = start + @as(usize, command.op_count);
        if (end > ops.len) return 0;

        const t0 = nowNs();
        const encoded = switch (command.kind) {
            .op => false,
            .projection_group => self.tryEncodeProjectionCommand(ops, command),
            .attention_batch => blk: {
                const n: usize = @intCast(command.op_count);
                if (self.attentionBatchRunLen(ops[start..]) < n) break :blk false;
                self.encodeAttentionBatch(ops[start..], n);
                break :blk true;
            },
            .rope_batch => blk: {
                const n: usize = @intCast(command.op_count);
                if (ropeBatchRunLen(ops[start..]) < n) break :blk false;
                self.encodeRopeBatch(ops[start..], n);
                break :blk true;
            },
            .movement_batch => blk: {
                const n: usize = @intCast(command.op_count);
                if (sliceAssignBatchRunLen(ops[start..]) < n) break :blk false;
                self.encodeSliceAssignBatch(ops[start..], n);
                break :blk true;
            },
            .elementwise_batch => blk: {
                for (command.anchorIndices()) |idx| {
                    if (idx >= ops.len) break :blk false;
                    const e = switch (ops[idx]) {
                        .elementwise => |e| e,
                        else => break :blk false,
                    };
                    if (!canEncodeElementwiseBatchOp(e)) break :blk false;
                }
                const t_batch = nowNs();
                self.encodeElementwiseBatch(ops, command.anchorIndices());
                self.recordRegionFusedRunFromIndices(ops, command.anchorIndices(), @intCast(nowNs() - t_batch));
                break :blk true;
            },
            .row_chain => switch (ops[start]) {
                .rmsnorm => |rn| switch (ops[start + 1]) {
                    .repeat => |rp| switch (ops[start + 2]) {
                        .elementwise => |e| self.encodeRmsnormRepeatMul(rn, rp, e),
                        else => false,
                    },
                    else => false,
                },
                else => false,
            },
            .rope_chain => switch (ops[start]) {
                .rope => |rr| switch (ops[start + 1]) {
                    .slice_assign => |sa| blk: {
                        if (!canFuseRopeSliceAssign(rr, sa)) break :blk false;
                        self.encodeRopeSliceAssign(rr, sa);
                        break :blk true;
                    },
                    else => false,
                },
                else => false,
            },
        };

        if (!encoded) return 0;
        if (command.kind != .projection_group and command.kind != .elementwise_batch) {
            self.recordRegionFusedRun(ops[start..end], @intCast(nowNs() - t0));
        }
        program_mod.markProgramCommandUsed(skipped, command);
        return command.advanceCount();
    }

    fn tryEncodeAttentionBatchRun(self: *CompiledProgram, ops: []const backend_mod.DeviceOp) usize {
        const n = self.attentionBatchRunLen(ops);
        if (n == 0) return 0;
        const t0 = nowNs();
        self.encodeAttentionBatch(ops, n);
        self.recordRegionFusedRun(ops[0..n], @intCast(nowNs() - t0));
        return n;
    }

    fn tryEncodeRopeBatchRun(self: *CompiledProgram, ops: []const backend_mod.DeviceOp) usize {
        const n = ropeBatchRunLen(ops);
        if (n == 0) return 0;
        const t0 = nowNs();
        self.encodeRopeBatch(ops, n);
        self.recordRegionFusedRun(ops[0..n], @intCast(nowNs() - t0));
        return n;
    }

    fn tryEncodeSliceAssignBatchRun(self: *CompiledProgram, ops: []const backend_mod.DeviceOp) usize {
        const n = sliceAssignBatchRunLen(ops);
        if (n == 0) return 0;
        const t0 = nowNs();
        self.encodeSliceAssignBatch(ops, n);
        self.recordRegionFusedRun(ops[0..n], @intCast(nowNs() - t0));
        return n;
    }

    fn tryEncodeRegionGpuOps(self: *CompiledProgram, ops: []const backend_mod.DeviceOp) bool {
        if (ops.len > 256) {
            for (ops) |op| {
                if (!self.tryEncodeRegionGpuOp(op)) return false;
            }
            return true;
        }

        var skipped = [_]bool{false} ** 256;
        var i: usize = 0;
        while (i < ops.len) {
            if (skipped[i]) {
                i += 1;
                continue;
            }
            const command_advance = self.tryEncodeProgramCommand(ops, i, skipped[0..ops.len]);
            if (command_advance > 0) {
                i += command_advance;
                continue;
            }
            const rope_batch_len = self.tryEncodeRopeBatchRun(ops[i..]);
            if (rope_batch_len > 0) {
                i += rope_batch_len;
                continue;
            }
            const slice_assign_batch_len = self.tryEncodeSliceAssignBatchRun(ops[i..]);
            if (slice_assign_batch_len > 0) {
                i += slice_assign_batch_len;
                continue;
            }
            const attention_batch_len = self.tryEncodeAttentionBatchRun(ops[i..]);
            if (attention_batch_len > 0) {
                i += attention_batch_len;
                continue;
            }
            if (i + 1 < ops.len and self.tryEncodeRegionFusedPair(ops[i], ops[i + 1])) {
                i += 2;
                continue;
            }
            if (self.tryEncodeElementwiseBatchRun(ops, i, skipped[0..ops.len])) {
                i += 1;
                continue;
            }
            if (!self.tryEncodeRegionGpuOp(ops[i])) return false;
            i += 1;
        }
        return true;
    }

    fn tryEncodePatternRegion(self: *CompiledProgram, unit: program_mod.ScheduleUnit) bool {
        return switch (unit.pattern_index) {
            metal_pattern_decode_layer_stage,
            metal_pattern_prefill_layer_stage,
            => self.tryEncodeGpuRegion(unit),
            else => false,
        };
    }

    fn tryEncodeGpuRegion(self: *CompiledProgram, unit: program_mod.ScheduleUnit) bool {
        const start_item: usize = @intCast(unit.start_item);
        const end_item = start_item + @as(usize, unit.item_count);
        if (end_item > self.schedule.len) return false;
        const items = self.schedule[start_item..end_item];

        for (items) |item| {
            const op_start: usize = @intCast(item.start);
            const op_end = op_start + @as(usize, item.len);
            if (op_end > self.ops.len) return false;
            for (self.ops[op_start..op_end]) |op| {
                if (!self.canEncodeRegionGpuOp(op)) return false;
            }
        }

        const op_start: usize = @intCast(unit.op_start);
        const op_end = op_start + @as(usize, unit.op_count);
        if (op_end > self.ops.len) return false;
        return self.tryEncodeRegionGpuOps(self.ops[op_start..op_end]);
    }

    fn tryEncodeGpuOp(self: *CompiledProgram, op: backend_mod.DeviceOp) bool {
        const fine_grained = self.backend.fine_grained_program_dispatch;

        if (fine_grained) {
            if (computeDispatchSpec(op)) |spec| {
                self.encodeComputeDispatch(spec);
                return true;
            }
        }

        switch (op) {
            .matmul => |m| {
                if (!fine_grained and m.geom.M < 16) return false;
                const buffers = [_]DeviceBuffer{
                    self.device_bufs[m.a],
                    self.device_bufs[m.b],
                    self.device_bufs[m.dst],
                };
                self.encodeTyped(MatMulParams, self.backend.matmul_pipeline, &buffers, matmulParams(m.geom), 3, matmulGrid(m.geom.M, m.geom.N), MATMUL_THREADS);
                return true;
            },
            .qmatmul => |q| {
                const w = self.qweight_views[q.weight_idx];
                const buffers = [_]DeviceBuffer{
                    w.data,
                    w.scales,
                    self.device_bufs[q.input],
                    self.device_bufs[q.dst],
                };
                if (q.M == 1) {
                    return self.encodeQMatvec(q);
                }
                if (!fine_grained and q.M < 16) return false;
                self.encodeTyped(QMatMulParams, self.backend.qmatmul_pipeline, &buffers, qmatmulParams(q, w.block_size), 4, matmulGrid(q.M, q.N), MATMUL_THREADS);
                return true;
            },
            .rope => |rr| {
                if (!fine_grained) return false;
                self.encodeRope(rr);
                return true;
            },
            .attention => |att| {
                if (!fine_grained) return false;
                if (!canEncodeAttention(att)) return false;
                self.encodeAttention(att);
                return true;
            },
            .elementwise, .softmax, .layernorm, .rmsnorm, .reduce, .repeat, .slice_assign => return false,
            .fused_elementwise => |fe| return self.tryEncodeFusedElementwise(fe),
        }
    }

    fn rebuildSchedule(self: *CompiledProgram, ops: []const backend_mod.DeviceOp) void {
        const policy = metalSchedulePolicy(self.backend.fine_grained_program_dispatch);
        if (program_mod.scheduleShapeMatches(ops, self.schedule, policy)) {
            self.ops = ops;
            self.runtime_profile.schedule_reuse_count +%= 1;
            return;
        }

        const schedule = program_mod.buildKernelSchedule(
            self.alloc,
            ops,
            policy,
        ) catch {
            if (self.schedule.len > 0) self.alloc.free(self.schedule);
            if (self.region_schedule.len > 0) self.alloc.free(self.region_schedule);
            self.ops = ops;
            self.schedule = &.{};
            self.region_schedule = &.{};
            self.runtime_profile.schedule_rebuild_count +%= 1;
            return;
        };

        const region_schedule = buildMetalRegionSchedule(self.alloc, schedule) catch &.{};

        if (self.schedule.len > 0) self.alloc.free(self.schedule);
        if (self.region_schedule.len > 0) self.alloc.free(self.region_schedule);
        self.ops = ops;
        self.schedule = schedule;
        self.region_schedule = region_schedule;
        self.runtime_profile.schedule_rebuild_count +%= 1;
    }
};

fn compileProgramFn(ctx: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const self = getState(ctx);
    const alloc = std.heap.page_allocator;

    // Allocate device buffers.
    const device_bufs = alloc.alloc(DeviceBuffer, program.n_buffers) catch return null;
    errdefer alloc.free(device_bufs);
    var n_device_bufs: usize = 0;
    errdefer releaseDeviceBuffers(device_bufs[0..n_device_bufs]);
    for (device_bufs, program.buffer_sizes) |*buf, size| {
        const byte_size = size * @sizeOf(f32);
        const ptr = c.mtl_create_buffer(self.device, byte_size) orelse return null;
        buf.* = .{ .ptr = ptr, .size = byte_size };
        n_device_bufs += 1;
    }

    // Upload initial data (weights, KV cache zeros).
    for (program.initial_uploads) |io| {
        const buf = device_bufs[io.buf_idx];
        const ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(buf.ptr));
        @memcpy(ptr[io.offset..][0..io.size], io.host_ptr[0..io.size]);
    }

    // Upload quantized weights.
    const qweight_views = alloc.alloc(DeviceQWeight, program.qweights.len) catch return null;
    errdefer if (qweight_views.len > 0) alloc.free(qweight_views);
    const ref_qweights = alloc.alloc(reference.QWeight, program.qweights.len) catch return null;
    errdefer if (ref_qweights.len > 0) alloc.free(ref_qweights);
    var n_qweight_views: usize = 0;
    errdefer releaseQWeightViews(qweight_views[0..n_qweight_views]);
    var n_ref_qweights: usize = 0;
    errdefer for (ref_qweights[0..n_ref_qweights]) |qw| reference.deinitTransposedQWeight(alloc, qw);
    for (program.qweights, 0..) |qw, i| {
        const data_raw = c.mtl_create_buffer(self.device, qw.data.len) orelse return null;
        const data_buf: DeviceBuffer = .{ .ptr = data_raw, .size = qw.data.len };
        const data_ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(data_buf.ptr));
        const i8_as_u8: [*]const u8 = @ptrCast(qw.data.ptr);
        @memcpy(data_ptr[0..qw.data.len], i8_as_u8[0..qw.data.len]);

        const scales_size = qw.scales.len * @sizeOf(f32);
        const scales_raw = c.mtl_create_buffer(self.device, scales_size) orelse {
            c.mtl_release(data_buf.ptr);
            return null;
        };
        const scales_buf: DeviceBuffer = .{ .ptr = scales_raw, .size = scales_size };
        const scales_ptr: [*]u8 = @ptrCast(c.mtl_buffer_contents(scales_buf.ptr));
        @memcpy(scales_ptr[0..scales_size], std.mem.sliceAsBytes(qw.scales));

        qweight_views[i] = .{ .data = data_buf, .scales = scales_buf, .block_size = qw.block_size };
        n_qweight_views += 1;
        const ref_data: [*]const i8 = @ptrCast(c.mtl_buffer_contents(data_buf.ptr));
        const ref_scales: [*]const f32 = @ptrCast(@alignCast(c.mtl_buffer_contents(scales_buf.ptr)));
        ref_qweights[i] = reference.prepareTransposedQWeight(alloc, .{
            .data = ref_data[0..qw.data.len],
            .scales = ref_scales[0..qw.scales.len],
            .rows = qw.rows,
            .cols = qw.cols,
            .block_size = qw.block_size,
        }) catch return null;
        n_ref_qweights += 1;
    }

    // Cache mtl_buffer_contents pointers — stable for shared-memory buffers.
    const ref_buffers = alloc.alloc(reference.Buffer, program.n_buffers) catch return null;
    errdefer alloc.free(ref_buffers);
    for (ref_buffers, device_bufs) |*rb, buf| {
        rb.* = .{ .ptr = @ptrCast(@alignCast(c.mtl_buffer_contents(buf.ptr))), .len = buf.size / @sizeOf(f32) };
    }

    const schedule = program_mod.buildKernelSchedule(
        alloc,
        program.ops,
        metalSchedulePolicy(self.fine_grained_program_dispatch),
    ) catch return null;
    errdefer if (schedule.len > 0) alloc.free(schedule);

    const region_schedule = buildMetalRegionSchedule(alloc, schedule) catch &.{};
    errdefer if (region_schedule.len > 0) alloc.free(region_schedule);

    const compiled = alloc.create(CompiledProgram) catch return null;
    compiled.* = .{
        .backend = self,
        .device_bufs = device_bufs,
        .ref_buffers = ref_buffers,
        .qweight_views = qweight_views,
        .ref_qweights = ref_qweights,
        .ops = program.ops,
        .schedule = schedule,
        .region_schedule = region_schedule,
        .alloc = alloc,
    };
    return @ptrCast(compiled);
}

fn executeProgramFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.execute(inputs, outputs);
}

fn refreshProgramFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, ops: []const backend_mod.DeviceOp) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.rebuildSchedule(ops);
}

fn freeProgramFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.deinit();
}

fn getRuntimeProfileFn(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) ?*profile_mod.RuntimeProfile {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    return &compiled.runtime_profile;
}

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .compile_program = compileProgramFn,
    .refresh_program = refreshProgramFn,
    .execute_program = executeProgramFn,
    .free_program = freeProgramFn,
    .get_runtime_profile = getRuntimeProfileFn,
};

// ── Tests ─────────────────────────────────────────────────────────

test "metal backend compiled program matmul" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setFineGrainedProgramDispatch(true);
    const be = metal.backend();

    // Program: buf0(A) × buf1(B) → buf2(dst). 2x3 × 3x2 = 2x2.
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };
    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{ .M = 2, .N = 2, .K = 3, .a_row_stride = 3, .a_col_stride = 1, .b_row_stride = 2, .b_col_stride = 1, .a_offset = 0, .b_offset = 0, .dst_offset = 0, .dst_row_stride = 2 },
    } }};
    const buf_sizes = [_]usize{ 6, 6, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&a_data), .size = 6 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&b_data), .size = 6 * 4 },
    };
    const program = backend_mod.DeviceProgram{ .ops = &ops, .n_buffers = 3, .buffer_sizes = &buf_sizes, .initial_uploads = &uploads };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var dst: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&dst), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, &dst);
}

test "metal backend compiled program qmatvec" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setFineGrainedProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 10, 20, 30 };
    const qdata = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const scales = [_]f32{1};
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 3, .cols = 2, .block_size = 6 }};
    const ops = [_]backend_mod.DeviceOp{.{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 1,
        .N = 2,
        .K = 3,
    } }};
    const buf_sizes = [_]usize{ 3, 2 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = 3 * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 2,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var dst: [2]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 1, .host_ptr = @ptrCast(&dst), .size = 2 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectApproxEqAbs(@as(f32, 220), dst[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 280), dst[1], 1e-4);
    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 1), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 1), rt.backend_dispatch_count);
}

test "metal backend region batches independent qmatvecs" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4 };
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 4, .cols = 4, .block_size = 4 }};

    var ops: [7]backend_mod.DeviceOp = undefined;
    for (&ops, 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 1),
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 4,
        } };
    }
    const buf_sizes = [_]usize{ 4, 4, 4, 4, 4, 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = 4 * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 8,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [7][4]f32 = undefined;
    var outs: [7]backend_mod.ProgramIO = undefined;
    for (&outs, 0..) |*out, i| {
        out.* = .{ .buf_idx = @intCast(i + 1), .host_ptr = @ptrCast(&got[i]), .size = 4 * 4 };
    }
    be.executeProgram(handle, &.{}, &outs);

    for (got) |row| {
        try std.testing.expectEqualSlices(f32, &input, &row);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 7), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 2), rt.backend_dispatch_count);
}

test "metal backend region batches independent qmatmuls" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    const scales = [_]f32{ 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 3, .cols = 4, .block_size = 4 }};

    var ops: [7]backend_mod.DeviceOp = undefined;
    for (&ops, 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 1),
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 3,
        } };
    }
    const buf_sizes = [_]usize{ 6, 8, 8, 8, 8, 8, 8, 8 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 8,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [7][8]f32 = undefined;
    var outs: [7]backend_mod.ProgramIO = undefined;
    for (&outs, 0..) |*out, i| {
        out.* = .{ .buf_idx = @intCast(i + 1), .host_ptr = @ptrCast(&got[i]), .size = 8 * 4 };
    }
    be.executeProgram(handle, &.{}, &outs);

    const expected = [_]f32{ 1, 2, 3, 0, 4, 5, 6, 0 };
    for (got) |row| {
        try std.testing.expectEqualSlices(f32, &expected, &row);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 7), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 2), rt.backend_dispatch_count);
}

test "metal backend qmatmul batch carries cache-store sidecars" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var cache = [_]f32{0} ** 12;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    const scales = [_]f32{ 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 3, .cols = 4, .block_size = 4 }};

    var ops: [8]backend_mod.DeviceOp = undefined;
    for (ops[0..7], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 1),
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 3,
        } };
    }
    ops[7] = .{ .slice_assign = .{
        .dst = 8,
        .src = 7,
        .rows = 4,
        .cols = 2,
        .dst_base_offset = 0,
        .dst_offset = 4,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 4,
        .patch_stride = 4,
    } };

    const buf_sizes = [_]usize{ 6, 8, 8, 8, 8, 8, 8, 8, 12 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&cache), .size = cache.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 9,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [12]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 8, .host_ptr = @ptrCast(&got), .size = got.len * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 0 }, got[4..8]);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6, 0 }, got[8..12]);

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 8), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 2), rt.backend_dispatch_count);
}

test "metal backend region batches independent elementwise ops" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var q_input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var lhs = [_]f32{ 1, 2, 3, 4 };
    var rhs = [_]f32{ 10, 20, 30, 40 };
    var zeros = [_]f32{0} ** 4;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    const scales = [_]f32{ 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 3, .cols = 4, .block_size = 4 }};

    var ops: [15]backend_mod.DeviceOp = undefined;
    for (ops[0..7], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 1),
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 3,
        } };
    }
    for (ops[7..], 0..) |*op, i| {
        op.* = .{ .elementwise = .{
            .op = .add,
            .dst = @intCast(10 + i),
            .src0 = 8,
            .src1 = 9,
            .n = 4,
        } };
    }

    const buf_sizes = [_]usize{ 6, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&q_input), .size = q_input.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&lhs), .size = lhs.len * 4 },
        .{ .buf_idx = 9, .host_ptr = @ptrCast(&rhs), .size = rhs.len * 4 },
        .{ .buf_idx = 10, .host_ptr = @ptrCast(&zeros), .size = zeros.len * 4 },
        .{ .buf_idx = 11, .host_ptr = @ptrCast(&zeros), .size = zeros.len * 4 },
        .{ .buf_idx = 12, .host_ptr = @ptrCast(&zeros), .size = zeros.len * 4 },
        .{ .buf_idx = 13, .host_ptr = @ptrCast(&zeros), .size = zeros.len * 4 },
        .{ .buf_idx = 14, .host_ptr = @ptrCast(&zeros), .size = zeros.len * 4 },
        .{ .buf_idx = 15, .host_ptr = @ptrCast(&zeros), .size = zeros.len * 4 },
        .{ .buf_idx = 16, .host_ptr = @ptrCast(&zeros), .size = zeros.len * 4 },
        .{ .buf_idx = 17, .host_ptr = @ptrCast(&zeros), .size = zeros.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 18,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [8][4]f32 = undefined;
    var outs: [8]backend_mod.ProgramIO = undefined;
    for (&outs, 0..) |*out, i| {
        out.* = .{ .buf_idx = @intCast(10 + i), .host_ptr = @ptrCast(&got[i]), .size = 4 * 4 };
    }
    be.executeProgram(handle, &.{}, &outs);

    const expected = [_]f32{ 11, 22, 33, 44 };
    for (got) |row| {
        try std.testing.expectEqualSlices(f32, &expected, &row);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 15), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 3), rt.backend_dispatch_count);
}

test "metal backend region fuses qmatmul cache-store sidecar" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var q_out = [_]f32{0} ** 8;
    var cache = [_]f32{0} ** 12;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    const scales = [_]f32{ 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 3, .cols = 4, .block_size = 4 }};

    var ops: [8]backend_mod.DeviceOp = undefined;
    for (ops[0..4], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 1),
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 3,
        } };
    }
    for (ops[4..7]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 5,
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 3,
        } };
    }
    ops[7] = .{ .slice_assign = .{
        .dst = 6,
        .src = 5,
        .rows = 4,
        .cols = 2,
        .dst_base_offset = 0,
        .dst_offset = 4,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 4,
        .patch_stride = 4,
    } };

    const buf_sizes = [_]usize{ 6, 8, 8, 8, 8, 8, 12 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&q_out), .size = q_out.len * 4 },
        .{ .buf_idx = 6, .host_ptr = @ptrCast(&cache), .size = cache.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 7,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [12]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 6, .host_ptr = @ptrCast(&got), .size = got.len * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 0 }, got[4..8]);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6, 0 }, got[8..12]);

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 8), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 4), rt.backend_dispatch_count);
}

test "metal backend region fuses qmatmul elementwise sidecar" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var q_out = [_]f32{0} ** 8;
    var residual = [_]f32{ 10, 10, 10, 10, 10, 10, 10, 10 };
    var fused_out = [_]f32{0} ** 8;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    const scales = [_]f32{ 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 3, .cols = 4, .block_size = 4 }};

    var ops: [8]backend_mod.DeviceOp = undefined;
    for (ops[0..4], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 1),
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 3,
        } };
    }
    for (ops[4..7]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 5,
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 3,
        } };
    }
    ops[7] = .{ .elementwise = .{
        .op = .add,
        .dst = 7,
        .src0 = 5,
        .src1 = 6,
        .n = 8,
    } };

    const buf_sizes = [_]usize{ 6, 8, 8, 8, 8, 8, 8, 8 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&q_out), .size = q_out.len * 4 },
        .{ .buf_idx = 6, .host_ptr = @ptrCast(&residual), .size = residual.len * 4 },
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&fused_out), .size = fused_out.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 8,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [8]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 7, .host_ptr = @ptrCast(&got), .size = got.len * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 11, 12, 13, 10, 14, 15, 16, 10 }, &got);

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 8), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 4), rt.backend_dispatch_count);
}

test "metal backend region fuses qmatmul fused-elementwise sidecar" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var q_out = [_]f32{0} ** 8;
    var scale = [_]f32{ 2, 2, 2, 2, 2, 2, 2, 2 };
    var fused_out = [_]f32{0} ** 8;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    const scales = [_]f32{ 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 3, .cols = 4, .block_size = 4 }};
    const fused_steps = [_]backend_mod.FusedEwStep{.{ .op = .mul, .is_swapped = false, .secondary_buf = 6, .secondary_offset = 0 }};

    var ops: [8]backend_mod.DeviceOp = undefined;
    for (ops[0..4], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 1),
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 3,
        } };
    }
    for (ops[4..7]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 5,
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 3,
        } };
    }
    ops[7] = .{ .fused_elementwise = .{
        .steps = &fused_steps,
        .n = 8,
        .dst = 7,
        .src = 5,
        .dst_offset = 0,
        .src_offset = 0,
    } };

    const buf_sizes = [_]usize{ 6, 8, 8, 8, 8, 8, 8, 8 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&q_out), .size = q_out.len * 4 },
        .{ .buf_idx = 6, .host_ptr = @ptrCast(&scale), .size = scale.len * 4 },
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&fused_out), .size = fused_out.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 8,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [8]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 7, .host_ptr = @ptrCast(&got), .size = got.len * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 0, 8, 10, 12, 0 }, &got);

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 8), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 4), rt.backend_dispatch_count);
}

test "metal backend region fuses cache-store sidecars" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4 };
    var cos_sin = [_]f32{ 1, 0, 0, 1 };
    var rope_tmp = [_]f32{0} ** 4;
    var q_cache = [_]f32{0} ** 12;
    var q_out = [_]f32{0} ** 4;
    var rope_cache = [_]f32{0} ** 12;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 4, .cols = 4, .block_size = 4 }};

    var ops: [10]backend_mod.DeviceOp = undefined;
    for (ops[0..7]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 4,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 4,
        } };
    }
    ops[7] = .{ .slice_assign = .{
        .dst = 3,
        .src = 4,
        .rows = 4,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 4,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 4,
        .patch_stride = 4,
    } };
    ops[8] = .{ .rope = .{
        .dst = 2,
        .src = 0,
        .cos_sin = 1,
        .half_d = 2,
        .seq_len = 1,
        .src_off = 0,
        .cs_off = 0,
        .dst_off = 0,
        .src_rs = 1,
        .src_cs = 4,
        .cs_cs = 4,
    } };
    ops[9] = .{ .slice_assign = .{
        .dst = 5,
        .src = 2,
        .rows = 4,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 4,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 4,
        .patch_stride = 4,
    } };

    const buf_sizes = [_]usize{ 4, 4, 4, 12, 4, 12 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = 4 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&cos_sin), .size = 4 * 4 },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&rope_tmp), .size = 4 * 4 },
        .{ .buf_idx = 3, .host_ptr = @ptrCast(&q_cache), .size = 12 * 4 },
        .{ .buf_idx = 4, .host_ptr = @ptrCast(&q_out), .size = 4 * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&rope_cache), .size = 12 * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 6,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var q_cache_out: [12]f32 = undefined;
    var rope_cache_out: [12]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 3, .host_ptr = @ptrCast(&q_cache_out), .size = 12 * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&rope_cache_out), .size = 12 * 4 },
    };
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, q_cache_out[4..8]);
    try std.testing.expectEqualSlices(f32, &.{ 1, -4, 3, 2 }, rope_cache_out[4..8]);

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 10), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 8), rt.backend_dispatch_count);
}

test "metal backend region fuses rmsnorm scale chain" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4 };
    var q_out = [_]f32{0} ** 4;
    var scale = [_]f32{ 2, 3, 4, 5 };
    var norm_out = [_]f32{0} ** 4;
    var repeat_out = [_]f32{0} ** 4;
    var scaled_out = [_]f32{0} ** 4;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 4, .cols = 4, .block_size = 4 }};

    var ops: [10]backend_mod.DeviceOp = undefined;
    for (ops[0..7]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 4,
        } };
    }
    ops[7] = .{ .rmsnorm = .{
        .dst = 3,
        .src = 0,
        .rows = 1,
        .cols = 4,
        .eps = 0,
    } };
    ops[8] = .{ .repeat = .{
        .dst = 4,
        .src = 2,
        .n = 4,
        .src_ne = .{ 4, 1, 1, 1 },
        .dst_ne = .{ 4, 1, 1, 1 },
        .src_strides = .{ 1, 4, 4, 4 },
        .dst_strides = .{ 1, 4, 4, 4 },
    } };
    ops[9] = .{ .elementwise = .{
        .op = .mul,
        .dst = 5,
        .src0 = 3,
        .src1 = 4,
        .n = 4,
    } };

    const buf_sizes = [_]usize{ 4, 4, 4, 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = 4 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&q_out), .size = 4 * 4 },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&scale), .size = 4 * 4 },
        .{ .buf_idx = 3, .host_ptr = @ptrCast(&norm_out), .size = 4 * 4 },
        .{ .buf_idx = 4, .host_ptr = @ptrCast(&repeat_out), .size = 4 * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&scaled_out), .size = 4 * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 6,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 5, .host_ptr = @ptrCast(&got), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    const inv_rms: f32 = 1.0 / @sqrt(@as(f32, 30.0 / 4.0));
    const expected = [_]f32{
        1 * inv_rms * 2,
        2 * inv_rms * 3,
        3 * inv_rms * 4,
        4 * inv_rms * 5,
    };
    for (expected, got) |want, actual| {
        try std.testing.expectApproxEqAbs(want, actual, 1e-4);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 10), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 8), rt.backend_dispatch_count);
}

test "metal backend region batches attention heads" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var q_input = [_]f32{ 1, 0, 0, 1 };
    var q_out = [_]f32{0} ** 4;
    var k_cache = [_]f32{
        1, 0, 0, 1,
        1, 0, 0, 1,
    };
    var v_cache = [_]f32{
        10, 20, 30, 40,
        50, 60, 70, 80,
    };
    var mask = [_]f32{ 0, 0 };
    var dst = [_]f32{0} ** 4;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 4, .cols = 4, .block_size = 4 }};

    var ops: [9]backend_mod.DeviceOp = undefined;
    for (ops[0..7]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 4,
        } };
    }
    ops[7] = .{ .attention = .{
        .dst = 5,
        .q = 0,
        .k = 2,
        .v = 3,
        .mask = 4,
        .has_mask = true,
        .d_head = 2,
        .seq_q = 1,
        .seq_kv = 2,
        .scale = 1,
        .q_off = 0,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = 0,
        .q_rs = 1,
        .q_cs = 2,
        .k_rs = 1,
        .k_cs = 2,
        .v_rs = 1,
        .v_cs = 2,
        .mask_rs = 1,
        .mask_cs = 2,
        .dst_rs = 1,
        .dst_cs = 2,
    } };
    ops[8] = ops[7];
    ops[8].attention.q_off = 2;
    ops[8].attention.k_off = 4;
    ops[8].attention.v_off = 4;
    ops[8].attention.dst_off = 2;

    const buf_sizes = [_]usize{ 4, 4, 8, 8, 2, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&q_input), .size = 4 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&q_out), .size = 4 * 4 },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&k_cache), .size = 8 * 4 },
        .{ .buf_idx = 3, .host_ptr = @ptrCast(&v_cache), .size = 8 * 4 },
        .{ .buf_idx = 4, .host_ptr = @ptrCast(&mask), .size = 2 * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&dst), .size = 4 * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 6,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 5, .host_ptr = @ptrCast(&got), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    const hi: f32 = @exp(@as(f32, 1.0)) / (@exp(@as(f32, 1.0)) + 1.0);
    const lo: f32 = 1.0 / (@exp(@as(f32, 1.0)) + 1.0);
    const expected = [_]f32{
        hi * 10 + lo * 30,
        hi * 20 + lo * 40,
        lo * 50 + hi * 70,
        lo * 60 + hi * 80,
    };
    for (expected, got) |want, actual| {
        try std.testing.expectApproxEqAbs(want, actual, 1e-4);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 9), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 8), rt.backend_dispatch_count);
}

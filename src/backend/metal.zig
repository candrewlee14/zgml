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
const DeviceOpTag = std.meta.Tag(backend_mod.DeviceOp);

fn nowNs() i96 {
    return std.Io.Clock.awake.now(std.Io.Threaded.global_single_threaded.io()).nanoseconds;
}

fn DeviceOpPayload(comptime tag: DeviceOpTag) type {
    inline for (@typeInfo(backend_mod.DeviceOp).@"union".fields) |field| {
        if (comptime std.mem.eql(u8, field.name, @tagName(tag))) return field.type;
    }
    @compileError("unknown DeviceOp tag: " ++ @tagName(tag));
}

fn deviceOpAs(comptime tag: DeviceOpTag, op: backend_mod.DeviceOp) ?DeviceOpPayload(tag) {
    return switch (op) {
        tag => |payload| payload,
        else => null,
    };
}

fn deviceOpAt(comptime tag: DeviceOpTag, ops: []const backend_mod.DeviceOp, idx: usize) ?DeviceOpPayload(tag) {
    if (idx >= ops.len) return null;
    return deviceOpAs(tag, ops[idx]);
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
    \\constant uint MAX_QMATMUL_BATCH_SIDECARS = 8;
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
    \\    uint write_primary[MAX_QMATMUL_BATCH];
    \\    uint sidecar_count[MAX_QMATMUL_BATCH];
    \\    uint sidecar_kind[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\    uint slice_rows[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\    uint slice_cols[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\    uint slice_src_col_start[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\    uint slice_dst_offset[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\    uint slice_dst_row_stride[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\    uint slice_dst_col_stride[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\    uint ew_op[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\    uint ew_is_swapped[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\    uint ew_dst_offset[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\    uint ew_secondary_offset[MAX_QMATMUL_BATCH * MAX_QMATMUL_BATCH_SIDECARS];
    \\};
    \\
    \\kernel void qmatmul_batch4_f32(
    \\    device const char*  w0 [[buffer(0)]],
    \\    device const float* s0 [[buffer(1)]],
    \\    device const float* i0 [[buffer(2)]],
    \\    device float*       o0 [[buffer(3)]],
    \\    device float*       side0 [[buffer(4)]],
    \\    device const float* sec0 [[buffer(5)]],
    \\    device const char*  w1 [[buffer(6)]],
    \\    device const float* s1 [[buffer(7)]],
    \\    device const float* i1 [[buffer(8)]],
    \\    device float*       o1 [[buffer(9)]],
    \\    device float*       side1 [[buffer(10)]],
    \\    device const float* sec1 [[buffer(11)]],
    \\    device const char*  w2 [[buffer(12)]],
    \\    device const float* s2 [[buffer(13)]],
    \\    device const float* i2 [[buffer(14)]],
    \\    device float*       o2 [[buffer(15)]],
    \\    device float*       side2 [[buffer(16)]],
    \\    device const float* sec2 [[buffer(17)]],
    \\    device const char*  w3 [[buffer(18)]],
    \\    device const float* s3 [[buffer(19)]],
    \\    device const float* i3 [[buffer(20)]],
    \\    device float*       o3 [[buffer(21)]],
    \\    device float*       side3 [[buffer(22)]],
    \\    device const float* sec3 [[buffer(23)]],
    \\    constant QMatmulBatch4Params& p [[buffer(24)]],
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
    \\    device float* sidecar_dst = side0;
    \\    device const float* secondary = sec0;
    \\    switch (slot) {
    \\        case 1: w = w1; s = s1; input = i1; output = o1; sidecar_dst = side1; secondary = sec1; break;
    \\        case 2: w = w2; s = s2; input = i2; output = o2; sidecar_dst = side2; secondary = sec2; break;
    \\        case 3: w = w3; s = s3; input = i3; output = o3; sidecar_dst = side3; secondary = sec3; break;
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
    \\            if (p.write_primary[slot] != 0) output[p.dst_offset[slot] + cr * p.dst_row_stride[slot] + cc] = val;
    \\            for (uint sid = 0; sid < p.sidecar_count[slot]; sid++) {
    \\                uint si = slot * MAX_QMATMUL_BATCH_SIDECARS + sid;
    \\                uint slice_col_start = p.slice_src_col_start[si];
    \\                if (p.sidecar_kind[si] == 1 && cc >= slice_col_start && cc < slice_col_start + p.slice_rows[si] && cr < p.slice_cols[si]) {
    \\                    uint slice_row = cc - slice_col_start;
    \\                    sidecar_dst[p.slice_dst_offset[si] + slice_row * p.slice_dst_row_stride[si] + cr * p.slice_dst_col_stride[si]] = val;
    \\                } else if (p.sidecar_kind[si] == 2) {
    \\                    uint linear = cr * N + cc;
    \\                    float other = secondary[p.ew_secondary_offset[si] + linear];
    \\                    float ew = val;
    \\                    if (p.ew_op[si] == 7) ew = (p.ew_is_swapped[si] != 0) ? other + val : val + other;
    \\                    else if (p.ew_op[si] == 8) ew = (p.ew_is_swapped[si] != 0) ? other * val : val * other;
    \\                    sidecar_dst[p.ew_dst_offset[si] + linear] = ew;
    \\                }
    \\            }
    \\        }
    \\    }
    \\}
    \\
    \\constant uint MAX_QMATMUL_ROPE_STORE_BATCH = 4;
    \\
    \\struct QMatmulRopeStoreBatch4Params {
    \\    uint n_ops; uint max_tiles_y;
    \\    uint M[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint N[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint K[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint block_size[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint input_offset[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint input_row_stride[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint dst_offset[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint dst_row_stride[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint write_primary[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint rope_half_d[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint rope_src_col_start[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint rope_cs_off[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint rope_cs_cs[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint slice_dst_offset[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint slice_dst_row_stride[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\    uint slice_dst_col_stride[MAX_QMATMUL_ROPE_STORE_BATCH];
    \\};
    \\
    \\kernel void qmatmul_rope_store_batch4_f32(
    \\    device const char*  w0 [[buffer(0)]],
    \\    device const float* s0 [[buffer(1)]],
    \\    device const float* i0 [[buffer(2)]],
    \\    device float*       o0 [[buffer(3)]],
    \\    device const float* cs0 [[buffer(4)]],
    \\    device float*       dst0 [[buffer(5)]],
    \\    device const char*  w1 [[buffer(6)]],
    \\    device const float* s1 [[buffer(7)]],
    \\    device const float* i1 [[buffer(8)]],
    \\    device float*       o1 [[buffer(9)]],
    \\    device const float* cs1 [[buffer(10)]],
    \\    device float*       dst1 [[buffer(11)]],
    \\    device const char*  w2 [[buffer(12)]],
    \\    device const float* s2 [[buffer(13)]],
    \\    device const float* i2 [[buffer(14)]],
    \\    device float*       o2 [[buffer(15)]],
    \\    device const float* cs2 [[buffer(16)]],
    \\    device float*       dst2 [[buffer(17)]],
    \\    device const char*  w3 [[buffer(18)]],
    \\    device const float* s3 [[buffer(19)]],
    \\    device const float* i3 [[buffer(20)]],
    \\    device float*       o3 [[buffer(21)]],
    \\    device const float* cs3 [[buffer(22)]],
    \\    device float*       dst3 [[buffer(23)]],
    \\    constant QMatmulRopeStoreBatch4Params& p [[buffer(24)]],
    \\    uint2 group_id [[threadgroup_position_in_grid]],
    \\    uint  simd_idx [[simdgroup_index_in_threadgroup]],
    \\    uint  lane     [[thread_index_in_simdgroup]],
    \\    uint  tid      [[thread_index_in_threadgroup]]
    \\) {
    \\    uint slot = group_id.y / p.max_tiles_y;
    \\    uint row_tile = group_id.y - slot * p.max_tiles_y;
    \\    if (slot >= p.n_ops) return;
    \\
    \\    uint M = p.M[slot], N = p.N[slot], K = p.K[slot];
    \\    uint rope_half = p.rope_half_d[slot];
    \\    uint rope_d = rope_half * 2;
    \\    const uint local_col_base = group_id.x * TILE;
    \\    if (local_col_base >= rope_d) return;
    \\    bool lower_half_tile = local_col_base < rope_half;
    \\    if (!lower_half_tile && p.write_primary[slot] == 0) return;
    \\
    \\    device const char* w = w0;
    \\    device const float* s = s0;
    \\    device const float* input = i0;
    \\    device float* output = o0;
    \\    device const float* cos_sin = cs0;
    \\    device float* slice_dst = dst0;
    \\    switch (slot) {
    \\        case 1: w = w1; s = s1; input = i1; output = o1; cos_sin = cs1; slice_dst = dst1; break;
    \\        case 2: w = w2; s = s2; input = i2; output = o2; cos_sin = cs2; slice_dst = dst2; break;
    \\        case 3: w = w3; s = s3; input = i3; output = o3; cos_sin = cs3; slice_dst = dst3; break;
    \\        default: break;
    \\    }
    \\
    \\    const uint gRow = row_tile * TILE;
    \\    const uint sRow = (simd_idx / 2) * 16;
    \\    const uint sCol = (simd_idx % 2) * 16;
    \\    const uint gCol = p.rope_src_col_start[slot] + local_col_base;
    \\    const uint pairCol = gCol + rope_half;
    \\
    \\    simdgroup_float8x8 acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\    simdgroup_float8x8 pair_acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\
    \\    threadgroup float tI[TILE * 8];
    \\    threadgroup float tW[8 * TILE];
    \\    threadgroup float tWPair[8 * TILE];
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
    \\            uint pc = pairCol + c;
    \\            if (lower_half_tile && kr < K && pc < N) {
    \\                uint pair_w_idx = kr * N + pc;
    \\                tWPair[i] = float(w[pair_w_idx]) * s[pair_w_idx / p.block_size[slot]];
    \\            } else {
    \\                tWPair[i] = 0.0f;
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
    \\        if (lower_half_tile) {
    \\            simdgroup_float8x8 pb0, pb1;
    \\            simdgroup_load(pb0, tWPair + (sCol + 0), TILE);
    \\            simdgroup_load(pb1, tWPair + (sCol + 8), TILE);
    \\            simdgroup_multiply_accumulate(pair_acc[0], a0, pb0, pair_acc[0]);
    \\            simdgroup_multiply_accumulate(pair_acc[1], a0, pb1, pair_acc[1]);
    \\            simdgroup_multiply_accumulate(pair_acc[2], a1, pb0, pair_acc[2]);
    \\            simdgroup_multiply_accumulate(pair_acc[3], a1, pb1, pair_acc[3]);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    threadgroup float tC[TILE * TILE];
    \\    threadgroup float tCPair[TILE * TILE];
    \\    simdgroup_store(acc[0], tC + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[1], tC + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(acc[2], tC + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(acc[3], tC + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    if (lower_half_tile) {
    \\        simdgroup_store(pair_acc[0], tCPair + (sRow + 0) * TILE + sCol + 0, TILE);
    \\        simdgroup_store(pair_acc[1], tCPair + (sRow + 0) * TILE + sCol + 8, TILE);
    \\        simdgroup_store(pair_acc[2], tCPair + (sRow + 8) * TILE + sCol + 0, TILE);
    \\        simdgroup_store(pair_acc[3], tCPair + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, local_cc = local_col_base + c;
    \\        if (cr >= M || local_cc >= rope_d) continue;
    \\        uint cc = p.rope_src_col_start[slot] + local_cc;
    \\        float val = tC[i];
    \\        if (p.write_primary[slot] != 0 && cc < N) {
    \\            output[p.dst_offset[slot] + cr * p.dst_row_stride[slot] + cc] = val;
    \\        }
    \\        if (lower_half_tile && local_cc < rope_half) {
    \\            float pair = tCPair[i];
    \\            float cos_val = cos_sin[p.rope_cs_off[slot] + cr * p.rope_cs_cs[slot] + local_cc];
    \\            float sin_val = cos_sin[p.rope_cs_off[slot] + cr * p.rope_cs_cs[slot] + rope_half + local_cc];
    \\            float y_lo = val * cos_val - pair * sin_val;
    \\            float y_hi = pair * cos_val + val * sin_val;
    \\            uint dst_col = p.slice_dst_offset[slot] + cr * p.slice_dst_col_stride[slot];
    \\            slice_dst[dst_col + local_cc * p.slice_dst_row_stride[slot]] = y_lo;
    \\            slice_dst[dst_col + (local_cc + rope_half) * p.slice_dst_row_stride[slot]] = y_hi;
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
    \\    uint write_primary[MAX_QMATVEC_BATCH];
    \\    uint sidecar_kind[MAX_QMATVEC_BATCH];
    \\    uint slice_rows[MAX_QMATVEC_BATCH];
    \\    uint slice_src_col_start[MAX_QMATVEC_BATCH];
    \\    uint slice_dst_offset[MAX_QMATVEC_BATCH];
    \\    uint slice_dst_row_stride[MAX_QMATVEC_BATCH];
    \\    uint rope_half_d[MAX_QMATVEC_BATCH];
    \\    uint rope_cs_off[MAX_QMATVEC_BATCH];
    \\    uint ew_op[MAX_QMATVEC_BATCH];
    \\    uint ew_is_swapped[MAX_QMATVEC_BATCH];
    \\    uint ew_dst_offset[MAX_QMATVEC_BATCH];
    \\    uint ew_secondary_offset[MAX_QMATVEC_BATCH];
    \\};
    \\
    \\kernel void qmatvec_batch4_f32(
    \\    device const char*  w0 [[buffer(0)]],
    \\    device const float* s0 [[buffer(1)]],
    \\    device const float* i0 [[buffer(2)]],
    \\    device float*       o0 [[buffer(3)]],
    \\    device const float* aux0 [[buffer(4)]],
    \\    device float*       d0 [[buffer(5)]],
    \\    device const char*  w1 [[buffer(6)]],
    \\    device const float* s1 [[buffer(7)]],
    \\    device const float* i1 [[buffer(8)]],
    \\    device float*       o1 [[buffer(9)]],
    \\    device const float* aux1 [[buffer(10)]],
    \\    device float*       d1 [[buffer(11)]],
    \\    device const char*  w2 [[buffer(12)]],
    \\    device const float* s2 [[buffer(13)]],
    \\    device const float* i2 [[buffer(14)]],
    \\    device float*       o2 [[buffer(15)]],
    \\    device const float* aux2 [[buffer(16)]],
    \\    device float*       d2 [[buffer(17)]],
    \\    device const char*  w3 [[buffer(18)]],
    \\    device const float* s3 [[buffer(19)]],
    \\    device const float* i3 [[buffer(20)]],
    \\    device float*       o3 [[buffer(21)]],
    \\    device const float* aux3 [[buffer(22)]],
    \\    device float*       d3 [[buffer(23)]],
    \\    constant QMatvecBatch4Params& p [[buffer(24)]],
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
    \\    device const float* sidecar_src = aux0;
    \\    device float* sidecar_dst = d0;
    \\    switch (slot) {
    \\        case 1: w = w1; s = s1; input = i1; output = o1; sidecar_src = aux1; sidecar_dst = d1; break;
    \\        case 2: w = w2; s = s2; input = i2; output = o2; sidecar_src = aux2; sidecar_dst = d2; break;
    \\        case 3: w = w3; s = s3; input = i3; output = o3; sidecar_src = aux3; sidecar_dst = d3; break;
    \\        default: break;
    \\    }
    \\
    \\    float sum = 0.0f;
    \\    for (uint k = 0; k < p.K[slot]; k++) {
    \\        uint w_idx = k * p.N[slot] + col;
    \\        sum += input[p.input_offset[slot] + k] * float(w[w_idx]) * s[w_idx / p.block_size[slot]];
    \\    }
    \\    if (p.write_primary[slot] != 0) output[p.dst_offset[slot] + col] = sum;
    \\    uint slice_col_start = p.slice_src_col_start[slot];
    \\    if (p.sidecar_kind[slot] == 1 && col >= slice_col_start && col < slice_col_start + p.slice_rows[slot]) {
    \\        uint slice_row = col - slice_col_start;
    \\        sidecar_dst[p.slice_dst_offset[slot] + slice_row * p.slice_dst_row_stride[slot]] = sum;
    \\    } else if (p.sidecar_kind[slot] == 2) {
    \\        uint rope_half = p.rope_half_d[slot];
    \\        if (col >= slice_col_start && col < slice_col_start + rope_half) {
    \\            uint local = col - slice_col_start;
    \\            uint pair_col = col + rope_half;
    \\            float pair_sum = 0.0f;
    \\            for (uint k = 0; k < p.K[slot]; k++) {
    \\                uint w_idx = k * p.N[slot] + pair_col;
    \\                pair_sum += input[p.input_offset[slot] + k] * float(w[w_idx]) * s[w_idx / p.block_size[slot]];
    \\            }
    \\            float cos_val = sidecar_src[p.rope_cs_off[slot] + local];
    \\            float sin_val = sidecar_src[p.rope_cs_off[slot] + rope_half + local];
    \\            sidecar_dst[p.slice_dst_offset[slot] + local * p.slice_dst_row_stride[slot]] = sum * cos_val - pair_sum * sin_val;
    \\            sidecar_dst[p.slice_dst_offset[slot] + (local + rope_half) * p.slice_dst_row_stride[slot]] = pair_sum * cos_val + sum * sin_val;
    \\        }
    \\    } else if (p.sidecar_kind[slot] == 3) {
    \\        float other = sidecar_src[p.ew_secondary_offset[slot] + col];
    \\        float ew = sum;
    \\        if (p.ew_op[slot] == 7) ew = (p.ew_is_swapped[slot] != 0) ? other + sum : sum + other;
    \\        else if (p.ew_op[slot] == 8) ew = (p.ew_is_swapped[slot] != 0) ? other * sum : sum * other;
    \\        sidecar_dst[p.ew_dst_offset[slot] + col] = ew;
    \\    }
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
    \\struct RopeSliceAssignBatchParams {
    \\    uint n_ops; uint max_n;
    \\    uint half_d[MAX_ROPE_BATCH];
    \\    uint seq_len[MAX_ROPE_BATCH];
    \\    uint src_off[MAX_ROPE_BATCH];
    \\    uint cs_off[MAX_ROPE_BATCH];
    \\    uint src_rs[MAX_ROPE_BATCH];
    \\    uint src_cs[MAX_ROPE_BATCH];
    \\    uint cs_cs[MAX_ROPE_BATCH];
    \\    uint dst_offset[MAX_ROPE_BATCH];
    \\    uint dst_row_stride[MAX_ROPE_BATCH];
    \\    uint dst_col_stride[MAX_ROPE_BATCH];
    \\};
    \\
    \\kernel void rope_slice_assign_batch_f32(
    \\    device const float* src0    [[buffer(0)]],
    \\    device const float* src1    [[buffer(1)]],
    \\    device const float* src2    [[buffer(2)]],
    \\    device const float* src3    [[buffer(3)]],
    \\    device const float* src4    [[buffer(4)]],
    \\    device const float* src5    [[buffer(5)]],
    \\    device const float* src6    [[buffer(6)]],
    \\    device const float* src7    [[buffer(7)]],
    \\    device const float* src8    [[buffer(8)]],
    \\    device const float* src9    [[buffer(9)]],
    \\    device const float* src10   [[buffer(10)]],
    \\    device const float* src11   [[buffer(11)]],
    \\    device const float* src12   [[buffer(12)]],
    \\    device const float* src13   [[buffer(13)]],
    \\    device const float* src14   [[buffer(14)]],
    \\    device const float* src15   [[buffer(15)]],
    \\    device const float* cos_sin [[buffer(16)]],
    \\    device float*       dst     [[buffer(17)]],
    \\    constant RopeSliceAssignBatchParams& p [[buffer(18)]],
    \\    uint2 gid [[thread_position_in_grid]]
    \\) {
    \\    uint local = gid.x;
    \\    uint slot = gid.y;
    \\    if (slot >= p.n_ops || local >= p.half_d[slot] * p.seq_len[slot]) return;
    \\    uint col = local / p.half_d[slot];
    \\    uint i   = local % p.half_d[slot];
    \\    device const float* src = src0;
    \\    switch (slot) {
    \\        case 1: src = src1; break;
    \\        case 2: src = src2; break;
    \\        case 3: src = src3; break;
    \\        case 4: src = src4; break;
    \\        case 5: src = src5; break;
    \\        case 6: src = src6; break;
    \\        case 7: src = src7; break;
    \\        case 8: src = src8; break;
    \\        case 9: src = src9; break;
    \\        case 10: src = src10; break;
    \\        case 11: src = src11; break;
    \\        case 12: src = src12; break;
    \\        case 13: src = src13; break;
    \\        case 14: src = src14; break;
    \\        case 15: src = src15; break;
    \\        default: break;
    \\    }
    \\
    \\    float cos_val = cos_sin[p.cs_off[slot] + col * p.cs_cs[slot] + i];
    \\    float sin_val = cos_sin[p.cs_off[slot] + col * p.cs_cs[slot] + p.half_d[slot] + i];
    \\    float x_lo = src[p.src_off[slot] + col * p.src_cs[slot] + i * p.src_rs[slot]];
    \\    float x_hi = src[p.src_off[slot] + col * p.src_cs[slot] + (i + p.half_d[slot]) * p.src_rs[slot]];
    \\    float y_lo = x_lo * cos_val - x_hi * sin_val;
    \\    float y_hi = x_hi * cos_val + x_lo * sin_val;
    \\
    \\    uint dst_col = p.dst_offset[slot] + col * p.dst_col_stride[slot];
    \\    dst[dst_col + i * p.dst_row_stride[slot]] = y_lo;
    \\    dst[dst_col + (i + p.half_d[slot]) * p.dst_row_stride[slot]] = y_hi;
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
    \\    uint write_primary;
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
    \\            if (p.write_primary != 0) output[p.dst_offset + cr * p.dst_row_stride + cc] = val;
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
    \\    uint write_primary;
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
    \\            if (p.write_primary != 0) output[p.dst_offset + cr * p.dst_row_stride + cc] = val;
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
    \\    if (p.write_primary != 0) output[p.dst_offset + gid] = sum;
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
    \\constant uint MAX_ATTENTION_STORE_BATCH_HEADS = 4;
    \\constant uint MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS = 16;
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
    \\struct AttentionSliceAssignParams {
    \\    uint d_head; uint seq_q; uint seq_kv;
    \\    float scale;
    \\    uint q_off; uint k_off; uint v_off; uint mask_off; uint dst_off;
    \\    uint q_rs; uint q_cs; uint k_rs; uint k_cs; uint v_rs; uint v_cs;
    \\    uint mask_rs; uint mask_cs; uint dst_rs; uint dst_cs;
    \\    uint slice_rows; uint slice_cols;
    \\    uint slice_src_offset; uint slice_src_row_stride; uint slice_src_col_stride;
    \\    uint slice_dst_offset; uint slice_dst_row_stride; uint slice_dst_col_stride;
    \\};
    \\
    \\kernel void attention_slice_assign_f32(
    \\    device const float* Q         [[buffer(0)]],
    \\    device const float* K         [[buffer(1)]],
    \\    device const float* V         [[buffer(2)]],
    \\    device const float* mask      [[buffer(3)]],
    \\    device float*       dst       [[buffer(4)]],
    \\    device const float* slice_src [[buffer(5)]],
    \\    device float*       slice_dst [[buffer(6)]],
    \\    constant AttentionSliceAssignParams& p [[buffer(7)]],
    \\    uint q_col   [[threadgroup_position_in_grid]],
    \\    uint tid     [[thread_index_in_threadgroup]]
    \\) {
    \\    if (q_col >= p.seq_q) return;
    \\    const uint tg_size = 256;
    \\
    \\    if (q_col == 0) {
    \\        uint copy_n = p.slice_rows * p.slice_cols;
    \\        for (uint i = tid; i < copy_n; i += tg_size) {
    \\            uint row = i % p.slice_rows;
    \\            uint col = i / p.slice_rows;
    \\            slice_dst[p.slice_dst_offset + row * p.slice_dst_row_stride + col * p.slice_dst_col_stride] =
    \\                slice_src[p.slice_src_offset + row * p.slice_src_row_stride + col * p.slice_src_col_stride];
    \\        }
    \\    }
    \\
    \\    threadgroup float scores[MAX_SEQ];
    \\    threadgroup float scratch[256];
    \\
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
    \\            val += scores[s] * V[p.v_off + r * p.v_rs + s * p.v_cs];
    \\        dst[p.dst_off + r * p.dst_rs + q_col * p.dst_cs] = val;
    \\    }
    \\}
    \\
    \\struct AttentionStoreParams {
    \\    uint d_head; uint seq_q; uint seq_kv;
    \\    float scale;
    \\    uint q_off; uint k_off; uint v_off; uint mask_off; uint dst_off;
    \\    uint q_rs; uint q_cs; uint k_rs; uint k_cs; uint v_rs; uint v_cs;
    \\    uint mask_rs; uint mask_cs; uint dst_rs; uint dst_cs;
    \\    uint slice_dst_offset; uint slice_dst_row_stride; uint slice_dst_col_stride;
    \\};
    \\
    \\kernel void attention_store_f32(
    \\    device const float* Q         [[buffer(0)]],
    \\    device const float* K         [[buffer(1)]],
    \\    device const float* V         [[buffer(2)]],
    \\    device const float* mask      [[buffer(3)]],
    \\    device float*       dst       [[buffer(4)]],
    \\    device float*       slice_dst [[buffer(5)]],
    \\    constant AttentionStoreParams& p [[buffer(6)]],
    \\    uint q_col   [[threadgroup_position_in_grid]],
    \\    uint tid     [[thread_index_in_threadgroup]]
    \\) {
    \\    if (q_col >= p.seq_q) return;
    \\    const uint tg_size = 256;
    \\
    \\    threadgroup float scores[MAX_SEQ];
    \\    threadgroup float scratch[256];
    \\
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
    \\            val += scores[s] * V[p.v_off + r * p.v_rs + s * p.v_cs];
    \\        dst[p.dst_off + r * p.dst_rs + q_col * p.dst_cs] = val;
    \\        slice_dst[p.slice_dst_offset + r * p.slice_dst_row_stride + q_col * p.slice_dst_col_stride] = val;
    \\    }
    \\}
    \\
    \\struct AttentionRopeStoreParams {
    \\    uint d_head; uint seq_q; uint seq_kv;
    \\    float scale;
    \\    uint rope_half_d; uint rope_src_off; uint rope_cs_off;
    \\    uint rope_src_rs; uint rope_src_cs; uint rope_cs_cs;
    \\    uint k_off; uint v_off; uint mask_off; uint dst_off;
    \\    uint k_rs; uint k_cs; uint v_rs; uint v_cs;
    \\    uint mask_rs; uint mask_cs; uint dst_rs; uint dst_cs;
    \\    uint slice_dst_offset; uint slice_dst_row_stride; uint slice_dst_col_stride;
    \\};
    \\
    \\inline float rope_q_value(
    \\    device const float* q_src,
    \\    device const float* cos_sin,
    \\    constant AttentionRopeStoreParams& p,
    \\    uint r,
    \\    uint q_col
    \\) {
    \\    uint i = r;
    \\    bool high = false;
    \\    if (i >= p.rope_half_d) {
    \\        i -= p.rope_half_d;
    \\        high = true;
    \\    }
    \\    float cos_val = cos_sin[p.rope_cs_off + q_col * p.rope_cs_cs + i];
    \\    float sin_val = cos_sin[p.rope_cs_off + q_col * p.rope_cs_cs + p.rope_half_d + i];
    \\    float x_lo = q_src[p.rope_src_off + q_col * p.rope_src_cs + i * p.rope_src_rs];
    \\    float x_hi = q_src[p.rope_src_off + q_col * p.rope_src_cs + (i + p.rope_half_d) * p.rope_src_rs];
    \\    return high ? (x_hi * cos_val + x_lo * sin_val) : (x_lo * cos_val - x_hi * sin_val);
    \\}
    \\
    \\kernel void attention_rope_store_f32(
    \\    device const float* q_src     [[buffer(0)]],
    \\    device const float* cos_sin   [[buffer(1)]],
    \\    device const float* K         [[buffer(2)]],
    \\    device const float* V         [[buffer(3)]],
    \\    device const float* mask      [[buffer(4)]],
    \\    device float*       dst       [[buffer(5)]],
    \\    device float*       slice_dst [[buffer(6)]],
    \\    constant AttentionRopeStoreParams& p [[buffer(7)]],
    \\    uint q_col   [[threadgroup_position_in_grid]],
    \\    uint tid     [[thread_index_in_threadgroup]]
    \\) {
    \\    if (q_col >= p.seq_q) return;
    \\    const uint tg_size = 256;
    \\
    \\    threadgroup float scores[MAX_SEQ];
    \\    threadgroup float scratch[256];
    \\
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size) {
    \\        float mv = mask[p.mask_off + s * p.mask_rs + q_col * p.mask_cs];
    \\        if (!isfinite(mv)) { scores[s] = -INFINITY; continue; }
    \\        float dot = 0.0f;
    \\        for (uint r = 0; r < p.d_head; r++)
    \\            dot += rope_q_value(q_src, cos_sin, p, r, q_col) * K[p.k_off + r * p.k_rs + s * p.k_cs];
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
    \\            val += scores[s] * V[p.v_off + r * p.v_rs + s * p.v_cs];
    \\        dst[p.dst_off + r * p.dst_rs + q_col * p.dst_cs] = val;
    \\        slice_dst[p.slice_dst_offset + r * p.slice_dst_row_stride + q_col * p.slice_dst_col_stride] = val;
    \\    }
    \\}
    \\
    \\struct AttentionRopeStoreBatchParams {
    \\    uint n_heads; uint d_head; uint seq_q; uint seq_kv;
    \\    float scale;
    \\    uint rope_half_d;
    \\    uint rope_src_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint rope_cs_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint k_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint v_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint mask_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint dst_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint slice_dst_offset[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint rope_src_rs; uint rope_src_cs; uint rope_cs_cs;
    \\    uint k_rs; uint k_cs; uint v_rs; uint v_cs;
    \\    uint mask_rs; uint mask_cs; uint dst_rs; uint dst_cs;
    \\    uint slice_dst_row_stride; uint slice_dst_col_stride;
    \\};
    \\
    \\inline float rope_batch_q_value(
    \\    device const float* q_src,
    \\    device const float* cos_sin,
    \\    constant AttentionRopeStoreBatchParams& p,
    \\    uint head,
    \\    uint r,
    \\    uint q_col
    \\) {
    \\    uint i = r;
    \\    bool high = false;
    \\    if (i >= p.rope_half_d) {
    \\        i -= p.rope_half_d;
    \\        high = true;
    \\    }
    \\    float cos_val = cos_sin[p.rope_cs_off[head] + q_col * p.rope_cs_cs + i];
    \\    float sin_val = cos_sin[p.rope_cs_off[head] + q_col * p.rope_cs_cs + p.rope_half_d + i];
    \\    float x_lo = q_src[p.rope_src_off[head] + q_col * p.rope_src_cs + i * p.rope_src_rs];
    \\    float x_hi = q_src[p.rope_src_off[head] + q_col * p.rope_src_cs + (i + p.rope_half_d) * p.rope_src_rs];
    \\    return high ? (x_hi * cos_val + x_lo * sin_val) : (x_lo * cos_val - x_hi * sin_val);
    \\}
    \\
    \\kernel void attention_rope_store_batch_f32(
    \\    device const float* q_src0     [[buffer(0)]],
    \\    device const float* q_src1     [[buffer(1)]],
    \\    device const float* q_src2     [[buffer(2)]],
    \\    device const float* q_src3     [[buffer(3)]],
    \\    device const float* cos_sin0   [[buffer(4)]],
    \\    device const float* cos_sin1   [[buffer(5)]],
    \\    device const float* cos_sin2   [[buffer(6)]],
    \\    device const float* cos_sin3   [[buffer(7)]],
    \\    device const float* K0         [[buffer(8)]],
    \\    device const float* K1         [[buffer(9)]],
    \\    device const float* K2         [[buffer(10)]],
    \\    device const float* K3         [[buffer(11)]],
    \\    device const float* V0         [[buffer(12)]],
    \\    device const float* V1         [[buffer(13)]],
    \\    device const float* V2         [[buffer(14)]],
    \\    device const float* V3         [[buffer(15)]],
    \\    device const float* mask0      [[buffer(16)]],
    \\    device const float* mask1      [[buffer(17)]],
    \\    device const float* mask2      [[buffer(18)]],
    \\    device const float* mask3      [[buffer(19)]],
    \\    device float*       dst0       [[buffer(20)]],
    \\    device float*       dst1       [[buffer(21)]],
    \\    device float*       dst2       [[buffer(22)]],
    \\    device float*       dst3       [[buffer(23)]],
    \\    device float*       slice_dst0 [[buffer(24)]],
    \\    device float*       slice_dst1 [[buffer(25)]],
    \\    device float*       slice_dst2 [[buffer(26)]],
    \\    device float*       slice_dst3 [[buffer(27)]],
    \\    constant AttentionRopeStoreBatchParams& p [[buffer(28)]],
    \\    uint2 group [[threadgroup_position_in_grid]],
    \\    uint tid     [[thread_index_in_threadgroup]]
    \\) {
    \\    uint q_col = group.x;
    \\    uint head = group.y;
    \\    if (q_col >= p.seq_q) return;
    \\    if (head >= p.n_heads) return;
    \\    const uint tg_size = 256;
    \\
    \\    device const float* q_src = q_src0;
    \\    device const float* cos_sin = cos_sin0;
    \\    device const float* K = K0;
    \\    device const float* V = V0;
    \\    device const float* mask = mask0;
    \\    device float* dst = dst0;
    \\    device float* slice_dst = slice_dst0;
    \\    if (head == 1) {
    \\        q_src = q_src1; cos_sin = cos_sin1; K = K1; V = V1; mask = mask1; dst = dst1; slice_dst = slice_dst1;
    \\    } else if (head == 2) {
    \\        q_src = q_src2; cos_sin = cos_sin2; K = K2; V = V2; mask = mask2; dst = dst2; slice_dst = slice_dst2;
    \\    } else if (head == 3) {
    \\        q_src = q_src3; cos_sin = cos_sin3; K = K3; V = V3; mask = mask3; dst = dst3; slice_dst = slice_dst3;
    \\    }
    \\
    \\    threadgroup float scores[MAX_SEQ];
    \\    threadgroup float scratch[256];
    \\
    \\    uint k_off = p.k_off[head];
    \\    uint v_off = p.v_off[head];
    \\    uint mask_off = p.mask_off[head];
    \\    uint dst_off = p.dst_off[head];
    \\    uint slice_dst_off = p.slice_dst_offset[head];
    \\
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size) {
    \\        float mv = mask[mask_off + s * p.mask_rs + q_col * p.mask_cs];
    \\        if (!isfinite(mv)) { scores[s] = -INFINITY; continue; }
    \\        float dot = 0.0f;
    \\        for (uint r = 0; r < p.d_head; r++)
    \\            dot += rope_batch_q_value(q_src, cos_sin, p, head, r, q_col) * K[k_off + r * p.k_rs + s * p.k_cs];
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
    \\        slice_dst[slice_dst_off + r * p.slice_dst_row_stride + q_col * p.slice_dst_col_stride] = val;
    \\    }
    \\}
    \\
    \\struct AttentionRopeStoreSharedBatchParams {
    \\    uint n_heads; uint d_head; uint seq_q; uint seq_kv;
    \\    float scale;
    \\    uint rope_half_d;
    \\    uint rope_src_off[MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS];
    \\    uint rope_cs_off[MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS];
    \\    uint k_off[MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS];
    \\    uint v_off[MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS];
    \\    uint mask_off[MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS];
    \\    uint slice_dst_offset[MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS];
    \\    uint rope_src_rs; uint rope_src_cs; uint rope_cs_cs;
    \\    uint k_rs; uint k_cs; uint v_rs; uint v_cs;
    \\    uint mask_rs; uint mask_cs;
    \\    uint slice_dst_row_stride; uint slice_dst_col_stride;
    \\};
    \\
    \\inline float rope_shared_batch_q_value(
    \\    device const float* q_src,
    \\    device const float* cos_sin,
    \\    constant AttentionRopeStoreSharedBatchParams& p,
    \\    uint head,
    \\    uint r,
    \\    uint q_col
    \\) {
    \\    uint i = r;
    \\    bool high = false;
    \\    if (i >= p.rope_half_d) {
    \\        i -= p.rope_half_d;
    \\        high = true;
    \\    }
    \\    float cos_val = cos_sin[p.rope_cs_off[head] + q_col * p.rope_cs_cs + i];
    \\    float sin_val = cos_sin[p.rope_cs_off[head] + q_col * p.rope_cs_cs + p.rope_half_d + i];
    \\    float x_lo = q_src[p.rope_src_off[head] + q_col * p.rope_src_cs + i * p.rope_src_rs];
    \\    float x_hi = q_src[p.rope_src_off[head] + q_col * p.rope_src_cs + (i + p.rope_half_d) * p.rope_src_rs];
    \\    return high ? (x_hi * cos_val + x_lo * sin_val) : (x_lo * cos_val - x_hi * sin_val);
    \\}
    \\
    \\kernel void attention_rope_store_shared_batch_f32(
    \\    device const float* q_src     [[buffer(0)]],
    \\    device const float* cos_sin   [[buffer(1)]],
    \\    device const float* K         [[buffer(2)]],
    \\    device const float* V         [[buffer(3)]],
    \\    device const float* mask      [[buffer(4)]],
    \\    device float*       slice_dst [[buffer(5)]],
    \\    constant AttentionRopeStoreSharedBatchParams& p [[buffer(6)]],
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
    \\    uint k_off = p.k_off[head];
    \\    uint v_off = p.v_off[head];
    \\    uint mask_off = p.mask_off[head];
    \\    uint slice_dst_off = p.slice_dst_offset[head];
    \\
    \\    for (uint s = tid; s < p.seq_kv; s += tg_size) {
    \\        float mv = mask[mask_off + s * p.mask_rs + q_col * p.mask_cs];
    \\        if (!isfinite(mv)) { scores[s] = -INFINITY; continue; }
    \\        float dot = 0.0f;
    \\        for (uint r = 0; r < p.d_head; r++)
    \\            dot += rope_shared_batch_q_value(q_src, cos_sin, p, head, r, q_col) * K[k_off + r * p.k_rs + s * p.k_cs];
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
    \\        slice_dst[slice_dst_off + r * p.slice_dst_row_stride + q_col * p.slice_dst_col_stride] = val;
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
    \\struct AttentionStoreBatchParams {
    \\    uint n_heads; uint d_head; uint seq_q; uint seq_kv;
    \\    float scale;
    \\    uint q_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint k_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint v_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint mask_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint dst_off[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint slice_dst_offset[MAX_ATTENTION_STORE_BATCH_HEADS];
    \\    uint q_rs; uint q_cs; uint k_rs; uint k_cs; uint v_rs; uint v_cs;
    \\    uint mask_rs; uint mask_cs; uint dst_rs; uint dst_cs;
    \\    uint slice_dst_row_stride; uint slice_dst_col_stride;
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
    \\kernel void attention_store_batch_f32(
    \\    device const float* Q0         [[buffer(0)]],
    \\    device const float* Q1         [[buffer(1)]],
    \\    device const float* Q2         [[buffer(2)]],
    \\    device const float* Q3         [[buffer(3)]],
    \\    device const float* K0         [[buffer(4)]],
    \\    device const float* K1         [[buffer(5)]],
    \\    device const float* K2         [[buffer(6)]],
    \\    device const float* K3         [[buffer(7)]],
    \\    device const float* V0         [[buffer(8)]],
    \\    device const float* V1         [[buffer(9)]],
    \\    device const float* V2         [[buffer(10)]],
    \\    device const float* V3         [[buffer(11)]],
    \\    device const float* mask0      [[buffer(12)]],
    \\    device const float* mask1      [[buffer(13)]],
    \\    device const float* mask2      [[buffer(14)]],
    \\    device const float* mask3      [[buffer(15)]],
    \\    device float*       dst0       [[buffer(16)]],
    \\    device float*       dst1       [[buffer(17)]],
    \\    device float*       dst2       [[buffer(18)]],
    \\    device float*       dst3       [[buffer(19)]],
    \\    device float*       slice_dst0 [[buffer(20)]],
    \\    device float*       slice_dst1 [[buffer(21)]],
    \\    device float*       slice_dst2 [[buffer(22)]],
    \\    device float*       slice_dst3 [[buffer(23)]],
    \\    constant AttentionStoreBatchParams& p [[buffer(24)]],
    \\    uint2 group [[threadgroup_position_in_grid]],
    \\    uint tid     [[thread_index_in_threadgroup]]
    \\) {
    \\    uint q_col = group.x;
    \\    uint head = group.y;
    \\    if (q_col >= p.seq_q) return;
    \\    if (head >= p.n_heads) return;
    \\    const uint tg_size = 256;
    \\
    \\    device const float* Q = Q0;
    \\    device const float* K = K0;
    \\    device const float* V = V0;
    \\    device const float* mask = mask0;
    \\    device float* dst = dst0;
    \\    device float* slice_dst = slice_dst0;
    \\    if (head == 1) {
    \\        Q = Q1; K = K1; V = V1; mask = mask1; dst = dst1; slice_dst = slice_dst1;
    \\    } else if (head == 2) {
    \\        Q = Q2; K = K2; V = V2; mask = mask2; dst = dst2; slice_dst = slice_dst2;
    \\    } else if (head == 3) {
    \\        Q = Q3; K = K3; V = V3; mask = mask3; dst = dst3; slice_dst = slice_dst3;
    \\    }
    \\
    \\    threadgroup float scores[MAX_SEQ];
    \\    threadgroup float scratch[256];
    \\
    \\    uint q_off = p.q_off[head];
    \\    uint k_off = p.k_off[head];
    \\    uint v_off = p.v_off[head];
    \\    uint mask_off = p.mask_off[head];
    \\    uint dst_off = p.dst_off[head];
    \\    uint slice_dst_off = p.slice_dst_offset[head];
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
    \\        slice_dst[slice_dst_off + r * p.slice_dst_row_stride + q_col * p.slice_dst_col_stride] = val;
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
    \\    uint secondary_is_repeat[MAX_FUSED_EW_STEPS];
    \\    uint secondary_repeat_dst_offset[MAX_FUSED_EW_STEPS];
    \\    uint secondary_repeat_src_offset[MAX_FUSED_EW_STEPS];
    \\    uint secondary_repeat_src_ne[MAX_FUSED_EW_STEPS][4];
    \\    uint secondary_repeat_src_strides[MAX_FUSED_EW_STEPS][4];
    \\    uint secondary_repeat_dst_strides[MAX_FUSED_EW_STEPS][4];
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
    \\uint fused_repeat_secondary_index(uint step, uint gid, constant FusedEwParams& p) {
    \\    uint idx = p.secondary_repeat_dst_offset[step] + gid;
    \\    uint src_idx = p.secondary_repeat_src_offset[step];
    \\    for (int d = 3; d >= 0; d--) {
    \\        uint ud = uint(d);
    \\        uint stride = p.secondary_repeat_dst_strides[step][ud];
    \\        uint coord = (stride == 0) ? 0 : idx / stride;
    \\        if (stride != 0) idx %= stride;
    \\        uint extent = p.secondary_repeat_src_ne[step][ud];
    \\        src_idx += ((extent == 0) ? 0 : coord % extent) * p.secondary_repeat_src_strides[step][ud];
    \\    }
    \\    return src_idx;
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
    \\            uint sec_idx = (p.secondary_is_repeat[step] != 0)
    \\                ? fused_repeat_secondary_index(step, gid, p)
    \\                : p.secondary_offset[step] + gid;
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
    \\    uint write_primary;
    \\    uint n_steps;
    \\    uint ew_dst_offset;
    \\    uint op[MAX_FUSED_EW_STEPS];
    \\    uint is_swapped[MAX_FUSED_EW_STEPS];
    \\    uint secondary_slot[MAX_FUSED_EW_STEPS];
    \\    uint secondary_offset[MAX_FUSED_EW_STEPS];
    \\    uint secondary_is_repeat[MAX_FUSED_EW_STEPS];
    \\    uint secondary_is_primary[MAX_FUSED_EW_STEPS];
    \\    uint secondary_repeat_dst_offset[MAX_FUSED_EW_STEPS];
    \\    uint secondary_repeat_src_offset[MAX_FUSED_EW_STEPS];
    \\    uint secondary_repeat_src_ne[MAX_FUSED_EW_STEPS][4];
    \\    uint secondary_repeat_src_strides[MAX_FUSED_EW_STEPS][4];
    \\    uint secondary_repeat_dst_strides[MAX_FUSED_EW_STEPS][4];
    \\};
    \\
    \\uint qmatmul_fused_repeat_secondary_index(uint step, uint linear, constant QMatmulFusedEwParams& p) {
    \\    uint idx = p.secondary_repeat_dst_offset[step] + linear;
    \\    uint src_idx = p.secondary_repeat_src_offset[step];
    \\    for (int d = 3; d >= 0; d--) {
    \\        uint ud = uint(d);
    \\        uint stride = p.secondary_repeat_dst_strides[step][ud];
    \\        uint coord = (stride == 0) ? 0 : idx / stride;
    \\        if (stride != 0) idx %= stride;
    \\        uint extent = p.secondary_repeat_src_ne[step][ud];
    \\        src_idx += ((extent == 0) ? 0 : coord % extent) * p.secondary_repeat_src_strides[step][ud];
    \\    }
    \\    return src_idx;
    \\}
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
    \\                    float other;
    \\                    if (p.secondary_is_primary[step] != 0) {
    \\                        other = base;
    \\                    } else {
    \\                        uint sec_idx = (p.secondary_is_repeat[step] != 0)
    \\                            ? qmatmul_fused_repeat_secondary_index(step, linear, p)
    \\                            : p.secondary_offset[step] + linear;
    \\                        other = fused_secondary(p.secondary_slot[step], sec_idx, s0, s1, s2, s3, s4, s5, s6, s7);
    \\                    }
    \\                    if (op == 7) v = (p.is_swapped[step] != 0) ? other + v : v + other;
    \\                    else v = (p.is_swapped[step] != 0) ? other * v : v * other;
    \\                } else {
    \\                    v = fused_unary(op, v);
    \\                }
    \\            }
    \\            if (p.write_primary != 0) output[p.dst_offset + cr * p.dst_row_stride + cc] = base;
    \\            ew_output[p.ew_dst_offset + linear] = v;
    \\        }
    \\    }
    \\}
    \\
    \\struct QMatmulPairFusedEwParams {
    \\    uint M; uint N; uint K;
    \\    uint left_block_size;
    \\    uint right_block_size;
    \\    uint input_offset;
    \\    uint input_row_stride;
    \\    uint dst_offset;
    \\    uint n_steps;
    \\    uint op[MAX_FUSED_EW_STEPS];
    \\    uint is_swapped[MAX_FUSED_EW_STEPS];
    \\    uint secondary_slot[MAX_FUSED_EW_STEPS];
    \\    uint secondary_offset[MAX_FUSED_EW_STEPS];
    \\    uint secondary_is_repeat[MAX_FUSED_EW_STEPS];
    \\    uint secondary_is_primary[MAX_FUSED_EW_STEPS];
    \\    uint secondary_repeat_dst_offset[MAX_FUSED_EW_STEPS];
    \\    uint secondary_repeat_src_offset[MAX_FUSED_EW_STEPS];
    \\    uint secondary_repeat_src_ne[MAX_FUSED_EW_STEPS][4];
    \\    uint secondary_repeat_src_strides[MAX_FUSED_EW_STEPS][4];
    \\    uint secondary_repeat_dst_strides[MAX_FUSED_EW_STEPS][4];
    \\};
    \\
    \\uint qmatmul_pair_repeat_secondary_index(uint step, uint linear, constant QMatmulPairFusedEwParams& p) {
    \\    uint idx = p.secondary_repeat_dst_offset[step] + linear;
    \\    uint src_idx = p.secondary_repeat_src_offset[step];
    \\    for (int d = 3; d >= 0; d--) {
    \\        uint ud = uint(d);
    \\        uint stride = p.secondary_repeat_dst_strides[step][ud];
    \\        uint coord = (stride == 0) ? 0 : idx / stride;
    \\        if (stride != 0) idx %= stride;
    \\        uint extent = p.secondary_repeat_src_ne[step][ud];
    \\        src_idx += ((extent == 0) ? 0 : coord % extent) * p.secondary_repeat_src_strides[step][ud];
    \\    }
    \\    return src_idx;
    \\}
    \\
    \\kernel void qmatmul_pair_fused_elementwise_f32(
    \\    device const char*  left_weight_data    [[buffer(0)]],
    \\    device const float* left_weight_scales  [[buffer(1)]],
    \\    device const char*  right_weight_data   [[buffer(2)]],
    \\    device const float* right_weight_scales [[buffer(3)]],
    \\    device const float* input               [[buffer(4)]],
    \\    device float*       output              [[buffer(5)]],
    \\    device const float* s0 [[buffer(6)]],
    \\    device const float* s1 [[buffer(7)]],
    \\    device const float* s2 [[buffer(8)]],
    \\    device const float* s3 [[buffer(9)]],
    \\    device const float* s4 [[buffer(10)]],
    \\    device const float* s5 [[buffer(11)]],
    \\    device const float* s6 [[buffer(12)]],
    \\    device const float* s7 [[buffer(13)]],
    \\    constant QMatmulPairFusedEwParams& p [[buffer(14)]],
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
    \\    simdgroup_float8x8 left_acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\    simdgroup_float8x8 right_acc[4] = {
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0),
    \\        simdgroup_float8x8(0), simdgroup_float8x8(0)
    \\    };
    \\    threadgroup float tI[TILE * 8];
    \\    threadgroup float tWL[8 * TILE];
    \\    threadgroup float tWR[8 * TILE];
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
    \\                tWL[i] = float(left_weight_data[w_idx]) * left_weight_scales[w_idx / p.left_block_size];
    \\                tWR[i] = float(right_weight_data[w_idx]) * right_weight_scales[w_idx / p.right_block_size];
    \\            } else {
    \\                tWL[i] = 0.0f;
    \\                tWR[i] = 0.0f;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        simdgroup_float8x8 a0, a1, lb0, lb1, rb0, rb1;
    \\        simdgroup_load(a0, tI + (sRow + 0) * 8, 8);
    \\        simdgroup_load(a1, tI + (sRow + 8) * 8, 8);
    \\        simdgroup_load(lb0, tWL + (sCol + 0), TILE);
    \\        simdgroup_load(lb1, tWL + (sCol + 8), TILE);
    \\        simdgroup_load(rb0, tWR + (sCol + 0), TILE);
    \\        simdgroup_load(rb1, tWR + (sCol + 8), TILE);
    \\        simdgroup_multiply_accumulate(left_acc[0], a0, lb0, left_acc[0]);
    \\        simdgroup_multiply_accumulate(left_acc[1], a0, lb1, left_acc[1]);
    \\        simdgroup_multiply_accumulate(left_acc[2], a1, lb0, left_acc[2]);
    \\        simdgroup_multiply_accumulate(left_acc[3], a1, lb1, left_acc[3]);
    \\        simdgroup_multiply_accumulate(right_acc[0], a0, rb0, right_acc[0]);
    \\        simdgroup_multiply_accumulate(right_acc[1], a0, rb1, right_acc[1]);
    \\        simdgroup_multiply_accumulate(right_acc[2], a1, rb0, right_acc[2]);
    \\        simdgroup_multiply_accumulate(right_acc[3], a1, rb1, right_acc[3]);
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    threadgroup float tL[TILE * TILE];
    \\    threadgroup float tR[TILE * TILE];
    \\    simdgroup_store(left_acc[0], tL + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(left_acc[1], tL + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(left_acc[2], tL + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(left_acc[3], tL + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(right_acc[0], tR + (sRow + 0) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(right_acc[1], tR + (sRow + 0) * TILE + sCol + 8, TILE);
    \\    simdgroup_store(right_acc[2], tR + (sRow + 8) * TILE + sCol + 0, TILE);
    \\    simdgroup_store(right_acc[3], tR + (sRow + 8) * TILE + sCol + 8, TILE);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint i = tid; i < TILE * TILE; i += 128) {
    \\        uint r = i / TILE, c = i % TILE;
    \\        uint cr = gRow + r, cc = gCol + c;
    \\        if (cr < p.M && cc < p.N) {
    \\            float left_base = tL[i];
    \\            float right_base = tR[i];
    \\            uint linear = cr * p.N + cc;
    \\            float v = left_base;
    \\            for (uint step = 0; step < p.n_steps; step++) {
    \\                uint op = p.op[step];
    \\                if (op == 7 || op == 8) {
    \\                    float other;
    \\                    if (p.secondary_is_primary[step] != 0) {
    \\                        other = left_base;
    \\                    } else {
    \\                        uint sec_idx = (p.secondary_is_repeat[step] != 0)
    \\                            ? qmatmul_pair_repeat_secondary_index(step, linear, p)
    \\                            : p.secondary_offset[step] + linear;
    \\                        other = fused_secondary(p.secondary_slot[step], sec_idx, s0, s1, s2, s3, s4, s5, s6, s7);
    \\                    }
    \\                    if (op == 7) v = (p.is_swapped[step] != 0) ? other + v : v + other;
    \\                    else v = (p.is_swapped[step] != 0) ? other * v : v * other;
    \\                } else {
    \\                    v = fused_unary(op, v);
    \\                }
    \\            }
    \\            output[p.dst_offset + linear] = v * right_base;
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
const MAX_QMATMUL_BATCH_SIDECARS: usize = 8;

fn BatchKernelLayout(comptime Buffer: type, comptime max_ops: usize) type {
    const buffers_per_op = switch (@typeInfo(Buffer)) {
        .@"enum" => |info| info.fields.len,
        else => @compileError("kernel buffer layout expects an enum"),
    };
    if (max_ops == 0) @compileError("kernel batch size must be non-zero");

    return struct {
        const buffer_count = max_ops * buffers_per_op;
        const params_index: u32 = @intCast(buffer_count);

        fn bufferIndex(slot: usize, buffer: Buffer) usize {
            std.debug.assert(slot < max_ops);
            return slot * buffers_per_op + @as(usize, @intFromEnum(buffer));
        }
    };
}

fn SlottedLayout(comptime max_slots: usize, comptime per_slot: usize) type {
    if (max_slots == 0 or per_slot == 0) @compileError("slotted layout dimensions must be non-zero");

    return struct {
        const slots = max_slots * per_slot;

        fn index(slot: usize, item_slot: usize) usize {
            std.debug.assert(slot < max_slots);
            std.debug.assert(item_slot < per_slot);
            return slot * per_slot + item_slot;
        }

        fn start(slot: usize) usize {
            std.debug.assert(slot < max_slots);
            return slot * per_slot;
        }
    };
}

fn requireShaderUintConst(comptime name: []const u8, comptime value: comptime_int) void {
    const needle = std.fmt.comptimePrint("constant uint {s} = {d};", .{ name, value });
    if (std.mem.indexOf(u8, shader_source, needle) == null) {
        @compileError("Metal shader constant drifted from Zig: " ++ name);
    }
}

const QMatmulBatchBuffer = enum(u8) {
    weight_data,
    weight_scales,
    input,
    output,
    sidecar_dst,
    secondary,
};

const QMatmulBatchKernel = BatchKernelLayout(QMatmulBatchBuffer, MAX_QMATMUL_BATCH);

const QMatmulBatchSidecarCode = enum(u32) {
    none = 0,
    slice = 1,
    elementwise = 2,
};

const QMatmulBatchSidecarLayout = SlottedLayout(MAX_QMATMUL_BATCH, MAX_QMATMUL_BATCH_SIDECARS);
const QMATMUL_BATCH_SIDECAR_SLOTS: usize = QMatmulBatchSidecarLayout.slots;

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
    write_primary: [MAX_QMATMUL_BATCH]u32,
    sidecar_count: [MAX_QMATMUL_BATCH]u32,
    sidecar_kind: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
    slice_rows: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
    slice_cols: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
    slice_src_col_start: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
    slice_dst_offset: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
    slice_dst_row_stride: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
    slice_dst_col_stride: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
    ew_op: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
    ew_is_swapped: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
    ew_dst_offset: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
    ew_secondary_offset: [QMATMUL_BATCH_SIDECAR_SLOTS]u32,
};

const MAX_QMATMUL_ROPE_STORE_BATCH: usize = 4;

const QMatmulRopeStoreBuffer = enum(u8) {
    weight_data,
    weight_scales,
    input,
    output,
    cos_sin,
    slice_dst,
};

const QMatmulRopeStoreKernel = BatchKernelLayout(QMatmulRopeStoreBuffer, MAX_QMATMUL_ROPE_STORE_BATCH);

const QMatmulRopeStoreBatch4Params = extern struct {
    n_ops: u32,
    max_tiles_y: u32,
    M: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    N: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    K: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    block_size: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    input_offset: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    input_row_stride: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    dst_offset: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    dst_row_stride: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    write_primary: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    rope_half_d: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    rope_src_col_start: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    rope_cs_off: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    rope_cs_cs: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    slice_dst_offset: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    slice_dst_row_stride: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
    slice_dst_col_stride: [MAX_QMATMUL_ROPE_STORE_BATCH]u32,
};

fn requireKernelBuffers(comptime Kernel: type, comptime expected: usize, comptime name: []const u8) void {
    if (Kernel.buffer_count != expected) @compileError(name ++ " MSL buffer layout changed; update Zig encoder and shader together");
    if (Kernel.buffer_count > max_encode_buffers) @compileError(name ++ " uses more buffers than the Metal encoder wrapper can pass");
}

const max_encode_buffers = 32;

comptime {
    requireShaderUintConst("MAX_QMATMUL_BATCH", MAX_QMATMUL_BATCH);
    requireShaderUintConst("MAX_QMATMUL_BATCH_SIDECARS", MAX_QMATMUL_BATCH_SIDECARS);
    requireShaderUintConst("MAX_QMATMUL_ROPE_STORE_BATCH", MAX_QMATMUL_ROPE_STORE_BATCH);
    requireKernelBuffers(QMatmulBatchKernel, 24, "qmatmul_batch4_f32");
    requireKernelBuffers(QMatmulRopeStoreKernel, 24, "qmatmul_rope_store_batch4_f32");
}

const MAX_QMATVEC_BATCH: usize = 4;

const QMatvecBatchBuffer = enum(u8) {
    weight_data,
    weight_scales,
    input,
    output,
    sidecar_src,
    sidecar_dst,
};

const QMatvecBatchKernel = BatchKernelLayout(QMatvecBatchBuffer, MAX_QMATVEC_BATCH);

comptime {
    requireShaderUintConst("MAX_QMATVEC_BATCH", MAX_QMATVEC_BATCH);
    requireKernelBuffers(QMatvecBatchKernel, 24, "qmatvec_batch4_f32");
}

const QMatvecBatch4Params = extern struct {
    n_ops: u32,
    max_n: u32,
    N: [MAX_QMATVEC_BATCH]u32,
    K: [MAX_QMATVEC_BATCH]u32,
    block_size: [MAX_QMATVEC_BATCH]u32,
    input_offset: [MAX_QMATVEC_BATCH]u32,
    dst_offset: [MAX_QMATVEC_BATCH]u32,
    write_primary: [MAX_QMATVEC_BATCH]u32,
    sidecar_kind: [MAX_QMATVEC_BATCH]u32,
    slice_rows: [MAX_QMATVEC_BATCH]u32,
    slice_src_col_start: [MAX_QMATVEC_BATCH]u32,
    slice_dst_offset: [MAX_QMATVEC_BATCH]u32,
    slice_dst_row_stride: [MAX_QMATVEC_BATCH]u32,
    rope_half_d: [MAX_QMATVEC_BATCH]u32,
    rope_cs_off: [MAX_QMATVEC_BATCH]u32,
    ew_op: [MAX_QMATVEC_BATCH]u32,
    ew_is_swapped: [MAX_QMATVEC_BATCH]u32,
    ew_dst_offset: [MAX_QMATVEC_BATCH]u32,
    ew_secondary_offset: [MAX_QMATVEC_BATCH]u32,
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

const RopeSliceAssignBatchParams = extern struct {
    n_ops: u32,
    max_n: u32,
    half_d: [MAX_ROPE_BATCH]u32,
    seq_len: [MAX_ROPE_BATCH]u32,
    src_off: [MAX_ROPE_BATCH]u32,
    cs_off: [MAX_ROPE_BATCH]u32,
    src_rs: [MAX_ROPE_BATCH]u32,
    src_cs: [MAX_ROPE_BATCH]u32,
    cs_cs: [MAX_ROPE_BATCH]u32,
    dst_offset: [MAX_ROPE_BATCH]u32,
    dst_row_stride: [MAX_ROPE_BATCH]u32,
    dst_col_stride: [MAX_ROPE_BATCH]u32,
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
    write_primary: u32,
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
    write_primary: u32,
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

const AttentionSliceAssignParams = extern struct {
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
    slice_rows: u32,
    slice_cols: u32,
    slice_src_offset: u32,
    slice_src_row_stride: u32,
    slice_src_col_stride: u32,
    slice_dst_offset: u32,
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
};

const AttentionStoreParams = extern struct {
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
    slice_dst_offset: u32,
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
};

const AttentionRopeStoreParams = extern struct {
    d_head: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    rope_half_d: u32,
    rope_src_off: u32,
    rope_cs_off: u32,
    rope_src_rs: u32,
    rope_src_cs: u32,
    rope_cs_cs: u32,
    k_off: u32,
    v_off: u32,
    mask_off: u32,
    dst_off: u32,
    k_rs: u32,
    k_cs: u32,
    v_rs: u32,
    v_cs: u32,
    mask_rs: u32,
    mask_cs: u32,
    dst_rs: u32,
    dst_cs: u32,
    slice_dst_offset: u32,
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
};

const MAX_ATTENTION_BATCH_HEADS: usize = 16;
const MAX_ATTENTION_STORE_BATCH_HEADS: usize = 4;
const MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS: usize = 16;

comptime {
    requireShaderUintConst("MAX_ATTENTION_BATCH_HEADS", MAX_ATTENTION_BATCH_HEADS);
    requireShaderUintConst("MAX_ATTENTION_STORE_BATCH_HEADS", MAX_ATTENTION_STORE_BATCH_HEADS);
    requireShaderUintConst("MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS", MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS);
}

const AttentionRopeStoreBatchParams = extern struct {
    n_heads: u32,
    d_head: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    rope_half_d: u32,
    rope_src_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    rope_cs_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    k_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    v_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    mask_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    dst_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    slice_dst_offset: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    rope_src_rs: u32,
    rope_src_cs: u32,
    rope_cs_cs: u32,
    k_rs: u32,
    k_cs: u32,
    v_rs: u32,
    v_cs: u32,
    mask_rs: u32,
    mask_cs: u32,
    dst_rs: u32,
    dst_cs: u32,
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
};

const AttentionRopeStoreSharedBatchParams = extern struct {
    n_heads: u32,
    d_head: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    rope_half_d: u32,
    rope_src_off: [MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS]u32,
    rope_cs_off: [MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS]u32,
    k_off: [MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS]u32,
    v_off: [MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS]u32,
    mask_off: [MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS]u32,
    slice_dst_offset: [MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS]u32,
    rope_src_rs: u32,
    rope_src_cs: u32,
    rope_cs_cs: u32,
    k_rs: u32,
    k_cs: u32,
    v_rs: u32,
    v_cs: u32,
    mask_rs: u32,
    mask_cs: u32,
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
};

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

const AttentionStoreBatchParams = extern struct {
    n_heads: u32,
    d_head: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    q_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    k_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    v_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    mask_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    dst_off: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
    slice_dst_offset: [MAX_ATTENTION_STORE_BATCH_HEADS]u32,
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
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
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
    secondary_is_repeat: [MAX_FUSED_EW_STEPS]u32,
    secondary_repeat_dst_offset: [MAX_FUSED_EW_STEPS]u32,
    secondary_repeat_src_offset: [MAX_FUSED_EW_STEPS]u32,
    secondary_repeat_src_ne: [MAX_FUSED_EW_STEPS][4]u32,
    secondary_repeat_src_strides: [MAX_FUSED_EW_STEPS][4]u32,
    secondary_repeat_dst_strides: [MAX_FUSED_EW_STEPS][4]u32,
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
    write_primary: u32,
    n_steps: u32,
    ew_dst_offset: u32,
    op: [MAX_FUSED_EW_STEPS]u32,
    is_swapped: [MAX_FUSED_EW_STEPS]u32,
    secondary_slot: [MAX_FUSED_EW_STEPS]u32,
    secondary_offset: [MAX_FUSED_EW_STEPS]u32,
    secondary_is_repeat: [MAX_FUSED_EW_STEPS]u32,
    secondary_is_primary: [MAX_FUSED_EW_STEPS]u32,
    secondary_repeat_dst_offset: [MAX_FUSED_EW_STEPS]u32,
    secondary_repeat_src_offset: [MAX_FUSED_EW_STEPS]u32,
    secondary_repeat_src_ne: [MAX_FUSED_EW_STEPS][4]u32,
    secondary_repeat_src_strides: [MAX_FUSED_EW_STEPS][4]u32,
    secondary_repeat_dst_strides: [MAX_FUSED_EW_STEPS][4]u32,
};

const QMatmulPairFusedEwParams = extern struct {
    M: u32,
    N: u32,
    K: u32,
    left_block_size: u32,
    right_block_size: u32,
    input_offset: u32,
    input_row_stride: u32,
    dst_offset: u32,
    n_steps: u32,
    op: [MAX_FUSED_EW_STEPS]u32,
    is_swapped: [MAX_FUSED_EW_STEPS]u32,
    secondary_slot: [MAX_FUSED_EW_STEPS]u32,
    secondary_offset: [MAX_FUSED_EW_STEPS]u32,
    secondary_is_repeat: [MAX_FUSED_EW_STEPS]u32,
    secondary_is_primary: [MAX_FUSED_EW_STEPS]u32,
    secondary_repeat_dst_offset: [MAX_FUSED_EW_STEPS]u32,
    secondary_repeat_src_offset: [MAX_FUSED_EW_STEPS]u32,
    secondary_repeat_src_ne: [MAX_FUSED_EW_STEPS][4]u32,
    secondary_repeat_src_strides: [MAX_FUSED_EW_STEPS][4]u32,
    secondary_repeat_dst_strides: [MAX_FUSED_EW_STEPS][4]u32,
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

fn attentionSliceAssignParams(att: anytype, sa: anytype) AttentionSliceAssignParams {
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
        .slice_rows = sa.rows,
        .slice_cols = sa.cols,
        .slice_src_offset = sa.src_offset,
        .slice_src_row_stride = sa.src_row_stride,
        .slice_src_col_stride = sa.src_col_stride,
        .slice_dst_offset = sa.dst_offset,
        .slice_dst_row_stride = sa.dst_row_stride,
        .slice_dst_col_stride = sa.dst_col_stride,
    };
}

fn attentionStoreParams(att: anytype, sa: anytype) AttentionStoreParams {
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
        .slice_dst_offset = sa.dst_offset,
        .slice_dst_row_stride = sa.dst_row_stride,
        .slice_dst_col_stride = sa.dst_col_stride,
    };
}

fn attentionRopeStoreParams(rr: anytype, att: anytype, sa: anytype) AttentionRopeStoreParams {
    return .{
        .d_head = att.d_head,
        .seq_q = att.seq_q,
        .seq_kv = att.seq_kv,
        .scale = att.scale,
        .rope_half_d = rr.half_d,
        .rope_src_off = rr.src_off,
        .rope_cs_off = rr.cs_off,
        .rope_src_rs = rr.src_rs,
        .rope_src_cs = rr.src_cs,
        .rope_cs_cs = rr.cs_cs,
        .k_off = att.k_off,
        .v_off = att.v_off,
        .mask_off = att.mask_off,
        .dst_off = att.dst_off,
        .k_rs = att.k_rs,
        .k_cs = att.k_cs,
        .v_rs = att.v_rs,
        .v_cs = att.v_cs,
        .mask_rs = att.mask_rs,
        .mask_cs = att.mask_cs,
        .dst_rs = att.dst_rs,
        .dst_cs = att.dst_cs,
        .slice_dst_offset = sa.dst_offset,
        .slice_dst_row_stride = sa.dst_row_stride,
        .slice_dst_col_stride = sa.dst_col_stride,
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

const MetalKernel = enum(u8) {
    matmul_f32,
    qmatmul_f32,
    qmatmul_batch4_f32,
    qmatmul_rope_store_batch4_f32,
    qmatvec_f32,
    qmatvec_batch4_f32,
    matvec_f16,
    matmul_f16,
    rope_f32,
    rope_batch_f32,
    rope_slice_assign_f32,
    rope_slice_assign_batch_f32,
    qmatmul_slice_assign_f32,
    qmatmul_elementwise_f32,
    qmatmul_fused_elementwise_f32,
    qmatmul_pair_fused_elementwise_f32,
    qmatvec_slice_assign_f32,
    slice_assign_batch_f32,
    rmsnorm_scale_f32,
    attention_f32,
    attention_slice_assign_f32,
    attention_store_f32,
    attention_rope_store_f32,
    attention_rope_store_batch_f32,
    attention_rope_store_shared_batch_f32,
    attention_store_batch_f32,
    attention_batch_f32,
    compute_f32,
    elementwise_batch8_f32,
    fused_elementwise_f32,
};

const metal_kernel_count = @typeInfo(MetalKernel).@"enum".fields.len;

fn metalKernelName(kernel: MetalKernel) [:0]const u8 {
    return @tagName(kernel);
}

fn requireShaderKernel(comptime name: []const u8) void {
    const needle = std.fmt.comptimePrint("kernel void {s}(", .{name});
    if (std.mem.indexOf(u8, shader_source, needle) == null) {
        @compileError("Metal kernel is listed in Zig but missing from shader source: " ++ name);
    }
}

comptime {
    @setEvalBranchQuota(500_000);
    for (@typeInfo(MetalKernel).@"enum".fields) |field| {
        const kernel: MetalKernel = @enumFromInt(field.value);
        requireShaderKernel(metalKernelName(kernel));
    }
}

fn releaseMetalPipelines(pipelines: *[metal_kernel_count]?*anyopaque) void {
    for (pipelines[0..]) |*maybe_pipeline| {
        if (maybe_pipeline.*) |pipeline| {
            c.mtl_release(pipeline);
            maybe_pipeline.* = null;
        }
    }
}

pub const MetalBackend = struct {
    device: *anyopaque,
    queue: *anyopaque,
    pipelines: [metal_kernel_count]?*anyopaque,
    library: *anyopaque,
    active_commands: ?*anyopaque = null,
    fine_grained_program_dispatch: bool = false,
    region_program_dispatch: bool = false,
    projection_rope_cache_sidecars: bool = false,

    pub fn init() !MetalBackend {
        const device = c.mtl_create_device() orelse return error.MetalNotAvailable;
        errdefer c.mtl_release(device);

        const queue = c.mtl_create_queue(device) orelse return error.MetalInitFailed;
        errdefer c.mtl_release(queue);

        const library = c.mtl_compile_source(device, shader_source.ptr, shader_source.len) orelse return error.ShaderCompileFailed;
        errdefer c.mtl_release(library);

        var pipelines = [_]?*anyopaque{null} ** metal_kernel_count;
        errdefer releaseMetalPipelines(&pipelines);
        inline for (@typeInfo(MetalKernel).@"enum".fields) |field| {
            const kernel: MetalKernel = @enumFromInt(field.value);
            pipelines[@intFromEnum(kernel)] = c.mtl_create_pipeline(device, library, metalKernelName(kernel).ptr) orelse return error.PipelineCreateFailed;
        }

        return .{
            .device = device,
            .queue = queue,
            .pipelines = pipelines,
            .library = library,
        };
    }

    pub fn deinit(self: *MetalBackend) void {
        self.flushCommands();
        releaseMetalPipelines(&self.pipelines);
        c.mtl_release(self.library);
        c.mtl_release(self.queue);
        c.mtl_release(self.device);
    }

    fn pipeline(self: *MetalBackend, kernel: MetalKernel) *anyopaque {
        return self.pipelines[@intFromEnum(kernel)].?;
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

    /// Enable the experimental projection -> RoPE -> KV-store lowering. This
    /// remains opt-in until benchmarks prove it wins on the target Apple GPU.
    pub fn setProjectionRopeCacheSidecars(self: *MetalBackend, enabled: bool) void {
        self.projection_rope_cache_sidecars = enabled;
    }

    pub fn setQMatmulRopeCacheSidecars(self: *MetalBackend, enabled: bool) void {
        self.setProjectionRopeCacheSidecars(enabled);
    }

    pub fn commandStreamPolicy(self: *const MetalBackend) program_mod.CommandStreamPolicy {
        var policy = program_mod.CommandStreamPolicy.fromCapabilities(backend_mod.Capabilities.metal);
        policy.projection_rope_cache_sidecars = self.projection_rope_cache_sidecars;
        return policy;
    }

    pub fn capabilities(self: *const MetalBackend) backend_mod.Capabilities {
        var caps = backend_mod.Capabilities.metal;
        caps.command_stream.projection_rope_cache_sidecars = self.projection_rope_cache_sidecars;
        return caps;
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
            .capabilities = self.capabilities(),
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

pub const MetalRegionPattern = enum(u32) {
    decode_layer_stage,
    prefill_layer_stage,

    pub fn index(self: MetalRegionPattern) u32 {
        return @intFromEnum(self);
    }

    pub fn stageName(self: MetalRegionPattern) []const u8 {
        return switch (self) {
            .decode_layer_stage => "decode-layer",
            .prefill_layer_stage => "prefill-layer",
        };
    }

    pub fn stagePolicy(self: MetalRegionPattern) program_mod.StagePolicy {
        return program_mod.StagePolicy.anchored(
            self.stageName(),
            self.index(),
            switch (self) {
                .decode_layer_stage => program_mod.RegionPolicy.qmatvecCluster(),
                .prefill_layer_stage => program_mod.RegionPolicy.qmatmulCluster(),
            },
            metal_projection_anchors_per_stage,
        );
    }
};

const metal_projection_anchors_per_stage: u32 = 7;
const metal_region_stages = [_]program_mod.StagePolicy{
    MetalRegionPattern.decode_layer_stage.stagePolicy(),
    MetalRegionPattern.prefill_layer_stage.stagePolicy(),
};

fn buildMetalExecutionPlan(
    alloc: std.mem.Allocator,
    ops: []const backend_mod.DeviceOp,
    schedule_policy: program_mod.SchedulePolicy,
    command_policy: program_mod.CommandStreamPolicy,
) !program_mod.ExecutionPlan {
    return program_mod.buildExecutionPlan(alloc, ops, schedule_policy, &metal_region_stages, command_policy);
}

const CompiledProgram = struct {
    backend: *MetalBackend,
    device_bufs: []DeviceBuffer,
    ref_buffers: []reference.Buffer,
    qweight_views: []DeviceQWeight,
    ref_qweights: []reference.QWeight,
    ops: []const backend_mod.DeviceOp,
    plan: program_mod.ExecutionPlan,
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
        self.plan.deinit(self.alloc);
        self.alloc.destroy(self);
    }

    fn execute(self: *CompiledProgram, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
        // Upload per-step inputs (token embed, pos, mask) via shared memory.
        reference.uploadToBuffers(self.ref_buffers, inputs);

        if (self.plan.schedule.len == 0 and self.ops.len > 0) {
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
        if (self.backend.region_program_dispatch and self.plan.regions.len > 0) {
            self.executeRegionScheduled();
            return;
        }

        for (self.plan.schedule) |item| {
            self.executeScheduleItem(item);
        }
    }

    fn executeRegionScheduled(self: *CompiledProgram) void {
        for (self.plan.regions, 0..) |unit, unit_index| {
            switch (unit.kind) {
                .item => self.executeScheduleItem(self.plan.schedule[@intCast(unit.start_item)]),
                .pattern_region => {
                    if (!self.tryEncodePatternRegion(unit, unit_index)) {
                        self.executeScheduleItems(unit.start_item, unit.item_count);
                    }
                },
            }
        }
    }

    fn executeScheduleItems(self: *CompiledProgram, start_item: u32, item_count: u32) void {
        const start: usize = @intCast(start_item);
        const end = start + @as(usize, item_count);
        for (self.plan.schedule[start..end]) |item| {
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
        var raw_buffers: [max_encode_buffers]?*anyopaque = undefined;
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

    fn encodeKernel(
        self: *CompiledProgram,
        kernel: MetalKernel,
        buffers: []const DeviceBuffer,
        params: anytype,
        params_index: u32,
        grid: DispatchGrid,
        threads_x: u32,
    ) void {
        const Params = @TypeOf(params);
        self.encode(self.backend.pipeline(kernel), buffers, &params, @sizeOf(Params), params_index, grid, threads_x);
    }

    fn canEncodeFusedElementwise(fe: anytype) bool {
        if (fe.steps.len > MAX_FUSED_EW_STEPS) return false;
        for (fe.steps) |step| {
            if (!isSupportedElementwiseOp(step.op)) return false;
        }
        return true;
    }

    const RepeatSecondaryView = struct {
        dst: u16,
        src: u16,
        n: u32,
        dst_offset: u32,
        src_offset: u32,
        src_ne: [4]u32,
        src_strides: [4]u32,
        dst_strides: [4]u32,
    };

    fn repeatSecondaryView(rp: anytype) RepeatSecondaryView {
        return .{
            .dst = rp.dst,
            .src = rp.src,
            .n = rp.n,
            .dst_offset = rp.dst_offset,
            .src_offset = rp.src_offset,
            .src_ne = rp.src_ne,
            .src_strides = rp.src_strides,
            .dst_strides = rp.dst_strides,
        };
    }

    fn encodeFusedElementwise(self: *CompiledProgram, fe: anytype) bool {
        return self.encodeFusedElementwiseWithRepeatView(fe, null);
    }

    fn encodeFusedElementwiseWithRepeatView(self: *CompiledProgram, fe: anytype, repeat_view: ?RepeatSecondaryView) bool {
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
                var secondary_buf = step.secondary_buf;
                if (repeat_view) |rp| {
                    if (step.secondary_buf == rp.dst) {
                        if (step.secondary_offset < rp.dst_offset) return false;
                        const rel = step.secondary_offset - rp.dst_offset;
                        if (@as(u64, rel) + @as(u64, fe.n) > @as(u64, rp.n)) return false;
                        secondary_buf = rp.src;
                        params.secondary_is_repeat[i] = 1;
                        params.secondary_repeat_dst_offset[i] = rel;
                        params.secondary_repeat_src_offset[i] = rp.src_offset;
                        params.secondary_repeat_src_ne[i] = rp.src_ne;
                        params.secondary_repeat_src_strides[i] = rp.src_strides;
                        params.secondary_repeat_dst_strides[i] = rp.dst_strides;
                    }
                }

                const slot = for (secondary_bufs[0..secondary_count], 0..) |buf_idx, slot_idx| {
                    if (buf_idx == secondary_buf) break slot_idx;
                } else blk: {
                    if (secondary_count >= MAX_FUSED_EW_SECONDARIES) return false;
                    const next = secondary_count;
                    secondary_bufs[next] = secondary_buf;
                    buffers[2 + next] = self.device_bufs[secondary_buf];
                    secondary_count += 1;
                    break :blk next;
                };
                params.secondary_slot[i] = @intCast(slot);
            }
        }

        self.encodeKernel(
            .fused_elementwise_f32,
            buffers[0..],
            params,
            10,
            .{ .gx = linearGrid(fe.n) },
            WG_SIZE,
        );
        return true;
    }

    fn encodeRepeatFusedElementwise(self: *CompiledProgram, rp: anytype, fe: anytype) bool {
        if (!program_mod.repeatFusedElementwiseCompatible(rp, fe)) return false;
        return self.encodeFusedElementwiseWithRepeatView(fe, repeatSecondaryView(rp));
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
        self.encodeKernel(.compute_f32, &buffers, spec.params, 3, spec.grid, WG_SIZE);
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

        self.encodeKernel(
            .elementwise_batch8_f32,
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
        self.encodeKernel(.qmatvec_f32, &buffers, qmatmulParams(q, w.block_size), 4, .{ .gx = linearGrid(q.N) }, WG_SIZE);
        return true;
    }

    fn canEncodeQMatmulBatchOp(self: *CompiledProgram, q: anytype) bool {
        return q.M != 1 and @as(usize, q.weight_idx) < self.qweight_views.len;
    }

    const QMatmulBatchSidecarKind = enum {
        slice,
        elementwise,
    };

    const QMatmulBatchSidecarPlan = struct {
        counts: [MAX_QMATMUL_BATCH]u32 = [_]u32{0} ** MAX_QMATMUL_BATCH,
        indices: [QMATMUL_BATCH_SIDECAR_SLOTS]?usize = [_]?usize{null} ** QMATMUL_BATCH_SIDECAR_SLOTS,
        kinds: [QMATMUL_BATCH_SIDECAR_SLOTS]QMatmulBatchSidecarKind = [_]QMatmulBatchSidecarKind{.slice} ** QMATMUL_BATCH_SIDECAR_SLOTS,

        fn append(self: *QMatmulBatchSidecarPlan, slot: usize, kind: QMatmulBatchSidecarKind, idx: usize) bool {
            if (slot >= MAX_QMATMUL_BATCH) return false;
            const count: usize = @intCast(self.counts[slot]);
            if (count >= MAX_QMATMUL_BATCH_SIDECARS) return false;
            const sidecar_slot = QMatmulBatchSidecarLayout.index(slot, count);
            self.indices[sidecar_slot] = idx;
            self.kinds[sidecar_slot] = kind;
            self.counts[slot] += 1;
            return true;
        }

        fn appendSlice(self: *QMatmulBatchSidecarPlan, slot: usize, idx: usize) bool {
            return self.append(slot, .slice, idx);
        }

        fn appendElementwise(self: *QMatmulBatchSidecarPlan, slot: usize, idx: usize) bool {
            return self.append(slot, .elementwise, idx);
        }

        fn slotIndices(self: *const QMatmulBatchSidecarPlan, slot: usize) []const ?usize {
            const start = QMatmulBatchSidecarLayout.start(slot);
            const count: usize = @intCast(self.counts[slot]);
            return self.indices[start .. start + count];
        }
    };

    fn encodeQMatmulBatch(
        self: *CompiledProgram,
        ops: []const backend_mod.DeviceOp,
        indices: []const usize,
        sidecar_indices: []const ?usize,
    ) void {
        var sidecars = QMatmulBatchSidecarPlan{};
        for (sidecar_indices, 0..) |maybe_idx, slot| {
            if (maybe_idx) |idx| switch (ops[idx]) {
                .slice_assign => _ = sidecars.appendSlice(slot, idx),
                .elementwise => _ = sidecars.appendElementwise(slot, idx),
                else => {},
            };
        }
        self.encodeQMatmulBatchWithSidecars(ops, indices, &sidecars);
    }

    fn encodeQMatmulBatchWithSidecars(
        self: *CompiledProgram,
        ops: []const backend_mod.DeviceOp,
        indices: []const usize,
        sidecars: *const QMatmulBatchSidecarPlan,
    ) void {
        const first_q = ops[indices[0]].qmatmul;
        const first_w = self.qweight_views[first_q.weight_idx];
        var buffers: [QMatmulBatchKernel.buffer_count]DeviceBuffer = undefined;
        for (0..MAX_QMATMUL_BATCH) |slot| {
            buffers[QMatmulBatchKernel.bufferIndex(slot, .weight_data)] = first_w.data;
            buffers[QMatmulBatchKernel.bufferIndex(slot, .weight_scales)] = first_w.scales;
            buffers[QMatmulBatchKernel.bufferIndex(slot, .input)] = self.device_bufs[first_q.input];
            buffers[QMatmulBatchKernel.bufferIndex(slot, .output)] = self.device_bufs[first_q.dst];
            buffers[QMatmulBatchKernel.bufferIndex(slot, .sidecar_dst)] = self.device_bufs[first_q.dst];
            buffers[QMatmulBatchKernel.bufferIndex(slot, .secondary)] = self.device_bufs[first_q.input];
        }

        var params = std.mem.zeroes(QMatmulBatch4Params);
        params.n_ops = @intCast(indices.len);
        var max_tiles_x: u32 = 1;
        var max_tiles_y: u32 = 1;
        for (indices, 0..) |op_index, slot| {
            const q = ops[op_index].qmatmul;
            const w = self.qweight_views[q.weight_idx];
            const qparams = qmatmulParams(q, w.block_size);
            buffers[QMatmulBatchKernel.bufferIndex(slot, .weight_data)] = w.data;
            buffers[QMatmulBatchKernel.bufferIndex(slot, .weight_scales)] = w.scales;
            buffers[QMatmulBatchKernel.bufferIndex(slot, .input)] = self.device_bufs[q.input];
            buffers[QMatmulBatchKernel.bufferIndex(slot, .output)] = self.device_bufs[q.dst];
            buffers[QMatmulBatchKernel.bufferIndex(slot, .sidecar_dst)] = self.device_bufs[q.dst];
            buffers[QMatmulBatchKernel.bufferIndex(slot, .secondary)] = self.device_bufs[q.input];
            params.M[slot] = qparams.M;
            params.N[slot] = qparams.N;
            params.K[slot] = qparams.K;
            params.block_size[slot] = qparams.block_size;
            params.input_offset[slot] = qparams.input_offset;
            params.input_row_stride[slot] = qparams.input_row_stride;
            params.dst_offset[slot] = qparams.dst_offset;
            params.dst_row_stride[slot] = qparams.dst_row_stride;
            params.write_primary[slot] = 1;
            params.sidecar_count[slot] = sidecars.counts[slot];
            if (params.sidecar_count[slot] > 0) {
                params.write_primary[slot] = @intFromBool(program_mod.projectionPrimaryOutputHasExternalUsersExcept(ops, op_index, sidecars.slotIndices(slot)));
            }
            for (sidecars.slotIndices(slot), 0..) |maybe_sidecar_index, sidecar_slot| {
                const sidecar_index = maybe_sidecar_index orelse continue;
                const param_slot = QMatmulBatchSidecarLayout.index(slot, sidecar_slot);
                switch (sidecars.kinds[param_slot]) {
                    .slice => {
                        const sa = ops[sidecar_index].slice_assign;
                        params.sidecar_kind[param_slot] = @intFromEnum(QMatmulBatchSidecarCode.slice);
                        params.slice_rows[param_slot] = sa.rows;
                        params.slice_cols[param_slot] = sa.cols;
                        params.slice_src_col_start[param_slot] = program_mod.qmatmulSliceSrcColStart(q, sa).?;
                        params.slice_dst_offset[param_slot] = sa.dst_offset;
                        params.slice_dst_row_stride[param_slot] = sa.dst_row_stride;
                        params.slice_dst_col_stride[param_slot] = sa.dst_col_stride;
                        buffers[QMatmulBatchKernel.bufferIndex(slot, .sidecar_dst)] = self.device_bufs[sa.dst];
                    },
                    .elementwise => {
                        const e = ops[sidecar_index].elementwise;
                        const q_is_src0 = e.src0 == q.dst and e.src0_offset == q.dst_offset;
                        const secondary_buf = if (q_is_src0) e.src1 else e.src0;
                        const secondary_offset = if (q_is_src0) e.src1_offset else e.src0_offset;
                        params.sidecar_kind[param_slot] = @intFromEnum(QMatmulBatchSidecarCode.elementwise);
                        params.ew_op[param_slot] = @intFromEnum(e.op);
                        params.ew_is_swapped[param_slot] = if (q_is_src0) 0 else 1;
                        params.ew_dst_offset[param_slot] = e.dst_offset;
                        params.ew_secondary_offset[param_slot] = secondary_offset;
                        buffers[QMatmulBatchKernel.bufferIndex(slot, .sidecar_dst)] = self.device_bufs[e.dst];
                        buffers[QMatmulBatchKernel.bufferIndex(slot, .secondary)] = self.device_bufs[secondary_buf];
                    },
                }
            }
            max_tiles_x = @max(max_tiles_x, (qparams.N + TILE - 1) / TILE);
            max_tiles_y = @max(max_tiles_y, (qparams.M + TILE - 1) / TILE);
        }
        params.max_tiles_y = max_tiles_y;

        self.encodeKernel(
            .qmatmul_batch4_f32,
            &buffers,
            params,
            QMatmulBatchKernel.params_index,
            .{ .gx = max_tiles_x, .gy = max_tiles_y * @as(u32, @intCast(indices.len)) },
            MATMUL_THREADS,
        );
    }

    const QMatmulRopeStorePair = struct {
        anchor_slot: usize,
        q_index: usize,
        rope_index: usize,
        store_index: usize,
    };

    fn canEncodeQMatmulRopeStorePair(self: *CompiledProgram, q: anytype, rr: anytype, sa: anytype) bool {
        return self.canEncodeQMatmulBatchOp(q) and
            program_mod.qmatmulRopeStoreTilePairCompatible(q, rr, sa, TILE);
    }

    fn encodeQMatmulRopeStoreBatch(
        self: *CompiledProgram,
        ops: []const backend_mod.DeviceOp,
        pairs: []const QMatmulRopeStorePair,
    ) void {
        if (pairs.len == 0) return;

        const first_q = ops[pairs[0].q_index].qmatmul;
        const first_rr = ops[pairs[0].rope_index].rope;
        const first_sa = ops[pairs[0].store_index].slice_assign;
        const first_w = self.qweight_views[first_q.weight_idx];
        var buffers: [QMatmulRopeStoreKernel.buffer_count]DeviceBuffer = undefined;
        for (0..MAX_QMATMUL_ROPE_STORE_BATCH) |slot| {
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .weight_data)] = first_w.data;
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .weight_scales)] = first_w.scales;
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .input)] = self.device_bufs[first_q.input];
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .output)] = self.device_bufs[first_q.dst];
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .cos_sin)] = self.device_bufs[first_rr.cos_sin];
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .slice_dst)] = self.device_bufs[first_sa.dst];
        }

        var params = std.mem.zeroes(QMatmulRopeStoreBatch4Params);
        params.n_ops = @intCast(pairs.len);
        var max_tiles_x: u32 = 1;
        var max_tiles_y: u32 = 1;
        for (pairs, 0..) |pair, slot| {
            const q = ops[pair.q_index].qmatmul;
            const rr = ops[pair.rope_index].rope;
            const sa = ops[pair.store_index].slice_assign;
            const w = self.qweight_views[q.weight_idx];
            const qparams = qmatmulParams(q, w.block_size);

            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .weight_data)] = w.data;
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .weight_scales)] = w.scales;
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .input)] = self.device_bufs[q.input];
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .output)] = self.device_bufs[q.dst];
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .cos_sin)] = self.device_bufs[rr.cos_sin];
            buffers[QMatmulRopeStoreKernel.bufferIndex(slot, .slice_dst)] = self.device_bufs[sa.dst];

            params.M[slot] = qparams.M;
            params.N[slot] = qparams.N;
            params.K[slot] = qparams.K;
            params.block_size[slot] = qparams.block_size;
            params.input_offset[slot] = qparams.input_offset;
            params.input_row_stride[slot] = qparams.input_row_stride;
            params.dst_offset[slot] = qparams.dst_offset;
            params.dst_row_stride[slot] = qparams.dst_row_stride;
            params.write_primary[slot] = 0;
            params.rope_half_d[slot] = rr.half_d;
            params.rope_src_col_start[slot] = program_mod.qmatmulRopeSrcColStart(q, rr).?;
            params.rope_cs_off[slot] = rr.cs_off;
            params.rope_cs_cs[slot] = rr.cs_cs;
            params.slice_dst_offset[slot] = sa.dst_offset;
            params.slice_dst_row_stride[slot] = sa.dst_row_stride;
            params.slice_dst_col_stride[slot] = sa.dst_col_stride;
            max_tiles_x = @max(max_tiles_x, (rr.half_d + TILE - 1) / TILE);
            max_tiles_y = @max(max_tiles_y, (qparams.M + TILE - 1) / TILE);
        }
        params.max_tiles_y = max_tiles_y;

        self.encodeKernel(
            .qmatmul_rope_store_batch4_f32,
            &buffers,
            params,
            QMatmulRopeStoreKernel.params_index,
            .{ .gx = max_tiles_x, .gy = max_tiles_y * @as(u32, @intCast(pairs.len)) },
            MATMUL_THREADS,
        );
    }

    const QMatvecBatchSidecarKind = enum(u32) {
        none = 0,
        slice = 1,
        rope = 2,
        elementwise = 3,
    };

    const QMatvecBatchSidecarPlan = struct {
        kinds: [MAX_QMATVEC_BATCH]QMatvecBatchSidecarKind = [_]QMatvecBatchSidecarKind{.none} ** MAX_QMATVEC_BATCH,
        rope_indices: [MAX_QMATVEC_BATCH]?usize = [_]?usize{null} ** MAX_QMATVEC_BATCH,
        store_indices: [MAX_QMATVEC_BATCH]?usize = [_]?usize{null} ** MAX_QMATVEC_BATCH,

        fn appendSlice(self: *QMatvecBatchSidecarPlan, slot: usize, idx: usize) bool {
            if (slot >= MAX_QMATVEC_BATCH or self.kinds[slot] != .none) return false;
            self.kinds[slot] = .slice;
            self.store_indices[slot] = idx;
            return true;
        }

        fn appendElementwise(self: *QMatvecBatchSidecarPlan, slot: usize, idx: usize) bool {
            if (slot >= MAX_QMATVEC_BATCH or self.kinds[slot] != .none) return false;
            self.kinds[slot] = .elementwise;
            self.store_indices[slot] = idx;
            return true;
        }

        fn appendRope(self: *QMatvecBatchSidecarPlan, slot: usize, rope_idx: usize, store_idx: ?usize) bool {
            if (slot >= MAX_QMATVEC_BATCH or self.kinds[slot] != .none) return false;
            self.kinds[slot] = .rope;
            self.rope_indices[slot] = rope_idx;
            self.store_indices[slot] = store_idx;
            return true;
        }

        fn appendRopeStore(self: *QMatvecBatchSidecarPlan, slot: usize, rope_idx: usize, store_idx: usize) bool {
            return self.appendRope(slot, rope_idx, store_idx);
        }
    };

    fn canEncodeQMatvecBatchOp(self: *CompiledProgram, q: anytype) bool {
        return q.M == 1 and
            @as(usize, q.weight_idx) < self.qweight_views.len and
            (q.input_row_stride == 0 or q.input_row_stride == q.K) and
            (q.dst_row_stride == 0 or q.dst_row_stride == q.N);
    }

    fn canEncodeQMatvecElementwiseSidecar(_: *CompiledProgram, q: anytype, e: anytype) bool {
        if (!program_mod.qmatvecElementwiseSidecarCompatible(q, e)) return false;
        const primary_is_src0 = e.src0 == q.dst and e.src0_offset == q.dst_offset;
        const secondary_buf = if (primary_is_src0) e.src1 else e.src0;
        return secondary_buf != q.dst;
    }

    fn encodeQMatvecBatch(
        self: *CompiledProgram,
        ops: []const backend_mod.DeviceOp,
        indices: []const usize,
        sidecar_indices: []const ?usize,
    ) void {
        var sidecars = QMatvecBatchSidecarPlan{};
        for (sidecar_indices, 0..) |maybe_idx, slot| {
            const idx = maybe_idx orelse continue;
            switch (ops[idx]) {
                .slice_assign => _ = sidecars.appendSlice(slot, idx),
                .elementwise => _ = sidecars.appendElementwise(slot, idx),
                else => {},
            }
        }
        self.encodeQMatvecBatchWithSidecars(ops, indices, &sidecars);
    }

    fn encodeQMatvecBatchWithSidecars(
        self: *CompiledProgram,
        ops: []const backend_mod.DeviceOp,
        indices: []const usize,
        sidecars: *const QMatvecBatchSidecarPlan,
    ) void {
        const first_q = ops[indices[0]].qmatmul;
        const first_w = self.qweight_views[first_q.weight_idx];
        var buffers: [QMatvecBatchKernel.buffer_count]DeviceBuffer = undefined;
        for (0..MAX_QMATVEC_BATCH) |slot| {
            buffers[QMatvecBatchKernel.bufferIndex(slot, .weight_data)] = first_w.data;
            buffers[QMatvecBatchKernel.bufferIndex(slot, .weight_scales)] = first_w.scales;
            buffers[QMatvecBatchKernel.bufferIndex(slot, .input)] = self.device_bufs[first_q.input];
            buffers[QMatvecBatchKernel.bufferIndex(slot, .output)] = self.device_bufs[first_q.dst];
            buffers[QMatvecBatchKernel.bufferIndex(slot, .sidecar_src)] = self.device_bufs[first_q.input];
            buffers[QMatvecBatchKernel.bufferIndex(slot, .sidecar_dst)] = self.device_bufs[first_q.dst];
        }

        var params = std.mem.zeroes(QMatvecBatch4Params);
        params.n_ops = @intCast(indices.len);
        for (indices, 0..) |op_index, slot| {
            const q = ops[op_index].qmatmul;
            const w = self.qweight_views[q.weight_idx];
            const qparams = qmatmulParams(q, w.block_size);
            buffers[QMatvecBatchKernel.bufferIndex(slot, .weight_data)] = w.data;
            buffers[QMatvecBatchKernel.bufferIndex(slot, .weight_scales)] = w.scales;
            buffers[QMatvecBatchKernel.bufferIndex(slot, .input)] = self.device_bufs[q.input];
            buffers[QMatvecBatchKernel.bufferIndex(slot, .output)] = self.device_bufs[q.dst];
            buffers[QMatvecBatchKernel.bufferIndex(slot, .sidecar_src)] = self.device_bufs[q.input];
            buffers[QMatvecBatchKernel.bufferIndex(slot, .sidecar_dst)] = self.device_bufs[q.dst];
            params.N[slot] = qparams.N;
            params.K[slot] = qparams.K;
            params.block_size[slot] = qparams.block_size;
            params.input_offset[slot] = qparams.input_offset;
            params.dst_offset[slot] = qparams.dst_offset;
            params.write_primary[slot] = 1;
            switch (sidecars.kinds[slot]) {
                .none => {},
                .slice => {
                    const sidecar_index = sidecars.store_indices[slot].?;
                    const sa = ops[sidecar_index].slice_assign;
                    const carried = [_]?usize{sidecar_index};
                    params.write_primary[slot] = @intFromBool(program_mod.projectionPrimaryOutputHasExternalUsersExcept(ops, op_index, &carried));
                    params.sidecar_kind[slot] = @intFromEnum(QMatvecBatchSidecarKind.slice);
                    params.slice_rows[slot] = sa.rows;
                    params.slice_src_col_start[slot] = program_mod.qmatmulSliceSrcColStart(q, sa).?;
                    params.slice_dst_offset[slot] = sa.dst_offset;
                    params.slice_dst_row_stride[slot] = sa.dst_row_stride;
                    buffers[QMatvecBatchKernel.bufferIndex(slot, .sidecar_dst)] = self.device_bufs[sa.dst];
                },
                .elementwise => {
                    const sidecar_index = sidecars.store_indices[slot].?;
                    const e = ops[sidecar_index].elementwise;
                    if (!program_mod.qmatvecElementwiseSidecarCompatible(q, e)) return;
                    const primary_is_src0 = e.src0 == q.dst and e.src0_offset == q.dst_offset;
                    const secondary_buf = if (primary_is_src0) e.src1 else e.src0;
                    const secondary_offset = if (primary_is_src0) e.src1_offset else e.src0_offset;
                    if (secondary_buf == q.dst) return;
                    const carried = [_]?usize{sidecar_index};
                    params.write_primary[slot] = @intFromBool(program_mod.projectionPrimaryOutputHasExternalUsersExcept(ops, op_index, &carried));
                    params.sidecar_kind[slot] = @intFromEnum(QMatvecBatchSidecarKind.elementwise);
                    params.ew_op[slot] = @intFromEnum(e.op);
                    params.ew_is_swapped[slot] = @intFromBool(!primary_is_src0);
                    params.ew_dst_offset[slot] = e.dst_offset;
                    params.ew_secondary_offset[slot] = secondary_offset;
                    buffers[QMatvecBatchKernel.bufferIndex(slot, .sidecar_src)] = self.device_bufs[secondary_buf];
                    buffers[QMatvecBatchKernel.bufferIndex(slot, .sidecar_dst)] = self.device_bufs[e.dst];
                },
                .rope => {
                    const rope_index = sidecars.rope_indices[slot].?;
                    const rr = ops[rope_index].rope;
                    const maybe_store_index = sidecars.store_indices[slot];
                    const carried = [_]?usize{ rope_index, maybe_store_index };
                    params.write_primary[slot] = @intFromBool(program_mod.projectionPrimaryOutputHasExternalUsersExcept(ops, op_index, &carried));
                    params.sidecar_kind[slot] = @intFromEnum(QMatvecBatchSidecarKind.rope);
                    params.slice_rows[slot] = rr.half_d * 2;
                    params.slice_src_col_start[slot] = program_mod.qmatmulRopeSrcColStart(q, rr).?;
                    params.slice_dst_offset[slot] = rr.dst_off;
                    params.slice_dst_row_stride[slot] = 1;
                    params.rope_half_d[slot] = rr.half_d;
                    params.rope_cs_off[slot] = rr.cs_off;
                    buffers[QMatvecBatchKernel.bufferIndex(slot, .sidecar_src)] = self.device_bufs[rr.cos_sin];
                    buffers[QMatvecBatchKernel.bufferIndex(slot, .sidecar_dst)] = self.device_bufs[rr.dst];
                    if (maybe_store_index) |store_index| {
                        const sa = ops[store_index].slice_assign;
                        params.slice_dst_offset[slot] = sa.dst_offset;
                        params.slice_dst_row_stride[slot] = sa.dst_row_stride;
                        buffers[QMatvecBatchKernel.bufferIndex(slot, .sidecar_dst)] = self.device_bufs[sa.dst];
                    }
                },
            }
            params.max_n = @max(params.max_n, qparams.N);
        }

        self.encodeKernel(
            .qmatvec_batch4_f32,
            &buffers,
            params,
            QMatvecBatchKernel.params_index,
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
        self.encodeKernel(.rope_f32, &buffers, params, 3, .{ .gx = linearGrid(rr.half_d * rr.seq_len) }, WG_SIZE);
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
        self.encodeKernel(.rope_batch_f32, &buffers, params, 3, .{ .gx = linearGrid(params.max_n), .gy = @intCast(n) }, WG_SIZE);
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
        self.encodeKernel(.rope_slice_assign_f32, &buffers, params, 3, .{ .gx = linearGrid(rr.half_d * rr.seq_len) }, WG_SIZE);
    }

    fn canEncodeRopeStoreGroupCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        _ = self;
        if (command.anchor_count < 2 or command.anchor_count > MAX_ROPE_BATCH) return false;
        if (command.sidecar_count != command.anchor_count) return false;
        const first_rope_idx = command.indices[0];
        const first_sa_idx = command.sidecar_indices[0] orelse return false;
        if (first_rope_idx >= ops.len or first_sa_idx >= ops.len) return false;
        const first_rope = switch (ops[first_rope_idx]) {
            .rope => |rr| rr,
            else => return false,
        };
        const first_sa = switch (ops[first_sa_idx]) {
            .slice_assign => |sa| sa,
            else => return false,
        };
        if (!program_mod.ropeSliceAssignCompatible(first_rope, first_sa)) return false;

        var i: usize = 1;
        while (i < command.anchor_count) : (i += 1) {
            const rope_idx = command.indices[i];
            const sa_idx = command.sidecar_indices[i] orelse return false;
            if (rope_idx >= ops.len or sa_idx >= ops.len) return false;
            const rr = switch (ops[rope_idx]) {
                .rope => |rr| rr,
                else => return false,
            };
            const sa = switch (ops[sa_idx]) {
                .slice_assign => |sa| sa,
                else => return false,
            };
            if (!program_mod.ropeSliceAssignCompatible(rr, sa)) return false;
            if (!program_mod.ropeStoreGroupCompatible(first_rope, first_sa, rr, sa)) return false;
        }
        return true;
    }

    fn encodeRopeStoreGroupCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) void {
        const first_rope = ops[command.indices[0]].rope;
        const first_sa = ops[command.sidecar_indices[0].?].slice_assign;
        var buffers: [MAX_ROPE_BATCH + 2]DeviceBuffer = undefined;
        for (buffers[0..MAX_ROPE_BATCH]) |*buffer| {
            buffer.* = self.device_bufs[first_rope.src];
        }
        buffers[MAX_ROPE_BATCH] = self.device_bufs[first_rope.cos_sin];
        buffers[MAX_ROPE_BATCH + 1] = self.device_bufs[first_sa.dst];
        var params = std.mem.zeroes(RopeSliceAssignBatchParams);
        params.n_ops = command.anchor_count;
        var i: usize = 0;
        while (i < command.anchor_count) : (i += 1) {
            const rr = ops[command.indices[i]].rope;
            const sa = ops[command.sidecar_indices[i].?].slice_assign;
            buffers[i] = self.device_bufs[rr.src];
            params.half_d[i] = rr.half_d;
            params.seq_len[i] = rr.seq_len;
            params.src_off[i] = rr.src_off;
            params.cs_off[i] = rr.cs_off;
            params.src_rs[i] = rr.src_rs;
            params.src_cs[i] = rr.src_cs;
            params.cs_cs[i] = rr.cs_cs;
            params.dst_offset[i] = sa.dst_offset;
            params.dst_row_stride[i] = sa.dst_row_stride;
            params.dst_col_stride[i] = sa.dst_col_stride;
            params.max_n = @max(params.max_n, rr.half_d * rr.seq_len);
        }
        self.encodeKernel(.rope_slice_assign_batch_f32, &buffers, params, MAX_ROPE_BATCH + 2, .{ .gx = linearGrid(params.max_n), .gy = command.anchor_count }, WG_SIZE);
    }

    fn canFuseQMatvecSliceAssign(self: *CompiledProgram, q: anytype, sa: anytype) bool {
        return @as(usize, q.weight_idx) < self.qweight_views.len and
            program_mod.qmatvecSliceSidecarCompatible(q, sa);
    }

    fn canFuseQMatmulSliceAssign(self: *CompiledProgram, q: anytype, sa: anytype) bool {
        return @as(usize, q.weight_idx) < self.qweight_views.len and
            program_mod.qmatmulSliceSidecarCompatible(q, sa);
    }

    fn encodeQMatvecSliceAssign(self: *CompiledProgram, q: anytype, sa: anytype, write_primary: bool) bool {
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
            .write_primary = @intFromBool(write_primary),
            .slice_rows = sa.rows,
            .slice_cols = sa.cols,
            .slice_src_col_start = program_mod.qmatmulSliceSrcColStart(q, sa).?,
            .slice_dst_offset = sa.dst_offset,
            .slice_dst_row_stride = sa.dst_row_stride,
            .slice_dst_col_stride = sa.dst_col_stride,
        };
        self.encodeKernel(.qmatvec_slice_assign_f32, &buffers, params, 5, .{ .gx = linearGrid(q.N) }, WG_SIZE);
        return true;
    }

    fn encodeQMatmulSliceAssign(self: *CompiledProgram, q: anytype, sa: anytype, write_primary: bool) bool {
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
            .write_primary = @intFromBool(write_primary),
            .slice_rows = sa.rows,
            .slice_cols = sa.cols,
            .slice_src_col_start = program_mod.qmatmulSliceSrcColStart(q, sa).?,
            .slice_dst_offset = sa.dst_offset,
            .slice_dst_row_stride = sa.dst_row_stride,
            .slice_dst_col_stride = sa.dst_col_stride,
        };
        self.encodeKernel(.qmatmul_slice_assign_f32, &buffers, params, 5, matmulGrid(q.M, q.N), MATMUL_THREADS);
        return true;
    }

    fn canFuseQMatmulElementwise(self: *CompiledProgram, q: anytype, e: anytype) bool {
        if (@as(usize, q.weight_idx) >= self.qweight_views.len) return false;
        return program_mod.qmatmulElementwiseSidecarCompatible(q, e);
    }

    fn encodeQMatmulElementwise(self: *CompiledProgram, q: anytype, e: anytype, write_primary: bool) bool {
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
            .write_primary = @intFromBool(write_primary),
            .ew_op = @intFromEnum(e.op),
            .ew_is_swapped = if (q_is_src0) 0 else 1,
            .ew_dst_offset = e.dst_offset,
            .ew_secondary_offset = secondary_offset,
        };
        self.encodeKernel(.qmatmul_elementwise_f32, &buffers, params, 6, matmulGrid(q.M, q.N), MATMUL_THREADS);
        return true;
    }

    fn canFuseQMatmulFusedElementwise(self: *CompiledProgram, q: anytype, fe: anytype) bool {
        if (@as(usize, q.weight_idx) >= self.qweight_views.len) return false;
        if (!canEncodeFusedElementwise(fe)) return false;
        for (fe.steps) |step| {
            if (step.op.isBinary() and step.secondary_buf == q.dst and step.secondary_offset != q.dst_offset) return false;
        }
        return program_mod.qmatmulFusedElementwiseSidecarCompatible(q, fe);
    }

    fn bindQMatmulFusedSecondary(
        self: *CompiledProgram,
        buffers: *[5 + MAX_FUSED_EW_SECONDARIES]DeviceBuffer,
        secondary_bufs: *[MAX_FUSED_EW_SECONDARIES]u16,
        secondary_count: *usize,
        params: *QMatmulFusedEwParams,
        q: anytype,
        step: backend_mod.FusedEwStep,
        step_index: usize,
        repeat_view: ?RepeatSecondaryView,
    ) bool {
        params.secondary_offset[step_index] = step.secondary_offset;
        if (!step.op.isBinary()) return true;

        if (step.secondary_buf == q.dst) {
            if (step.secondary_offset != q.dst_offset) return false;
            params.secondary_is_primary[step_index] = 1;
            return true;
        }

        var secondary_buf = step.secondary_buf;
        if (repeat_view) |rp| {
            if (step.secondary_buf == rp.dst) {
                if (step.secondary_offset < rp.dst_offset) return false;
                const rel = step.secondary_offset - rp.dst_offset;
                if (@as(u64, rel) + @as(u64, params.M * params.N) > @as(u64, rp.n)) return false;
                secondary_buf = rp.src;
                params.secondary_is_repeat[step_index] = 1;
                params.secondary_repeat_dst_offset[step_index] = rel;
                params.secondary_repeat_src_offset[step_index] = rp.src_offset;
                params.secondary_repeat_src_ne[step_index] = rp.src_ne;
                params.secondary_repeat_src_strides[step_index] = rp.src_strides;
                params.secondary_repeat_dst_strides[step_index] = rp.dst_strides;
            }
        }

        const slot = for (secondary_bufs[0..secondary_count.*], 0..) |buf_idx, slot_idx| {
            if (buf_idx == secondary_buf) break slot_idx;
        } else blk: {
            if (secondary_count.* >= MAX_FUSED_EW_SECONDARIES) return false;
            const next = secondary_count.*;
            secondary_bufs[next] = secondary_buf;
            buffers[5 + next] = self.device_bufs[secondary_buf];
            secondary_count.* += 1;
            break :blk next;
        };
        params.secondary_slot[step_index] = @intCast(slot);
        return true;
    }

    fn encodeQMatmulFusedElementwise(self: *CompiledProgram, q: anytype, fe: anytype, write_primary: bool) bool {
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
        params.write_primary = @intFromBool(write_primary);
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
            if (!self.bindQMatmulFusedSecondary(&buffers, &secondary_bufs, &secondary_count, &params, q, step, i, null)) return false;
        }

        self.encodeKernel(
            .qmatmul_fused_elementwise_f32,
            &buffers,
            params,
            13,
            matmulGrid(q.M, q.N),
            MATMUL_THREADS,
        );
        return true;
    }

    fn encodeQMatmulFusedElementwiseChain(self: *CompiledProgram, q: anytype, first: anytype, rp: anytype, second: anytype) bool {
        if (@as(usize, q.weight_idx) >= self.qweight_views.len) return false;
        if (!program_mod.projectionFusedElementwiseChainCompatible(q, first, rp, second)) return false;
        if (first.steps.len + second.steps.len > MAX_FUSED_EW_STEPS) return false;

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
        params.write_primary = 0;
        params.n_steps = @intCast(first.steps.len + second.steps.len);
        params.ew_dst_offset = second.dst_offset;

        var buffers: [5 + MAX_FUSED_EW_SECONDARIES]DeviceBuffer = undefined;
        buffers[0] = w.data;
        buffers[1] = w.scales;
        buffers[2] = self.device_bufs[q.input];
        buffers[3] = self.device_bufs[q.dst];
        buffers[4] = self.device_bufs[second.dst];
        for (buffers[5..]) |*buf| buf.* = self.device_bufs[q.input];

        var secondary_bufs: [MAX_FUSED_EW_SECONDARIES]u16 = undefined;
        var secondary_count: usize = 0;

        var step_index: usize = 0;
        for (first.steps) |step| {
            params.op[step_index] = @intFromEnum(step.op);
            params.is_swapped[step_index] = @intFromBool(step.is_swapped);
            if (!self.bindQMatmulFusedSecondary(&buffers, &secondary_bufs, &secondary_count, &params, q, step, step_index, null)) return false;
            step_index += 1;
        }
        const repeat_view = repeatSecondaryView(rp);
        for (second.steps) |step| {
            params.op[step_index] = @intFromEnum(step.op);
            params.is_swapped[step_index] = @intFromBool(step.is_swapped);
            if (!self.bindQMatmulFusedSecondary(&buffers, &secondary_bufs, &secondary_count, &params, q, step, step_index, repeat_view)) return false;
            step_index += 1;
        }

        self.encodeKernel(
            .qmatmul_fused_elementwise_f32,
            &buffers,
            params,
            13,
            matmulGrid(q.M, q.N),
            MATMUL_THREADS,
        );
        return true;
    }

    fn bindQMatmulPairFusedSecondary(
        self: *CompiledProgram,
        buffers: *[6 + MAX_FUSED_EW_SECONDARIES]DeviceBuffer,
        secondary_bufs: *[MAX_FUSED_EW_SECONDARIES]u16,
        secondary_count: *usize,
        params: *QMatmulPairFusedEwParams,
        gate: anytype,
        up: anytype,
        step: backend_mod.FusedEwStep,
        step_index: usize,
        repeat_view: ?RepeatSecondaryView,
    ) bool {
        params.secondary_offset[step_index] = step.secondary_offset;
        if (!step.op.isBinary()) return true;

        if (step.secondary_buf == gate.dst) {
            if (step.secondary_offset != gate.dst_offset) return false;
            params.secondary_is_primary[step_index] = 1;
            return true;
        }
        if (step.secondary_buf == up.dst) return false;

        var secondary_buf = step.secondary_buf;
        if (repeat_view) |rp| {
            if (step.secondary_buf == rp.dst) {
                if (step.secondary_offset < rp.dst_offset) return false;
                const rel = step.secondary_offset - rp.dst_offset;
                if (@as(u64, rel) + @as(u64, params.M * params.N) > @as(u64, rp.n)) return false;
                secondary_buf = rp.src;
                params.secondary_is_repeat[step_index] = 1;
                params.secondary_repeat_dst_offset[step_index] = rel;
                params.secondary_repeat_src_offset[step_index] = rp.src_offset;
                params.secondary_repeat_src_ne[step_index] = rp.src_ne;
                params.secondary_repeat_src_strides[step_index] = rp.src_strides;
                params.secondary_repeat_dst_strides[step_index] = rp.dst_strides;
            }
        }

        const slot = for (secondary_bufs[0..secondary_count.*], 0..) |buf_idx, slot_idx| {
            if (buf_idx == secondary_buf) break slot_idx;
        } else blk: {
            if (secondary_count.* >= MAX_FUSED_EW_SECONDARIES) return false;
            const next = secondary_count.*;
            secondary_bufs[next] = secondary_buf;
            buffers[6 + next] = self.device_bufs[secondary_buf];
            secondary_count.* += 1;
            break :blk next;
        };
        params.secondary_slot[step_index] = @intCast(slot);
        return true;
    }

    fn encodeQMatmulPairFusedElementwiseChain(
        self: *CompiledProgram,
        gate: anytype,
        first: anytype,
        rp: anytype,
        second: anytype,
        up: anytype,
        product: anytype,
    ) bool {
        if (@as(usize, gate.weight_idx) >= self.qweight_views.len) return false;
        if (@as(usize, up.weight_idx) >= self.qweight_views.len) return false;
        if (!program_mod.projectionPairFusedElementwiseChainCompatible(gate, first, rp, second, up, product)) return false;
        if (first.steps.len + second.steps.len > MAX_FUSED_EW_STEPS) return false;

        const left_w = self.qweight_views[gate.weight_idx];
        const right_w = self.qweight_views[up.weight_idx];
        const gate_params = qmatmulParams(gate, left_w.block_size);
        const up_params = qmatmulParams(up, right_w.block_size);
        if (gate_params.M != up_params.M or gate_params.N != up_params.N or gate_params.K != up_params.K) return false;
        if (gate_params.input_offset != up_params.input_offset or gate_params.input_row_stride != up_params.input_row_stride) return false;
        var params = std.mem.zeroes(QMatmulPairFusedEwParams);
        params.M = gate_params.M;
        params.N = gate_params.N;
        params.K = gate_params.K;
        params.left_block_size = gate_params.block_size;
        params.right_block_size = up_params.block_size;
        params.input_offset = gate_params.input_offset;
        params.input_row_stride = gate_params.input_row_stride;
        params.dst_offset = product.dst_offset;
        params.n_steps = @intCast(first.steps.len + second.steps.len);

        var buffers: [6 + MAX_FUSED_EW_SECONDARIES]DeviceBuffer = undefined;
        buffers[0] = left_w.data;
        buffers[1] = left_w.scales;
        buffers[2] = right_w.data;
        buffers[3] = right_w.scales;
        buffers[4] = self.device_bufs[gate.input];
        buffers[5] = self.device_bufs[product.dst];
        for (buffers[6..]) |*buf| buf.* = self.device_bufs[product.dst];

        var secondary_bufs: [MAX_FUSED_EW_SECONDARIES]u16 = undefined;
        var secondary_count: usize = 0;

        var step_index: usize = 0;
        for (first.steps) |step| {
            params.op[step_index] = @intFromEnum(step.op);
            params.is_swapped[step_index] = @intFromBool(step.is_swapped);
            if (!self.bindQMatmulPairFusedSecondary(&buffers, &secondary_bufs, &secondary_count, &params, gate, up, step, step_index, null)) return false;
            step_index += 1;
        }
        const repeat_view = repeatSecondaryView(rp);
        for (second.steps) |step| {
            params.op[step_index] = @intFromEnum(step.op);
            params.is_swapped[step_index] = @intFromBool(step.is_swapped);
            if (!self.bindQMatmulPairFusedSecondary(&buffers, &secondary_bufs, &secondary_count, &params, gate, up, step, step_index, repeat_view)) return false;
            step_index += 1;
        }

        self.encodeKernel(
            .qmatmul_pair_fused_elementwise_f32,
            &buffers,
            params,
            14,
            matmulGrid(gate.M, gate.N),
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
        var indices: [MAX_SLICE_ASSIGN_BATCH]usize = undefined;
        for (0..n) |i| indices[i] = i;
        self.encodeSliceAssignBatchIndices(ops, indices[0..n]);
    }

    fn canEncodeSliceAssignBatchIndices(_: *CompiledProgram, ops: []const backend_mod.DeviceOp, indices: []const usize) bool {
        if (indices.len < 2 or indices.len > MAX_SLICE_ASSIGN_BATCH) return false;
        if (indices[0] >= ops.len) return false;
        const first = switch (ops[indices[0]]) {
            .slice_assign => |sa| sa,
            else => return false,
        };
        for (indices[1..]) |idx| {
            if (idx >= ops.len) return false;
            const next = switch (ops[idx]) {
                .slice_assign => |sa| sa,
                else => return false,
            };
            if (!sliceAssignBatchCompatible(first, next)) return false;
        }
        return true;
    }

    fn encodeSliceAssignBatchIndices(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, indices: []const usize) void {
        const first = ops[indices[0]].slice_assign;
        const buffers = [_]DeviceBuffer{
            self.device_bufs[first.src],
            self.device_bufs[first.dst],
        };
        var params = std.mem.zeroes(SliceAssignBatchParams);
        params.n_ops = @intCast(indices.len);
        for (indices, 0..) |idx, i| {
            const sa = ops[idx].slice_assign;
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
        self.encodeKernel(.slice_assign_batch_f32, &buffers, params, 2, .{ .gx = linearGrid(params.max_n), .gy = @intCast(indices.len) }, WG_SIZE);
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
        self.encodeKernel(.rmsnorm_scale_f32, &buffers, params, 5, .{ .gx = linearGrid(rn.rows) }, WG_SIZE);
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
        self.encodeKernel(.attention_f32, &buffers, attentionParams(att), 5, .{ .gx = att.seq_q }, WG_SIZE);
    }

    fn encodeAttentionSliceAssign(self: *CompiledProgram, sa: anytype, att: anytype) bool {
        const operand = program_mod.attentionSliceAssignOperand(sa, att) orelse return false;
        if (!canEncodeAttention(att)) return false;

        var buffers = [_]DeviceBuffer{
            self.device_bufs[att.q],
            self.device_bufs[att.k],
            self.device_bufs[att.v],
            self.device_bufs[att.mask],
            self.device_bufs[att.dst],
            self.device_bufs[sa.src],
            self.device_bufs[sa.dst],
        };
        var params = attentionSliceAssignParams(att, sa);
        switch (operand) {
            .q => {
                buffers[0] = self.device_bufs[sa.src];
                params.q_off = sa.src_offset;
                params.q_rs = sa.src_row_stride;
                params.q_cs = sa.src_col_stride;
            },
            .k => {
                buffers[1] = self.device_bufs[sa.src];
                params.k_off = sa.src_offset;
                params.k_rs = sa.src_row_stride;
                params.k_cs = sa.src_col_stride;
            },
            .v => {
                buffers[2] = self.device_bufs[sa.src];
                params.v_off = sa.src_offset;
                params.v_rs = sa.src_row_stride;
                params.v_cs = sa.src_col_stride;
            },
        }
        self.encodeKernel(.attention_slice_assign_f32, &buffers, params, 7, .{ .gx = att.seq_q }, WG_SIZE);
        return true;
    }

    fn encodeAttentionSliceStore(self: *CompiledProgram, att: anytype, sa: anytype) bool {
        if (!program_mod.attentionSliceStoreCompatible(att, sa)) return false;
        if (!canEncodeAttention(att)) return false;
        const buffers = [_]DeviceBuffer{
            self.device_bufs[att.q],
            self.device_bufs[att.k],
            self.device_bufs[att.v],
            self.device_bufs[att.mask],
            self.device_bufs[att.dst],
            self.device_bufs[sa.dst],
        };
        self.encodeKernel(.attention_store_f32, &buffers, attentionStoreParams(att, sa), 6, .{ .gx = att.seq_q }, WG_SIZE);
        return true;
    }

    fn encodeAttentionRopeStore(self: *CompiledProgram, rr: anytype, att: anytype, sa: anytype) bool {
        if (!program_mod.ropeAttentionCompatible(rr, att)) return false;
        if (!program_mod.attentionSliceStoreCompatible(att, sa)) return false;
        if (!canEncodeAttention(att)) return false;
        if (rr.half_d * 2 != att.d_head) return false;
        const buffers = [_]DeviceBuffer{
            self.device_bufs[rr.src],
            self.device_bufs[rr.cos_sin],
            self.device_bufs[att.k],
            self.device_bufs[att.v],
            self.device_bufs[att.mask],
            self.device_bufs[att.dst],
            self.device_bufs[sa.dst],
        };
        self.encodeKernel(.attention_rope_store_f32, &buffers, attentionRopeStoreParams(rr, att, sa), 7, .{ .gx = att.seq_q }, WG_SIZE);
        return true;
    }

    fn canEncodeAttentionRopeStoreSharedBatchCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        _ = self;
        if (command.sidecar_count < 2 or command.sidecar_count > MAX_ATTENTION_ROPE_STORE_SHARED_BATCH_HEADS) return false;
        if (!program_mod.ropeAttentionStoreCompactBatchCompatible(ops, &command)) return false;
        var i: usize = 0;
        while (i < command.sidecar_count) : (i += 1) {
            const att_idx = command.indices[i * 2 + 1];
            const sa_idx = command.sidecar_indices[i] orelse return false;
            const att = switch (ops[att_idx]) {
                .attention => |att| att,
                else => return false,
            };
            const sa = switch (ops[sa_idx]) {
                .slice_assign => |sa| sa,
                else => return false,
            };
            if (!canEncodeAttention(att)) return false;
            if (!program_mod.canFuseAttentionStoreSidecar(ops, att_idx, sa_idx, sa)) return false;
        }
        return true;
    }

    fn encodeAttentionRopeStoreSharedBatchCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) void {
        const first_rope = ops[command.indices[0]].rope;
        const first_att = ops[command.indices[1]].attention;
        const first_sa = ops[command.sidecar_indices[0].?].slice_assign;
        const buffers = [_]DeviceBuffer{
            self.device_bufs[first_rope.src],
            self.device_bufs[first_rope.cos_sin],
            self.device_bufs[first_att.k],
            self.device_bufs[first_att.v],
            self.device_bufs[first_att.mask],
            self.device_bufs[first_sa.dst],
        };

        var params = std.mem.zeroes(AttentionRopeStoreSharedBatchParams);
        params.n_heads = @intCast(command.sidecar_count);
        params.d_head = first_att.d_head;
        params.seq_q = first_att.seq_q;
        params.seq_kv = first_att.seq_kv;
        params.scale = first_att.scale;
        params.rope_half_d = first_rope.half_d;
        params.rope_src_rs = first_rope.src_rs;
        params.rope_src_cs = first_rope.src_cs;
        params.rope_cs_cs = first_rope.cs_cs;
        params.k_rs = first_att.k_rs;
        params.k_cs = first_att.k_cs;
        params.v_rs = first_att.v_rs;
        params.v_cs = first_att.v_cs;
        params.mask_rs = first_att.mask_rs;
        params.mask_cs = first_att.mask_cs;
        params.slice_dst_row_stride = first_sa.dst_row_stride;
        params.slice_dst_col_stride = first_sa.dst_col_stride;

        var i: usize = 0;
        while (i < command.sidecar_count) : (i += 1) {
            const rr = ops[command.indices[i * 2]].rope;
            const att = ops[command.indices[i * 2 + 1]].attention;
            const sa = ops[command.sidecar_indices[i].?].slice_assign;
            params.rope_src_off[i] = rr.src_off;
            params.rope_cs_off[i] = rr.cs_off;
            params.k_off[i] = att.k_off;
            params.v_off[i] = att.v_off;
            params.mask_off[i] = att.mask_off;
            params.slice_dst_offset[i] = sa.dst_offset;
        }

        self.encodeKernel(.attention_rope_store_shared_batch_f32, &buffers, params, @intCast(buffers.len), .{ .gx = first_att.seq_q, .gy = @intCast(command.sidecar_count) }, WG_SIZE);
    }

    fn canEncodeAttentionRopeStoreBatchCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        _ = self;
        if (command.sidecar_count < 2 or command.sidecar_count > MAX_ATTENTION_STORE_BATCH_HEADS) return false;
        if (command.anchor_count != command.sidecar_count * 2) return false;
        const first_rope_idx = command.indices[0];
        const first_att_idx = command.indices[1];
        const first_sa_idx = command.sidecar_indices[0] orelse return false;
        if (first_rope_idx >= ops.len or first_att_idx >= ops.len or first_sa_idx >= ops.len) return false;
        const first_rope = switch (ops[first_rope_idx]) {
            .rope => |rr| rr,
            else => return false,
        };
        const first_att = switch (ops[first_att_idx]) {
            .attention => |att| att,
            else => return false,
        };
        const first_sa = switch (ops[first_sa_idx]) {
            .slice_assign => |sa| sa,
            else => return false,
        };
        if (!canEncodeAttention(first_att)) return false;
        if (!program_mod.ropeAttentionCompatible(first_rope, first_att)) return false;
        if (!program_mod.attentionSliceStoreCompatible(first_att, first_sa)) return false;
        if (!program_mod.canFuseAttentionStoreSidecar(ops, first_att_idx, first_sa_idx, first_sa)) return false;

        var i: usize = 1;
        while (i < command.sidecar_count) : (i += 1) {
            const rope_idx = command.indices[i * 2];
            const att_idx = command.indices[i * 2 + 1];
            const sa_idx = command.sidecar_indices[i] orelse return false;
            if (rope_idx >= ops.len or att_idx >= ops.len or sa_idx >= ops.len) return false;
            const rr = switch (ops[rope_idx]) {
                .rope => |rr| rr,
                else => return false,
            };
            const att = switch (ops[att_idx]) {
                .attention => |att| att,
                else => return false,
            };
            const sa = switch (ops[sa_idx]) {
                .slice_assign => |sa| sa,
                else => return false,
            };
            if (!canEncodeAttention(att)) return false;
            if (!program_mod.ropeAttentionCompatible(rr, att)) return false;
            if (!program_mod.ropeStoreBatchGeometryCompatible(first_rope, rr)) return false;
            if (!program_mod.attentionGeometryCompatible(first_att, att)) return false;
            if (!program_mod.attentionSliceStoreCompatible(att, sa)) return false;
            if (!program_mod.canFuseAttentionStoreSidecar(ops, att_idx, sa_idx, sa)) return false;
            if (sa.dst_row_stride != first_sa.dst_row_stride or
                sa.dst_col_stride != first_sa.dst_col_stride) return false;
        }
        return true;
    }

    fn encodeAttentionRopeStoreBatchCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) void {
        const first_rope = ops[command.indices[0]].rope;
        const first_att = ops[command.indices[1]].attention;
        const first_sa = ops[command.sidecar_indices[0].?].slice_assign;
        const q_src_base = 0;
        const cos_sin_base = q_src_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        const k_base = cos_sin_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        const v_base = k_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        const mask_base = v_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        const dst_base = mask_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        const slice_dst_base = dst_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        var buffers: [MAX_ATTENTION_STORE_BATCH_HEADS * 7]DeviceBuffer = undefined;
        for (0..MAX_ATTENTION_STORE_BATCH_HEADS) |i| {
            buffers[q_src_base + i] = self.device_bufs[first_rope.src];
            buffers[cos_sin_base + i] = self.device_bufs[first_rope.cos_sin];
            buffers[k_base + i] = self.device_bufs[first_att.k];
            buffers[v_base + i] = self.device_bufs[first_att.v];
            buffers[mask_base + i] = self.device_bufs[first_att.mask];
            buffers[dst_base + i] = self.device_bufs[first_att.dst];
            buffers[slice_dst_base + i] = self.device_bufs[first_sa.dst];
        }

        var params = std.mem.zeroes(AttentionRopeStoreBatchParams);
        params.n_heads = @intCast(command.sidecar_count);
        params.d_head = first_att.d_head;
        params.seq_q = first_att.seq_q;
        params.seq_kv = first_att.seq_kv;
        params.scale = first_att.scale;
        params.rope_half_d = first_rope.half_d;
        params.rope_src_rs = first_rope.src_rs;
        params.rope_src_cs = first_rope.src_cs;
        params.rope_cs_cs = first_rope.cs_cs;
        params.k_rs = first_att.k_rs;
        params.k_cs = first_att.k_cs;
        params.v_rs = first_att.v_rs;
        params.v_cs = first_att.v_cs;
        params.mask_rs = first_att.mask_rs;
        params.mask_cs = first_att.mask_cs;
        params.dst_rs = first_att.dst_rs;
        params.dst_cs = first_att.dst_cs;
        params.slice_dst_row_stride = first_sa.dst_row_stride;
        params.slice_dst_col_stride = first_sa.dst_col_stride;

        var i: usize = 0;
        while (i < command.sidecar_count) : (i += 1) {
            const rr = ops[command.indices[i * 2]].rope;
            const att = ops[command.indices[i * 2 + 1]].attention;
            const sa = ops[command.sidecar_indices[i].?].slice_assign;
            buffers[q_src_base + i] = self.device_bufs[rr.src];
            buffers[cos_sin_base + i] = self.device_bufs[rr.cos_sin];
            buffers[k_base + i] = self.device_bufs[att.k];
            buffers[v_base + i] = self.device_bufs[att.v];
            buffers[mask_base + i] = self.device_bufs[att.mask];
            buffers[dst_base + i] = self.device_bufs[att.dst];
            buffers[slice_dst_base + i] = self.device_bufs[sa.dst];
            params.rope_src_off[i] = rr.src_off;
            params.rope_cs_off[i] = rr.cs_off;
            params.k_off[i] = att.k_off;
            params.v_off[i] = att.v_off;
            params.mask_off[i] = att.mask_off;
            params.dst_off[i] = att.dst_off;
            params.slice_dst_offset[i] = sa.dst_offset;
        }

        self.encodeKernel(.attention_rope_store_batch_f32, &buffers, params, @intCast(buffers.len), .{ .gx = first_att.seq_q, .gy = @intCast(command.sidecar_count) }, WG_SIZE);
    }

    fn canEncodeAttentionStoreBatchCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        _ = self;
        if (command.anchor_count < 2 or command.anchor_count > MAX_ATTENTION_STORE_BATCH_HEADS) return false;
        if (command.sidecar_count != command.anchor_count) return false;
        const first_idx = command.indices[0];
        const first_sa_idx = command.sidecar_indices[0] orelse return false;
        if (first_idx >= ops.len or first_sa_idx >= ops.len) return false;
        const first = switch (ops[first_idx]) {
            .attention => |att| att,
            else => return false,
        };
        const first_sa = switch (ops[first_sa_idx]) {
            .slice_assign => |sa| sa,
            else => return false,
        };
        if (!canEncodeAttention(first)) return false;
        if (!program_mod.attentionSliceStoreCompatible(first, first_sa)) return false;
        if (!program_mod.canFuseAttentionStoreSidecar(ops, first_idx, first_sa_idx, first_sa)) return false;

        for (command.anchorIndices()[1..], command.sidecarIndices()[1..]) |idx, maybe_sa_idx| {
            const sa_idx = maybe_sa_idx orelse return false;
            if (idx >= ops.len or sa_idx >= ops.len) return false;
            const att = switch (ops[idx]) {
                .attention => |att| att,
                else => return false,
            };
            const sa = switch (ops[sa_idx]) {
                .slice_assign => |sa| sa,
                else => return false,
            };
            if (!canEncodeAttention(att)) return false;
            if (!program_mod.attentionGeometryCompatible(first, att)) return false;
            if (!program_mod.attentionSliceStoreCompatible(att, sa)) return false;
            if (!program_mod.canFuseAttentionStoreSidecar(ops, idx, sa_idx, sa)) return false;
            if (sa.dst_row_stride != first_sa.dst_row_stride or
                sa.dst_col_stride != first_sa.dst_col_stride) return false;
        }
        return true;
    }

    fn encodeAttentionStoreBatchCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) void {
        const first = ops[command.indices[0]].attention;
        const first_sa = ops[command.sidecar_indices[0].?].slice_assign;
        const q_base = 0;
        const k_base = q_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        const v_base = k_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        const mask_base = v_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        const dst_base = mask_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        const slice_dst_base = dst_base + MAX_ATTENTION_STORE_BATCH_HEADS;
        var buffers: [MAX_ATTENTION_STORE_BATCH_HEADS * 6]DeviceBuffer = undefined;
        for (0..MAX_ATTENTION_STORE_BATCH_HEADS) |i| {
            buffers[q_base + i] = self.device_bufs[first.q];
            buffers[k_base + i] = self.device_bufs[first.k];
            buffers[v_base + i] = self.device_bufs[first.v];
            buffers[mask_base + i] = self.device_bufs[first.mask];
            buffers[dst_base + i] = self.device_bufs[first.dst];
            buffers[slice_dst_base + i] = self.device_bufs[first_sa.dst];
        }
        var params = std.mem.zeroes(AttentionStoreBatchParams);
        params.n_heads = @intCast(command.anchor_count);
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
        params.slice_dst_row_stride = first_sa.dst_row_stride;
        params.slice_dst_col_stride = first_sa.dst_col_stride;
        for (command.anchorIndices(), command.sidecarIndices(), 0..) |idx, maybe_sa_idx, i| {
            const att = ops[idx].attention;
            const sa = ops[maybe_sa_idx.?].slice_assign;
            buffers[q_base + i] = self.device_bufs[att.q];
            buffers[k_base + i] = self.device_bufs[att.k];
            buffers[v_base + i] = self.device_bufs[att.v];
            buffers[mask_base + i] = self.device_bufs[att.mask];
            buffers[dst_base + i] = self.device_bufs[att.dst];
            buffers[slice_dst_base + i] = self.device_bufs[sa.dst];
            params.q_off[i] = att.q_off;
            params.k_off[i] = att.k_off;
            params.v_off[i] = att.v_off;
            params.mask_off[i] = att.mask_off;
            params.dst_off[i] = att.dst_off;
            params.slice_dst_offset[i] = sa.dst_offset;
        }
        self.encodeKernel(.attention_store_batch_f32, &buffers, params, @intCast(buffers.len), .{ .gx = first.seq_q, .gy = @intCast(command.anchor_count) }, WG_SIZE);
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

    fn canEncodeAttentionBatchIndices(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, indices: []const usize) bool {
        _ = self;
        if (indices.len < 2 or indices.len > MAX_ATTENTION_BATCH_HEADS) return false;
        if (indices[0] >= ops.len) return false;
        const first = switch (ops[indices[0]]) {
            .attention => |att| att,
            else => return false,
        };
        if (!canEncodeAttention(first)) return false;
        for (indices[1..]) |idx| {
            if (idx >= ops.len) return false;
            const next = switch (ops[idx]) {
                .attention => |att| att,
                else => return false,
            };
            if (!canEncodeAttention(next) or !attentionBatchCompatible(first, next)) return false;
        }
        return true;
    }

    fn encodeAttentionBatch(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, n: usize) void {
        var indices: [MAX_ATTENTION_BATCH_HEADS]usize = undefined;
        for (0..n) |i| indices[i] = i;
        self.encodeAttentionBatchIndices(ops, indices[0..n]);
    }

    fn encodeAttentionBatchIndices(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, indices: []const usize) void {
        const first = ops[indices[0]].attention;
        const buffers = [_]DeviceBuffer{
            self.device_bufs[first.q],
            self.device_bufs[first.k],
            self.device_bufs[first.v],
            self.device_bufs[first.mask],
            self.device_bufs[first.dst],
        };
        var params = std.mem.zeroes(AttentionBatchParams);
        params.n_heads = @intCast(indices.len);
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
        for (indices, 0..) |idx, i| {
            const att = ops[idx].attention;
            params.q_off[i] = att.q_off;
            params.k_off[i] = att.k_off;
            params.v_off[i] = att.v_off;
            params.mask_off[i] = att.mask_off;
            params.dst_off[i] = att.dst_off;
        }
        self.encodeKernel(.attention_batch_f32, &buffers, params, 5, .{ .gx = first.seq_q, .gy = @intCast(indices.len) }, WG_SIZE);
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

    fn recordRegionFusedRunFromCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand, elapsed_ns: u64) void {
        var indices = command.coveredIndexIterator();
        const count = indices.remainingCount();
        if (count == 0) return;
        const per_op = elapsed_ns / count;
        var used: u64 = 0;
        var i: usize = 0;
        while (indices.next()) |idx| : (i += 1) {
            if (idx >= ops.len) continue;
            const t = if (i + 1 == count) elapsed_ns - used else per_op;
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
                for (command.sidecarIndices(), 0..) |maybe_idx, slot| {
                    const idx = maybe_idx orelse continue;
                    const q = ops[command.indices[slot]].qmatmul;
                    switch (ops[idx]) {
                        .slice_assign => |sa| if (!program_mod.qmatvecSliceSidecarCompatible(q, sa)) return false,
                        .elementwise => |e| if (!self.canEncodeQMatvecElementwiseSidecar(q, e)) return false,
                        else => return false,
                    }
                }

                const t0 = nowNs();
                self.encodeQMatvecBatch(ops, command.anchorIndices(), command.sidecarIndices());
                self.recordRegionFusedRunFromIndices(ops, command.anchorIndices(), @intCast(nowNs() - t0));
                for (command.sidecarIndices()) |maybe_idx| {
                    if (maybe_idx) |idx| self.recordRegionBackendOp(ops[idx], 0);
                }
                return true;
            },
            .qmatmul => {
                for (command.anchorIndices()) |idx| {
                    if (!self.canEncodeQMatmulBatchOp(ops[idx].qmatmul)) return false;
                }
                for (command.sidecarIndices(), 0..) |maybe_idx, slot| {
                    const idx = maybe_idx orelse continue;
                    const q = ops[command.indices[slot]].qmatmul;
                    switch (ops[idx]) {
                        .slice_assign => |sa| if (!self.canFuseQMatmulSliceAssign(q, sa)) return false,
                        .elementwise => |e| if (!self.canFuseQMatmulElementwise(q, e)) return false,
                        else => return false,
                    }
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

    fn tryEncodeProjectionCacheCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (command.projection_kind == .qmatvec) return self.tryEncodeQMatvecProjectionCacheCommand(ops, command);
        if (command.projection_kind != .qmatmul) return false;
        if (command.anchor_count == 0 or command.anchor_count > MAX_QMATMUL_BATCH) return false;
        for (command.anchorIndices()) |idx| {
            if (idx >= ops.len) return false;
            if (!self.canEncodeQMatmulBatchOp(ops[idx].qmatmul)) return false;
        }

        var direct_sidecars = QMatmulBatchSidecarPlan{};
        var direct_sidecar_dst = [_]?u16{null} ** MAX_QMATMUL_BATCH;
        var rope_pairs: [MAX_QMATMUL_ROPE_STORE_BATCH]QMatmulRopeStorePair = undefined;
        var rope_pair_count: usize = 0;
        var anchor_has_rope = [_]bool{false} ** MAX_QMATMUL_BATCH;

        var flat_i: usize = 0;
        while (flat_i < command.sidecar_count) : (flat_i += 1) {
            const idx = command.sidecar_indices[flat_i] orelse continue;
            if (idx >= ops.len) return false;
            switch (ops[idx]) {
                .slice_assign => |sa| {
                    const slot = for (command.anchorIndices(), 0..) |anchor_idx, slot| {
                        const q = ops[anchor_idx].qmatmul;
                        if (program_mod.qmatmulSliceSidecarCompatible(q, sa)) break slot;
                    } else return false;
                    if (direct_sidecar_dst[slot]) |dst| {
                        if (dst != sa.dst) return false;
                    } else {
                        direct_sidecar_dst[slot] = sa.dst;
                    }
                    if (!direct_sidecars.appendSlice(slot, idx)) return false;
                },
                .rope => |rr| {
                    if (flat_i + 1 >= command.sidecar_count) return false;
                    const store_idx = command.sidecar_indices[flat_i + 1] orelse return false;
                    if (store_idx >= ops.len) return false;
                    const sa = switch (ops[store_idx]) {
                        .slice_assign => |sa| sa,
                        else => return false,
                    };
                    const slot = for (command.anchorIndices(), 0..) |anchor_idx, slot| {
                        const q = ops[anchor_idx].qmatmul;
                        if (self.canEncodeQMatmulRopeStorePair(q, rr, sa)) break slot;
                    } else return false;
                    if (anchor_has_rope[slot]) return false;
                    if (rope_pair_count >= MAX_QMATMUL_ROPE_STORE_BATCH) return false;
                    rope_pairs[rope_pair_count] = .{
                        .anchor_slot = slot,
                        .q_index = command.indices[slot],
                        .rope_index = idx,
                        .store_index = store_idx,
                    };
                    rope_pair_count += 1;
                    anchor_has_rope[slot] = true;
                    flat_i += 1;
                },
                else => return false,
            }
        }

        const t0 = nowNs();
        if (rope_pair_count == 0) {
            self.encodeQMatmulBatchWithSidecars(ops, command.anchorIndices(), &direct_sidecars);
        } else {
            var normal_indices: [MAX_QMATMUL_BATCH]usize = undefined;
            var normal_slot_for_anchor = [_]?usize{null} ** MAX_QMATMUL_BATCH;
            var normal_count: usize = 0;
            for (command.anchorIndices(), 0..) |anchor_idx, anchor_slot| {
                const has_direct_sidecars = direct_sidecars.counts[anchor_slot] != 0;
                const primary_needed = program_mod.projectionPrimaryOutputHasExternalUsersExcept(ops, anchor_idx, command.carriedSidecarIndices());
                if (!anchor_has_rope[anchor_slot] or has_direct_sidecars or primary_needed) {
                    normal_slot_for_anchor[anchor_slot] = normal_count;
                    normal_indices[normal_count] = anchor_idx;
                    normal_count += 1;
                }
            }

            var normal_sidecars = QMatmulBatchSidecarPlan{};
            for (0..command.anchor_count) |anchor_slot| {
                const normal_slot = normal_slot_for_anchor[anchor_slot] orelse {
                    if (direct_sidecars.counts[anchor_slot] != 0) return false;
                    continue;
                };
                for (direct_sidecars.slotIndices(anchor_slot)) |maybe_idx| {
                    if (maybe_idx) |sidecar_idx| {
                        if (!normal_sidecars.appendSlice(normal_slot, sidecar_idx)) return false;
                    }
                }
            }

            if (normal_count > 0) {
                self.encodeQMatmulBatchWithSidecars(ops, normal_indices[0..normal_count], &normal_sidecars);
            }
            self.encodeQMatmulRopeStoreBatch(ops, rope_pairs[0..rope_pair_count]);
        }
        self.recordRegionFusedRunFromIndices(ops, command.anchorIndices(), @intCast(nowNs() - t0));
        for (command.carriedSidecarIndices()) |maybe_idx| {
            if (maybe_idx) |idx| self.recordRegionBackendOp(ops[idx], 0);
        }
        return true;
    }

    fn tryEncodeQMatvecProjectionCacheCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (command.anchor_count == 0 or command.anchor_count > MAX_QMATVEC_BATCH) return false;
        for (command.anchorIndices()) |idx| {
            if (idx >= ops.len) return false;
            if (!self.canEncodeQMatvecBatchOp(ops[idx].qmatmul)) return false;
        }

        var sidecars = QMatvecBatchSidecarPlan{};
        var flat_i: usize = 0;
        while (flat_i < command.sidecar_count) : (flat_i += 1) {
            const idx = command.sidecar_indices[flat_i] orelse continue;
            if (idx >= ops.len) return false;
            switch (ops[idx]) {
                .slice_assign => |sa| {
                    const slot = for (command.anchorIndices(), 0..) |anchor_idx, slot| {
                        const q = ops[anchor_idx].qmatmul;
                        if (program_mod.qmatvecSliceSidecarCompatible(q, sa)) break slot;
                    } else return false;
                    if (!sidecars.appendSlice(slot, idx)) return false;
                },
                .elementwise => |e| {
                    const slot = for (command.anchorIndices(), 0..) |anchor_idx, slot| {
                        const q = ops[anchor_idx].qmatmul;
                        if (self.canEncodeQMatvecElementwiseSidecar(q, e)) break slot;
                    } else return false;
                    if (!sidecars.appendElementwise(slot, idx)) return false;
                },
                .rope => |rr| {
                    var maybe_store_idx: ?usize = null;
                    const slot = for (command.anchorIndices(), 0..) |anchor_idx, slot| {
                        const q = ops[anchor_idx].qmatmul;
                        if (program_mod.qmatvecRopeSidecarCompatible(q, rr)) break slot;
                    } else return false;
                    if (flat_i + 1 < command.sidecar_count) {
                        const store_idx = command.sidecar_indices[flat_i + 1] orelse return false;
                        if (store_idx >= ops.len) return false;
                        if (ops[store_idx] == .slice_assign) {
                            const store = ops[store_idx].slice_assign;
                            const q = ops[command.indices[slot]].qmatmul;
                            if (program_mod.qmatvecRopeStoreSidecarCompatible(q, rr, store)) {
                                maybe_store_idx = store_idx;
                                flat_i += 1;
                            }
                        }
                    }
                    if (!sidecars.appendRope(slot, idx, maybe_store_idx)) return false;
                },
                else => return false,
            }
        }

        const t0 = nowNs();
        self.encodeQMatvecBatchWithSidecars(ops, command.anchorIndices(), &sidecars);
        self.recordRegionFusedRunFromIndices(ops, command.anchorIndices(), @intCast(nowNs() - t0));
        for (command.carriedSidecarIndices()) |maybe_idx| {
            if (maybe_idx) |idx| self.recordRegionBackendOp(ops[idx], 0);
        }
        return true;
    }

    fn tryEncodeProjectionChainCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (command.anchor_count != 1 or command.sidecar_count != 1) return false;
        const anchor_idx = command.indices[0];
        const sidecar_idx = command.sidecar_indices[0] orelse return false;
        if (anchor_idx >= ops.len or sidecar_idx >= ops.len) return false;
        const q = switch (ops[anchor_idx]) {
            .qmatmul => |q| q,
            else => return false,
        };
        const write_primary = program_mod.projectionPrimaryOutputHasExternalUsers(ops, anchor_idx, sidecar_idx);
        return switch (ops[sidecar_idx]) {
            .slice_assign => |sa| if (q.M == 1) self.encodeQMatvecSliceAssign(q, sa, write_primary) else self.encodeQMatmulSliceAssign(q, sa, write_primary),
            .elementwise => |e| self.encodeQMatmulElementwise(q, e, write_primary),
            .fused_elementwise => |fe| self.encodeQMatmulFusedElementwise(q, fe, write_primary),
            else => false,
        };
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
                self.encodeKernel(.matmul_f32, &buffers, matmulParams(m.geom), 3, matmulGrid(m.geom.M, m.geom.N), MATMUL_THREADS);
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
                self.encodeKernel(.qmatmul_f32, &buffers, qmatmulParams(q, w.block_size), 4, matmulGrid(q.M, q.N), MATMUL_THREADS);
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

    fn tryEncodeAttentionChainCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (command.anchor_count != 1 or command.sidecar_count != 1) return false;
        const att_idx = command.indices[0];
        const sa_idx = command.sidecar_indices[0] orelse return false;
        const sa = deviceOpAt(.slice_assign, ops, sa_idx) orelse return false;
        const att = deviceOpAt(.attention, ops, att_idx) orelse return false;
        return self.encodeAttentionSliceAssign(sa, att);
    }

    fn tryEncodeAttentionStoreChainCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (command.anchor_count != 1 or command.sidecar_count != 1) return false;
        const att_idx = command.indices[0];
        const sa_idx = command.sidecar_indices[0] orelse return false;
        const att = deviceOpAt(.attention, ops, att_idx) orelse return false;
        const sa = deviceOpAt(.slice_assign, ops, sa_idx) orelse return false;
        return self.encodeAttentionSliceStore(att, sa);
    }

    fn tryEncodeAttentionStoreGroupCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (!self.canEncodeAttentionStoreBatchCommand(ops, command)) return false;
        self.encodeAttentionStoreBatchCommand(ops, command);
        return true;
    }

    fn tryEncodeRopeAttentionStoreChainCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (command.anchor_count != 2 or command.sidecar_count != 1) return false;
        const rope_idx = command.indices[0];
        const att_idx = command.indices[1];
        const sa_idx = command.sidecar_indices[0] orelse return false;
        const rr = deviceOpAt(.rope, ops, rope_idx) orelse return false;
        const att = deviceOpAt(.attention, ops, att_idx) orelse return false;
        const sa = deviceOpAt(.slice_assign, ops, sa_idx) orelse return false;
        return self.encodeAttentionRopeStore(rr, att, sa);
    }

    fn tryEncodeRopeAttentionStoreGroupCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (self.canEncodeAttentionRopeStoreSharedBatchCommand(ops, command)) {
            self.encodeAttentionRopeStoreSharedBatchCommand(ops, command);
            return true;
        }
        if (!self.canEncodeAttentionRopeStoreBatchCommand(ops, command)) return false;
        self.encodeAttentionRopeStoreBatchCommand(ops, command);
        return true;
    }

    fn tryEncodeAttentionBatchCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        const start: usize = @intCast(command.op_start);
        const n: usize = @intCast(command.op_count);
        if (self.attentionBatchRunLen(ops[start..]) < n) return false;
        self.encodeAttentionBatch(ops[start..], n);
        return true;
    }

    fn tryEncodeAttentionGroupCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (!self.canEncodeAttentionBatchIndices(ops, command.anchorIndices())) return false;
        self.encodeAttentionBatchIndices(ops, command.anchorIndices());
        return true;
    }

    fn tryEncodeRopeBatchCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        const start: usize = @intCast(command.op_start);
        const n: usize = @intCast(command.op_count);
        if (ropeBatchRunLen(ops[start..]) < n) return false;
        self.encodeRopeBatch(ops[start..], n);
        return true;
    }

    fn tryEncodeRopeStoreGroupCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (!self.canEncodeRopeStoreGroupCommand(ops, command)) return false;
        self.encodeRopeStoreGroupCommand(ops, command);
        return true;
    }

    fn tryEncodeMovementBatchCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        const start: usize = @intCast(command.op_start);
        const n: usize = @intCast(command.op_count);
        if (sliceAssignBatchRunLen(ops[start..]) < n) return false;
        self.encodeSliceAssignBatch(ops[start..], n);
        return true;
    }

    fn tryEncodeMovementGroupCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        if (!self.canEncodeSliceAssignBatchIndices(ops, command.anchorIndices())) return false;
        self.encodeSliceAssignBatchIndices(ops, command.anchorIndices());
        return true;
    }

    fn tryEncodeElementwiseBatchCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        for (command.anchorIndices()) |idx| {
            const e = deviceOpAt(.elementwise, ops, idx) orelse return false;
            if (!canEncodeElementwiseBatchOp(e)) return false;
        }
        self.encodeElementwiseBatch(ops, command.anchorIndices());
        return true;
    }

    fn tryEncodeRepeatFusedElementwiseChainCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        const start: usize = @intCast(command.op_start);
        const rp = deviceOpAt(.repeat, ops, start) orelse return false;
        const fe = deviceOpAt(.fused_elementwise, ops, start + 1) orelse return false;
        return self.encodeRepeatFusedElementwise(rp, fe);
    }

    fn tryEncodeProjectionFusedElementwiseChainCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        const start: usize = @intCast(command.op_start);
        const q = deviceOpAt(.qmatmul, ops, start) orelse return false;
        const first = deviceOpAt(.fused_elementwise, ops, start + 1) orelse return false;
        const rp = deviceOpAt(.repeat, ops, start + 2) orelse return false;
        const second = deviceOpAt(.fused_elementwise, ops, start + 3) orelse return false;
        return self.encodeQMatmulFusedElementwiseChain(q, first, rp, second);
    }

    fn tryEncodeProjectionPairFusedElementwiseChainCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        const start: usize = @intCast(command.op_start);
        const gate = deviceOpAt(.qmatmul, ops, start) orelse return false;
        const first = deviceOpAt(.fused_elementwise, ops, start + 1) orelse return false;
        const rp = deviceOpAt(.repeat, ops, start + 2) orelse return false;
        const second = deviceOpAt(.fused_elementwise, ops, start + 3) orelse return false;
        const up = deviceOpAt(.qmatmul, ops, start + 4) orelse return false;
        const product = deviceOpAt(.elementwise, ops, start + 5) orelse return false;
        return self.encodeQMatmulPairFusedElementwiseChain(gate, first, rp, second, up, product);
    }

    fn tryEncodeRowChainCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        const start: usize = @intCast(command.op_start);
        const rn = deviceOpAt(.rmsnorm, ops, start) orelse return false;
        const rp = deviceOpAt(.repeat, ops, start + 1) orelse return false;
        const e = deviceOpAt(.elementwise, ops, start + 2) orelse return false;
        return self.encodeRmsnormRepeatMul(rn, rp, e);
    }

    fn tryEncodeRopeChainCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        const start: usize = @intCast(command.op_start);
        const rr = deviceOpAt(.rope, ops, start) orelse return false;
        const sa = deviceOpAt(.slice_assign, ops, start + 1) orelse return false;
        if (!canFuseRopeSliceAssign(rr, sa)) return false;
        self.encodeRopeSliceAssign(rr, sa);
        return true;
    }

    const ProgramCommandLoweringFn = *const fn (*CompiledProgram, []const backend_mod.DeviceOp, program_mod.ProgramCommand) bool;
    const ProgramCommandLowering = struct {
        kind: program_mod.ProgramCommandKind,
        encode: ProgramCommandLoweringFn,
        records_run: bool = false,
    };

    const exact_program_command_lowerings = [_]ProgramCommandLowering{
        .{ .kind = .projection_group, .encode = tryEncodeProjectionCommand, .records_run = true },
        .{ .kind = .projection_cache_group, .encode = tryEncodeProjectionCacheCommand, .records_run = true },
        .{ .kind = .projection_chain, .encode = tryEncodeProjectionChainCommand },
        .{ .kind = .attention_chain, .encode = tryEncodeAttentionChainCommand },
        .{ .kind = .attention_store_chain, .encode = tryEncodeAttentionStoreChainCommand },
        .{ .kind = .attention_store_group, .encode = tryEncodeAttentionStoreGroupCommand },
        .{ .kind = .rope_attention_store_chain, .encode = tryEncodeRopeAttentionStoreChainCommand },
        .{ .kind = .rope_attention_store_group, .encode = tryEncodeRopeAttentionStoreGroupCommand },
        .{ .kind = .attention_batch, .encode = tryEncodeAttentionBatchCommand },
        .{ .kind = .attention_group, .encode = tryEncodeAttentionGroupCommand },
        .{ .kind = .rope_batch, .encode = tryEncodeRopeBatchCommand },
        .{ .kind = .rope_store_group, .encode = tryEncodeRopeStoreGroupCommand },
        .{ .kind = .movement_batch, .encode = tryEncodeMovementBatchCommand },
        .{ .kind = .movement_group, .encode = tryEncodeMovementGroupCommand },
        .{ .kind = .elementwise_batch, .encode = tryEncodeElementwiseBatchCommand },
        .{ .kind = .repeat_fused_elementwise_chain, .encode = tryEncodeRepeatFusedElementwiseChainCommand },
        .{ .kind = .projection_fused_elementwise_chain, .encode = tryEncodeProjectionFusedElementwiseChainCommand },
        .{ .kind = .projection_pair_fused_elementwise_chain, .encode = tryEncodeProjectionPairFusedElementwiseChainCommand },
        .{ .kind = .row_chain, .encode = tryEncodeRowChainCommand },
        .{ .kind = .rope_chain, .encode = tryEncodeRopeChainCommand },
    };

    comptime {
        const command_count = @typeInfo(program_mod.ProgramCommandKind).@"enum".fields.len;
        var seen = [_]bool{false} ** command_count;
        seen[@intFromEnum(program_mod.ProgramCommandKind.op)] = true;
        for (exact_program_command_lowerings) |lowering| {
            const idx = @intFromEnum(lowering.kind);
            if (seen[idx]) @compileError("duplicate exact Metal command lowering: " ++ @tagName(lowering.kind));
            seen[idx] = true;
        }
        for (@typeInfo(program_mod.ProgramCommandKind).@"enum".fields) |field| {
            if (!seen[field.value]) @compileError("missing exact Metal command lowering: " ++ field.name);
        }
    }

    fn exactProgramCommandLowering(kind: program_mod.ProgramCommandKind) ?ProgramCommandLowering {
        inline for (exact_program_command_lowerings) |lowering| {
            if (lowering.kind == kind) return lowering;
        }
        return null;
    }

    fn recordExactProgramCommandRun(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand, lowering: ProgramCommandLowering, elapsed_ns: u64) void {
        if (lowering.records_run) return;
        self.recordRegionFusedRunFromCommand(ops, command, elapsed_ns);
    }

    fn tryEncodeExactProgramCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        const start: usize = @intCast(command.op_start);
        const end = start + @as(usize, command.op_count);
        if (end > ops.len) return false;

        self.runtime_profile.recordProgramCommandAttempt(command.kind);
        const t0 = nowNs();
        const lowering = exactProgramCommandLowering(command.kind) orelse return false;
        const encoded = lowering.encode(self, ops, command);

        if (!encoded) {
            self.runtime_profile.recordProgramCommandFailed(command.kind);
            return false;
        }
        self.recordExactProgramCommandRun(ops, command, lowering, @intCast(nowNs() - t0));
        self.runtime_profile.recordProgramCommand(command.kind);
        return true;
    }

    fn canEncodeRegionGpuOpDirect(self: *CompiledProgram, op: backend_mod.DeviceOp) bool {
        if (computeDispatchSpec(op) != null) return true;

        return switch (op) {
            .matmul => true,
            .qmatmul => |q| {
                if (@as(usize, q.weight_idx) >= self.qweight_views.len) return false;
                if (q.M == 1) return self.canEncodeQMatvecBatchOp(q);
                return true;
            },
            .rope => true,
            .attention => |att| canEncodeAttention(att),
            .fused_elementwise => |fe| canEncodeFusedElementwise(fe),
            else => false,
        };
    }

    fn canEncodeRegionGpuOpsDirect(self: *CompiledProgram, ops: []const backend_mod.DeviceOp) bool {
        for (ops) |op| {
            if (!self.canEncodeRegionGpuOpDirect(op)) return false;
        }
        return true;
    }

    fn canEncodeProgramCommandIndividually(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        var indices = command.coveredIndexIterator();
        while (indices.next()) |idx| {
            if (idx >= ops.len or !self.canEncodeRegionGpuOpDirect(ops[idx])) return false;
        }
        return true;
    }

    fn canEncodeProgramCommand(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        return self.canEncodeProgramCommandIndividually(ops, command);
    }

    fn canEncodeProgramCommands(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, commands: []const program_mod.ProgramCommand) bool {
        for (commands) |command| {
            if (!self.canEncodeProgramCommand(ops, command)) return false;
        }
        return true;
    }

    fn tryEncodeProgramCommandIndividually(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, command: program_mod.ProgramCommand) bool {
        var indices = command.coveredIndexIterator();
        while (indices.next()) |idx| {
            if (idx >= ops.len) return false;
            if (!self.tryEncodeRegionGpuOp(ops[idx])) return false;
        }
        return true;
    }

    fn tryEncodeRegionGpuCommands(self: *CompiledProgram, ops: []const backend_mod.DeviceOp, commands: []const program_mod.ProgramCommand) bool {
        if (ops.len > 256) return false;
        if (!self.canEncodeProgramCommands(ops, commands)) return false;

        var skipped = [_]bool{false} ** 256;
        for (commands) |command| {
            self.runtime_profile.recordProgramCommandPlanned(command.kind);
            const start: usize = @intCast(command.op_start);
            if (start >= ops.len) return false;
            if (command.kind == .op) {
                if (skipped[start]) continue;
                if (!self.tryEncodeRegionGpuOp(ops[start])) return false;
                program_mod.markProgramCommandUsed(skipped[0..ops.len], command);
                continue;
            }

            if (self.tryEncodeExactProgramCommand(ops, command)) {
                program_mod.markProgramCommandUsed(skipped[0..ops.len], command);
                continue;
            }

            if (!self.tryEncodeProgramCommandIndividually(ops, command)) return false;
            program_mod.markProgramCommandUsed(skipped[0..ops.len], command);
        }
        return true;
    }

    fn tryEncodeRegionGpuOps(self: *CompiledProgram, ops: []const backend_mod.DeviceOp) bool {
        if (ops.len > 256) {
            if (!self.canEncodeRegionGpuOpsDirect(ops)) return false;
            for (ops) |op| {
                if (!self.tryEncodeRegionGpuOp(op)) return false;
            }
            return true;
        }

        const commands = program_mod.buildProgramCommands(self.alloc, ops, self.backend.commandStreamPolicy()) catch return false;
        defer self.alloc.free(commands);
        return self.tryEncodeRegionGpuCommands(ops, commands);
    }

    const RegionLoweringFn = *const fn (*CompiledProgram, program_mod.ScheduleUnit, []const program_mod.ProgramCommand) bool;
    const RegionLowering = struct {
        pattern: MetalRegionPattern,
        encode: RegionLoweringFn,
    };

    const region_lowerings = [_]RegionLowering{
        .{ .pattern = .decode_layer_stage, .encode = tryEncodeLayerStageRegion },
        .{ .pattern = .prefill_layer_stage, .encode = tryEncodeLayerStageRegion },
    };

    comptime {
        const pattern_count = @typeInfo(MetalRegionPattern).@"enum".fields.len;
        var seen = [_]bool{false} ** pattern_count;
        for (region_lowerings) |lowering| {
            const idx = @intFromEnum(lowering.pattern);
            if (seen[idx]) @compileError("duplicate Metal region lowering: " ++ @tagName(lowering.pattern));
            seen[idx] = true;
        }
        for (@typeInfo(MetalRegionPattern).@"enum".fields) |field| {
            if (!seen[field.value]) {
                @compileError("missing Metal region lowering: " ++ field.name);
            }
        }
    }

    fn regionLowering(pattern_index: u32) ?RegionLowering {
        inline for (region_lowerings) |lowering| {
            if (@intFromEnum(lowering.pattern) == pattern_index) return lowering;
        }
        return null;
    }

    fn regionCommandPlan(self: *const CompiledProgram, unit_index: usize) []const program_mod.ProgramCommand {
        return self.plan.regionCommandPlan(unit_index, self.backend.commandStreamPolicy());
    }

    fn tryEncodePatternRegion(self: *CompiledProgram, unit: program_mod.ScheduleUnit, unit_index: usize) bool {
        self.runtime_profile.recordScheduleRegionAttempt(unit);
        const lowering = regionLowering(unit.pattern_index) orelse {
            self.runtime_profile.recordScheduleRegionFailed(unit);
            return false;
        };
        const encoded = lowering.encode(self, unit, self.regionCommandPlan(unit_index));
        if (encoded) {
            self.runtime_profile.recordScheduleRegionLowered(unit);
        } else {
            self.runtime_profile.recordScheduleRegionFailed(unit);
        }
        return encoded;
    }

    fn tryEncodeLayerStageRegion(self: *CompiledProgram, unit: program_mod.ScheduleUnit, commands: []const program_mod.ProgramCommand) bool {
        const start_item: usize = @intCast(unit.start_item);
        const end_item = start_item + @as(usize, unit.item_count);
        if (end_item > self.plan.schedule.len) return false;
        const items = self.plan.schedule[start_item..end_item];

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
        const ops = self.ops[op_start..op_end];
        if (commands.len > 0) {
            self.runtime_profile.recordCachedRegionCommandPlan(commands.len);
            return self.tryEncodeRegionGpuCommands(ops, commands);
        }

        self.runtime_profile.recordDynamicRegionCommandPlan();
        return self.tryEncodeRegionGpuOps(ops);
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
                self.encodeKernel(.matmul_f32, &buffers, matmulParams(m.geom), 3, matmulGrid(m.geom.M, m.geom.N), MATMUL_THREADS);
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
                self.encodeKernel(.qmatmul_f32, &buffers, qmatmulParams(q, w.block_size), 4, matmulGrid(q.M, q.N), MATMUL_THREADS);
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
        if (self.plan.shapeMatches(ops, policy)) {
            self.ops = ops;
            self.runtime_profile.schedule_reuse_count +%= 1;
            return;
        }

        const plan = buildMetalExecutionPlan(
            self.alloc,
            ops,
            policy,
            self.backend.commandStreamPolicy(),
        ) catch {
            self.plan.deinit(self.alloc);
            self.ops = ops;
            self.plan = .{};
            self.runtime_profile.schedule_rebuild_count +%= 1;
            return;
        };

        self.plan.deinit(self.alloc);
        self.ops = ops;
        self.plan = plan;
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

    const plan = buildMetalExecutionPlan(
        alloc,
        program.ops,
        metalSchedulePolicy(self.fine_grained_program_dispatch),
        self.commandStreamPolicy(),
    ) catch return null;
    errdefer plan.deinit(alloc);

    const compiled = alloc.create(CompiledProgram) catch return null;
    compiled.* = .{
        .backend = self,
        .device_bufs = device_bufs,
        .ref_buffers = ref_buffers,
        .qweight_views = qweight_views,
        .ref_qweights = ref_qweights,
        .ops = program.ops,
        .plan = plan,
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

test "metal backend capabilities reflect projection rope cache sidecar knob" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();

    try std.testing.expect(!metal.backend().capabilities.command_stream.projection_rope_cache_sidecars);
    metal.setProjectionRopeCacheSidecars(true);
    try std.testing.expect(metal.backend().capabilities.command_stream.projection_rope_cache_sidecars);
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
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_regions.attempted);
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_regions.lowered);
    try std.testing.expectEqual(@as(u64, 0), rt.schedule_regions.failed);
    try std.testing.expectEqual(@as(u64, 7), rt.schedule_regions.lowered_ops);
    try std.testing.expectEqual(@as(u64, 1), rt.region_command_plan_cached_count);
    try std.testing.expectEqual(@as(u64, 0), rt.region_command_plan_dynamic_count);
    try std.testing.expect(rt.region_command_plan_cached_command_count > 0);
    const decode_pattern = MetalRegionPattern.decode_layer_stage.index();
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_region_patterns[decode_pattern].attempted);
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_region_patterns[decode_pattern].lowered);
    try std.testing.expectEqual(@as(u64, 7), rt.schedule_region_patterns[decode_pattern].lowered_ops);
}

test "metal backend qmatvec projection group carries cache store sidecars" {
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

    var ops: [8]backend_mod.DeviceOp = undefined;
    ops[0] = .{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 1,
        .N = 4,
        .K = 4,
    } };
    ops[1] = .{ .slice_assign = .{
        .dst = 8,
        .src = 1,
        .rows = 4,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 0,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 4,
        .patch_stride = 4,
    } };
    for (ops[2..], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 2),
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 4,
        } };
    }

    const buf_sizes = [_]usize{ 4, 4, 4, 4, 4, 4, 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = 4 * 4 },
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

    var got: [7][4]f32 = undefined;
    var outs: [7]backend_mod.ProgramIO = undefined;
    for (outs[0..6], 0..) |*out, i| {
        out.* = .{ .buf_idx = @intCast(i + 2), .host_ptr = @ptrCast(&got[i]), .size = 4 * 4 };
    }
    outs[6] = .{ .buf_idx = 8, .host_ptr = @ptrCast(&got[6]), .size = 4 * 4 };
    be.executeProgram(handle, &.{}, &outs);

    for (got) |row| {
        try std.testing.expectEqualSlices(f32, &input, &row);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 8), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 2), rt.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 1), rt.region_command_plan_cached_count);
}

test "metal backend qmatvec projection group carries elementwise sidecars" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4 };
    var bias = [_]f32{ 10, 20, 30, 40 };
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 4, .cols = 4, .block_size = 4 }};

    var ops: [8]backend_mod.DeviceOp = undefined;
    ops[0] = .{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 1,
        .N = 4,
        .K = 4,
    } };
    ops[1] = .{ .elementwise = .{
        .op = .add,
        .dst = 9,
        .src0 = 1,
        .src1 = 8,
        .n = 4,
    } };
    for (ops[2..], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 2),
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 4,
        } };
    }

    const buf_sizes = [_]usize{ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&bias), .size = bias.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 10,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 9, .host_ptr = @ptrCast(&got), .size = got.len * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, &got);

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 8), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 2), rt.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 1), rt.region_command_plan_cached_count);
    try std.testing.expectEqual(@as(u64, 2), rt.program_command_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_group)]);
}

test "metal backend qmatvec projection cache group carries rope store sidecars" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    metal.setProjectionRopeCacheSidecars(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4 };
    var q_out_seed = [_]f32{99} ** 4;
    var rope_out_seed = [_]f32{77} ** 4;
    var cos_sin = [_]f32{ 0, 0, 1, 1 };
    var cache = [_]f32{0} ** 4;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 4, .cols = 4, .block_size = 4 }};

    var ops: [9]backend_mod.DeviceOp = undefined;
    for (ops[0..7], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 1),
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 4,
        } };
    }
    ops[7] = .{ .rope = .{
        .dst = 8,
        .src = 7,
        .cos_sin = 9,
        .half_d = 2,
        .seq_len = 1,
        .src_off = 0,
        .dst_off = 0,
        .cs_off = 0,
        .src_rs = 1,
        .src_cs = 4,
        .cs_cs = 4,
    } };
    ops[8] = .{ .slice_assign = .{
        .dst = 10,
        .src = 8,
        .rows = 4,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 0,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 4,
        .patch_stride = 4,
    } };

    const buf_sizes = [_]usize{ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&q_out_seed), .size = q_out_seed.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&rope_out_seed), .size = rope_out_seed.len * 4 },
        .{ .buf_idx = 9, .host_ptr = @ptrCast(&cos_sin), .size = cos_sin.len * 4 },
        .{ .buf_idx = 10, .host_ptr = @ptrCast(&cache), .size = cache.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 11,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got_q: [4]f32 = undefined;
    var got_rope: [4]f32 = undefined;
    var got_cache: [4]f32 = undefined;
    const outs = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&got_q), .size = got_q.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&got_rope), .size = got_rope.len * 4 },
        .{ .buf_idx = 10, .host_ptr = @ptrCast(&got_cache), .size = got_cache.len * 4 },
    };
    be.executeProgram(handle, &.{}, &outs);

    try std.testing.expectEqualSlices(f32, &q_out_seed, &got_q);
    try std.testing.expectEqualSlices(f32, &rope_out_seed, &got_rope);
    try std.testing.expectEqualSlices(f32, &.{ -3, -4, 1, 2 }, &got_cache);

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 9), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 2), rt.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 1), rt.region_command_plan_cached_count);
}

test "metal backend qmatvec projection cache group materializes rope sidecars" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    metal.setProjectionRopeCacheSidecars(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4 };
    var q_out_seed = [_]f32{99} ** 4;
    var rope_out_seed = [_]f32{77} ** 4;
    var cos_sin = [_]f32{ 0, 0, 1, 1 };
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 4, .cols = 4, .block_size = 4 }};

    var ops: [8]backend_mod.DeviceOp = undefined;
    ops[0] = .{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 1,
        .N = 4,
        .K = 4,
    } };
    ops[1] = .{ .rope = .{
        .dst = 8,
        .src = 1,
        .cos_sin = 9,
        .half_d = 2,
        .seq_len = 1,
        .src_off = 0,
        .dst_off = 0,
        .cs_off = 0,
        .src_rs = 1,
        .src_cs = 4,
        .cs_cs = 4,
    } };
    for (ops[2..], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 2),
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 4,
        } };
    }

    const buf_sizes = [_]usize{ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&q_out_seed), .size = q_out_seed.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&rope_out_seed), .size = rope_out_seed.len * 4 },
        .{ .buf_idx = 9, .host_ptr = @ptrCast(&cos_sin), .size = cos_sin.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 10,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got_q: [4]f32 = undefined;
    var got_rope: [4]f32 = undefined;
    const outs = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&got_q), .size = got_q.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&got_rope), .size = got_rope.len * 4 },
    };
    be.executeProgram(handle, &.{}, &outs);

    try std.testing.expectEqualSlices(f32, &q_out_seed, &got_q);
    try std.testing.expectEqualSlices(f32, &.{ -3, -4, 1, 2 }, &got_rope);

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 8), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 2), rt.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 1), rt.region_command_plan_cached_count);
}

test "metal backend region lowers mixed direct ops without fallback" {
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

    var ops: [8]backend_mod.DeviceOp = undefined;
    for (ops[0..7], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 1),
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 4,
        } };
    }
    ops[7] = .{ .elementwise = .{
        .op = .add,
        .dst = 8,
        .src0 = 1,
        .src1 = 2,
        .n = 4,
    } };

    const buf_sizes = [_]usize{ 4, 4, 4, 4, 4, 4, 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = 4 * 4 },
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

    var got: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 8, .host_ptr = @ptrCast(&got), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    for (got, [_]f32{ 2, 4, 6, 8 }) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 8), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_regions.attempted);
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_regions.lowered);
    try std.testing.expectEqual(@as(u64, 0), rt.schedule_regions.failed);
    try std.testing.expectEqual(@as(u64, 8), rt.schedule_regions.lowered_ops);
    try std.testing.expectEqual(@as(u64, 1), rt.region_command_plan_cached_count);
    try std.testing.expectEqual(@as(u64, 0), rt.region_command_plan_dynamic_count);
    try std.testing.expect(rt.region_command_plan_cached_command_count > 0);
    const decode_pattern = MetalRegionPattern.decode_layer_stage.index();
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_region_patterns[decode_pattern].attempted);
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_region_patterns[decode_pattern].lowered);
    try std.testing.expectEqual(@as(u64, 0), rt.schedule_region_patterns[decode_pattern].failed);
    try std.testing.expectEqual(@as(u64, 8), rt.schedule_region_patterns[decode_pattern].lowered_ops);
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
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_regions.attempted);
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_regions.lowered);
    try std.testing.expectEqual(@as(u64, 0), rt.schedule_regions.failed);
    try std.testing.expectEqual(@as(u64, 7), rt.schedule_regions.lowered_ops);
    try std.testing.expectEqual(@as(u64, 1), rt.region_command_plan_cached_count);
    try std.testing.expectEqual(@as(u64, 0), rt.region_command_plan_dynamic_count);
    try std.testing.expect(rt.region_command_plan_cached_command_count > 0);
    const prefill_pattern = MetalRegionPattern.prefill_layer_stage.index();
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_region_patterns[prefill_pattern].attempted);
    try std.testing.expectEqual(@as(u64, 1), rt.schedule_region_patterns[prefill_pattern].lowered);
    try std.testing.expectEqual(@as(u64, 7), rt.schedule_region_patterns[prefill_pattern].lowered_ops);
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

test "metal backend opt-in qmatmul rope cache store sidecar" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setFineGrainedProgramDispatch(true);
    metal.setRegionProgramDispatch(true);
    metal.setProjectionRopeCacheSidecars(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2 };
    var q_out_seed = [_]f32{99} ** 128;
    var rope_out_seed = [_]f32{77} ** 128;
    var cos_sin = [_]f32{0} ** 128;
    for (0..2) |row| {
        const base = row * 64;
        for (0..32) |i| cos_sin[base + i] = 1;
    }
    var cache = [_]f32{0} ** 128;
    var qdata: [64]i8 = undefined;
    for (&qdata, 0..) |*v, i| v.* = @intCast(i + 1);
    const scales = [_]f32{1};
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 1, .cols = 64, .block_size = 64 }};

    var ops: [9]backend_mod.DeviceOp = undefined;
    for (ops[0..7], 0..) |*op, i| {
        op.* = .{ .qmatmul = .{
            .dst = @intCast(i + 1),
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 64,
            .K = 1,
        } };
    }
    ops[7] =
        .{ .rope = .{
            .dst = 8,
            .src = 7,
            .cos_sin = 9,
            .half_d = 32,
            .seq_len = 2,
            .src_off = 0,
            .dst_off = 0,
            .cs_off = 0,
            .src_rs = 1,
            .src_cs = 64,
            .cs_cs = 64,
        } };
    ops[8] =
        .{ .slice_assign = .{
            .dst = 10,
            .src = 8,
            .rows = 64,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 64,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 64,
            .patch_stride = 64,
        } };

    const buf_sizes = [_]usize{ 2, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&q_out_seed), .size = q_out_seed.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&rope_out_seed), .size = rope_out_seed.len * 4 },
        .{ .buf_idx = 9, .host_ptr = @ptrCast(&cos_sin), .size = cos_sin.len * 4 },
        .{ .buf_idx = 10, .host_ptr = @ptrCast(&cache), .size = cache.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 11,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got_q: [128]f32 = undefined;
    var got_rope: [128]f32 = undefined;
    var got_cache: [128]f32 = undefined;
    const outs = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&got_q), .size = got_q.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&got_rope), .size = got_rope.len * 4 },
        .{ .buf_idx = 10, .host_ptr = @ptrCast(&got_cache), .size = got_cache.len * 4 },
    };
    be.executeProgram(handle, &.{}, &outs);

    for (got_q) |v| try std.testing.expectEqual(@as(f32, 99), v);
    for (got_rope) |v| try std.testing.expectEqual(@as(f32, 77), v);
    for (0..2) |row| {
        for (0..64) |col| {
            const expected: f32 = @floatFromInt((row + 1) * (col + 1));
            try std.testing.expectEqual(expected, got_cache[row * 64 + col]);
        }
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 9), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 3), rt.backend_dispatch_count);
}

test "metal backend qmatmul batch carries elementwise sidecars" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var bias = [_]f32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var sidecar_out = [_]f32{0} ** 8;
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
    ops[7] = .{ .elementwise = .{
        .op = .add,
        .dst = 8,
        .src0 = 7,
        .src1 = 9,
        .n = 8,
    } };

    const buf_sizes = [_]usize{ 6, 8, 8, 8, 8, 8, 8, 8, 8, 8 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&sidecar_out), .size = sidecar_out.len * 4 },
        .{ .buf_idx = 9, .host_ptr = @ptrCast(&bias), .size = bias.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 10,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [8]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 8, .host_ptr = @ptrCast(&got), .size = got.len * 4 }};
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 40, 54, 65, 76, 80 }, &got);

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

test "metal qmatmul sidecar elides dead primary output" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var q_out = [_]f32{99} ** 8;
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
    ops[7] = .{ .elementwise = .{
        .op = .add,
        .dst = 9,
        .src0 = 7,
        .src1 = 8,
        .n = 8,
    } };

    const buf_sizes = [_]usize{ 6, 8, 8, 8, 8, 8, 8, 8, 8, 8 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&q_out), .size = q_out.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&residual), .size = residual.len * 4 },
        .{ .buf_idx = 9, .host_ptr = @ptrCast(&fused_out), .size = fused_out.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 10,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got_q: [8]f32 = undefined;
    var got_fused: [8]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&got_q), .size = got_q.len * 4 },
        .{ .buf_idx = 9, .host_ptr = @ptrCast(&got_fused), .size = got_fused.len * 4 },
    };
    be.executeProgram(handle, &.{}, &out);

    try std.testing.expectEqualSlices(f32, &.{ 99, 99, 99, 99, 99, 99, 99, 99 }, &got_q);
    try std.testing.expectEqualSlices(f32, &.{ 11, 12, 13, 10, 14, 15, 16, 10 }, &got_fused);

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 8), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 2), rt.backend_dispatch_count);
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

test "metal backend region fuses qmatmul activation expression chain" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var filler = [_]f32{0} ** 8;
    var q_out = [_]f32{0} ** 8;
    var exp_tmp = [_]f32{0} ** 8;
    var one = [_]f32{1};
    var repeat_out = [_]f32{0} ** 8;
    var silu_out = [_]f32{0} ** 8;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 4, .cols = 4, .block_size = 4 }};
    const exp_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const silu_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 7, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 5, .secondary_offset = 0 },
    };

    var ops: [10]backend_mod.DeviceOp = undefined;
    for (ops[0..4]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 4,
        } };
    }
    ops[4] = .{ .qmatmul = .{
        .dst = 5,
        .input = 0,
        .weight_idx = 0,
        .M = 2,
        .N = 4,
        .K = 4,
    } };
    ops[5] = .{ .fused_elementwise = .{
        .steps = &exp_steps,
        .n = 8,
        .dst = 6,
        .src = 5,
        .dst_offset = 0,
        .src_offset = 0,
    } };
    ops[6] = .{ .repeat = .{
        .dst = 7,
        .src = 8,
        .n = 8,
        .src_ne = .{ 1, 1, 1, 1 },
        .dst_ne = .{ 4, 2, 1, 1 },
        .src_strides = .{ 1, 1, 1, 1 },
        .dst_strides = .{ 1, 4, 8, 8 },
    } };
    ops[7] = .{ .fused_elementwise = .{
        .steps = &silu_steps,
        .n = 8,
        .dst = 9,
        .src = 6,
        .dst_offset = 0,
        .src_offset = 0,
    } };
    for (ops[8..10]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 4,
        } };
    }

    const buf_sizes = [_]usize{ 8, 8, 8, 8, 8, 8, 8, 8, 1, 8 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&filler), .size = filler.len * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&q_out), .size = q_out.len * 4 },
        .{ .buf_idx = 6, .host_ptr = @ptrCast(&exp_tmp), .size = exp_tmp.len * 4 },
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&repeat_out), .size = repeat_out.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&one), .size = one.len * 4 },
        .{ .buf_idx = 9, .host_ptr = @ptrCast(&silu_out), .size = silu_out.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 10,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [8]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 9, .host_ptr = @ptrCast(&got), .size = got.len * 4 }};
    be.executeProgram(handle, &.{}, &out);

    for (input, got) |x, actual| {
        const want = x / (1.0 + @exp(-x));
        try std.testing.expectApproxEqAbs(want, actual, 1e-4);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 10), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
}

test "metal backend region fuses paired qmatmul activation product chain" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var input = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var filler = [_]f32{0} ** 8;
    var shared_q_out = [_]f32{0} ** 8;
    var exp_tmp = [_]f32{0} ** 8;
    var one = [_]f32{1};
    var repeat_out = [_]f32{0} ** 8;
    var silu_out = [_]f32{0} ** 8;
    var product_out = [_]f32{0} ** 8;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 4, .cols = 4, .block_size = 4 }};
    const exp_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const silu_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 7, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 5, .secondary_offset = 0 },
    };

    var ops: [11]backend_mod.DeviceOp = undefined;
    ops[0] = .{ .qmatmul = .{
        .dst = 5,
        .input = 0,
        .weight_idx = 0,
        .M = 2,
        .N = 4,
        .K = 4,
    } };
    ops[1] = .{ .fused_elementwise = .{
        .steps = &exp_steps,
        .n = 8,
        .dst = 6,
        .src = 5,
        .dst_offset = 0,
        .src_offset = 0,
    } };
    ops[2] = .{ .repeat = .{
        .dst = 7,
        .src = 8,
        .n = 8,
        .src_ne = .{ 1, 1, 1, 1 },
        .dst_ne = .{ 4, 2, 1, 1 },
        .src_strides = .{ 1, 1, 1, 1 },
        .dst_strides = .{ 1, 4, 8, 8 },
    } };
    ops[3] = .{ .fused_elementwise = .{
        .steps = &silu_steps,
        .n = 8,
        .dst = 9,
        .src = 6,
        .dst_offset = 0,
        .src_offset = 0,
    } };
    ops[4] = .{ .qmatmul = .{
        .dst = 5,
        .input = 0,
        .weight_idx = 0,
        .M = 2,
        .N = 4,
        .K = 4,
    } };
    ops[5] = .{ .elementwise = .{
        .op = .mul,
        .dst = 10,
        .src0 = 9,
        .src1 = 5,
        .n = 8,
    } };
    for (ops[6..11]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 4,
            .K = 4,
        } };
    }

    const buf_sizes = [_]usize{ 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&input), .size = input.len * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&filler), .size = filler.len * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&shared_q_out), .size = shared_q_out.len * 4 },
        .{ .buf_idx = 6, .host_ptr = @ptrCast(&exp_tmp), .size = exp_tmp.len * 4 },
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&repeat_out), .size = repeat_out.len * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&one), .size = one.len * 4 },
        .{ .buf_idx = 9, .host_ptr = @ptrCast(&silu_out), .size = silu_out.len * 4 },
        .{ .buf_idx = 10, .host_ptr = @ptrCast(&product_out), .size = product_out.len * 4 },
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = 11,
        .buffer_sizes = &buf_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const handle = be.compileProgram(program) orelse return error.CompileFailed;
    defer be.freeProgram(handle);

    var got: [8]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 10, .host_ptr = @ptrCast(&got), .size = got.len * 4 }};
    be.executeProgram(handle, &.{}, &out);

    for (input, got) |x, actual| {
        const want = x * x / (1.0 + @exp(-x));
        try std.testing.expectApproxEqAbs(want, actual, 1e-4);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 11), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 1), rt.program_command_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_pair_fused_elementwise_chain)]);
}

test "metal backend region fuses repeated secondary into fused elementwise" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var q_input = [_]f32{ 1, 2, 3, 4 };
    var q_out = [_]f32{0} ** 4;
    var one = [_]f32{1};
    var x = [_]f32{ 1, 3, 7, 15 };
    var repeat_out = [_]f32{0} ** 4;
    var fused_out = [_]f32{0} ** 4;
    const qdata = [_]i8{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 4, .cols = 4, .block_size = 4 }};
    const fused_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 3, .secondary_offset = 0 },
    };

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
    ops[7] = .{ .repeat = .{
        .dst = 4,
        .src = 2,
        .n = 4,
        .src_ne = .{ 1, 1, 1, 1 },
        .dst_ne = .{ 4, 1, 1, 1 },
        .src_strides = .{ 1, 1, 1, 1 },
        .dst_strides = .{ 1, 4, 4, 4 },
    } };
    ops[8] = .{ .fused_elementwise = .{
        .steps = &fused_steps,
        .n = 4,
        .dst = 5,
        .src = 3,
        .dst_offset = 0,
        .src_offset = 0,
    } };

    const buf_sizes = [_]usize{ 4, 4, 1, 4, 4, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&q_input), .size = q_input.len * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&q_out), .size = q_out.len * 4 },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&one), .size = one.len * 4 },
        .{ .buf_idx = 3, .host_ptr = @ptrCast(&x), .size = x.len * 4 },
        .{ .buf_idx = 4, .host_ptr = @ptrCast(&repeat_out), .size = repeat_out.len * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&fused_out), .size = fused_out.len * 4 },
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
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 5, .host_ptr = @ptrCast(&got), .size = got.len * 4 }};
    be.executeProgram(handle, &.{}, &out);

    const expected = [_]f32{ 0.5, 0.75, 0.875, 0.9375 };
    for (expected, got) |want, actual| {
        try std.testing.expectApproxEqAbs(want, actual, 1e-5);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 9), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 8), rt.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 1), rt.program_command_counts[@intFromEnum(program_mod.ProgramCommandKind.repeat_fused_elementwise_chain)]);
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

test "metal backend fuses attention output store chains" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var q_input = [_]f32{ 1, 0, 0, 1 };
    var q_out = [_]f32{0} ** 4;
    var k_cache = [_]f32{ 1, 0, 0, 1 };
    var v_cache = [_]f32{ 10, 20, 30, 40 };
    var mask = [_]f32{ 0, 0 };
    var dst = [_]f32{0} ** 2;
    var slice_dst = [_]f32{ -1, -1, -1, -1 };
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
    ops[8] = .{ .slice_assign = .{
        .dst = 6,
        .src = 5,
        .rows = 2,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 2,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 2,
        .patch_stride = 4,
    } };

    const buf_sizes = [_]usize{ 4, 4, 4, 4, 2, 2, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&q_input), .size = 4 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&q_out), .size = 4 * 4 },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&k_cache), .size = 4 * 4 },
        .{ .buf_idx = 3, .host_ptr = @ptrCast(&v_cache), .size = 4 * 4 },
        .{ .buf_idx = 4, .host_ptr = @ptrCast(&mask), .size = 2 * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&dst), .size = 2 * 4 },
        .{ .buf_idx = 6, .host_ptr = @ptrCast(&slice_dst), .size = 4 * 4 },
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

    var got_dst: [2]f32 = undefined;
    var got_slice: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&got_dst), .size = 2 * 4 },
        .{ .buf_idx = 6, .host_ptr = @ptrCast(&got_slice), .size = 4 * 4 },
    };
    be.executeProgram(handle, &.{}, &out);

    const hi: f32 = @exp(@as(f32, 1.0)) / (@exp(@as(f32, 1.0)) + 1.0);
    const lo: f32 = 1.0 / (@exp(@as(f32, 1.0)) + 1.0);
    const expected = [_]f32{
        hi * 10 + lo * 30,
        hi * 20 + lo * 40,
    };
    for (expected, got_dst) |want, actual| {
        try std.testing.expectApproxEqAbs(want, actual, 1e-4);
    }
    try std.testing.expectEqual(@as(f32, -1), got_slice[0]);
    try std.testing.expectEqual(@as(f32, -1), got_slice[1]);
    for (expected, got_slice[2..4]) |want, actual| {
        try std.testing.expectApproxEqAbs(want, actual, 1e-4);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 9), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 8), rt.backend_dispatch_count);
}

test "metal backend fuses rope attention output store chains" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var q_input = [_]f32{ 1, 0 };
    var q_rope = [_]f32{0} ** 2;
    var cos_sin = [_]f32{ 1, 0 };
    var k_cache = [_]f32{ 1, 0, 0, 1 };
    var v_cache = [_]f32{ 10, 20, 30, 40 };
    var mask = [_]f32{ 0, 0 };
    var dst = [_]f32{0} ** 2;
    var slice_dst = [_]f32{ -1, -1, -1, -1 };
    const qdata = [_]i8{
        1, 0,
        0, 1,
    };
    const scales = [_]f32{ 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 2, .cols = 2, .block_size = 2 }};

    var ops: [10]backend_mod.DeviceOp = undefined;
    for (ops[0..7]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 2,
            .K = 2,
        } };
    }
    ops[7] = .{ .rope = .{
        .dst = 1,
        .src = 0,
        .cos_sin = 2,
        .half_d = 1,
        .seq_len = 1,
        .src_off = 0,
        .cs_off = 0,
        .dst_off = 0,
        .src_rs = 1,
        .src_cs = 2,
        .cs_cs = 1,
    } };
    ops[8] = .{ .attention = .{
        .dst = 6,
        .q = 1,
        .k = 3,
        .v = 4,
        .mask = 5,
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
    ops[9] = .{ .slice_assign = .{
        .dst = 7,
        .src = 6,
        .rows = 2,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 2,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 2,
        .patch_stride = 4,
    } };

    const buf_sizes = [_]usize{ 2, 2, 2, 4, 4, 2, 2, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&q_input), .size = 2 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&q_rope), .size = 2 * 4 },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&cos_sin), .size = 2 * 4 },
        .{ .buf_idx = 3, .host_ptr = @ptrCast(&k_cache), .size = 4 * 4 },
        .{ .buf_idx = 4, .host_ptr = @ptrCast(&v_cache), .size = 4 * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&mask), .size = 2 * 4 },
        .{ .buf_idx = 6, .host_ptr = @ptrCast(&dst), .size = 2 * 4 },
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&slice_dst), .size = 4 * 4 },
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

    var got_dst: [2]f32 = undefined;
    var got_slice: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 6, .host_ptr = @ptrCast(&got_dst), .size = 2 * 4 },
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&got_slice), .size = 4 * 4 },
    };
    be.executeProgram(handle, &.{}, &out);

    const hi: f32 = @exp(@as(f32, 1.0)) / (@exp(@as(f32, 1.0)) + 1.0);
    const lo: f32 = 1.0 / (@exp(@as(f32, 1.0)) + 1.0);
    const expected = [_]f32{
        hi * 10 + lo * 30,
        hi * 20 + lo * 40,
    };
    for (expected, got_dst) |want, actual| {
        try std.testing.expectApproxEqAbs(want, actual, 1e-4);
    }
    try std.testing.expectEqual(@as(f32, -1), got_slice[0]);
    try std.testing.expectEqual(@as(f32, -1), got_slice[1]);
    for (expected, got_slice[2..4]) |want, actual| {
        try std.testing.expectApproxEqAbs(want, actual, 1e-4);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 1), rt.program_command_counts[@intFromEnum(program_mod.ProgramCommandKind.rope_attention_store_chain)]);
}

test "metal backend groups rope slice stores" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var src0 = [_]f32{ 1, 0 };
    var src1 = [_]f32{ 0, 1 };
    var cos_sin = [_]f32{ 1, 0 };
    var dst = [_]f32{ -1, -1, -1, -1 };
    var scratch = [_]f32{ -9, -9 };
    var q_out = [_]f32{0} ** 2;
    const qdata = [_]i8{
        1, 0,
        0, 1,
    };
    const scales = [_]f32{ 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 2, .cols = 2, .block_size = 2 }};

    var ops: [11]backend_mod.DeviceOp = undefined;
    for (ops[0..7]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 4,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 2,
            .K = 2,
        } };
    }
    ops[7] = .{ .rope = .{
        .dst = 3,
        .src = 0,
        .cos_sin = 1,
        .half_d = 1,
        .seq_len = 1,
        .src_off = 0,
        .cs_off = 0,
        .dst_off = 0,
        .src_rs = 1,
        .src_cs = 2,
        .cs_cs = 1,
    } };
    ops[8] = .{ .slice_assign = .{
        .dst = 2,
        .src = 3,
        .rows = 2,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 0,
        .dst_row_stride = 1,
        .dst_col_stride = 2,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 2,
        .patch_stride = 2,
    } };
    ops[9] = .{ .rope = .{
        .dst = 3,
        .src = 5,
        .cos_sin = 1,
        .half_d = 1,
        .seq_len = 1,
        .src_off = 0,
        .cs_off = 0,
        .dst_off = 0,
        .src_rs = 1,
        .src_cs = 2,
        .cs_cs = 1,
    } };
    ops[10] = .{ .slice_assign = .{
        .dst = 2,
        .src = 3,
        .rows = 2,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 2,
        .dst_row_stride = 1,
        .dst_col_stride = 2,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 2,
        .patch_stride = 2,
    } };
    const buf_sizes = [_]usize{ 2, 2, 4, 2, 2, 2 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&src0), .size = 2 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&cos_sin), .size = 2 * 4 },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&dst), .size = 4 * 4 },
        .{ .buf_idx = 3, .host_ptr = @ptrCast(&scratch), .size = 2 * 4 },
        .{ .buf_idx = 4, .host_ptr = @ptrCast(&q_out), .size = 2 * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&src1), .size = 2 * 4 },
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
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 2, .host_ptr = @ptrCast(&got), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    const expected = [_]f32{ 1, 0, 0, 1 };
    for (expected, got) |want, actual| {
        try std.testing.expectApproxEqAbs(want, actual, 1e-5);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 1), rt.program_command_counts[@intFromEnum(program_mod.ProgramCommandKind.rope_store_group)]);
}

test "metal backend groups rope attention output stores with aliased scratch" {
    var metal = MetalBackend.init() catch |err| switch (err) {
        error.MetalNotAvailable => return,
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);
    const be = metal.backend();

    var q0 = [_]f32{ 1, 0 };
    var q1 = [_]f32{ 0, 1 };
    var cos_sin = [_]f32{ 1, 0 };
    var k_cache = [_]f32{ 1, 0, 0, 1 };
    var v_cache = [_]f32{ 10, 20, 30, 40 };
    var mask = [_]f32{ 0, 0 };
    var scratch = [_]f32{ -9, -9 };
    var slice_dst = [_]f32{ -1, -1, -1, -1 };
    var unused = [_]f32{0} ** 2;
    const qdata = [_]i8{
        1, 0,
        0, 1,
    };
    const scales = [_]f32{ 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{ .data = &qdata, .scales = &scales, .rows = 2, .cols = 2, .block_size = 2 }};

    var ops: [13]backend_mod.DeviceOp = undefined;
    for (ops[0..7]) |*op| {
        op.* = .{ .qmatmul = .{
            .dst = 8,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 2,
            .K = 2,
        } };
    }
    ops[7] = .{ .rope = .{
        .dst = 6,
        .src = 0,
        .cos_sin = 2,
        .half_d = 1,
        .seq_len = 1,
        .src_off = 0,
        .cs_off = 0,
        .dst_off = 0,
        .src_rs = 1,
        .src_cs = 2,
        .cs_cs = 1,
    } };
    ops[8] = .{ .attention = .{
        .dst = 6,
        .q = 6,
        .k = 3,
        .v = 4,
        .mask = 5,
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
    ops[9] = .{ .slice_assign = .{
        .dst = 7,
        .src = 6,
        .rows = 2,
        .cols = 1,
        .dst_base_offset = 0,
        .dst_offset = 0,
        .dst_row_stride = 1,
        .dst_col_stride = 4,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 2,
        .patch_stride = 4,
    } };
    ops[10] = .{ .rope = .{
        .dst = 6,
        .src = 1,
        .cos_sin = 2,
        .half_d = 1,
        .seq_len = 1,
        .src_off = 0,
        .cs_off = 0,
        .dst_off = 0,
        .src_rs = 1,
        .src_cs = 2,
        .cs_cs = 1,
    } };
    ops[11] = ops[8];
    ops[12] = ops[9];
    ops[12].slice_assign.dst_offset = 2;

    const buf_sizes = [_]usize{ 2, 2, 2, 4, 4, 2, 2, 4, 2 };
    const uploads = [_]backend_mod.ProgramIO{
        .{ .buf_idx = 0, .host_ptr = @ptrCast(&q0), .size = 2 * 4 },
        .{ .buf_idx = 1, .host_ptr = @ptrCast(&q1), .size = 2 * 4 },
        .{ .buf_idx = 2, .host_ptr = @ptrCast(&cos_sin), .size = 2 * 4 },
        .{ .buf_idx = 3, .host_ptr = @ptrCast(&k_cache), .size = 4 * 4 },
        .{ .buf_idx = 4, .host_ptr = @ptrCast(&v_cache), .size = 4 * 4 },
        .{ .buf_idx = 5, .host_ptr = @ptrCast(&mask), .size = 2 * 4 },
        .{ .buf_idx = 6, .host_ptr = @ptrCast(&scratch), .size = 2 * 4 },
        .{ .buf_idx = 7, .host_ptr = @ptrCast(&slice_dst), .size = 4 * 4 },
        .{ .buf_idx = 8, .host_ptr = @ptrCast(&unused), .size = 2 * 4 },
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

    var got_slice: [4]f32 = undefined;
    var out = [_]backend_mod.ProgramIO{.{ .buf_idx = 7, .host_ptr = @ptrCast(&got_slice), .size = 4 * 4 }};
    be.executeProgram(handle, &.{}, &out);

    const hi: f32 = @exp(@as(f32, 1.0)) / (@exp(@as(f32, 1.0)) + 1.0);
    const lo: f32 = 1.0 / (@exp(@as(f32, 1.0)) + 1.0);
    const expected = [_]f32{
        hi * 10 + lo * 30,
        hi * 20 + lo * 40,
        lo * 10 + hi * 30,
        lo * 20 + hi * 40,
    };
    for (expected, got_slice) |want, actual| {
        try std.testing.expectApproxEqAbs(want, actual, 1e-4);
    }

    const rt = be.getRuntimeProfile(handle).?;
    try std.testing.expectEqual(@as(u64, 0), rt.fallback_op_count);
    try std.testing.expectEqual(@as(u64, 1), rt.program_command_counts[@intFromEnum(program_mod.ProgramCommandKind.rope_attention_store_group)]);
}
